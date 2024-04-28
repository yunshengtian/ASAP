import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import numpy as np
import random
import networkx as nx
from time import time
import traceback

from plan_sequence.planner.base import SequencePlanner
from plan_sequence.stable_pose import get_combined_mesh, get_stable_poses


class RandomSequencePlanner(SequencePlanner):

    def _update_tree_recursive(self, tree, parts_parent): # update n_gripper accordingly
        for parts_child in tree.successors(tuple(parts_parent)):
            edge_info = tree.edges[tuple(parts_parent), tuple(parts_child)]['sim_info']
            if edge_info['feasible']:
                child_n_gripper, parent_n_gripper = tree.nodes[tuple(parts_child)]['n_gripper'], tree.nodes[tuple(parts_parent)]['n_gripper']
                child_n_gripper_new = max(parent_n_gripper, len(edge_info['parts_fix']) + 1)
                if child_n_gripper is None or child_n_gripper_new < child_n_gripper:
                    tree.nodes[tuple(parts_child)]['n_gripper'] = child_n_gripper_new
                    self._update_tree_recursive(tree, parts_child) # recursively update children

    def _update_tree(self, tree, parts_parent, parts_child, n_eval, sim_info):
        if tree.has_node(tuple(parts_child)):
            assert tree.nodes[tuple(parts_child)]['n_eval'] < n_eval,  f'[error] child {parts_child} incorrectly updated'
            if sim_info['feasible']:
                child_n_gripper, parent_n_gripper = tree.nodes[tuple(parts_child)]['n_gripper'], tree.nodes[tuple(parts_parent)]['n_gripper']
                if parent_n_gripper is not None:
                    child_n_gripper_new = max(parent_n_gripper, len(sim_info['parts_fix']) + 1)
                    if child_n_gripper is None or child_n_gripper_new < child_n_gripper: # found better plan
                        tree.nodes[tuple(parts_child)]['n_gripper'] = child_n_gripper_new
                        self._update_tree_recursive(tree, parts_child) # recursively update children
                for pose in tree.nodes[tuple(parts_child)]['poses']: # check if pose is already in node attr
                    if sim_info['pose'] is None:
                        if pose is None:
                            break # both None -> in node attr
                    else:
                        if pose is not None and np.allclose(pose, sim_info['pose']):
                            break # both not None and allclose -> in node attr
                else:
                    tree.nodes[tuple(parts_child)]['poses'].insert(0, sim_info['pose']) # not in node attr, prioritize later poses
        else:
            parent_n_gripper = tree.nodes[tuple(parts_parent)]['n_gripper']
            child_n_gripper = max(parent_n_gripper, len(sim_info['parts_fix']) + 1) if parent_n_gripper is not None and sim_info['feasible'] else None
            tree.add_node(tuple(parts_child), n_eval=n_eval, n_gripper=child_n_gripper, poses=[sim_info['pose']] if sim_info['feasible'] else [])
        tree.add_edge(tuple(parts_parent), tuple(parts_child), n_eval=n_eval, sim_info=sim_info)

    def eval_seq_fitness(self, tree, sequence, budget, max_grippers, max_poses, pose_reuse, early_term, timeout, debug=0, render=False):

        solution_found = False
        fitness = 0

        G0 = self.parts.copy()
        G = self.parts.copy()

        for p in sequence:
            G_prime = G.copy()
            G_prime.remove(p)
            if len(G_prime) <= 1: break

            parts_removed = [part for part in G0 if part not in G]

            if early_term and solution_found:
                self.stop_msg = 'solution found'
                break
            
            if budget is not None and self.n_eval >= budget:
                self.stop_msg = 'budget reached'
                break

            if self._check_fully_explored(tree, G0):
                self.stop_msg = 'tree fully explored'
                break

            if timeout is not None and (time() - self.t_start) > timeout:
                self.stop_msg = 'timeout'
                break

            poses = tree.nodes[tuple(G)]['poses'][:pose_reuse]
            G_mesh = get_combined_mesh(self.assembly_dir, G)
            poses.extend(get_stable_poses(G_mesh, max_num=max_poses - pose_reuse))
            if len(poses) == 0:
                poses = [None]
            # poses = [None]

            if not tree.has_edge(tuple(G), tuple(G_prime)):
                for pose in poses:

                    sim_timeout = None if timeout is None else timeout - (time() - self.t_start) # allocate for this step of simulation
                    if sim_timeout is not None and sim_timeout < 0:
                        break

                    sim_info = self._simulate(p, G_prime, parts_removed, pose, max_grippers=max_grippers, timeout=sim_timeout, debug=debug - 1, render=render)
                    self.n_eval += 1
                    self._update_tree(tree, G, G_prime, self.n_eval, sim_info)

                    if self.find_sequence(tree, leaf_size=2) is not None:
                        solution_found = True

                    if sim_info['feasible']: # NOTE: edge color issue (feasble child link but infeasible parent link -> red)
                        fitness += 1
                        break

                if debug > 0:
                    print(f'[planner.random.plan] add edge: ({G}, {G_prime}), feasible: {sim_info["feasible"]}')
                    print(f'[planner.random.plan] progress: {self.n_eval}/{budget} evaluations')
            else:
                if tree.edges[tuple(G), tuple(G_prime)]['sim_info']['feasible']:
                    fitness += 1
            
            G = G_prime.copy()

        # self.plot_tree(tree)

        return fitness

    def plan(self, budget, max_grippers, max_poses=3, pose_reuse=0, early_term=False, timeout=None, debug=0, render=False):
        '''
        Main planning function
        Input:
            budget: max # simulations allowed
            max_grippers: max # grippers allowed to use
        Output:
            tree: disassembly tree including all disassembly attempts
        '''
        self.t_start = time()
        self.stop_msg = None
        assert budget is not None or timeout is not None

        self._reset()

        self.n_eval = 0
        G0 = self.parts.copy()
        tree = nx.DiGraph()
        tree.add_node(tuple(G0), n_eval=0, n_gripper=1, poses=[])

        sequences = []

        try:

            while self.stop_msg is None:

                if self._check_fully_explored(tree, G0):
                    self.stop_msg = 'tree fully explored'
                    break
            
                sequence = None
                max_attempt = 1e4
                attempt_count = 0
                while sequence is None: # generate a new random sequence
                    rand_seq = np.random.permutation(self.parts)
                    for exist_seq in sequences:
                        if (rand_seq == exist_seq).all():
                            break
                    else:
                        sequence = rand_seq
                    attempt_count += 1
                    if attempt_count > max_attempt:
                        self.stop_msg = 'tree fully explored'
                        break
                
                if self.stop_msg is not None:
                    break
                
                self.eval_seq_fitness(tree, sequence, budget, max_grippers, max_poses, pose_reuse, early_term, timeout, debug, render)
            
            self._expand_leaf(tree, max_poses, pose_reuse, debug, render)
            # self.plot_tree(tree)

        except (Exception, KeyboardInterrupt) as e:
            if type(e) == KeyboardInterrupt:
                self.stop_msg = 'interrupt'
            else:
                self.stop_msg = 'exception'
                print(e, f'from {self.assembly_dir}')
                print(traceback.format_exc())

        assert self.stop_msg is not None, '[planner.base.plan] bug: unexpectedly stopped'
        if debug > 0:
            print(f'[planner.base.plan] stopped: {self.stop_msg}')

        return tree

    def _expand_leaf(self, tree, max_poses, pose_reuse, debug, render):
        for node in tree.nodes:
            assert len(node) >= 2

        node_expand_list = []
        for node in tree.nodes:
            node_info = tree.nodes[node]
            if len(node) == 2 and node_info['n_gripper'] is not None:
                node_expand_list.append(node)

        G0 = self.parts.copy()

        for node in node_expand_list:
            part_a, part_b = node
            mass_a, mass_b = self.part_mass[part_a], self.part_mass[part_b]
            part_fix, part_move = (part_a, part_b) if mass_a > mass_b else (part_b, part_a)
            if part_move == self.base_part: part_fix, part_move = part_move, part_fix
            parts_removed = [part for part in G0 if part != part_a and part != part_b]
            poses = tree.nodes[tuple(node)]['poses'][:pose_reuse]
            poses.extend(get_stable_poses(get_combined_mesh(self.assembly_dir, node), max_num=max_poses - pose_reuse))
            if len(poses) == 0 or self.base_part is not None: poses = [None]
            for pose in poses:
                sim_info = self._simulate(part_move, [part_fix], parts_removed, pose=pose, max_grippers=2, debug=debug - 1, render=render)
                if sim_info['feasible']:
                    node_info = tree.nodes[node]
                    self._update_tree(tree, list(node), [part_fix], node_info['n_eval'], sim_info)
                    break

    @staticmethod
    def find_sequence(tree, leaf_size=1):

        # find leaf node
        leaf_node = None
        for node in tree.nodes:
            node_info = tree.nodes[node]
            if len(node) == leaf_size and node_info['n_gripper'] is not None:
                leaf_node = node
                break
        else:
            return None

        # find sequence in tree from bottom to top
        sequence = []
        node = leaf_node
        while tree.in_degree(node) > 0:
            node_info = tree.nodes[node]
            for parent_node in tree.predecessors(node):
                parent_node_info = tree.nodes[parent_node]
                if parent_node_info['n_gripper'] is not None and parent_node_info['n_gripper'] <= node_info['n_gripper']:
                    part_move = tree.edges[parent_node, node]['sim_info']['part_move']
                    sequence.insert(0, part_move)
                    node = parent_node
                    break
        return sequence

    @staticmethod
    def get_stats(tree):
        success, n_eval, n_gripper = RandomSequencePlanner.check_success(tree)
        if success:
            sequence = RandomSequencePlanner.find_sequence(tree)
        else:
            sequence = None
        return {
            'success': success,
            'n_eval': n_eval,
            'n_gripper': n_gripper,
            'sequence': sequence,
        }
