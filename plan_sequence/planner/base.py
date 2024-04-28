import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import numpy as np
import random
import torch
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from time import time
import traceback

from assets.load import load_part_ids
from plan_sequence.sim_string import get_body_color_dict
from plan_sequence.physics_planner import MultiPartPathPlanner, get_body_mass
from plan_sequence.feasibility_check import check_assemblable, get_stable_plan_1pose_parallel, get_stable_plan_1pose_serial
from plan_sequence.stable_pose import get_combined_mesh, get_stable_poses
from plan_robot.run_grasp_plan import GraspPlanner
from plan_robot.run_grasp_arm_plan import GraspArmPlanner


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class SequencePlanner:
    '''
    Disassembly sequence planning (without ground, 1 direction gravity, 1 part at a time)
    '''
    def __init__(self, seq_generator, num_proc=1, save_sdf=False):
        self.seq_generator = seq_generator
        self.asset_folder = seq_generator.asset_folder
        self.assembly_dir = seq_generator.assembly_dir
        self.base_part = seq_generator.base_part
        self.parts = sorted(load_part_ids(self.assembly_dir))
        assert len(self.parts) >= 2
        self.num_proc = num_proc
        self.save_sdf = save_sdf
        self.part_mass = get_body_mass(self.asset_folder, self.assembly_dir, self.parts, save_sdf=self.save_sdf)
        self.t_start = None
        self.n_eval = None
        self.stop_msg = None

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _simulate(self, part_move, parts_rest, parts_removed, pose, max_grippers, timeout=None, grasp_planner=None, optimizer='L-BFGS-B', debug=0, render=False):
        assert len(parts_rest) > 0
        action, path = check_assemblable(self.asset_folder, self.assembly_dir, parts_rest, part_move, pose=pose, save_sdf=self.save_sdf, 
            return_path=True, debug=debug, render=render)

        if action is not None:
            max_fix = max_grippers - 1 if max_grippers is not None else None
            parts_fix_list = get_stable_plan_1pose_serial(self.asset_folder, self.assembly_dir, parts_rest, self.base_part, pose=pose, max_fix=max_fix, save_sdf=self.save_sdf, timeout=timeout, debug=debug, render=render)
            if parts_fix_list is not None: parts_fix_list = [parts_fix_list]
        else:
            parts_fix_list = None

        if parts_fix_list is None:
            parts_fix = None
        elif len(parts_fix_list) == 0:
            parts_fix = []
        else:
            parts_fix = parts_fix_list[0] # NOTE: only pick the first feasible fix list, can be changed

        feasible = action is not None and parts_fix is not None

        if feasible and grasp_planner is not None:
            grasps = grasp_planner.plan(part_move, parts_rest, parts_removed, pose, path, optimizer)
            if len(grasps) == 0:
                grasps = None
                feasible = False
        else:
            grasps = None

        sim_info = {
            'feasible': feasible,
            'action': action,
            'base_part': self.base_part,
            'parts_fix': parts_fix,
            'part_move': part_move,
            'pose': pose,
            'grasp': grasps,
        }
        return sim_info

    def _update_tree(self, tree, parts_parent, parts_child, n_eval, sim_info):
        if sim_info['feasible']: 
            assert tree.nodes[tuple(parts_parent)]['n_gripper'] is not None, f'[error] parent {parts_parent} not feasible'
        if tree.has_node(tuple(parts_child)):
            assert tree.nodes[tuple(parts_child)]['n_eval'] < n_eval,  f'[error] child {parts_child} incorrectly updated'
            if sim_info['feasible']:
                child_n_gripper, parent_n_gripper = tree.nodes[tuple(parts_child)]['n_gripper'], tree.nodes[tuple(parts_parent)]['n_gripper']
                child_n_gripper_new = max(parent_n_gripper, len(sim_info['parts_fix']) + 1)
                if child_n_gripper is None:
                    tree.nodes[tuple(parts_child)]['n_gripper'] = child_n_gripper_new
                else:
                    tree.nodes[tuple(parts_child)]['n_gripper'] = min(child_n_gripper, child_n_gripper_new)
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
            if sim_info['feasible']:
                tree.add_node(tuple(parts_child), n_eval=n_eval, n_gripper=\
                    max(tree.nodes[tuple(parts_parent)]['n_gripper'], len(sim_info['parts_fix']) + 1), poses=[sim_info['pose']])
            else:
                tree.add_node(tuple(parts_child), n_eval=n_eval, n_gripper=None, poses=[])
        tree.add_edge(tuple(parts_parent), tuple(parts_child), n_eval=n_eval, sim_info=sim_info)

    def _check_fully_explored(self, tree, root_node): # code can be optimized
        # if all childs are recursively explored and failed, then fully explored
        assert tree.has_node(tuple(root_node))
        assert len(root_node) >= 2
        if len(root_node) == 2: return True # NOTE: assume max_gripper >= 2

        for part_move in root_node:
            if self.base_part is not None and part_move == self.base_part: continue
            child_node = root_node.copy()
            child_node.remove(part_move)
            if tree.has_edge(tuple(root_node), tuple(child_node)):
                if tree.edges[tuple(root_node), tuple(child_node)]['sim_info']['feasible']:
                    if not self._check_fully_explored(tree, child_node): return False # child not fully explored
            else:
                return False # no such child
        return True

    def plan(self, budget, max_grippers, max_poses=3, pose_reuse=0, early_term=False, timeout=None, plan_grasp=False, plan_arm=False, gripper_type=None, gripper_scale=None, optimizer='L-BFGS-B', debug=0, render=False, log_dir=None):
        '''
        Main planning function
        Input:
            budget: max # simulations allowed
            max_grippers: max # grippers allowed to use
        Output:
            tree: disassembly tree including all disassembly attempts
        '''
        self.t_start = time()
        solution_found = False
        self.stop_msg = None
        assert budget is not None or timeout is not None

        self._reset()

        self.n_eval = 0
        G0 = self.parts.copy()
        tree = nx.DiGraph()
        tree.add_node(tuple(G0), n_eval=0, n_gripper=1, poses=[])

        if plan_arm:
            grasp_planner = GraspArmPlanner(self.asset_folder, self.assembly_dir, gripper_type, gripper_scale)
        elif plan_grasp:
            grasp_planner = GraspPlanner(self.asset_folder, self.assembly_dir, gripper_type, gripper_scale)
        else:
            grasp_planner = None

        try:

            while True:

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

                G = self._select_node(tree)
                parts_removed = [part for part in G0 if part not in G]
                
                if self.base_part is not None:
                    poses = [None]
                else:
                    poses = tree.nodes[tuple(G)]['poses'][:pose_reuse]
                    G_mesh = get_combined_mesh(self.assembly_dir, G)
                    poses.extend(get_stable_poses(G_mesh, max_num=max_poses - pose_reuse))
                    if len(poses) == 0:
                        poses = [None]

                for p in self.seq_generator.generate_candidate_part(G): # NOTE: maybe can specify which parts to exclude
                    G_prime = G.copy()
                    G_prime.remove(p)

                    if tree.has_edge(tuple(G), tuple(G_prime)): continue

                    for pose in poses:

                        sim_timeout = None if timeout is None else timeout - (time() - self.t_start) # allocate for this step of simulation
                        if sim_timeout is not None and sim_timeout < 0:
                            break

                        sim_info = self._simulate(p, G_prime, parts_removed, pose, max_grippers=max_grippers, timeout=sim_timeout, 
                            grasp_planner=grasp_planner, optimizer=optimizer, debug=debug - 1, render=render)
                        self.n_eval += 1
                        self._update_tree(tree, G, G_prime, self.n_eval, sim_info)

                        if sim_info['feasible']:
                            if len(G_prime) == 2:
                                solution_found = True
                            break

                    if debug > 0:
                        print(f'[planner.base.plan] add edge: ({G}, {G_prime}), feasible: {sim_info["feasible"]}')
                        print(f'[planner.base.plan] progress: {self.n_eval}/{budget} evaluations')

                    # self.plot_tree(tree)
                    break

                if log_dir is not None:
                    stats = self.get_stats(tree)
                    self.log(tree, stats, log_dir)
            
            self._expand_leaf(tree, max_poses, pose_reuse, grasp_planner, optimizer, debug, render)
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

    def _reset(self):
        pass

    def _select_node(self, tree):
        raise NotImplementedError

    def _expand_leaf(self, tree, max_poses, pose_reuse, grasp_planner, optimizer, debug, render):
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
                sim_info = self._simulate(part_move, [part_fix], parts_removed, pose=pose, max_grippers=2, grasp_planner=grasp_planner, optimizer=optimizer, debug=debug - 1, render=render)
                if sim_info['feasible']:
                    node_info = tree.nodes[node]
                    SequencePlanner._update_tree(self, tree, list(node), [part_fix], node_info['n_eval'], sim_info)
                    break

    @staticmethod
    def plot_tree(tree, save_path=None):
        from networkx.drawing.nx_agraph import graphviz_layout
        node_colors = ['g' if tree.nodes[node]['n_gripper'] is not None else 'r' for node in tree.nodes]
        edge_colors = ['g' if tree.edges[edge]['sim_info']['feasible'] else 'r' for edge in tree.edges]
        edge_labels = {edge: ','.join([str(x) for x in set(edge[0]) - set(edge[1])]) for edge in tree.edges}
        pos = graphviz_layout(tree, prog='dot')
        nx.draw(tree, pos, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    @staticmethod
    def plot_tree_with_budget(tree, budget, save_path=None):
        if budget is None: return SequencePlanner.plot_tree(tree, save_path=save_path)

        from networkx.drawing.nx_agraph import graphviz_layout
        budget_tree = nx.DiGraph()
        # for node in tree.nodes:
        #     node_info = tree.nodes[node]
        #     if node_info['n_eval'] <= budget:
        #         budget_tree.add_node(node)
        for edge in tree.edges:
            edge_info = tree.edges[edge]
            if edge_info['n_eval'] <= budget:
                budget_tree.add_edge(*edge, feasible=edge_info['sim_info']['feasible'])
                
        # node_colors = ['g' if budget_tree.nodes[node]['feasible'] else 'r' for node in budget_tree.nodes]
        edge_colors = ['g' if budget_tree.edges[edge]['feasible'] else 'r' for edge in budget_tree.edges]
        edge_labels = {edge: ','.join([str(x) for x in set(edge[0]) - set(edge[1])]) for edge in budget_tree.edges}
        pos = graphviz_layout(budget_tree, prog='dot')
        nx.draw(budget_tree, pos, edge_color=edge_colors, with_labels=True)
        nx.draw_networkx_edge_labels(budget_tree, pos, edge_labels=edge_labels)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    @staticmethod
    def find_sequence(tree):

        # find leaf node
        leaf_node = None
        for node in tree.nodes:
            node_info = tree.nodes[node]
            if len(node) == 1 and node_info['n_gripper'] is not None:
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
                if parent_node_info['n_gripper'] <= node_info['n_gripper']:
                    part_move = tree.edges[parent_node, node]['sim_info']['part_move']
                    sequence.insert(0, part_move)
                    node = parent_node
                    break
        return sequence

    @staticmethod
    def check_success(tree):

        success = False
        n_eval = None # min n_eval
        n_gripper = None # min n_gripper

        for node in tree.nodes:
            node_info = tree.nodes[node]
            if len(node) == 1 and node_info['n_gripper'] is not None:
                success = True

                if n_eval is None:
                    n_eval = node_info['n_eval']
                else:
                    n_eval = min(n_eval, node_info['n_eval'])

                if n_gripper is None:
                    n_gripper = node_info['n_gripper']
                else:
                    n_gripper = min(n_gripper, node_info['n_gripper'])

        return success, n_eval, n_gripper

    @staticmethod
    def get_stats(tree):
        success, n_eval, n_gripper = SequencePlanner.check_success(tree)
        if success:
            sequence = SequencePlanner.find_sequence(tree)
        else:
            sequence = None
        return {
            'success': success,
            'n_eval': n_eval,
            'n_gripper': n_gripper,
            'sequence': [x for x in sequence] if sequence is not None else None,
        }

    def log(self, tree, stats, log_dir, plot=False):
        '''
        Log planned disassembly sequence and gripper statistics
        '''
        t_plan = time() - self.t_start # NOTE: a bit hacky
        stats['time'] = round(t_plan, 2)
        stats['total_n_eval'] = self.n_eval
        stats['stop_msg'] = self.stop_msg

        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'tree.pkl'), 'wb') as fp:
            pickle.dump(tree, fp)
        with open(os.path.join(log_dir, 'stats.json'), 'w') as fp:
            json.dump(stats, fp, cls=NumpyEncoder)
        if plot:
            self.plot_tree(tree, save_path=os.path.join(log_dir, 'tree.png'))

    def render(self, sequence, tree, record_dir=None):
        '''
        Render planned disassembly sequence
        '''
        parts_assembled = self.parts.copy()
        parts_removed = []

        if record_dir is not None:
            os.makedirs(record_dir, exist_ok=True)

        for i, part_move in enumerate(sequence):
            parts_rest = parts_assembled.copy()
            parts_rest.remove(part_move)

            sim_info = tree.edges[tuple(parts_assembled), tuple(parts_rest)]['sim_info']
            assert part_move == sim_info['part_move']
            parts_fix = sim_info['parts_fix']
            parts_free = [part_i for part_i in parts_rest if part_i not in parts_fix] + [part_move]
            action = np.array(sim_info['action'])
            pose = np.array(sim_info['pose']) if sim_info['pose'] is not None else None

            if record_dir is not None:
                record_path = os.path.join(record_dir, f'{i}_{part_move}.gif')
            else:
                record_path = None

            path_planner = MultiPartPathPlanner(self.asset_folder, self.assembly_dir, parts_rest, part_move, parts_removed=parts_removed, pose=pose, save_sdf=self.save_sdf)
            # path_planner.check_success(action)
            success, path = path_planner.plan_path(action, rotation=True)

            body_color_dict = get_body_color_dict(parts_fix, parts_free) # visualize fixes
            path_planner.sim.set_body_color_map(body_color_dict)
            path_planner.render(path=path, record_path=record_path)

            parts_assembled = parts_rest
            parts_removed.append(part_move)
