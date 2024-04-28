import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import random
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import redmax_py as redmax
import time

from assets.load import load_assembly_all_transformed
from assets.save import sample_path
from assets.mesh_distance import compute_move_mesh_distance_from_mat, compute_ground_distance_from_mat
from assets.transform import transform_pts_by_matrix, get_transform_matrix, get_transform_matrix_euler
from plan_path.pyplanners.rrt_connect import rrt_connect, birrt
from plan_path.pyplanners.smoothing import smooth_path


class ConnectPathPlanner:

    def __init__(self, assembly_dir, 
        step_size=0.1, min_sep=0.5, sdf_dx=0.05):

        self.assembly = load_assembly_all_transformed(assembly_dir)
        for part_id, part_data in self.assembly.items():

            mesh_final = part_data['mesh_final']
            phys_mesh_final = redmax.SDFMesh(mesh_final.vertices.T, mesh_final.faces.T, sdf_dx)
            self.assembly[part_id]['phys_mesh_final'] = phys_mesh_final

            mesh_initial = part_data['mesh_initial']
            if mesh_initial is not None:
                phys_mesh_initial = redmax.SDFMesh(mesh_initial.vertices.T, mesh_initial.faces.T, sdf_dx)
                self.assembly[part_id]['phys_mesh_initial'] = phys_mesh_initial
            else:
                self.assembly[part_id]['phys_mesh_initial'] = None
        
        self.step_size = step_size
        self.min_sep = min_sep
        self.sdf_dx = sdf_dx

    def get_fns(self, move_id, still_ids, removed_ids, rotation=False):

        mesh_move = self.assembly[move_id]['mesh']
        min_box_move = mesh_move.vertices.min(axis=0)
        max_box_move = mesh_move.vertices.max(axis=0)

        phys_mesh_move = self.assembly[move_id]['phys_mesh_final']
        phys_meshes_still = [self.assembly[still_id]['phys_mesh_final'] for still_id in still_ids]
        phys_meshes_removed = [self.assembly[removed_id]['phys_mesh_initial'] for removed_id in removed_ids]
        phys_meshes_removed = [phys_mesh for phys_mesh in phys_meshes_removed if phys_mesh is not None]

        phys_meshes_all = [phys_mesh_move] + phys_meshes_still + phys_meshes_removed
        min_box_all = np.min([phys_mesh.vertices.T.min(axis=0) for phys_mesh in phys_meshes_all], axis=0)
        max_box_all = np.max([phys_mesh.vertices.T.max(axis=0) for phys_mesh in phys_meshes_all], axis=0)
        extend_box_all = max_box_all - min_box_all
        state_lower_bound = min_box_all - 0.5 * extend_box_all
        state_upper_bound = max_box_all + 0.5 * extend_box_all

        if rotation:
            def distance_fn(q1, q2):
                mat1 = get_transform_matrix_euler(q1[:3], q1[3:])
                mat2 = get_transform_matrix_euler(q2[:3], q2[3:])
                boxes1 = transform_pts_by_matrix(np.vstack([min_box_move, max_box_move]), mat1)
                boxes2 = transform_pts_by_matrix(np.vstack([min_box_move, max_box_move]), mat2)
                return np.linalg.norm(boxes1 - boxes2, axis=1).sum()
        else:
            def distance_fn(q1, q2):
                return np.linalg.norm(q1 - q2)

        def collision_fn(q):
            mat = get_transform_matrix(q) @ np.linalg.inv(get_transform_matrix(self.assembly[move_id]['final_state']))
            d_m = compute_move_mesh_distance_from_mat(phys_mesh_move, phys_meshes_still + phys_meshes_removed, mat)
            d_g = compute_ground_distance_from_mat(phys_mesh_move, mat)
            return d_m <= self.min_sep or d_g <= -self.sdf_dx

        if rotation:
            def sample_fn():
                while True:
                    ratio = np.random.random(3)
                    translation = ratio * state_lower_bound + (1.0 - ratio) * state_upper_bound
                    rotation = Rotation.random().as_euler('xyz')
                    state = np.concatenate([translation, rotation])
                    if not collision_fn(state):
                        return state
        else:
            def sample_fn():
                while True:
                    ratio = np.random.random(3)
                    state = ratio * state_lower_bound + (1.0 - ratio) * state_upper_bound
                    if not collision_fn(state):
                        return state

        def extend_fn(q1, q2):
            q_dist = distance_fn(q1, q2)
            num_steps = int(np.ceil(q_dist / self.step_size))
            for i in range(num_steps):
                yield q1 + (q2 - q1) / q_dist * (i + 1) * self.step_size

        return distance_fn, sample_fn, extend_fn, collision_fn

    def plan(self, move_id, still_ids, removed_ids, rotation=False, initial_state=None, final_state=None, max_time=120, smooth=True, verbose=False):

        if initial_state is None: 
            if 'initial_state' in self.assembly[move_id]:
                initial_state = self.assembly[move_id]['initial_state']
        if initial_state is None: return None
        if final_state is None: 
            if 'final_state' in self.assembly[move_id]:
                final_state = self.assembly[move_id]['final_state']
            else:
                final_state = np.zeros(6)
        
        if not rotation:
            initial_state = initial_state[:3]
            final_state = final_state[:3]

        distance_fn, sample_fn, extend_fn, collision_fn = self.get_fns(move_id, still_ids, removed_ids, rotation=rotation)
        if collision_fn(initial_state):
            print('initial state in collision')
            return None
        if collision_fn(final_state):
            print('final state in collision')
            return None

        connected_path = None
        n_failed_attempt = 0
        while connected_path is None:
            if verbose:
                print('planning connecting path')
            connected_path = rrt_connect(final_state, initial_state, # NOTE: disassembly from final to initial
                distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=100, max_time=max_time)
            n_failed_attempt += 1
            if n_failed_attempt == 3:
                if verbose:
                    print('path connector gets stuck')
                    break
        else:
            in_collision = False
            for state in connected_path:
                if collision_fn(state):
                    in_collision = True

            if verbose:
                print(f'connecting path planned (len: {len(connected_path)}, collision: {in_collision})')

        if smooth and connected_path is not None:
            if verbose:
                print('smoothing path')
            connected_path = smooth_path(connected_path, extend_fn, collision_fn, distance_fn,
                cost_fn=None, sample_fn=sample_fn, max_iterations=100, tolerance=1e-5, verbose=False) # NOTE: max_time and converge_time not specified
            
            in_collision = False
            for state in connected_path:
                if collision_fn(state):
                    in_collision = True

            if verbose:
                print(f'smoothing path completed (len: {len(connected_path)}, collision: {in_collision})')

        return connected_path

    def visualize_state(self, move_id, still_ids, removed_ids, state):
        mesh_move = self.assembly[move_id]['mesh_final']
        meshes_still = [self.assembly[still_id]['mesh_final'] for still_id in still_ids]
        meshes_removed = [self.assembly[removed_id]['mesh_initial'] for removed_id in removed_ids]
        meshes_removed = [mesh for mesh in meshes_removed if mesh is not None]

        viz_new_mesh = mesh_move.copy()
        viz_new_mesh.apply_transform(get_transform_matrix(state) @ np.linalg.inv(get_transform_matrix(self.assembly[move_id]['final_state'])))
        trimesh.Scene([viz_new_mesh, *meshes_still, *meshes_removed]).show()

    def visualize_path(self, move_id, still_ids, removed_ids, path, n_frame=10):
        for state in sample_path(path, n_frame):
            self.visualize_state(move_id, still_ids, removed_ids, state)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--id', type=str, required=True, help='assembly id (e.g. 00000)')
    parser.add_argument('--dir', type=str, default='joint_assembly', help='directory storing all assemblies')
    parser.add_argument('--move-id', type=str, default='0')
    parser.add_argument('--still-ids', type=str, nargs='+', default=['1'])
    parser.add_argument('--removed-ids', type=str, nargs='+', default=[])
    parser.add_argument('--rotation', default=False, action='store_true')
    parser.add_argument('--initial-state', type=float, nargs='+', default=None)
    parser.add_argument('--final-state', type=float, nargs='+', default=None)
    parser.add_argument('--sdf-dx', type=float, default=0.05, help='grid resolution of SDF')
    parser.add_argument('--step-size', type=float, default=0.1, help='step size for state extension in sampling')
    parser.add_argument('--min-sep', type=float, default=0.5, help='min part separation')
    parser.add_argument('--max-time', type=float, default=120, help='timeout')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--disable-smooth', default=False, action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    asset_folder = os.path.join(project_base_dir, './assets')
    assembly_dir = os.path.join(asset_folder, args.dir, args.id)

    planner = ConnectPathPlanner(assembly_dir, 
        step_size=args.step_size, min_sep=args.min_sep, sdf_dx=args.sdf_dx)
    initial_state = np.array(args.initial_state) if args.initial_state is not None else planner.assembly[args.move_id]['initial_state']
    final_state = np.array(args.final_state) if args.final_state is not None else planner.assembly[args.move_id]['final_state']
    
    planner.visualize_state(args.move_id, args.still_ids, args.removed_ids, initial_state)
    planner.visualize_state(args.move_id, args.still_ids, args.removed_ids, final_state)

    path = planner.plan(args.move_id, args.still_ids, args.removed_ids, args.rotation, initial_state, final_state, max_time=args.max_time, smooth=not args.disable_smooth, verbose=True)
    
    if path is not None:
        planner.visualize_path(args.move_id, args.still_ids, args.removed_ids, path)
