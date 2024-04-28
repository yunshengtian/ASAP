'''
Grasp planning with stable pose, accelerated collision detection by specifying assembly directory and move/still parts
'''

import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import os
from argparse import ArgumentParser
from tqdm import tqdm
import trimesh
from time import time
from scipy.spatial.transform import Rotation as R

from assets.load import load_assembly_all_transformed, load_pos_quat_dict
from assets.transform import get_pos_quat_from_pose, get_transform_from_path
from plan_sequence.stable_pose import get_stable_poses
from plan_sequence.feasibility_check import check_assemblable
from plan_robot.util_grasp import Grasp, compute_antipodal_pairs, generate_gripper_states, get_gripper_open_ratio
from plan_robot.render_grasp import render_path_with_grasp
from plan_robot.geometry import load_gripper_meshes, transform_gripper_meshes, transform_part_meshes
from utils.seed import set_seed


class GraspPlanner:

    def __init__(self, asset_folder, assembly_dir, gripper_type=None, gripper_scale=None, seed=0, n_surface_pt=100, n_angle=10):
        set_seed(seed)

        # load assembly meshes
        self.assembly = load_assembly_all_transformed(assembly_dir)
        self.part_pos_dict, self.part_quat_dict = load_pos_quat_dict(assembly_dir, transform='final')
        self.part_meshes = {part_id: self.assembly[part_id]['mesh'] for part_id in self.assembly.keys()}
        self.part_meshes_initial = {part_id: self.assembly[part_id]['mesh_initial'] for part_id in self.assembly.keys()}
        self.part_meshes_final = {part_id: self.assembly[part_id]['mesh_final'] for part_id in self.assembly.keys()}

        # load gripper meshes
        self.gripper_type = gripper_type
        self.gripper_meshes = load_gripper_meshes(self.gripper_type, asset_folder)
        self.gripper_scale = gripper_scale
        
        # sampling budget
        self.n_surface_pt = n_surface_pt
        self.n_angle = n_angle

    def generate_grasps(self, part_mesh, disassembly_direction=None):
        grasps = []

        # compute antipodal points
        antipodal_pairs = compute_antipodal_pairs(part_mesh, sample_budget=self.n_surface_pt)
        for antipodal_points in antipodal_pairs:
            open_ratio = get_gripper_open_ratio(self.gripper_type, antipodal_points, self.gripper_scale)
            if open_ratio is None: continue

            # compute grasps
            gripper_pos_list, gripper_quat_list = generate_gripper_states(self.gripper_type, antipodal_points, self.gripper_scale, open_ratio, self.n_angle, disassembly_direction=disassembly_direction)
            for gripper_pos, gripper_quat in zip(gripper_pos_list, gripper_quat_list):
                grasps.append(Grasp(gripper_pos, gripper_quat, open_ratio))

        return grasps

    def transform_grasp(self, grasp, pose):
        grasp = grasp.copy()
        grasp.pos, grasp.quat = get_pos_quat_from_pose(grasp.pos, grasp.quat, pose)
        return grasp

    def check_grasp_feasible(self, grasp, move_mesh, still_mesh, verbose=False, render=False):
        gripper_pos, gripper_quat, open_ratio = grasp.pos, grasp.quat, grasp.open_ratio

        # gripper collision manager (loose hold)
        gripper_meshes_i = transform_gripper_meshes(self.gripper_type, self.gripper_meshes, gripper_pos, gripper_quat, self.gripper_scale, np.eye(4), open_ratio)
        gripper_col_manager = trimesh.collision.CollisionManager()
        for gripper_part_name, gripper_part_mesh in gripper_meshes_i.items():
            gripper_col_manager.add_object(gripper_part_name, gripper_part_mesh)

        # check gripper-ground collision
        for gripper_part_mesh in gripper_meshes_i.values():
            if gripper_part_mesh.vertices[:, 2].min() <= 0.0:
                if verbose: print('[check_grasp_feasible] gripper-ground collision')
                return False
        
        # check gripper-move part collision
        if gripper_col_manager.in_collision_single(move_mesh):
            if verbose: print('[check_grasp_feasible] gripper-move part collision')
            return False

        # gripper collision manager (tight hold)
        gripper_meshes_i = transform_gripper_meshes(self.gripper_type, self.gripper_meshes, gripper_pos, gripper_quat, self.gripper_scale, np.eye(4), open_ratio - 0.005)
        gripper_col_manager = trimesh.collision.CollisionManager()
        for gripper_part_name, gripper_part_mesh in gripper_meshes_i.items():
            gripper_col_manager.add_object(gripper_part_name, gripper_part_mesh)

        # check gripper-still part collision
        if gripper_col_manager.in_collision_single(still_mesh):
            if verbose: print('[check_grasp_feasible] gripper-still part collision')
            return False

        return True

    def render(self, grasp, object_meshes):
        gripper_meshes_i = transform_gripper_meshes(self.gripper_type, self.gripper_meshes, grasp.pos, grasp.quat, self.gripper_scale, np.eye(4), grasp.open_ratio)
        trimesh.Scene(object_meshes + list(gripper_meshes_i.values())).show()
    
    def plan(self, move_id, still_ids, removed_ids, pose, path, early_terminate=True, seed=0, verbose=False):
        set_seed(seed)

        part_meshes_final = transform_part_meshes(self.part_meshes, self.part_pos_dict, self.part_quat_dict, pose)
        move_mesh_final = part_meshes_final[move_id]
        still_mesh_final = trimesh.util.concatenate([part_meshes_final[still_id] for still_id in still_ids])
        self.center = trimesh.util.concatenate(list(part_meshes_final.values())).centroid.copy()
        self.center[2] = 0

        part_transforms = get_transform_from_path(path, n_sample=None)
        part_rel_transforms = [T @ np.linalg.inv(part_transforms[0]) for T in part_transforms]
        
        disassembly_direction = path[-1][:3] - path[0][:3]
        disassembly_direction /= np.linalg.norm(disassembly_direction)
        disassembly_direction_local = pose[:3, :3].T @ disassembly_direction if pose is not None else disassembly_direction
        grasps_final = self.generate_grasps(move_mesh_final, disassembly_direction_local)

        grasps = []
        success = False

        # Every grasp
        for grasp_idx, grasp in enumerate(grasps_final):

            if verbose:
                print('-' * 10 + f' {grasp_idx + 1}/{len(grasps_final)} gripper state ' + '-' * 10)
            
            # Every time step
            grasps_t = []
            for part_transform in part_rel_transforms:
                grasp_t = self.transform_grasp(grasp, part_transform)
                move_mesh_t = move_mesh_final.copy()
                move_mesh_t.apply_transform(part_transform)

                feasible = self.check_grasp_feasible(grasp_t, move_mesh_t, still_mesh_final, verbose=verbose)
                if not feasible:
                    # self.render(grasp_t, [move_mesh_t, still_mesh_final])
                    break
                grasps_t.append(grasp_t)
                # self.render(grasp_t, [move_mesh_t, still_mesh_final])

            else:
                grasps.append(grasps_t)
                success = True

            if early_terminate and success:
                break

        return grasps


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True, help='directory of assembly')
    parser.add_argument('--move-id', type=str, required=True, help='move part id')
    parser.add_argument('--still-ids', type=str, nargs='+', default=None, help='still part ids')
    parser.add_argument('--removed-ids', type=str, nargs='+', default=[], help='removed part ids')
    parser.add_argument('--gripper', type=str, default='robotiq-140', choices=['panda', 'robotiq-85', 'robotiq-140'], help='gripper type')
    parser.add_argument('--scale', type=float, default=0.4, help='gripper scale')
    parser.add_argument('--n-pose', type=int, default=5, help='number of pose samples')
    parser.add_argument('--n-surface-pt', type=int, default=100, help='number of surface point samples for generating antipodal pairs')
    parser.add_argument('--n-angle', type=int, default=10, help='number of grasp angle samples')
    parser.add_argument('--early-terminate', action='store_true', default=False, help='early terminate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=None, help='camera lookat')
    parser.add_argument('--camera-pos', type=float, nargs=3, default=None, help='camera position')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
    parser.add_argument('--render', action='store_true', default=False, help='render grasps')
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')

    grasp_planner = GraspPlanner(asset_folder, args.assembly_dir, args.gripper, args.scale, 
        args.seed, args.n_surface_pt, args.n_angle)

    move_id = args.move_id
    still_ids = [part_id for part_id in grasp_planner.assembly.keys() if part_id != move_id] if args.still_ids is None else args.still_ids
    removed_ids = args.removed_ids

    part_meshes_combined = trimesh.util.concatenate(list(grasp_planner.part_meshes_final.values()))
    poses = get_stable_poses(part_meshes_combined, max_num=args.n_pose)

    for pose_idx, pose in enumerate(poses):
        print('=' * 10 + f' {pose_idx + 1}/{len(poses)} pose ' + '=' * 10)

        _, path = check_assemblable(asset_folder, args.assembly_dir, still_ids, move_id, pose=pose, return_path=True)
        if path is None: continue
        
        grasps = grasp_planner.plan(move_id, still_ids, removed_ids, pose, path, early_terminate=args.early_terminate, seed=args.seed, verbose=args.verbose)
        print(f'{len(grasps)} feasible grasps found')

        # import pickle
        # grasp_path = os.path.join('grasps', f'grasp.pkl')
        # os.makedirs(os.path.dirname(grasp_path), exist_ok=True)
        # with open(grasp_path, 'wb') as f:
        #     pickle.dump(grasps, f)
        # with open(grasp_path, 'rb') as f:
        #     grasps = pickle.load(f)

        if args.render:
            n_render = min(1, len(grasps))
            random_indices = np.random.choice(len(grasps), n_render, replace=False)
            for idx in random_indices:
                grasp = grasps[idx]
                render_path_with_grasp(asset_folder, args.assembly_dir, move_id, still_ids, removed_ids, pose, path, args.gripper, args.scale, grasp, args.camera_lookat, args.camera_pos)
