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

from assets.transform import get_transform_from_path
from plan_sequence.stable_pose import get_stable_poses
from plan_sequence.feasibility_check import check_assemblable
from plan_robot.util_grasp import get_gripper_base_name
from plan_robot.render_grasp_arm import render_path_with_grasp_and_arm
from plan_robot.util_arm import get_arm_chain, get_arm_pos_candidates, get_arm_euler, inverse_kinematics_correction, get_default_arm_rest_q
from plan_robot.geometry import load_arm_meshes, transform_gripper_meshes, transform_arm_meshes, transform_part_meshes
from plan_robot.run_grasp_plan import GraspPlanner
from utils.seed import set_seed


class GraspArmPlanner(GraspPlanner):

    def __init__(self, asset_folder, assembly_dir, gripper_type=None, gripper_scale=None, seed=0, n_surface_pt=100, n_angle=10, ik_optimizer='L-BFGS-B'):
        super().__init__(asset_folder, assembly_dir, gripper_type=gripper_type, gripper_scale=gripper_scale, seed=seed, n_surface_pt=n_surface_pt, n_angle=n_angle)

        # load arm meshes
        self.arm_meshes = load_arm_meshes(asset_folder)
        self.arm_scale = gripper_scale
        self.arm_q_default = [0] + list(get_default_arm_rest_q()) # full
        self.ik_optimizer = ik_optimizer
        self.center = None
        
    def check_arm_feasible(self, grasp, move_mesh, still_mesh, arm_pos, arm_q_default=None, verbose=False, render=False):
        gripper_pos, gripper_quat, open_ratio = grasp.pos, grasp.quat, grasp.open_ratio
        gripper_ori = R.from_quat(gripper_quat[[1, 2, 3, 0]]).apply([0, 0, 1])

        # check IK
        arm_euler = get_arm_euler(arm_pos, center=self.center)
        arm_chain = get_arm_chain(base_pos=arm_pos, base_euler=arm_euler, scale=self.arm_scale)
        if arm_q_default is None: arm_q_default = self.arm_q_default
        arm_q, ik_success = arm_chain.inverse_kinematics(target_position=gripper_pos, target_orientation=gripper_ori, orientation_mode='Z', n_restart=3, initial_position=arm_q_default, optimizer=self.ik_optimizer)
        if ik_success:
            arm_q = inverse_kinematics_correction(arm_chain, arm_q, self.gripper_type, gripper_quat)
        else:
            if verbose: print('[check_arm_feasible] IK failed')
            return False

        # gripper collision manager
        gripper_meshes_i = transform_gripper_meshes(self.gripper_type, self.gripper_meshes, gripper_pos, gripper_quat, self.gripper_scale, np.eye(4), open_ratio)
        gripper_col_manager = trimesh.collision.CollisionManager()
        for gripper_part_name, gripper_part_mesh in gripper_meshes_i.items():
            gripper_col_manager.add_object(gripper_part_name, gripper_part_mesh)

        # arm collision manager
        arm_meshes_i = transform_arm_meshes(self.arm_meshes, arm_chain, arm_q, self.arm_scale)
        arm_col_manager = trimesh.collision.CollisionManager()
        for name, mesh in arm_meshes_i.items():
            arm_col_manager.add_object(name, mesh)

        # check arm-ground collision
        for arm_part_name, arm_part_mesh in arm_meshes_i.items():
            if arm_part_name != 'linkbase' and arm_part_mesh.vertices[:, 2].min() <= 0.0:
                if verbose: print('[check_arm_feasible] arm-ground collision')
                return False

        # check arm-move part collision
        if arm_col_manager.in_collision_single(move_mesh):
            if verbose: print('[check_arm_feasible] arm-move part collision')
            return False
        
        # check arm-still part collision
        if arm_col_manager.in_collision_single(still_mesh):
            if verbose: print('[check_arm_feasible] arm-still part collision')
            return False
        
        # check arm-gripper collision
        _, collision_names = gripper_col_manager.in_collision_other(arm_col_manager, return_names=True)
        for (col_gripper_name, col_arm_name) in collision_names:
            if col_arm_name == 'link7':
                if col_gripper_name != get_gripper_base_name(self.gripper_type):
                    if verbose: print('[check_arm_feasible] arm-gripper collision')
                    return False
            else:
                if verbose: print('[check_arm_feasible] arm-gripper collision')
                return False

        # check arm self-collision
        _, collision_names = arm_col_manager.in_collision_internal(return_names=True)
        for (col_arm_name1, col_arm_name2) in collision_names:
            if not arm_chain.check_neighboring_links(col_arm_name1, col_arm_name2):
                if verbose: print('[check_arm_feasible] arm self-collision')
                return False

        grasp.arm_pos = arm_pos
        grasp.arm_euler = arm_euler
        grasp.arm_q = arm_q
        return True

    def render(self, grasp, object_meshes):
        gripper_meshes_i = transform_gripper_meshes(self.gripper_type, self.gripper_meshes, grasp.pos, grasp.quat, self.gripper_scale, np.eye(4), grasp.open_ratio)
        if grasp.arm_pos is not None:
            arm_chain = get_arm_chain(base_pos=grasp.arm_pos, base_euler=grasp.arm_euler, scale=self.arm_scale)
            arm_meshes_i = transform_arm_meshes(self.arm_meshes, arm_chain, grasp.arm_q, self.arm_scale)
        else:
            arm_meshes_i = {}
        trimesh.Scene(object_meshes + list(gripper_meshes_i.values()) + list(arm_meshes_i.values())).show()
    
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
        grasps_final = self.generate_grasps(move_mesh_final, disassembly_direction)

        grasps = []
        success = False

        # Every grasp
        for grasp_idx, grasp in enumerate(grasps_final):

            gripper_pos, gripper_quat, open_ratio = grasp.pos, grasp.quat, grasp.open_ratio
            if verbose:
                print('-' * 10 + f' {grasp_idx + 1}/{len(grasps_final)} gripper state ' + '-' * 10)
            
            # Every arm position
            gripper_ori = R.from_quat(gripper_quat[[1, 2, 3, 0]]).apply([0, 0, 1])
            arm_pos_candidates = get_arm_pos_candidates(gripper_pos, gripper_ori, self.gripper_scale, center=self.center)
            for arm_pos in arm_pos_candidates:

                # Every time step
                grasps_t = []
                arm_q_default = None
                for part_transform in part_rel_transforms:
                    grasp_t = self.transform_grasp(grasp, part_transform)
                    move_mesh_t = move_mesh_final.copy()
                    move_mesh_t.apply_transform(part_transform)

                    feasible = self.check_grasp_feasible(grasp_t, move_mesh_t, still_mesh_final, verbose=verbose) and \
                        self.check_arm_feasible(grasp_t, move_mesh_t, still_mesh_final, arm_pos, arm_q_default, verbose=verbose)
                    if not feasible:
                        # self.render(grasp_t, [move_mesh_t, still_mesh_final])
                        break
                    grasps_t.append(grasp_t)
                    arm_q_default = grasp_t.arm_q
                    # self.render(grasp_t, [move_mesh_t, still_mesh_final])

                else:
                    grasps.append(grasps_t)
                    success = True

                if early_terminate and success:
                    break

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
    parser.add_argument('--optimizer', type=str, default='L-BFGS-B', help='optimizer')
    parser.add_argument('--early-terminate', action='store_true', default=False, help='early terminate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=None, help='camera lookat')
    parser.add_argument('--camera-pos', type=float, nargs=3, default=None, help='camera position')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
    parser.add_argument('--render', action='store_true', default=False, help='render grasps')
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')

    grasp_planner = GraspArmPlanner(asset_folder, args.assembly_dir, args.gripper, args.scale, 
        args.seed, args.n_surface_pt, args.n_angle, args.optimizer)

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
                render_path_with_grasp_and_arm(asset_folder, args.assembly_dir, move_id, still_ids, removed_ids, pose, path, args.gripper, args.scale, grasp, args.camera_lookat, args.camera_pos)
