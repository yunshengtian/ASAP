import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import random
import json
import pickle
from tqdm import tqdm
import traceback
import shutil
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from plan_sequence.sim_string import get_body_color_dict
from plan_sequence.physics_planner import MultiPartPathPlanner, MultiPartStabilityPlanner
from plan_sequence.planner.base import SequencePlanner
from plan_robot.render_grasp import render_path_with_grasp
from plan_robot.render_grasp_arm import render_path_with_grasp_and_arm
from plan_robot.geometry import load_part_meshes, load_gripper_meshes, load_arm_meshes, save_meshes
from assets.save import clear_saved_sdfs, save_path_all_objects
from assets.load import load_part_ids


def interpolate_path(states):
    interpolated_path = []

    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        interpolated_path.append(current_state)

        if len(current_state) == 3:
            average_state = (current_state + next_state) / 2
            interpolated_path.append(average_state)
        elif len(current_state) == 6:
            current_pos, current_euler = current_state[:3], current_state[3:]
            next_pos, next_euler = next_state[:3], next_state[3:]
            average_pos = (current_pos + next_pos) / 2
            rotations = R.from_euler('xyz', [current_euler, next_euler])
            slerp = Slerp([0, 1], rotations)
            average_euler = slerp([0.5]).as_euler('xyz')[0]
            average_state = np.concatenate([average_pos, average_euler])
            interpolated_path.append(average_state)
        else:
            raise NotImplementedError

    interpolated_path.append(states[-1])

    return interpolated_path


def play_logged_plan(asset_folder, assembly_dir, sequence, tree, result_dir, save_mesh, save_pose, save_part, save_path, save_record, save_all, 
    reverse=False, show_fix=False, show_grasp=False, show_arm=False, gripper_type=None, gripper_scale=None, optimizer='L-BFGS-B', save_sdf=False, clear_sdf=False, make_video=False, budget=None, camera_pos=None, camera_lookat=None):

    parts_assembled = sorted(load_part_ids(assembly_dir))

    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)

    if save_mesh or save_all: # save object centric mesh
        mesh_dir = os.path.join(result_dir, 'mesh')
        os.makedirs(mesh_dir, exist_ok=True)
        all_meshes = load_part_meshes(assembly_dir, transform='none')
        # shutil.copyfile(os.path.join(assembly_dir, 'config.json'), os.path.join(mesh_dir, 'config.json'))
        if show_grasp:
            gripper_meshes = load_gripper_meshes(gripper_type, asset_folder, visual=True)
            all_meshes.update(gripper_meshes)
        if show_arm:
            arm_meshes = load_arm_meshes(asset_folder, visual=True, convex=False)
            all_meshes.update(arm_meshes)
        save_meshes(all_meshes, mesh_dir)
    else:
        mesh_dir = None

    if save_pose or save_all:
        pose_dir = os.path.join(result_dir, 'pose')
        os.makedirs(pose_dir, exist_ok=True)
    else:
        pose_dir = None
    
    if save_part or save_all:
        part_dir = os.path.join(result_dir, 'part_fix')
        os.makedirs(part_dir, exist_ok=True)
    else:
        part_dir = None

    if save_path or save_all:
        path_dir = os.path.join(result_dir, 'path')
        os.makedirs(path_dir, exist_ok=True)
    else:
        path_dir = None

    if save_record or save_all:
        record_dir = os.path.join(result_dir, 'record')
        os.makedirs(record_dir, exist_ok=True)
        record_dir_grasp = None
        if show_grasp:
            record_dir_grasp = os.path.join(result_dir, 'record_grasp')
            os.makedirs(record_dir_grasp, exist_ok=True)
    else:
        record_dir = None
        record_dir_grasp = None

    try:
        parts_removed = []

        for i, part_move in enumerate(tqdm(sequence)):
            parts_rest = parts_assembled.copy()
            parts_rest.remove(part_move)

            sim_info = tree.edges[tuple(parts_assembled), tuple(parts_rest)]['sim_info']
            assert part_move == sim_info['part_move']
            action = np.array(sim_info['action'])
            pose = np.array(sim_info['pose']) if sim_info['pose'] is not None else None
            if show_grasp:
                grasps = sim_info['grasp']
            else:
                grasps = None

            parts_fix = sim_info['parts_fix']
            parts_free = [part_i for part_i in parts_rest if parts_fix is None or part_i not in parts_fix] + [part_move]
            
            if show_fix:
                body_color_dict = get_body_color_dict(parts_fix, parts_free) # visualize fixes
            else:
                body_color_dict = get_body_color_dict([], parts_assembled)

            if record_dir is not None:
                record_path = os.path.join(record_dir, f'{i}_{part_move}.mp4' if make_video else f'{i}_{part_move}.gif')
            else:
                record_path = None
            
            # print(f'[play_logged_plan] {i}-th step, part_move: {part_move}')

            if save_path or save_record or save_all:
                path_planner = MultiPartPathPlanner(asset_folder, assembly_dir, parts_rest, part_move, parts_removed=parts_removed, pose=pose, save_sdf=save_sdf,
                    camera_pos=camera_pos, camera_lookat=camera_lookat)
                # success, path = path_planner.check_success(action, return_path=True)
                success, path = path_planner.plan_path(action, rotation=True)
                assert success, f'[play_logged_plan] path planner: part_move {part_move} with action {action} is not successful'

                min_path_len = 300
                while len(path) < min_path_len:
                    path = interpolate_path(path)
                
                if (show_grasp or show_arm) and grasps is not None:
                    n_render = min(3, len(grasps))
                    random_indices = np.random.choice(len(grasps), n_render, replace=False)
                    for idx in random_indices:
                        grasp = grasps[idx][0]
                        if record_dir_grasp is not None:
                            record_path_grasp = os.path.join(record_dir_grasp, f'{i}_{part_move}_g{idx}.mp4' if make_video else f'{i}_{part_move}_g{idx}.gif')
                        else:
                            record_path_grasp = None
                        if show_arm:
                            body_matrices = render_path_with_grasp_and_arm(asset_folder, assembly_dir, part_move, parts_rest, parts_removed, pose, path, gripper_type, gripper_scale, grasp, optimizer, camera_lookat, camera_pos,
                                body_color_dict, reverse, save_record or save_all, record_path_grasp, make_video)
                        else:
                            body_matrices = render_path_with_grasp(asset_folder, assembly_dir, part_move, parts_rest, parts_removed, pose, path, gripper_type, gripper_scale, grasp, camera_lookat, camera_pos,
                                body_color_dict, reverse, save_record or save_all, record_path_grasp, make_video)

                        if path_dir is not None:
                            path_i_dir = os.path.join(path_dir, f'{i}_{part_move}_g{idx}')
                            save_path_all_objects(path_i_dir, body_matrices, n_frame=300)
                            
                path_planner.sim.set_body_color_map(body_color_dict)
                if record_path is not None:
                    body_matrices = path_planner.render(path=path, reverse=reverse, record_path=record_path, make_video=make_video)

                    if path_dir is not None:
                        path_i_dir = os.path.join(path_dir, f'{i}_{part_move}')
                        save_path_all_objects(path_i_dir, body_matrices, n_frame=300)

            if pose_dir is not None:
                pose_path = os.path.join(pose_dir, f'{i}_{part_move}.npy')
                np.save(pose_path, pose, allow_pickle=True)

            if part_dir is not None:
                part_path = os.path.join(part_dir, f'{i}_{part_move}.json')
                with open(part_path, 'w') as fp:
                    json.dump(parts_fix, fp)

            parts_assembled = parts_rest
            parts_removed.append(part_move)

    except (Exception, KeyboardInterrupt) as e:
        if type(e) == KeyboardInterrupt:
            print('[play_logged_plan] interrupt')
        else:
            print('[play_logged_plan] exception:', e, f'from {assembly_dir}')
            print(traceback.format_exc())
        
        if clear_sdf:
            clear_saved_sdfs(assembly_dir)
        raise e

    if clear_sdf:
        clear_saved_sdfs(assembly_dir)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--result-dir', type=str, required=True)
    parser.add_argument('--save-mesh', default=False, action='store_true')
    parser.add_argument('--save-pose', default=False, action='store_true')
    parser.add_argument('--save-part', default=False, action='store_true')
    parser.add_argument('--save-path', default=False, action='store_true')
    parser.add_argument('--save-record', default=False, action='store_true')
    parser.add_argument('--save-all', default=False, action='store_true')
    parser.add_argument('--reverse', default=False, action='store_true')
    parser.add_argument('--show-fix', default=False, action='store_true')
    parser.add_argument('--show-grasp', default=False, action='store_true')
    parser.add_argument('--show-arm', default=False, action='store_true')
    parser.add_argument('--gripper', type=str, default='robotiq-140', choices=['panda', 'robotiq-85', 'robotiq-140'])
    parser.add_argument('--scale', type=float, default=0.4)
    parser.add_argument('--optimizer', type=str, default='L-BFGS-B')
    parser.add_argument('--disable-save-sdf', default=False, action='store_true')
    parser.add_argument('--clear-sdf', default=False, action='store_true')
    parser.add_argument('--plot-tree', default=False, action='store_true')
    parser.add_argument('--make-video', default=False, action='store_true')
    parser.add_argument('--budget', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=[-1, 1, 0], help='camera lookat')
    parser.add_argument('--camera-pos', type=float, nargs=3, default=[1.25, -1.5, 1.5], help='camera position')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(os.path.join(args.log_dir, 'tree.pkl'), 'rb') as fp:
        tree = pickle.load(fp)
    with open(os.path.join(args.log_dir, 'stats.json'), 'r') as fp:
        stats = json.load(fp)
        sequence = stats['sequence']

    if args.plot_tree:
        SequencePlanner.plot_tree_with_budget(tree, budget=args.budget)

    if sequence is None:
        print('[play_logged_plan] failed plan')
    else:
        asset_folder = os.path.join(project_base_dir, './assets')
        play_logged_plan(asset_folder, args.assembly_dir, sequence, tree, args.result_dir, args.save_mesh, args.save_pose, args.save_part, args.save_path, args.save_record, args.save_all, 
            args.reverse, args.show_fix, args.show_grasp, args.show_arm, args.gripper, args.scale, args.optimizer, not args.disable_save_sdf, args.clear_sdf, args.make_video, args.budget, args.camera_pos, args.camera_lookat)
