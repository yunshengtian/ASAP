import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
from itertools import combinations
from time import time

from utils.renderer import SimRenderer
from utils.parallel import parallel_execute
from plan_sequence.physics_planner import MultiPartPathPlanner, MultiPartStabilityPlanner, MultiPartNoForceStabilityPlanner, get_contact_graph, CONTACT_EPS


def get_R3_actions():
    actions = [
        np.array([0, 0, 1]), # +Z
        np.array([0, 0, -1]), # -Z
        np.array([1, 0, 0]), # +X
        np.array([-1, 0, 0]), # -X
        np.array([0, 1, 0]), # +Y
        np.array([0, -1, 0]), # -Y   
    ]
    return actions


def check_assemblable(asset_folder, assembly_dir, parts_fix, part_move, pose=None, save_sdf=False, debug=0, render=False, return_path=False, optimize_path=False, min_sep=None):
    '''
    Check if certain parts are disassemblable
    '''
    planner = MultiPartPathPlanner(asset_folder, assembly_dir, parts_fix, part_move, pose=pose, save_sdf=save_sdf)

    actions = get_R3_actions()
    best_action = None
    best_path = None
    best_path_len = np.inf
    for action in actions:
        success, path = planner.check_success(action, return_path=True, min_sep=None if optimize_path else min_sep)
        if debug > 0:
            print(f'[check_assemblable] success: {success}, parts_fix: {parts_fix}, part_move: {part_move}, action: {action}, path_len: {len(path)}')
            if render:
                SimRenderer().replay(planner.sim)
        if success:
            if len(path) < best_path_len:
                best_path_len = len(path)
                best_path = path
                best_action = action

    if best_path is not None:
        best_path = np.array(best_path)
        if optimize_path: # optimize action based on the path found
            best_dirs = best_path[1:, :3] - best_path[0, :3]
            opt_action = (best_dirs / np.linalg.norm(best_dirs, axis=1)[:, None]).mean(axis=0)
            opt_action = opt_action / np.linalg.norm(opt_action)
            success, opt_path = planner.check_success(opt_action, return_path=True, min_sep=min_sep)
            if debug > 0:
                print(f'[check_assemblable] success: {success}, parts_fix: {parts_fix}, part_move: {part_move}, action (optimized): {opt_action}, path_len (optimized): {len(opt_path)}')
                if render:
                    SimRenderer().replay(planner.sim)
            if success:
                best_path_len = len(opt_path)
                best_path = opt_path
                best_action = opt_action
            else: # just in case, plan again with min_sep
                success, best_path = planner.check_success(best_action, return_path=True, min_sep=min_sep)
                assert success
        best_path = np.array(best_path)

    if return_path:
        return best_action, best_path
    else:
        return best_action


def check_all_connection_assemblable(asset_folder, assembly_dir, parts=None, contact_eps=CONTACT_EPS, save_sdf=False, num_proc=1, debug=0, render=False):
    '''
    Check if all connected pairs of parts are disassemblable
    '''
    G = get_contact_graph(asset_folder, assembly_dir, parts, contact_eps=contact_eps, save_sdf=save_sdf)

    worker_args = []
    for pair in G.edges:
        part_a, part_b = pair
        worker_args.append([asset_folder, assembly_dir, [part_a], part_b, None, save_sdf, debug, render])

    failures = []
    for action, args in parallel_execute(check_assemblable, worker_args, num_proc, show_progress=debug > 0, desc='check_all_connection_assemblable', return_args=True):
        success = action is not None
        part_fix, part_move = args[2][0], args[3]
        if debug > 0:
            print(f'[check_all_connection_assemblable] success: {success}, part_fix: {part_fix}, part_move: {part_move}, action: {action}')
        if not success:
            failures.append((part_fix, part_move))

    all_success = len(failures) == 0
    return all_success, failures


def check_given_connection_assemblable(asset_folder, assembly_dir, part_pairs, bidirection=False, save_sdf=False, num_proc=1, debug=0, render=False):
    '''
    Check if given connected pairs of parts are disassemblable
    '''
    worker_args = []
    for pair in part_pairs:
        part_a, part_b = pair
        worker_args.append([asset_folder, assembly_dir, [part_a], part_b, None, save_sdf, debug, render])
        if bidirection:
            worker_args.append([asset_folder, assembly_dir, [part_b], part_a, None, save_sdf, debug, render])

    failures = []
    for action, args in parallel_execute(check_assemblable, worker_args, num_proc, show_progress=debug > 0, desc='check_given_connection_assemblable', return_args=True):
        success = action is not None
        part_fix, part_move = args[2][0], args[3]
        if debug > 0:
            print(f'[check_given_connection_assemblable] success: {success}, part_fix: {part_fix}, part_move: {part_move}, action: {action}')
        if not success:
            failures.append((part_fix, part_move))

    all_success = len(failures) == 0
    return all_success, failures


def check_stable_noforce(asset_folder, assembly_dir, parts, save_sdf=False, timeout=None, allow_gap=False, debug=0, render=False):
    '''
    Check if stable without any external force
    '''
    planner = MultiPartNoForceStabilityPlanner(asset_folder, assembly_dir, parts, save_sdf=save_sdf, allow_gap=allow_gap)
    
    success, G = planner.check_success(timeout=timeout)
    if debug > 0:
        print(f'[check_stable_noforce] success: {success}')
        if render:
            SimRenderer().replay(planner.sim)

    return success, G


def check_stable(asset_folder, assembly_dir, parts_fix, parts_move, pose=None, save_sdf=False, timeout=None, allow_gap=False, debug=0, render=False):
    '''
    Check if gravitationally stable for a given fixed part
    '''
    planner = MultiPartStabilityPlanner(asset_folder, assembly_dir, parts_fix, parts_move, pose=pose, save_sdf=save_sdf, allow_gap=allow_gap)

    success, parts_fall = planner.check_success(timeout=timeout)
    if debug > 0:
        print(f'[check_stable] success: {success}, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts_move}')
        if render:
            SimRenderer().replay(planner.sim)

    return success, parts_fall


def get_stable_plan_1pose_serial(asset_folder, assembly_dir, parts, base_part, pose, max_fix=None, save_sdf=False, timeout=None, allow_gap=False, debug=0, render=False, return_count=False):
    '''
    Get all gravitationally stable plans given 1 pose through serial greedy search
    '''
    t_start = time()
    count = 0

    max_fix = len(parts) if max_fix is None else min(max_fix, len(parts))
    parts_fix = [] if base_part is None else [base_part]
    
    while True:

        parts_move = parts.copy()
        for part_fix in parts_fix:
            parts_move.remove(part_fix)

        if timeout is not None:
            timeout -= (time() - t_start)
            if timeout < 0:
                if return_count:
                    return None, count
                else:
                    return None
            t_start = time()

        success, parts_fall = check_stable(asset_folder, assembly_dir, parts_fix, parts_move, pose, save_sdf, timeout, allow_gap, debug, render)
        count += 1

        if debug > 0:
            print(f'[get_stable_plan_1pose_serial] success: {success}, n_fix: {len(parts_fix)}, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts_move}')

        if success:
            break
        else:
            if parts_fall is None:
                if return_count:
                    return None, count # timeout
                else:
                    return None
            parts_fix.extend(parts_fall)
        
        if len(parts_fix) > max_fix:
            if return_count:
                return None, count # failed
            else:
                return None

    if base_part is not None:
        parts_fix.remove(base_part)

    if return_count:
        return parts_fix, count
    else:
        return parts_fix


def get_stable_plan_1pose_parallel(asset_folder, assembly_dir, parts, base_part, pose=None, max_fix=None, save_sdf=False, timeout=None, allow_gap=False, num_proc=1, debug=0, render=False):
    '''
    Get all gravitationally stable plans given 1 pose through parallel greedy search
    '''
    t_start = time()

    max_fix = len(parts) if max_fix is None else min(max_fix, len(parts))

    if pose is not None:
        parts_fix = [] if base_part is None else [base_part]
        success, parts_fall = check_stable(asset_folder, assembly_dir, parts_fix, parts, pose, save_sdf, timeout, allow_gap, debug, render) # check if stable without any grippers
        if debug > 0:
            print(f'[get_stable_plan_1pose_parallel] success: {success}, n_fix: 0, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts}')
        if success:
            return []
        else:
            if parts_fall is None:
                return None # timeout

    if base_part is None:
        parts_fix_list = [[part_fix] for part_fix in parts]
    else:
        parts_fix_list = [[part_fix, base_part] for part_fix in parts if part_fix != base_part]
    
    while True:
        success_any = False

        if timeout is not None:
            timeout -= (time() - t_start)
            if timeout < 0:
                return None
            t_start = time()

        worker_args = []
        for parts_fix in parts_fix_list:
            if len(parts_fix) > max_fix: continue
            parts_move = parts.copy()
            for part_fix in parts_fix:
                parts_move.remove(part_fix)
            worker_args.append([asset_folder, assembly_dir, parts_fix, parts_move, pose, save_sdf, timeout, allow_gap, debug, render])

        if len(worker_args) == 0:
            return None # failed

        for (success, parts_fall), args in parallel_execute(check_stable, worker_args, num_proc, show_progress=debug > 0, desc='get_stable_plan_1pose_parallel', return_args=True):
            parts_fix, parts_move = args[2], args[3]
            if debug > 0:
                print(f'[get_stable_plan_1pose_parallel] success: {success}, n_fix: {len(parts_fix)}, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts_move}')
            if success:
                success_any = True
            else:
                if parts_fall is None:
                    return None # timeout
                index = parts_fix_list.index(parts_fix)
                parts_fix_list[index].extend(parts_fall)
            if timeout is not None and time() - t_start > timeout:
                return None

        if success_any:
            break

    parts_fix_list = [parts_fix for parts_fix in parts_fix_list if len(parts_fix) <= max_fix]
    for parts_fix in parts_fix_list:
        if base_part is not None:
            parts_fix.remove(base_part)
    parts_fix_list = sorted(parts_fix_list, key=lambda x: len(x))
    return parts_fix_list
