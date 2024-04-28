import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

from plan_sequence.stable_pose import get_combined_mesh, get_stable_poses
from plan_sequence.feasibility_check import get_stable_plan_1pose_serial, check_stable_noforce
from assets.load import load_part_ids
from assets.save import clear_saved_sdfs


def check_stability(assembly_dir, save_sdf=False, debug=0, render=False, verbose=False):

    asset_folder = os.path.join(project_base_dir, 'assets')
    parts = load_part_ids(assembly_dir)

    try:
        success, _ = check_stable_noforce(asset_folder, assembly_dir, parts, save_sdf=save_sdf, debug=debug, render=render)

        if verbose:
            print(f'[check_stability] no force, success: {success}')
        if not success:
            if save_sdf:
                clear_saved_sdfs(assembly_dir)
            return False

        mesh = get_combined_mesh(assembly_dir, parts)
        poses = get_stable_poses(mesh)
        
        for i, pose in enumerate(poses):

            parts_fix_list = get_stable_plan_1pose_serial(asset_folder, assembly_dir, parts, None, pose, max_fix=3, save_sdf=save_sdf, debug=debug, render=render)
            success = parts_fix_list is not None
            if verbose:
                print(f'[check_stability] {i+1}/{len(poses)} stable pose, success: {success}')

            if success:
                if save_sdf:
                    clear_saved_sdfs(assembly_dir)
                return True

    except (Exception, KeyboardInterrupt) as e:
        if type(e) == KeyboardInterrupt:
            print('[check_stability] interrupt')
        else:
            print('[check_stability] exception:', e)

    if save_sdf:
        clear_saved_sdfs(assembly_dir)

    return False


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='assembly dir')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--disable-save-sdf', default=False, action='store_true')
    args = parser.parse_args()

    success = check_stability(args.dir, save_sdf=not args.disable_save_sdf, debug=args.debug, render=args.render, verbose=True)
    print(f'Success: {success}')
