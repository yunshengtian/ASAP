import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import json
import traceback
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning) # for trimesh stable pose computation

from plan_sequence.generator import generators
from plan_sequence.planner import planners
from assets.save import clear_saved_sdfs


def seq_plan(asset_folder, assembly_dir, generator_name, planner_name, num_proc, seed, budget, max_gripper, max_pose, pose_reuse, early_term, timeout, base_part,
    save_sdf, clear_sdf, plan_grasp, plan_arm, gripper_type, gripper_scale, optimizer, debug, render, record_dir, log_dir):

    try:
        generator_cls = generators[generator_name]
        generator = generator_cls(asset_folder, assembly_dir, base_part=base_part, save_sdf=save_sdf)
        planner_cls = planners[planner_name]
        planner = planner_cls(generator, num_proc, save_sdf=save_sdf)
        planner.seed(seed)

        setup = {
            'budget': budget, 'max_grippers': max_gripper, 'max_poses': max_pose, 'pose_reuse': pose_reuse, 'early_term': early_term, 'timeout': timeout,
            'plan_grasp': plan_grasp, 'plan_arm': plan_arm, 'gripper_type': gripper_type, 'gripper_scale': gripper_scale, 'optimizer': optimizer,
        }
        tree = planner.plan(**setup, debug=debug - 1, render=False)

        stats = planner.get_stats(tree)

        if log_dir is not None:
            planner.log(tree, stats, log_dir)
            with open(os.path.join(log_dir, 'setup.json'), 'w') as fp:
                json.dump(setup, fp)

        if debug:
            print(f'[seq_plan] stats: {stats}')

        if render and stats['success']:
            planner.render(stats['sequence'], tree, record_dir)

    except (Exception, KeyboardInterrupt) as e:
        if type(e) == KeyboardInterrupt:
            print('[seq_plan] interrupt')
        else:
            print('[seq_plan] exception:', e, f'from {assembly_dir}')
            print(traceback.format_exc())
        if clear_sdf:
            clear_saved_sdfs(assembly_dir)
        raise e

    if clear_sdf:
        clear_saved_sdfs(assembly_dir)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='multi_assembly', help='directory storing all assemblies')
    parser.add_argument('--id', type=str, required=True, help='assembly id (e.g. 00000)')
    parser.add_argument('--planner', type=str, required=True, choices=list(planners.keys()), help='name of planner (node selection algorithm)')
    parser.add_argument('--generator', type=str, required=True, choices=list(generators.keys()), help='name of part generator (part selection algorithm)')
    parser.add_argument('--num-proc', type=int, default=1, help='number of processes for parallel planning')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--budget', type=int, default=400, help='maximum evaluation (feasibility check) budget')
    parser.add_argument('--max-gripper', type=int, default=3)
    parser.add_argument('--max-pose', type=int, default=3)
    parser.add_argument('--pose-reuse', type=int, default=0)
    parser.add_argument('--early-term', default=False, action='store_true')
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--base-part', type=str, default=None)
    parser.add_argument('--debug', type=int, default=3, help='depth of debug message')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--record-dir', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--disable-save-sdf', default=False, action='store_true')
    parser.add_argument('--clear-sdf', default=False, action='store_true')
    parser.add_argument('--plan-grasp', default=False, action='store_true')
    parser.add_argument('--plan-arm', default=False, action='store_true')
    parser.add_argument('--gripper', type=str, default='robotiq-140', choices=['panda', 'robotiq-85', 'robotiq-140'])
    parser.add_argument('--scale', type=float, default=0.4)
    parser.add_argument('--optimizer', type=str, default='L-BFGS-B')

    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')
    assembly_dir = os.path.join(asset_folder, args.dir, args.id)
    exp_name = f'{args.planner}-{args.generator}'

    if args.record_dir is None:
        record_dir = None
    else:
        record_dir = os.path.join(args.record_dir, exp_name, f's{args.seed}', args.id)

    if args.log_dir is None:
        log_dir = None
    else:
        log_dir = os.path.join(args.log_dir, exp_name, f's{args.seed}', f'{args.id}')

    seq_plan(asset_folder, assembly_dir, args.generator, args.planner, args.num_proc, args.seed, args.budget, args.max_gripper, args.max_pose, args.pose_reuse, args.early_term, args.timeout, args.base_part,
        not args.disable_save_sdf, args.clear_sdf, args.plan_grasp, args.plan_arm, args.gripper, args.scale, args.optimizer, args.debug, args.render, record_dir, log_dir)
