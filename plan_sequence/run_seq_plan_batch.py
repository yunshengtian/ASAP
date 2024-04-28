import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from plan_sequence.run_seq_plan import seq_plan
from utils.parallel import parallel_execute


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='multi_assembly', help='directory storing all assemblies')
    parser.add_argument('--id-path', type=str, default=None)
    parser.add_argument('--planner', type=str, nargs='+', required=True)
    parser.add_argument('--generator', type=str, nargs='+', required=True)
    parser.add_argument('--num-proc', type=int, default=1)
    parser.add_argument('--inner-num-proc', type=int, default=1)
    parser.add_argument('--n-seed', type=int, default=1)
    parser.add_argument('--budget', type=int, default=400)
    parser.add_argument('--max-gripper', type=int, nargs='+', default=(2,))
    parser.add_argument('--max-pose', type=int, default=3)
    parser.add_argument('--pose-reuse', type=int, default=0)
    parser.add_argument('--early-term', default=False, action='store_true')
    parser.add_argument('--timeout', type=int, default=None)
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
    assemblies_dir = os.path.join(asset_folder, args.dir)
    assembly_ids = []
    if args.id_path is None:
        for assembly_id in os.listdir(assemblies_dir):
            assembly_dir = os.path.join(assemblies_dir, assembly_id)
            if os.path.isdir(assembly_dir):
                assembly_ids.append(assembly_id)
    else:
        with open(args.id_path, 'r') as fp:
            assembly_ids = [x.replace('\n', '') for x in fp.readlines()]
        real_assembly_ids = []
        for assembly_id in assembly_ids:
            assembly_dir = os.path.join(assemblies_dir, assembly_id)
            if os.path.isdir(assembly_dir):
                real_assembly_ids.append(assembly_id)
        assembly_ids = real_assembly_ids
    assembly_ids.sort()
    
    worker_args = []

    for max_gripper in args.max_gripper:
        for assembly_id in assembly_ids:
            assembly_dir = os.path.join(assemblies_dir, assembly_id)
            for generator_name in args.generator:
                for planner_name in args.planner:
                    for seed in range(args.n_seed):
                        exp_name = f'{planner_name}-{generator_name}'

                        if args.record_dir is None:
                            record_dir = None
                        else:
                            record_dir = os.path.join(args.record_dir, f'g{max_gripper}', exp_name, f's{seed}', assembly_id)

                        if args.log_dir is None:
                            log_dir = None
                        else:
                            log_dir = os.path.join(args.log_dir, f'g{max_gripper}', exp_name, f's{seed}', f'{assembly_id}')

                        worker_args.append([asset_folder, assembly_dir, generator_name, planner_name, args.inner_num_proc, seed, args.budget, max_gripper, args.max_pose, args.pose_reuse, args.early_term, args.timeout, None, 
                            not args.disable_save_sdf, args.clear_sdf, args.plan_grasp, args.plan_arm, args.gripper, args.scale, args.optimizer, 0, args.render, record_dir, log_dir])

    try:
        for _ in parallel_execute(seq_plan, worker_args, args.num_proc):
            pass
    except KeyboardInterrupt:
        exit()
