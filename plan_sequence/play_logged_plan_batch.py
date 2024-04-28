import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import pickle
import json

from plan_sequence.play_logged_plan import play_logged_plan
from utils.parallel import parallel_execute


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--id-path', type=str, default=None)
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
    parser.add_argument('--make-video', default=False, action='store_true')
    parser.add_argument('--num-proc', type=int, default=8)
    parser.add_argument('--budget', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=[-1, 1, 0], help='camera lookat')
    parser.add_argument('--camera-pos', type=float, nargs=3, default=[1.25, -1.5, 1.5], help='camera position')
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')

    if args.id_path is not None:
        with open(args.id_path, 'r') as fp:
            if args.id_path.endswith('.txt'):
                target_ids = [x.replace('\n', '') for x in fp.readlines()]
            elif args.id_path.endswith('.json'):
                target_ids = json.load(fp)
            else:
                raise Exception
    else:
        target_ids = None
    
    worker_args = []

    for assembly_id in os.listdir(args.log_dir):
        if target_ids is not None and assembly_id not in target_ids: continue

        log_dir = os.path.join(args.log_dir, assembly_id)
        if not os.path.isdir(log_dir): continue

        assembly_dir = os.path.join(args.assembly_dir, assembly_id)
        assert os.path.isdir(assembly_dir), f'{assembly_dir} does not exist'

        result_i_dir = os.path.join(args.result_dir, assembly_id)

        with open(os.path.join(log_dir, 'tree.pkl'), 'rb') as fp:
            tree = pickle.load(fp)
        with open(os.path.join(log_dir, 'stats.json'), 'r') as fp:
            stats = json.load(fp)
            sequence = stats['sequence'] 

        if sequence is not None:
            worker_args.append([
                asset_folder, assembly_dir, sequence, tree, result_i_dir, args.save_mesh, args.save_pose, args.save_part, args.save_path, args.save_record, args.save_all,
                args.reverse, args.show_fix, args.show_grasp, args.show_arm, args.gripper, args.scale, args.optimizer, not args.disable_save_sdf, args.clear_sdf, args.make_video, args.budget, args.camera_pos, args.camera_lookat
            ])

    try:
        for _ in parallel_execute(play_logged_plan, worker_args, args.num_proc):
            pass
    except KeyboardInterrupt:
        exit()
