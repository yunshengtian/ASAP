import os
import json
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True, help='directory storing logs')
    parser.add_argument('--budget', type=int, default=None, help='maximum number of evaluations')
    args = parser.parse_args()

    for method in os.listdir(args.log_dir):
        method_dir = os.path.join(args.log_dir, method)
        if not os.path.isdir(method_dir): continue
        for seed in os.listdir(method_dir):
            seed_dir = os.path.join(method_dir, seed)
            if not os.path.isdir(seed_dir): continue

            success_ids = {}
            failed_ids = {}

            for assembly_id in os.listdir(seed_dir):
                assembly_dir = os.path.join(seed_dir, assembly_id)
                if os.path.isdir(assembly_dir):
                    json_path = os.path.join(assembly_dir, 'stats.json')
                    if not os.path.exists(json_path):
                        print(f'{assembly_dir} does not have stats.json')
                        continue
                    with open(json_path, 'r') as fp:
                        stats = json.load(fp)
                    if stats['success']:
                        if args.budget is not None:
                            success = stats['n_eval'] <= args.budget
                            if success:
                                success_ids[assembly_id] = stats['time']
                            else:
                                failed_ids[assembly_id] = stats['time']
                            
                        else:
                            success_ids[assembly_id] = stats['time']
                    else:
                        failed_ids[assembly_id] = stats['time']

            success_rate = '{:.2f}'.format(100 * len(success_ids) / (len(success_ids) + len(failed_ids)))
            print(f'Method: {method}, seed: {seed} | success: {len(success_ids)}, failed: {len(failed_ids)}, success rate: ' + success_rate)