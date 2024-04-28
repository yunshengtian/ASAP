import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import networkx as nx

from assets.load import load_part_ids
from plan_sequence.feasibility_check import check_stable_noforce
from plan_sequence.assets.fix_invalid import save_new_parts


def fix_explode(source_dir, target_dir, debug=0, render=False, verbose=False):

    asset_folder = os.path.join(project_base_dir, 'assets')
    parts = load_part_ids(source_dir)

    success, G = check_stable_noforce(asset_folder, source_dir, parts, debug=debug, render=render)
    if verbose:
        if success:
            print(f'[fix_explode] all parts are stable')
        else:
            print(f'[fix_explode] exist unstable parts')

    if len(G.nodes) < 3:
        if verbose:
            print(f'[fix_explode] less than 3 parts after fix')
        return False

    Gs = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gs[0])
    
    if len(G.nodes) < 3:
        if verbose:
            print(f'[fix_explode] less than 3 parts after fix')
        return False

    save_new_parts(source_dir, target_dir, [str(node) for node in G.nodes])
    
    return True


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--source-dir', type=str, required=True)
    parser.add_argument('--target-dir', type=str, required=True)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--render', default=False, action='store_true')
    args = parser.parse_args()

    success = fix_explode(args.source_dir, args.target_dir, debug=args.debug, render=args.render, verbose=True)
    print(f'Success: {success}')
