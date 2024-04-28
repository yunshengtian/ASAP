import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import pickle
import json

from plan_sequence.planner.base import SequencePlanner


def plot_logged_tree(log_dir, save, verbose=False):
    with open(os.path.join(log_dir, 'tree.pkl'), 'rb') as fp:
        tree = pickle.load(fp)
    if verbose:
        with open(os.path.join(log_dir, 'stats.json'), 'r') as fp:
            stats = json.load(fp)
        print(stats)

    SequencePlanner.plot_tree(tree, save_path=os.path.join(log_dir, 'tree.png') if save else None)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--save', default=False, action='store_true')
    args = parser.parse_args()

    plot_logged_tree(args.log_dir, args.save, True)
