"""

Command line args

"""

import argparse
import os

curr_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def get_args():

    args = argparse.Namespace()

    args.max_pc_size = 1000
    args.feat_len = 512
    args.dropout = 0.0
    args.batch_norm = 1

    return args
