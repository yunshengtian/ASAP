import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from argparse import ArgumentParser
from make_sdf import make_sdf

from utils.parallel import parallel_execute


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='multi_assembly', help='directory storing all assemblies')
    parser.add_argument('--sdf-dx', type=float, default=0.05, help='grid resolution of SDF')
    parser.add_argument('--num-proc', type=int, default=8)
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')
    assemblies_dir = os.path.join(asset_folder, args.dir)
    assembly_ids = []
    for assembly_id in os.listdir(assemblies_dir):
        assembly_dir = os.path.join(assemblies_dir, assembly_id)
        if os.path.isdir(assembly_dir):
            assembly_ids.append(assembly_id)
    assembly_ids.sort()

    worker_args = []
    for assembly_id in assembly_ids:
        assembly_dir = os.path.join(assemblies_dir, assembly_id)
        worker_args.append([asset_folder, assembly_dir, args.sdf_dx])

    for _ in parallel_execute(make_sdf, worker_args, args.num_proc, show_progress=True):
        pass
