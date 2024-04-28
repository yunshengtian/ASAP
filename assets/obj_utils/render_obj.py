import trimesh
import os
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--obj-path', type=str, required=True)
    args = parser.parse_args()

    assert os.path.isfile(args.obj_path) and args.obj_path.endswith('.obj')
    mesh = trimesh.load(args.obj_path)
    mesh.show()
