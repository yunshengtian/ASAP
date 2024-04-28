import os
import trimesh
import numpy as np
from argparse import ArgumentParser


def visualize_meshes(mesh_dir, path_dir):
    times = sorted([int(time) for time in os.listdir(path_dir) if time.isnumeric()])
    for time in times:
        mat_dir = os.path.join(path_dir, str(time))
        meshes = {}
        for name in os.listdir(mat_dir):
            if name.endswith('.npy'):
                mat = np.load(os.path.join(mat_dir, name))
                mesh = trimesh.load(os.path.join(mesh_dir, name[:-4] + '.obj'))
                mesh.apply_transform(mat)
                meshes[name] = mesh
        trimesh.Scene(list(meshes.values())).show()
    return meshes


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mesh-dir', type=str, required=True)
    parser.add_argument('--path-dir', type=str, required=True)
    args = parser.parse_args()

    visualize_meshes(args.mesh_dir, args.path_dir)
