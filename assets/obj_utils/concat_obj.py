import trimesh
import os
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--batch', action='store_true', default=False)
    args = parser.parse_args()

    if args.batch:
        for obj_file in os.listdir(args.input_path):
            obj_file = obj_file.lower()
            if not obj_file.endswith('.obj'):
                continue
            input_path = os.path.join(args.input_path, obj_file)
            output_path = os.path.join(args.output_path, os.path.basename(obj_file))
            os.makedirs(args.output_path, exist_ok=True)
            mesh = trimesh.load(input_path)
            if type(mesh) == trimesh.Scene:
                meshes = list(mesh.geometry.values())
                mesh = trimesh.util.concatenate(meshes)
            else:
                print('Not a scene, no need to concatenate.')
            mesh.export(output_path)
    else:
        assert os.path.isfile(args.input_path) and args.input_path.endswith('.obj')
        mesh = trimesh.load(args.input_path)
        if type(mesh) == trimesh.Scene:
            meshes = list(mesh.geometry.values())
            mesh = trimesh.util.concatenate(meshes)
        else:
            print('Not a scene, no need to concatenate.')
        mesh.export(args.output_path)
