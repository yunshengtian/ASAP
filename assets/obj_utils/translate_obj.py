import numpy as np
import trimesh
import os
from argparse import ArgumentParser


def translate_obj(input_path, output_path, translation):
    mesh = trimesh.load(input_path)
    mesh.apply_translation(translation)
    mesh.export(output_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--translation', type=float, nargs='+', required=True)
    parser.add_argument('--batch', default=False, action='store_true')
    args = parser.parse_args()

    if args.batch:
        os.makedirs(args.output_path, exist_ok=True)
        for obj_file in os.listdir(args.input_path):
            obj_file = obj_file.lower()
            if not obj_file.lower().endswith('.obj'):
                continue
            input_path = os.path.join(args.input_path, obj_file)
            output_path = os.path.join(args.output_path, obj_file)
            translate_obj(input_path, output_path, args.translation)
    else:
        assert os.path.isfile(args.input_path) and args.input_path.endswith('.obj')
        translate_obj(args.input_path, args.output_path, args.translation)
