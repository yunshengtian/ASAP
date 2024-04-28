import numpy as np
import trimesh
import os
from argparse import ArgumentParser


def dae_to_obj(input_path, output_path, material):
    if material:
        mtl_path = output_path.replace('.obj', '.mtl')
        obj, mtl = trimesh.exchange.obj.export_obj(trimesh.load(input_path), include_color=True, include_texture=True, return_texture=True, header='', mtl_name=mtl_path)
        with open(output_path, 'w') as fp:
            fp.write(obj)
        for mtl_path, mtl_data in mtl.items():
            with open(mtl_path, 'wb') as fp:
                fp.write(mtl_data)
    else:
        obj = trimesh.exchange.obj.export_obj(trimesh.load(input_path), header='')
        with open(output_path, 'w') as fp:
            fp.write(obj)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--batch', default=False, action='store_true')
    parser.add_argument('--material', default=False, action='store_true')
    args = parser.parse_args()

    if args.batch:
        for dae_file in os.listdir(args.input_path):
            dae_file = dae_file.lower()
            if not dae_file.lower().endswith('.dae'):
                continue
            input_path = os.path.join(args.input_path, dae_file)
            output_path = os.path.join(args.output_path, os.path.basename(dae_file).replace('.dae', '.obj'))
            dae_to_obj(input_path, output_path, args.material)
    else:
        assert os.path.isfile(args.input_path) and args.input_path.endswith('.dae')
        dae_to_obj(args.input_path, args.output_path, args.material)
