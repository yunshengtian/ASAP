import numpy as np
import stl
from stl import mesh
import pywavefront
import glob
import os
from argparse import ArgumentParser


def stl_to_obj(stl_file, obj_file):
    # Load the STL files and add the vectors to the plot
    stl_mesh = mesh.Mesh.from_file(stl_file)
    with open(obj_file, 'w') as file:
        for i, facet in enumerate(stl_mesh.vectors):
            # Write vertices
            for vertex in facet:
                file.write('v {0} {1} {2}\n'.format(vertex[0], vertex[1], vertex[2]))
            # Write faces
            file.write('f {0} {1} {2}\n'.format(i*3+1, i*3+2, i*3+3))
        print(f"{stl_file} has been converted to {obj_file}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--batch', default=False, action='store_true')
    args = parser.parse_args()

    if args.batch:
        for stl_file in os.listdir(args.input_path):
            stl_file = stl_file.lower()
            if not stl_file.endswith('.stl'):
                continue
            input_path = os.path.join(args.input_path, stl_file)
            output_path = os.path.join(args.output_path, os.path.basename(stl_file).replace('.stl', '.obj'))
            os.makedirs(args.output_path, exist_ok=True)
            stl_to_obj(input_path, output_path)
    else:
        assert os.path.isfile(args.input_path) and args.input_path.endswith('.stl')
        stl_to_obj(args.input_path, args.output_path)
