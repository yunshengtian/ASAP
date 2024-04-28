'''
Postprocessing for multi assembly
'''

import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import json
import numpy as np
import trimesh

from assets.subdivide import subdivide_to_size


def norm_to_transform(center, scale):
    transform0 = np.eye(4)
    transform0[:3, 3] = -center
    transform1 = np.eye(4)
    transform1[:3, :3] = np.eye(3) * scale
    return transform1.dot(transform0)


def scale_to_transform(scale):
    transform = np.eye(4)
    transform[:3, :3] = np.eye(3) * scale
    return transform


def com_to_transform(com):
    transform = np.eye(4)
    transform[:3, 3] = com
    return transform


def normalize(meshes, bbox_size=10):

    # compute normalization factors
    vertex_all_stacked = np.vstack([mesh.vertices for mesh in meshes])
    min_box = vertex_all_stacked.min(axis=0)
    max_box = vertex_all_stacked.max(axis=0)
    center = (min_box + max_box) / 2
    scale = max_box - min_box
    scale_factor = bbox_size / np.max(scale)
    scale_transform = norm_to_transform(center, scale_factor)

    # normalization
    for mesh in meshes:
        mesh.apply_transform(scale_transform)
    return meshes


def rescale(meshes, scale_factor):

    scale_transform = scale_to_transform(scale_factor)

    # normalization
    for mesh in meshes:
        mesh.apply_transform(scale_transform)
    return meshes


def get_scale(meshes):
    vertex_all_stacked = np.vstack([mesh.vertices for mesh in meshes])
    min_box = vertex_all_stacked.min(axis=0)
    max_box = vertex_all_stacked.max(axis=0)
    scale = max_box - min_box
    return np.min(scale), np.max(scale)


def get_oriented_bounding_box(mesh):
    _, extents = trimesh.bounds.oriented_bounds(mesh)
    return extents


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def process_mesh(source_dir, target_dir, subdivide, rescale_factor=None, verbose=False):
    '''
    1. Scale to unit bounding box
    2. Check thickness (optional)
    3. Subdivide (optional)
    4. Translate
    '''
    # order objs
    source_files = []
    for source_file in os.listdir(source_dir):
        if not source_file.endswith('.obj'): continue
        source_files.append(source_file)
    source_files.sort()
    
    # load meshes
    meshes = []
    obj_ids = {}
    watertight = True
    for source_file in source_files:
        source_id = source_file.replace('.obj', '')
        obj_ids[source_id] = source_file
        source_path = os.path.join(source_dir, source_file)
        mesh = trimesh.load_mesh(source_path, process=False, maintain_order=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = as_mesh(mesh)
            if not isinstance(mesh, trimesh.Trimesh):
                print(f'Failed to load {source_path} as mesh')
                return False
            
        if not mesh.is_watertight:
            print(f'Mesh {source_path} is not watertight')
            
            # mesh.fix_normals()
            # mesh.fill_holes()
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)
            if not mesh.is_watertight:
                watertight = False
                print('non-watertight mesh not fixed')
            else:
                print('non-watertight mesh fixed')
            
        meshes.append(mesh)
    if not watertight:
        return False

    # scale
    # meshes = normalize(meshes)
    if rescale_factor is not None:
        meshes = rescale(meshes, rescale_factor)

    # get scale
    scale_min, scale_max = get_scale(meshes)

    # subdivide mesh
    # max_edge = 0.5
    max_edge = min(scale_min, scale_max / 20)
    if subdivide:
        for i in range(len(meshes)):
            meshes[i] = subdivide_to_size(meshes[i], max_edge=max_edge)

    # make target directory
    os.makedirs(target_dir, exist_ok=True)

    # write processed objs
    os.makedirs(target_dir, exist_ok=True)
    assert len(meshes) == len(obj_ids)
    for mesh, obj_id in zip(meshes, obj_ids.keys()):
        obj_target_path = os.path.join(target_dir, f'{obj_id}.obj')
        mesh.export(obj_target_path, header=None, include_color=False)
        if verbose:
            print(f'Processed obj written to {obj_target_path}')

    return True


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--source-dir', type=str, required=True)
    parser.add_argument('--target-dir', type=str, required=True)
    parser.add_argument('--subdivide', default=False, action='store_true')
    parser.add_argument('--rescale', type=float, default=None)
    args = parser.parse_args()

    success = process_mesh(args.source_dir, args.target_dir, args.subdivide, rescale_factor=args.rescale, verbose=True)
    print(f'Success: {success}')
