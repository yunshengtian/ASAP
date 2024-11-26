'''
Load assembly meshes and transform
'''
import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import json
import trimesh
from scipy.spatial.transform import Rotation as R

from assets.color import get_color
from assets.transform import get_transform_matrix, q_to_pos_quat


def load_config(obj_dir):
    '''
    Load config from dir (initial/final states)
    '''
    config_path = os.path.join(obj_dir, 'config.json')
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r') as fp:
        config = json.load(fp)

    return config


def load_pos_quat_dict(obj_dir, transform='final'):
    config = load_config(obj_dir)
    pos_dict, quat_dict = {}, {}
    part_ids = load_part_ids(obj_dir)
    for part_id in part_ids:
        if transform == 'final':
            if config is None or part_id not in config:
                pos_dict[part_id], quat_dict[part_id] = np.array([0., 0., 0.]), np.array([1., 0., 0., 0.])
            else:
                part_cfg = config[part_id]
                state = part_cfg['final_state']
                if state is not None:
                    pos_dict[part_id], quat_dict[part_id] = q_to_pos_quat(state)
                else:
                    pos_dict[part_id], quat_dict[part_id] = np.array([0., 0., 0.]), np.array([1., 0., 0., 0.])
        elif transform == 'initial':
            if config is None or part_id not in config:
                pos_dict[part_id], quat_dict[part_id] = None, None
            else:
                part_cfg = config[part_id]
                state = part_cfg['initial_state'] if 'initial_state' in part_cfg else None
                if state is not None:
                    pos_dict[part_id], quat_dict[part_id] = q_to_pos_quat(state)
                else:
                    pos_dict[part_id], quat_dict[part_id] = None, None
        else:
            raise Exception(f'Unknown transform type: {transform}')
    return pos_dict, quat_dict


def load_part_ids(obj_dir):
    part_ids = []
    for obj_name in os.listdir(obj_dir):
        if obj_name.endswith('.obj'):
            part_id = obj_name.replace('.obj', '')
            part_ids.append(part_id)
    part_ids.sort()
    return part_ids


def load_assembly(obj_dir, transform='final'):
    '''
    Load the entire assembly from dir
    transform: 'final', 'initial' or 'none'
    '''
    obj_ids = load_part_ids(obj_dir)
    obj_ids = sorted(obj_ids)
    color_map = get_color(obj_ids, normalize=False)

    assembly = {}
    config = load_config(obj_dir)

    for obj_id in obj_ids:
        obj_name = f'{obj_id}.obj'
        obj_path = os.path.join(obj_dir, obj_name)
        mesh = trimesh.load_mesh(obj_path, process=False, maintain_order=True)
        mesh.visual.face_colors = color_map[obj_id]

        assembly[obj_id] = {
            'mesh': mesh,
            'name': obj_name,
            'transform': transform,
        }

        if config is not None and obj_id in config:
            if transform == 'final':
                mat = get_transform_matrix(config[obj_id]['final_state'])
                mesh.apply_transform(mat)
            elif transform == 'initial':
                mat = get_transform_matrix(config[obj_id]['initial_state'])
                mesh.apply_transform(mat)
            elif transform == 'none':
                pass
            else:
                raise Exception(f'Unknown transform type: {transform}')
            
            assembly[obj_id]['initial_state'] = config[obj_id]['initial_state'] if 'initial_state' in config[obj_id] else None
            assembly[obj_id]['final_state'] = config[obj_id]['final_state']
        else:
            assembly[obj_id]['initial_state'] = None
            assembly[obj_id]['final_state'] = None

    return assembly


def load_assembly_all_transformed(obj_dir):
    '''
    Load the entire assembly from dir with all transforms applied
    '''
    obj_ids = load_part_ids(obj_dir)
    obj_ids = sorted(obj_ids)
    color_map = get_color(obj_ids, normalize=False)

    assembly = {}
    config = load_config(obj_dir)

    for obj_id in obj_ids:
        obj_name = f'{obj_id}.obj'
        obj_path = os.path.join(obj_dir, obj_name)
        mesh = trimesh.load_mesh(obj_path, process=False, maintain_order=True)
        mesh.visual.face_colors = color_map[obj_id]

        mesh_none = mesh.copy()
        mesh_final = mesh.copy()
        mesh_initial = None
        if config is not None and obj_id in config:
            mat_final = get_transform_matrix(config[obj_id]['final_state'])
            mat_initial = get_transform_matrix(config[obj_id]['initial_state']) if 'initial_state' in config[obj_id] else None
            mesh_final.apply_transform(mat_final)
            if mat_initial is not None:
                mesh_initial = mesh.copy()
                mesh_initial.apply_transform(mat_initial)

        assembly[obj_id] = {
            'name': obj_name,
            'mesh': mesh_none,
            'mesh_final': mesh_final,
            'mesh_initial': mesh_initial,
        }
        if config is not None and obj_id in config:
            assembly[obj_id]['initial_state'] = config[obj_id]['initial_state'] if 'initial_state' in config[obj_id] else None
            assembly[obj_id]['final_state'] = config[obj_id]['final_state']

    return assembly


def load_paths(path_dir):
    '''
    Load motion of assembly meshes at every time step
    '''
    paths = {}
    for step in os.listdir(path_dir):
        obj_id = step.split('_')[1]
        step_dir = os.path.join(path_dir, step)
        if os.path.isdir(step_dir):
            path = []
            frame_files = []
            for frame_file in os.listdir(step_dir):
                if frame_file.endswith('.npy'):
                    frame_files.append(frame_file)
            frame_files.sort(key=lambda x: int(x.replace('.npy', '')))
            for frame_file in frame_files:
                frame_path = os.path.join(step_dir, frame_file)
                frame_transform = np.load(frame_path)
                path.append(frame_transform)
            paths[obj_id] = path
    return paths


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    assembly = load_assembly(args.dir)
    meshes = [assembly[obj_id]['mesh'] for obj_id in assembly]
    trimesh.Scene(meshes).show()
