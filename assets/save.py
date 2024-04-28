'''
Save transformed assembly meshes
'''
import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np

from assets.transform import get_transform_matrix
from assets.load import load_assembly


def sample_path(path, n_frame=None):
    if n_frame is None: return path
    if len(path) > n_frame:
        interp_path = []
        for i in range(n_frame - 1):
            interp_path.append(path[int(i / n_frame * len(path))])
        interp_path.append(path[-1])
    else:
        interp_path = path
    return interp_path


def extrapolate_path(path, n_frame=None):
    if n_frame is None: return path
    direction = path[-1] - path[-2]
    extrap_path = path.copy()
    for i in range(n_frame):
        extrap_path.append(path[-1] + direction * (i + 1))
    return extrap_path


def save_path(obj_dir, path, n_frame=None):
    '''
    Save motion of assembly meshes at every time step
    '''
    if path is None: return
    path = sample_path(path, n_frame)

    os.makedirs(obj_dir, exist_ok=True)
    for frame, state in enumerate(path):
        frame_transform = get_transform_matrix(state)
        np.save(os.path.join(obj_dir, f'{frame}.npy'), frame_transform)


def save_path_new(obj_dir, path, n_frame=None):
    '''
    Save motion of assembly meshes at every time step
    '''
    if path is None: return
    path = sample_path(path, n_frame)

    os.makedirs(obj_dir, exist_ok=True)
    for frame, transform in enumerate(path):
        np.save(os.path.join(obj_dir, f'{frame}.npy'), transform)


def save_path_all_objects(obj_dir, matrices, n_frame=None):
    '''
    Save motion of all meshes at every time step
    '''
    if matrices is None: return
    matrices = sample_path(matrices, n_frame)

    os.makedirs(obj_dir, exist_ok=True)
    for frame, matrix_dict in enumerate(matrices):
        frame_dir = os.path.join(obj_dir, f'{frame}')
        os.makedirs(frame_dir, exist_ok=True)
        for name, matrix in matrix_dict.items():
            np.save(os.path.join(frame_dir, f'{name}.npy'), matrix)


def save_posed_mesh(obj_dir, assembly_dir, parts, pose):
    '''
    Save posed assembly meshes at a single step
    '''
    assembly = load_assembly(assembly_dir, transform='none')
    os.makedirs(obj_dir)
    for part_id, part_data in assembly.items():
        name = part_data['name']
        mesh = part_data['mesh']
        if part_id in parts:
            matrix = get_transform_matrix(part_data['final_state'])
            if pose is not None:
                matrix = pose @ matrix
            mesh.apply_transform(matrix)
            obj_target_path = os.path.join(obj_dir, name)
            mesh.export(obj_target_path, header=None, include_color=False)


def clear_saved_sdfs(obj_dir):
    for file in os.listdir(obj_dir):
        file_path = os.path.join(obj_dir, file)
        if file_path.endswith('.sdf'):
            try:
                os.remove(file_path)
            except:
                pass
