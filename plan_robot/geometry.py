import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import trimesh
from assets.load import load_assembly
from assets.transform import get_scale_matrix, get_translate_matrix, get_revolute_matrix, get_transform_matrix_quat, get_pos_quat_from_pose


def load_panda_meshes(asset_folder, visual=False):
    meshes = {}
    dir_name = 'visual' if visual else 'collision'
    meshes['panda_hand'] = trimesh.load(os.path.join(asset_folder, 'panda', dir_name, 'hand.obj'))
    meshes['panda_leftfinger'] = trimesh.load(os.path.join(asset_folder, 'panda', dir_name, 'finger.obj'))
    meshes['panda_rightfinger'] = trimesh.load(os.path.join(asset_folder, 'panda', dir_name, 'finger.obj'))
    return meshes


def load_robotiq_85_meshes(asset_folder, visual=False):
    meshes = {}
    dir_name = 'visual' if visual else 'collision'
    postfix = 'fine' if visual else 'coarse'
    meshes['robotiq_base'] = trimesh.load(os.path.join(asset_folder, 'robotiq_85', dir_name, f'robotiq_base_{postfix}.obj'))
    for side_i in ['left', 'right']:
        for side_j in ['outer', 'inner']:
            for link in ['knuckle', 'finger']:
                meshes[f'robotiq_{side_i}_{side_j}_{link}'] = trimesh.load(os.path.join(asset_folder, 'robotiq_85', dir_name, f'{side_j}_{link}_{postfix}.obj'))
    return meshes


def load_robotiq_140_meshes(asset_folder, visual=False):
    meshes = {}
    dir_name = 'visual' if visual else 'collision'
    postfix = 'fine' if visual else 'coarse'
    meshes['robotiq_base'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'robotiq_base_{postfix}.obj'))
    for side in ['left', 'right']:
        for link in ['outer_knuckle', 'outer_finger', 'inner_finger']:
            meshes[f'robotiq_{side}_{link}'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'{link}_{postfix}.obj'))
        meshes[f'robotiq_{side}_pad'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'pad_{postfix}.obj'))
        meshes[f'robotiq_{side}_inner_knuckle'] = trimesh.load(os.path.join(asset_folder, 'robotiq_140', dir_name, f'inner_knuckle_{postfix}.obj'))
    return meshes


def load_gripper_meshes(gripper_type, asset_folder, visual=False):
    if gripper_type == 'panda':
        return load_panda_meshes(asset_folder, visual=visual)
    elif gripper_type == 'robotiq-85':
        return load_robotiq_85_meshes(asset_folder, visual=visual)
    elif gripper_type == 'robotiq-140':
        return load_robotiq_140_meshes(asset_folder, visual=visual)
    else:
        raise NotImplementedError


def load_arm_meshes(asset_folder, visual=False, convex=True):
    meshes = {}
    if visual:
        meshes['linkbase'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'visual', 'linkbase_smooth.obj'))
        for i in range(1, 8):
            meshes[f'link{i}'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'visual', f'link{i}_smooth.obj'))
    else:
        meshes['linkbase'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'collision', 'linkbase_vhacd.obj'))
        if convex: meshes['linkbase'] = meshes['linkbase'].convex_hull
        for i in range(1, 8):
            meshes[f'link{i}'] = trimesh.load(os.path.join(asset_folder, 'xarm7', 'collision', f'link{i}_vhacd.obj'))
            if convex: meshes[f'link{i}'] = meshes[f'link{i}'].convex_hull
    return meshes


def load_part_meshes(assembly_dir, transform='none'):
    assembly = load_assembly(assembly_dir, transform=transform)
    part_meshes = {f'part{part_id}': assembly[part_id]['mesh'] for part_id in assembly}
    return part_meshes


def transform_panda_meshes(meshes, pos, quat, scale, pose, open_ratio):
    meshes = {k: v.copy() for k, v in meshes.items()}
    meshes['panda_rightfinger'].apply_transform(get_scale_matrix([1, -1, 1]))
    meshes['panda_leftfinger'].apply_transform(get_translate_matrix([0, 4 * open_ratio, 5.84]))
    meshes['panda_rightfinger'].apply_transform(get_translate_matrix([0, -4 * open_ratio, 5.84]))
    pos, quat = get_pos_quat_from_pose(pos, quat, pose)
    for name, mesh in meshes.items():
        mesh.apply_transform(get_scale_matrix(scale))
        mesh.apply_transform(get_transform_matrix_quat(pos, quat))
    return meshes


def transform_robotiq_85_meshes(meshes, pos, quat, scale, pose, open_ratio):
    meshes = {k: v.copy() for k, v in meshes.items()}
    mats = {name: np.eye(4) for name in meshes}

    close_extent = 0.8757 * (1 - open_ratio)

    mats['robotiq_left_outer_knuckle'] = get_translate_matrix([3.06011444260539, 0.0, 6.27920162695395]) @ get_revolute_matrix('Y', -close_extent)
    mats['robotiq_left_outer_finger'] = mats['robotiq_left_outer_knuckle'] @ get_translate_matrix([3.16910442266543, 0.0, -0.193396375724605])
    mats['robotiq_left_inner_knuckle'] = get_translate_matrix([1.27000000001501, 0.0, 6.93074999999639]) @ get_revolute_matrix('Y', -close_extent)
    mats['robotiq_left_inner_finger'] = mats['robotiq_left_inner_knuckle'] @ get_translate_matrix([3.4585310861294003, 0.0, 4.5497019381797505]) @ get_revolute_matrix('Y', close_extent)

    mats['robotiq_right_outer_knuckle'] = get_transform_matrix_quat([-3.06011444260539, 0.0, 6.27920162695395], [0, 0, 0, 1]) @ get_revolute_matrix('Y', -close_extent)
    mats['robotiq_right_outer_finger'] = mats['robotiq_right_outer_knuckle'] @ get_translate_matrix([3.16910442266543, 0.0, -0.193396375724605])
    mats['robotiq_right_inner_knuckle'] = get_transform_matrix_quat([-1.27000000001501, 0.0, 6.93074999999639], [0, 0, 0, 1]) @ get_revolute_matrix('Y', -close_extent)
    mats['robotiq_right_inner_finger'] = mats['robotiq_right_inner_knuckle'] @ get_translate_matrix([3.4585310861294003, 0.0, 4.5497019381797505]) @ get_revolute_matrix('Y', close_extent)

    pos, quat = get_pos_quat_from_pose(pos, quat, pose)
    for name, mesh in meshes.items():
        mesh.apply_transform(mats[name])
        mesh.apply_transform(get_scale_matrix(scale))
        mesh.apply_transform(get_transform_matrix_quat(pos, quat))
    return meshes


def transform_robotiq_140_meshes(meshes, pos, quat, scale, pose, open_ratio):
    meshes = {k: v.copy() for k, v in meshes.items()}
    mats = {name: np.eye(4) for name in meshes}

    close_extent = 0.8757 * (1 - open_ratio)

    mats['robotiq_left_outer_knuckle'] = get_transform_matrix_quat([0.0, -3.0601, 5.4905], [0.41040502, 0.91190335, 0.0, 0.0]) @ get_revolute_matrix('X', -close_extent)
    mats['robotiq_left_outer_finger'] = mats['robotiq_left_outer_knuckle'] @ get_translate_matrix([0.0, 1.821998610742, 2.60018192872234])
    mats['robotiq_left_inner_finger'] = mats['robotiq_left_outer_finger'] @ get_transform_matrix_quat([0.0, 8.17554015893473, -2.82203446692936], [0.93501321, -0.35461287, 0.0, 0.0]) @ get_revolute_matrix('X', close_extent)
    mats['robotiq_left_pad'] = mats['robotiq_left_inner_finger'] @ get_transform_matrix_quat([0.0, 3.8, -2.3], [0, 0, 0.70710678, 0.70710678])
    mats['robotiq_left_inner_knuckle'] = get_transform_matrix_quat([0.0, -1.27, 6.142], [0.41040502, 0.91190335, 0.0, 0.0]) @ get_revolute_matrix('X', -close_extent)

    mats['robotiq_right_outer_knuckle'] = get_transform_matrix_quat([0.0, 3.0601, 5.4905], [0.0, 0.0, 0.91190335, 0.41040502]) @ get_revolute_matrix('X', -close_extent)
    mats['robotiq_right_outer_finger'] = mats['robotiq_right_outer_knuckle'] @ get_translate_matrix([0.0, 1.821998610742, 2.60018192872234])
    mats['robotiq_right_inner_finger'] = mats['robotiq_right_outer_finger'] @ get_transform_matrix_quat([0.0, 8.17554015893473, -2.82203446692936], [0.93501321, -0.35461287, 0.0, 0.0]) @ get_revolute_matrix('X', close_extent)
    mats['robotiq_right_pad'] = mats['robotiq_right_inner_finger'] @ get_transform_matrix_quat([0.0, 3.8, -2.3], [0, 0, 0.70710678, 0.70710678])
    mats['robotiq_right_inner_knuckle'] = get_transform_matrix_quat([0.0, 1.27, 6.142], [0.0, 0.0, -0.91190335, -0.41040502]) @ get_revolute_matrix('X', -close_extent)

    pos, quat = get_pos_quat_from_pose(pos, quat, pose)
    for name, mesh in meshes.items():
        mesh.apply_transform(mats[name])
        mesh.apply_transform(get_scale_matrix(scale))
        mesh.apply_transform(get_transform_matrix_quat(pos, quat))
    return meshes


def transform_gripper_meshes(gripper_type, meshes, pos, quat, scale, pose, open_ratio):
    if gripper_type == 'panda':
        return transform_panda_meshes(meshes, pos, quat, scale, pose, open_ratio)
    elif gripper_type == 'robotiq-85':
        return transform_robotiq_85_meshes(meshes, pos, quat, scale, pose, open_ratio)
    elif gripper_type == 'robotiq-140':
        return transform_robotiq_140_meshes(meshes, pos, quat, scale, pose, open_ratio)
    else:
        raise NotImplementedError


def transform_arm_meshes(meshes, chain, q, scale):
    meshes = {k: v.copy() for k, v in meshes.items()}
    matrices = chain.forward_kinematics(q, full_kinematics=True)
    for (name, mesh), matrix in zip(meshes.items(), matrices):
        mesh.apply_scale(scale)
        mesh.apply_transform(matrix)
    return meshes


def transform_part_mesh(mesh, pos, quat, pose=None):
    mesh = mesh.copy()
    if pose is not None:
        pos, quat = get_pos_quat_from_pose(pos, quat, pose)
    mesh.apply_transform(get_transform_matrix_quat(pos, quat))
    return mesh


def transform_part_meshes(meshes, pos_dict, quat_dict, pose=None):
    meshes = {k: v.copy() for k, v in meshes.items()}
    for name, mesh in meshes.items():
        pos, quat = pos_dict[name], quat_dict[name]
        if pose is not None:
            pos, quat = get_pos_quat_from_pose(pos, quat, pose)
        mesh.apply_transform(get_transform_matrix_quat(pos, quat))
    return meshes


def save_meshes(meshes, folder, include_color=False):
    for name, mesh in meshes.items():
        mesh.export(os.path.join(folder, f'{name}.obj'), header=None, include_color=include_color)
