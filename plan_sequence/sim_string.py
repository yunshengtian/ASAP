import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import json
import numpy as np
from scipy.spatial.transform import Rotation

from assets.load import load_config, load_pos_quat_dict, load_part_ids
from assets.transform import mat_dict_to_pos_quat_dict, pos_quat_to_mat


GROUND_Z = 0

KN = 1e3
KT = 1e3
MU = 0.5
DAMPING = 1e3


def _get_color(index, alpha=1.0):
    colors = [
        [210, 87, 89, 255 * alpha],
        [237, 204, 73, 255 * alpha],
        [60, 167, 221, 255 * alpha],
        [190, 126, 208, 255 * alpha],
        [108, 192, 90, 255 * alpha],
    ]
    colors = np.array(colors) / 255.0
    return colors[int(index) % 5]


def _get_fixed_color():
    return np.array([120, 120, 120, 255]) / 255.0


def get_body_color_dict(parts_fix, parts_free): # parts_free = parts_rest - parts_fix + [part_move]
    body_color_dict = {}
    for part_id in [*parts_fix, *parts_free]:
        if part_id in parts_free:
            color = _get_color(part_id)[:3]
        else:
            color = _get_fixed_color()[:3]
        body_color_dict[f'part{part_id}'] = color
    return body_color_dict


def _arr_to_str(arr):
    return ' '.join([str(x) for x in arr])


def _get_basic_sim_substring(gravity=False):
    substring = f'''
<redmax model="assemble">
<option integrator="BDF1" timestep="1e-3" gravity="0. 0. {-9.8 if gravity else 1e-12}"/>
<ground pos="0 0 {GROUND_Z}" normal="0 0 1"/>
'''
    return substring


def _get_path_sim_substring():
    substring = _get_basic_sim_substring(gravity=False)
    substring += f'''
<default>
    <ground_contact kn="{KN}" kt="0" mu="0" damping="{DAMPING}"/>
    <general_SDF_contact kn="{KN}" kt="0" mu="0.0" damping="{DAMPING}"/>
    <general_MultiSDF_contact kn="{KN}" kt="0" mu="0.0" damping="{DAMPING}"/>
</default>
'''
    return substring


def _get_stablility_sim_substring(gravity=True):
    substring = _get_basic_sim_substring(gravity=gravity)
    substring += f'''
<default>
    <ground_contact kn="{KN}" kt="{KT}" mu="{MU}" damping="{DAMPING}"/>
    <general_SDF_contact kn="{KN}" kt="{KT}" mu="{MU}" damping="{DAMPING}"/>
    <general_MultiSDF_contact kn="{KN}" kt="{KT}" mu="{MU}" damping="{DAMPING}"/>
</default>
'''
    return substring


def get_contact_sim_string(assembly_dir, parts=None, save_sdf=False, mat_dict=None):
    '''
    Simulation string for checking contact info
    '''
    if mat_dict is None:
        pos_dict, quat_dict = load_pos_quat_dict(assembly_dir)
    else:
        pos_dict, quat_dict = mat_dict_to_pos_quat_dict(mat_dict)

    sdf_args = 'load_sdf="true" save_sdf="true"' if save_sdf else ''
    string = _get_basic_sim_substring()
    if parts is None: parts = load_part_ids(assembly_dir)
    for part_id in parts:
        joint_type = 'fixed'
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{_arr_to_str(pos_dict[part_id])}" quat="{_arr_to_str(quat_dict[part_id])}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="SDF" filename="{assembly_dir}/{part_id}.obj" {sdf_args} pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" dx="0.05" res="20" mu="0" rgba="{_arr_to_str(_get_color(part_id))}"/>
    </link>
</robot>
'''
    string += f'''
</redmax>
'''
    return string


def get_path_sim_string(assembly_dir, parts_fix, part_move, parts_removed=[], save_sdf=False, pose=None, mat_dict=None, col_th=0.01, arm_string=None):
    '''
    Simulation string for checking path assemblability
    '''
    if pose is None: pose = np.eye(4)

    if mat_dict is None:
        pos_dict, quat_dict = load_pos_quat_dict(assembly_dir)
    else:
        pos_dict, quat_dict = mat_dict_to_pos_quat_dict(mat_dict)

    if len(parts_removed) > 0: # set removed parts to initial states
        pos_init_dict, quat_init_dict = load_pos_quat_dict(assembly_dir, transform='initial')
        for part_id in parts_removed:
            pos_dict[part_id] = pos_init_dict[part_id]
            quat_dict[part_id] = quat_init_dict[part_id]

    sdf_args = 'load_sdf="true" save_sdf="true"' if save_sdf else ''
    string = _get_path_sim_substring()
    for part_id in [part_move, *parts_fix, *parts_removed]:

        if part_id in parts_removed:
            if pos_dict[part_id] is None or quat_dict[part_id] is None:
                continue

        if part_id == part_move:
            joint_type = 'free3d-exp'
            color = _get_color(part_id)
        else:
            joint_type = 'fixed'
            # color = _get_fixed_color()
            color = _get_color(part_id)

        matrix = pos_quat_to_mat(pos_dict[part_id], quat_dict[part_id])
        matrix = pose @ matrix
        pos = matrix[:3, 3] + np.array([0, 0, GROUND_Z]) # NOTE: pay attention when combining mat_dict and pose
        quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]

        if pos is None or quat is None:
            continue

        if type(col_th) == dict:
            col_th_i = col_th[part_id]
        else:
            col_th_i = col_th

        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{_arr_to_str(pos)}" quat="{_arr_to_str(quat)}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="SDF" filename="{assembly_dir}/{part_id}.obj" {sdf_args} pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" dx="0.05" res="20" col_th="{col_th_i}" mu="0" rgba="{_arr_to_str(color)}"/>
    </link>
</robot>
'''
    if arm_string is not None:
        string += arm_string
    string += f'''
<contact>
'''
    string += f'''
    <ground_contact body="part{part_move}"/>
'''
    for part_id in parts_fix:
        string += f'''
    <general_SDF_contact general_body="part{part_id}" SDF_body="part{part_move}"/>
    <general_SDF_contact general_body="part{part_move}" SDF_body="part{part_id}"/>
'''
    string += f'''
</contact>
</redmax>
'''
    return string


def get_stability_sim_string(assembly_dir, parts_fix, parts_move, gravity=True, save_sdf=False, pose=None, mat_dict=None, col_th=0.01):
    '''
    Simulation string for checking stability
    '''
    if pose is None: pose = np.eye(4)

    if mat_dict is None:
        pos_dict, quat_dict = load_pos_quat_dict(assembly_dir)
    else:
        pos_dict, quat_dict = mat_dict_to_pos_quat_dict(mat_dict)

    sdf_args = 'load_sdf="true" save_sdf="true"' if save_sdf else ''
    string = _get_stablility_sim_substring(gravity=gravity)

    for part_id in [*parts_fix, *parts_move]:
        if part_id in parts_fix:
            joint_type = 'fixed'
            color = _get_fixed_color()
        else:
            joint_type = 'free3d-exp'
            color = _get_color(part_id)

        if mat_dict is None: # ground init
            matrix = pos_quat_to_mat(pos_dict[part_id], quat_dict[part_id])
            matrix = pose @ matrix
            pos = matrix[:3, 3] + np.array([0, 0, GROUND_Z]) # NOTE: pay attention when combining mat_dict and pose
            quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
        else: # ground cont
            pos = pos_dict[part_id]
            quat = quat_dict[part_id]

        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{_arr_to_str(pos)}" quat="{_arr_to_str(quat)}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="SDF" filename="{assembly_dir}/{part_id}.obj" {sdf_args} pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" dx="0.05" res="20" col_th="{col_th}" mu="0" rgba="{_arr_to_str(color)}"/>
    </link>
</robot>
'''
    string += f'''
<contact>
'''
    part_pairs_in_contact = []
    for i, part_move in enumerate(parts_move):
        for part_fix in parts_fix:
            part_pairs_in_contact.append((part_move, part_fix))
        for j in range(i + 1, len(parts_move)):
            part_pairs_in_contact.append((part_move, parts_move[j]))
        string += f'''
    <ground_contact body="part{part_move}"/>
'''
    for part_pair in part_pairs_in_contact:
        string += f'''
    <general_SDF_contact general_body="part{part_pair[0]}" SDF_body="part{part_pair[1]}"/>
    <general_SDF_contact general_body="part{part_pair[1]}" SDF_body="part{part_pair[0]}"/>
'''
    string += f'''
</contact>
</redmax>
'''
    return string
