import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import copy
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from argparse import ArgumentParser

from assets.transform import get_transform_matrix_quat, get_transform_matrix_euler, get_pos_euler_from_transform_matrix


class Grasp:

    def __init__(self, pos, quat, open_ratio, arm_pos=None, arm_euler=None, arm_q=None, arm_q_init=None):
        self.pos = pos
        self.quat = quat
        self.open_ratio = open_ratio
        self.arm_pos = arm_pos
        self.arm_euler = arm_euler
        self.arm_q = arm_q
        self.arm_q_init = arm_q_init

    def copy(self):
        return copy.deepcopy(self)


def compute_antipodal_pairs(obj, sample_budget=100, visualize=False, verbose=False):
    '''
    Compute pairs of antipodal points for a given mesh through ray casting and surface sampling
    '''
    # load mesh
    if type(obj) == str:
        mesh = trimesh.load(obj)
    elif type(obj) == trimesh.Trimesh:
        mesh = obj
    else:
        raise NotImplementedError

    # randomly sample surface points
    sample_points, sample_face_idx = mesh.sample(sample_budget, return_index=True)
    sample_normals = mesh.face_normals[sample_face_idx]

    # ray casting for computing antipodal points
    init_offset = 0.05 # move ray origins slightly inside the surface
    ray_caster = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    intersect_points, ray_idx, intersect_face_idx = ray_caster.intersects_location(sample_points - init_offset * sample_normals, -sample_normals)
    intersect_normals = mesh.face_normals[intersect_face_idx]

    antipodal_idx = np.einsum('ij,ij->i', intersect_normals, -sample_normals[ray_idx]) > 0.95
    antipodal_pairs = np.stack([sample_points[ray_idx][antipodal_idx], intersect_points[antipodal_idx]], axis=1)
    antipodal_pairs = sort_antipodal_pairs(mesh, antipodal_pairs)
    grasping_widths = np.linalg.norm(antipodal_pairs[:, 0, :] - antipodal_pairs[:, 1, :], axis=1)
    # TODO: what if cannot find antipodal pairs?

    if visualize:
        pts = [antipodal_pairs[i][0] for i in range(len(antipodal_pairs))]
        pc = trimesh.PointCloud(pts, colors=[255, 0, 0, 255])
        trimesh.Scene([mesh, pc]).show()

    if verbose:
        print(f'Found {len(antipodal_pairs)} pairs of antipodal points')

    return antipodal_pairs


def sort_antipodal_pairs(mesh, antipodal_pairs):
    '''
    Sort antipodal pairs based on the distance from center of mesh to center of antipodal points
    '''
    if len(antipodal_pairs) == 0: return antipodal_pairs
    antipodal_center = np.mean(antipodal_pairs, axis=1)
    dist = np.linalg.norm(antipodal_center - mesh.centroid, axis=1)
    sorted_idx = np.argsort(dist)
    return antipodal_pairs[sorted_idx]


def get_panda_grasp_base_offset():
    return 10.4


def get_robotiq_85_grasp_base_offset(open_ratio):
    return 3.5 + np.sin(0.9208 + (1 - open_ratio) * 0.8757) * 5.715 + 6.93075


def get_robotiq_140_grasp_base_offset(open_ratio):
    return 1.5 + 3.8 + np.sin(0.8680 + (1 - open_ratio) * 0.8757) * 10 + 5.4905 + 0.5 # NOTE: extra 0.5 for the finger tip


def get_gripper_grasp_base_offset(gripper_type, open_ratio):
    if gripper_type == 'panda':
        return get_panda_grasp_base_offset()
    elif gripper_type == 'robotiq-85':
        return get_robotiq_85_grasp_base_offset(open_ratio)
    elif gripper_type == 'robotiq-140':
        return get_robotiq_140_grasp_base_offset(open_ratio)
    else:
        raise NotImplementedError


def get_panda_basis_directions():
    return [0, 0, -1], [0, -1, 0]


def get_robotiq_85_basis_directions():
    return [0, 0, -1], [-1, 0, 0]


def get_robotiq_140_basis_directions():
    return [0, 0, -1], [0, 1, 0]


def get_gripper_basis_directions(gripper_type):
    if gripper_type == 'panda':
        return get_panda_basis_directions()
    elif gripper_type == 'robotiq-85':
        return get_robotiq_85_basis_directions()
    elif gripper_type == 'robotiq-140':
        return get_robotiq_140_basis_directions()
    else:
        raise NotImplementedError


def get_gripper_pos_quat(gripper_type, grasp_center, base_direction, l2r_direction, open_ratio, scale):
    offset = get_gripper_grasp_base_offset(gripper_type, open_ratio)
    pos = grasp_center + base_direction * offset * scale
    rotation = R.align_vectors([base_direction, l2r_direction], [*get_gripper_basis_directions(gripper_type)])[0]
    quat = rotation.as_quat()[[3, 0, 1, 2]]
    return pos, quat


def generate_gripper_states(gripper_type, antipodal_points, scale, open_ratio, sample_budget=10, disassembly_direction=None):
    antipodal_points = np.array(antipodal_points, dtype=float)
    grasp_center = np.mean(antipodal_points, axis=0)
    l2r_direction = antipodal_points[1] - antipodal_points[0]
    l2r_direction /= np.linalg.norm(l2r_direction)

    # get one base direction
    if disassembly_direction is None:
        random_direction = np.random.rand(3)
        random_direction /= np.linalg.norm(random_direction)
        base_direction_basis = np.cross(l2r_direction, random_direction)
    else:
        disassembly_direction = np.array(disassembly_direction, dtype=float)
        disassembly_direction /= np.linalg.norm(disassembly_direction)
        if np.dot(disassembly_direction, l2r_direction) in [1, -1]: # disassembly direction is parallel to l2r direction
            random_direction = np.random.rand(3)
            random_direction /= np.linalg.norm(random_direction)
            base_direction_basis = np.cross(disassembly_direction, random_direction)
        else:
            base_direction_basis = np.cross(disassembly_direction, l2r_direction)
    base_direction_basis /= np.linalg.norm(base_direction_basis)
    
    base_direction_list = []
    # generate base directions by sampling angles
    for angle in np.linspace(0, 2 * np.pi, sample_budget + 1)[:-1]:
        base_rotation = R.from_rotvec(angle * l2r_direction)
        base_direction = base_rotation.apply(base_direction_basis)
        base_direction /= np.linalg.norm(base_direction)
        if disassembly_direction is not None and np.dot(base_direction, disassembly_direction) < 0:
            continue
        base_direction_list.append(base_direction)
    
    # sort base directions by angle with disassembly direction
    if disassembly_direction is not None:
        base_direction_list.sort(key=lambda x: np.dot(x, disassembly_direction), reverse=True)

    # generate state candidates from base directions
    pos_list, quat_list = [], []
    for base_direction in base_direction_list:
        pos, quat = get_gripper_pos_quat(gripper_type, grasp_center, base_direction, l2r_direction, open_ratio, scale)
        pos_list.append(pos)
        quat_list.append(quat)
    return pos_list, quat_list


def get_panda_open_ratio(antipodal_points, scale):
    antipodal_points = np.array(antipodal_points, dtype=float)
    antipodal_width = np.linalg.norm(antipodal_points[1] - antipodal_points[0])
    open_ratio = antipodal_width / 8 / scale
    if open_ratio > 1:
        return None
    else:
        return open_ratio


def get_robotiq_85_open_ratio(antipodal_points, scale):
    antipodal_points = np.array(antipodal_points, dtype=float)
    antipodal_width = np.linalg.norm(antipodal_points[1] - antipodal_points[0])
    if antipodal_width / scale > 3.92853109 * 2:
        return None
    else:
        return 1.0 - (np.arccos((antipodal_width / scale / 2 + 0.8 - 1.27) / 5.715) - 0.9208) / 0.8757


def get_robotiq_140_open_ratio(antipodal_points, scale):
    antipodal_points = np.array(antipodal_points, dtype=float)
    antipodal_width = np.linalg.norm(antipodal_points[1] - antipodal_points[0])
    if antipodal_width / scale > 7.22376574 * 2:
        return None
    else:
        return 1.0 - (np.arccos((antipodal_width / scale / 2 + 0.325 + 2.3 - 1.7901 - 1.27) / 10) - 0.8680) / 0.8757


def get_gripper_open_ratio(gripper_type, antipodal_points, scale):
    if gripper_type == 'panda':
        return get_panda_open_ratio(antipodal_points, scale)
    elif gripper_type == 'robotiq-85':
        return get_robotiq_85_open_ratio(antipodal_points, scale)
    elif gripper_type == 'robotiq-140':
        return get_robotiq_140_open_ratio(antipodal_points, scale)
    else:
        raise NotImplementedError


def get_panda_finger_states(open_ratio, gripper_scale):
    finger_open_extent = 4 * open_ratio * gripper_scale
    return {
        'panda_leftfinger': [finger_open_extent],
        'panda_rightfinger': [finger_open_extent],
    }


def get_robotiq_85_finger_states(open_ratio):
    finger_states = {}
    for side_i in ['left', 'right']:
        for side_j in ['outer', 'inner']:
            for link in ['knuckle', 'finger']:
                name = f'{side_i}_{side_j}_{link}'
                if name in ['left_outer_finger', 'right_outer_finger']: continue
                sign = -1 if name in ['left_inner_finger', 'right_inner_knuckle'] else 1
                finger_states[f'robotiq_{name}'] = [sign * (1 - open_ratio) * 0.8757]
    return finger_states


def get_robotiq_140_finger_states(open_ratio):
    finger_states = {}
    for side in ['left', 'right']:
        for link in ['outer_knuckle', 'inner_finger', 'inner_knuckle']:
            name = f'{side}_{link}'
            sign = 1 if link == 'inner_finger' or name == 'left_outer_knuckle' else -1
            finger_states[f'robotiq_{name}'] = [sign * (1 - open_ratio) * 0.8757]
    return finger_states


def get_gripper_finger_states(gripper_type, open_ratio, gripper_scale, suffix=None):
    if gripper_type == 'panda':
        finger_states = get_panda_finger_states(open_ratio, gripper_scale)
    elif gripper_type == 'robotiq-85':
        finger_states = get_robotiq_85_finger_states(open_ratio)
    elif gripper_type == 'robotiq-140':
        finger_states = get_robotiq_140_finger_states(open_ratio)
    else:
        raise NotImplementedError
    if suffix is not None:
        finger_states = {f'{key}_{suffix}': value for key, value in finger_states.items()}
    return finger_states


def get_gripper_base_name(gripper_type, suffix=None):
    if gripper_type == 'panda':
        name = 'panda_hand'
    elif gripper_type in ['robotiq-85', 'robotiq-140']:
        name = 'robotiq_base'
    else:
        raise NotImplementedError
    if suffix is not None:
        name = f'{name}_{suffix}'
    return name


def get_gripper_hand_names(gripper_type, suffix=None):
    if gripper_type == 'panda':
        names = ['panda_hand']
    elif gripper_type == 'robotiq-85':
        names = ['robotiq_base']
        for side_i in ['left', 'right']:
            for side_j in ['outer', 'inner']:
                for link in ['knuckle', 'finger']:
                    name = f'{side_i}_{side_j}_{link}'
                    if name not in ['left_inner_finger', 'right_inner_finger']:
                        names.append(f'robotiq_{name}')
    elif gripper_type == 'robotiq-140':
        names = ['robotiq_base']
        for side_i in ['left', 'right']:
            for side_j in ['outer', 'inner']:
                for link in ['knuckle', 'finger']:
                    name = f'{side_i}_{side_j}_{link}'
                    names.append(f'robotiq_{name}')
    else:
        raise NotImplementedError
    if suffix is not None:
        names = [f'{name}_{suffix}' for name in names]
    return names


def get_gripper_finger_names(gripper_type, suffix=None):
    if gripper_type == 'panda':
        names = ['panda_leftfinger', 'panda_rightfinger']
    elif gripper_type == 'robotiq-85':
        names = ['robotiq_left_inner_finger', 'robotiq_right_inner_finger']
    elif gripper_type == 'robotiq-140':
        names = ['robotiq_left_pad', 'robotiq_right_pad']
    else:
        raise NotImplementedError
    if suffix is not None:
        names = [f'{name}_{suffix}' for name in names]
    return names


def get_gripper_path_from_part_path(part_path, gripper_pos, gripper_quat):
    T_g_0 = get_transform_matrix_quat(gripper_pos, gripper_quat) # get initial transform matrix of gripper
    T_p = [get_transform_matrix_euler(p[:3], p[3:]) for p in part_path] # get transform matrix of part path
    T_p_0 = T_p[0] # initial transform matrix of part
    T_p_g = np.linalg.inv(T_p_0) @ T_g_0 # get transform matrix from path to gripper
    T_g = [T_p_i @ T_p_g for T_p_i in T_p] # get transform matrix of gripper in global coordinate
    gripper_path = [get_pos_euler_from_transform_matrix(T_g_i) for T_g_i in T_g] # get gripper path in global coordinate
    return gripper_path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--obj-path', type=str, required=True)
    parser.add_argument('--sample-budget', type=int, default=100)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    compute_antipodal_pairs(args.obj_path, args.sample_budget, args.visualize)
