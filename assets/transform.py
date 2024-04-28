'''
Transformation utilities
'''

import numpy as np
from scipy.spatial.transform import Rotation


def get_transform_matrix(state):
    '''
    Get transformation matrix of the given state and center of mass
    COM (or previous translation) is necessary when applying rotation on translated meshes
    '''
    if len(state) == 3: # translation only
        transform = np.eye(4)
        transform[:3, 3] = state
        return transform
    elif len(state) == 6: # translation + rotation
        translation, rotation = state[:3], state[3:]
        rotation = Rotation.from_euler('xyz', rotation).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform
    else:
        raise NotImplementedError


def transform_pt_by_matrix(pt, matrix):
    '''
    Transform a single xyz pt by a 4x4 matrix
    '''
    pt = np.array(pt)
    if len(pt) == 3:
        v = np.append(pt, 1.0)
        v = matrix @ v
        return v[0:3]
    elif len(pt) == 6:
        return  get_pos_euler_from_transform_matrix(matrix @ get_transform_matrix(pt))
    else:
        raise NotImplementedError


def transform_pts_by_matrix(pts, matrix):
    '''
    Transform an array of xyz pts (n, 3) by a 4x4 matrix
    '''
    pts = np.array(pts)
    if len(pts.shape) == 1:
        if len(pts) == 3:
            v = np.append(pts, 1.0)
        elif len(pts) == 4:
            v = pts
        else:
            raise NotImplementedError
        v = matrix @ v
        return v[0:3]
    elif len(pts.shape) == 2:
        # transpose first
        if pts.shape[1] == 3:
            # pad the points with ones to be (n, 4)
            v = np.hstack([pts, np.ones((len(pts), 1))]).T
        elif pts.shape[1] == 4:
            v = pts.T
        else:
            raise NotImplementedError
        v = matrix @ v
        # transpose and crop back to (n, 3)
        return v.T[:, 0:3]
    else:
        raise NotImplementedError


def transform_pts_by_state(pts, state):
    matrix = get_transform_matrix(state)
    return transform_pts_by_matrix(pts, matrix)


def pos_quat_to_mat(pos, quat):
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = Rotation.from_quat(np.array(quat)[[1, 2, 3, 0]]).as_matrix()
    return mat


def mat_to_pos_quat(mat):
    pos = mat[:3, 3]
    quat = Rotation.from_matrix(mat[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return pos, quat


def mat_dict_to_pos_quat_dict(mat_dict):
    pos_dict, quat_dict = {}, {}
    for part_id, mat in mat_dict.items():
        pos, quat = mat_to_pos_quat(mat)
        pos_dict[part_id] = pos
        quat_dict[part_id] = quat
    return pos_dict, quat_dict


def q_to_pos_quat(q):
    pos = q[:3]
    if len(q) == 3:
        quat = [1, 0, 0, 0]
    elif len(q) == 6:
        quat = Rotation.from_euler('xyz', q[3:]).as_quat()[[3, 0, 1, 2]].tolist()
    else:
        raise Exception(f'incorrect state dimension')
    return pos, quat


def pos_quat_to_q(pos, quat):
    euler = Rotation.from_quat(np.array(quat)[[1, 2, 3, 0]]).as_euler('xyz')
    return np.concatenate([pos, euler])


def q_dict_to_pos_quat_dict(q_dict):
    pos_dict, quat_dict = {}, {}
    for part_id, q in q_dict.items():
        pos, quat = q_to_pos_quat(q)
        pos_dict[part_id] = pos
        quat_dict[part_id] = quat
    return pos_dict, quat_dict


def get_pos_quat_from_pose(pos, quat, pose):
    if pose is None: return pos, quat
    matrix = pos_quat_to_mat(pos, quat)
    matrix = pose @ matrix
    pos = matrix[:3, 3]
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return pos, quat


def get_pos_quat_from_pose_global(pos, quat, pose):
    if pose is None: return pos, quat
    matrix = pos_quat_to_mat(pos, quat)
    matrix = matrix @ pose
    pos = matrix[:3, 3]
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return pos, quat


def get_translate_matrix(translate):
    translate_matrix = np.eye(4)
    translate_matrix[:3, 3] = translate
    return translate_matrix


def get_scale_matrix(scale):
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale
    return scale_matrix


def get_revolute_matrix(axis, angle):
    if axis == 'X':
        axis_arr = np.array([1, 0, 0])
    elif axis == 'Y':
        axis_arr = np.array([0, 1, 0])
    elif axis == 'Z':
        axis_arr = np.array([0, 0, 1])
    else:
        raise NotImplementedError
    revolute_matrix = np.eye(4)
    revolute_matrix[:3, :3] = Rotation.from_rotvec(axis_arr * angle).as_matrix()
    return revolute_matrix


def get_transform_matrix_quat(pos, quat): # quat: wxyz
    pos = np.array(pos)
    quat = np.array(quat)
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation.as_matrix()
    transform_matrix[:3, 3] = pos
    return transform_matrix


def get_transform_matrix_euler(pos, euler): # euler: xyz
    pos = np.array(pos)
    euler = np.array(euler)
    rotation = Rotation.from_euler('xyz', euler)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation.as_matrix()
    transform_matrix[:3, 3] = pos
    return transform_matrix


def get_pos_euler_from_transform_matrix(transform_matrix):
    pos = transform_matrix[:3, 3]
    rotation = Rotation.from_matrix(transform_matrix[:3, :3])
    euler = rotation.as_euler('xyz')
    return np.concatenate([pos, euler])


def quat_to_euler(quat):
    return Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler('xyz')


def get_transform_from_path(path, n_sample=None):
    if n_sample is not None:
        path = path.copy()[::len(path) // n_sample]
    T = [get_transform_matrix_euler(p[:3], p[3:]) for p in path]
    return T


def get_sequential_transform_from_path(path, n_sample=None):
    if n_sample is not None:
        path = path.copy()[::len(path) // n_sample]
    T = [get_transform_matrix_euler(p[:3], p[3:]) for p in path]
    T_seq = [np.eye(4)] + [T[i] @ np.linalg.inv(T[i-1]) for i in range(1, len(T))]
    return T_seq
