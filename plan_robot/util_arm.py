import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
from scipy.spatial.transform import Rotation as R
from .ik.chain import Chain
from ikpy.link import URDFLink

from assets.transform import get_transform_matrix_quat, get_pos_euler_from_transform_matrix
from plan_robot.util_grasp import get_gripper_basis_directions


def get_arm_chain(base_pos=[0, 0, 0], base_euler=[0, 0, 0], scale=1):
    
    scale *= 100 # since urdf info are defined in m, convert to cm
    symbolic = True

    links = [
        URDFLink(
            name="linkbase",
            bounds=tuple([-np.inf, np.inf]),
            origin_translation=np.array(base_pos),
            origin_orientation=np.array(base_euler),
            rotation=None,
            translation=None,
            use_symbolic_matrix=symbolic,
            joint_type='fixed'
        ),

        URDFLink(
            name="link1",
            bounds=tuple([-6.28318530718, 6.28318530718]),
            origin_translation=np.array([0, 0, 0.267]) * scale,
            origin_orientation=np.array([0, 0, 0]),
            rotation=np.array([0, 0, 1]),
            translation=None,
            use_symbolic_matrix=symbolic,
            joint_type='revolute'
        ),

        URDFLink(
            name="link2",
            bounds=tuple([-2.059, 2.0944]),
            origin_translation=np.array([0, 0, 0]) * scale,
            origin_orientation=np.array([-1.5708, 0, 0]),
            rotation=np.array([0, 0, 1]),
            translation=None,
            use_symbolic_matrix=symbolic,
            joint_type='revolute'
        ),

        URDFLink(
            name="link3",
            bounds=tuple([-6.28318530718, 6.28318530718]),
            origin_translation=np.array([0, -0.293, 0]) * scale,
            origin_orientation=np.array([1.5708, 0, 0]),
            rotation=np.array([0, 0, 1]),
            translation=None,
            use_symbolic_matrix=symbolic,
            joint_type='revolute'
        ),

        URDFLink(
            name="link4",
            bounds=tuple([-0.19198, 3.927]),
            origin_translation=np.array([0.0525, 0, 0]) * scale,
            origin_orientation=np.array([1.5708, 0, 0]),
            rotation=np.array([0, 0, 1]),
            translation=None,
            use_symbolic_matrix=symbolic,
            joint_type='revolute'
        ),

        URDFLink(
            name="link5",
            bounds=tuple([-6.28318530718, 6.28318530718]),
            origin_translation=np.array([0.0775, -0.3425, 0]) * scale,
            origin_orientation=np.array([1.5708, 0, 0]),
            rotation=np.array([0, 0, 1]),
            translation=None,
            use_symbolic_matrix=symbolic,
            joint_type='revolute'
        ),

        URDFLink(
            name="link6",
            bounds=tuple([-1.69297, 3.14159265359]),
            origin_translation=np.array([0, 0, 0]) * scale,
            origin_orientation=np.array([1.5708, 0, 0]),
            rotation=np.array([0, 0, 1]),
            translation=None,
            use_symbolic_matrix=symbolic,
            joint_type='revolute'
        ),

        URDFLink(
            name="link7",
            bounds=tuple([-6.28318530718, 6.28318530718]),
            origin_translation=np.array([0.076, 0.097, 0]) * scale,
            origin_orientation=np.array([-1.5708, 0, 0]),
            rotation=np.array([0, 0, 1]),
            translation=None,
            use_symbolic_matrix=symbolic,
            joint_type='revolute'
        ),
    ]

    return Chain(links, active_links_mask=[False] + [True] * (len(links) - 1))


def get_default_arm_rest_q():
    return [2.35618014, -0.56582579, 6.28318531, 0.35527904, -6.28318531, 0.92109843, 0.] # NOTE: active, hardcoded


def intersecting_point_on_circle(x, y, ray_x, ray_y, R):
    # Calculate the coefficients of the quadratic equation in t
    a = ray_x**2 + ray_y**2
    b = 2 * (x * ray_x + y * ray_y)
    c = x**2 + y**2 - R**2
    
    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c
    assert discriminant >= 0, 'No intersection between the ray and the circle'
        
    # Calculate the positive solution for t
    t = (-b + np.sqrt(discriminant)) / (2 * a)
    
    # Calculate the intersecting point coordinates
    x_int = x + t * ray_x
    y_int = y + t * ray_y
    
    return x_int, y_int


def get_arm_pos(gripper_pos, gripper_ori, arm_scale, center=np.zeros(3)):

    radius = 50 * arm_scale # TODO: update radius

    # project all directions onto a 2d plane
    gripper_pos = gripper_pos.copy() - center
    gripper_pos[2] = 0
    gripper_ori = gripper_ori.copy()
    gripper_ori[2] = 0

    # calculate norms
    gripper_pos_norm = np.linalg.norm(gripper_pos)
    gripper_ori_norm = np.linalg.norm(gripper_ori)

    # determine arm direction (gripper -> arm)
    if np.isclose(gripper_ori_norm, 0):
        if np.isclose(gripper_pos_norm, 0):

            # if gripper is top-down at origin, use random direction
            arm_direction = np.random.random(3)
            arm_direction[2] = 0
            arm_direction /= np.linalg.norm(arm_direction)

        else:
            # if gripper is top-down at non-origin, use gripper position
            arm_direction = gripper_pos / gripper_pos_norm

    else:
        # if gripper is not top-down, arm should be aligned with gripper orientation
        x_int, y_int = intersecting_point_on_circle(gripper_pos[0], gripper_pos[1], -gripper_ori[0], -gripper_ori[1], radius)
        arm_direction = np.array([x_int, y_int, 0]) / radius
    
    arm_pos = radius * arm_direction + center
    return arm_pos


def get_arm_pos_candidates(gripper_pos, gripper_ori, arm_scale, center=np.zeros(3), n_angle=4):

    arm_pos_init = get_arm_pos(gripper_pos, gripper_ori, arm_scale, center) - center
    arm_pos_candidates = []

    for i in range(n_angle):
        angle = 2 * np.pi * i / n_angle
        rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]])
        arm_pos_candidates.append(rot_mat.dot(arm_pos_init) + center)

    return arm_pos_candidates


def get_arm_euler(arm_pos, center=np.zeros(3)):

    # get orientation
    arm_ori = arm_pos - center
    arm_ori[2] = 0
    arm_ori /= np.linalg.norm(arm_ori)

    # get euler angle
    yaw = np.arccos(np.clip(-arm_ori[0], -1, 1))
    if arm_ori[1] > 0:
        yaw = -yaw
    euler = np.array([0, 0, yaw])

    return euler


def get_gripper_pos_quat_from_arm_q(arm_chain, arm_q, gripper_type):

    ef_target_matrix = arm_chain.forward_kinematics(arm_q)
    ef_init_matrix = arm_chain.forward_kinematics([0] * len(arm_chain.links))[:3, :3]
    arm_euler = arm_chain.links[0].origin_orientation
    base_init_direction, l2r_init_direction = [0, 0, 1], R.from_euler('xyz', arm_euler).apply([0, -1, 0])
    gripper_init_matrix = R.align_vectors([base_init_direction, l2r_init_direction], [*get_gripper_basis_directions(gripper_type)])[0].as_matrix()
    gripper_target_matrix = ef_target_matrix[:3, :3] @ ef_init_matrix.T @ gripper_init_matrix
    gripper_pos, gripper_quat = ef_target_matrix[:3, 3], R.from_matrix(gripper_target_matrix).as_quat()[[3, 0, 1, 2]]
    
    return gripper_pos, gripper_quat


def get_gripper_qm_from_arm_q(arm_chain, arm_q, gripper_type):

    gripper_pos, gripper_quat = get_gripper_pos_quat_from_arm_q(arm_chain, arm_q, gripper_type)
    gripper_matrix = get_transform_matrix_quat(gripper_pos, gripper_quat)
    gripper_qm = get_pos_euler_from_transform_matrix(gripper_matrix)

    return gripper_qm


def get_gripper_path_from_arm_path(arm_chain, arm_path, gripper_type):
    
    gripper_path = []
    for arm_q in arm_path:
        gripper_path.append(get_gripper_qm_from_arm_q(arm_chain, arm_q, gripper_type))

    return gripper_path


def get_gripper_part_qm_from_arm_q(arm_chain, arm_q, gripper_type, part_transform):

    gripper_pos, gripper_quat = get_gripper_pos_quat_from_arm_q(arm_chain, arm_q, gripper_type)
    gripper_matrix = get_transform_matrix_quat(gripper_pos, gripper_quat)
    gripper_qm = get_pos_euler_from_transform_matrix(gripper_matrix)

    part_matrix = gripper_matrix @ part_transform
    part_qm = get_pos_euler_from_transform_matrix(part_matrix)

    return gripper_qm, part_qm


def get_gripper_part_path_from_arm_path(arm_chain, arm_path, gripper_type, part_transform):
    
    gripper_path, part_path = [], []
    for arm_q in arm_path:
        gripper_qm, part_qm = get_gripper_part_qm_from_arm_q(arm_chain, arm_q, gripper_type, part_transform)
        gripper_path.append(gripper_qm)
        part_path.append(part_qm)

    return gripper_path, part_path


def inverse_kinematics_correction(arm_chain, arm_q, gripper_type, gripper_quat):
    '''
    Computes the inverse kinematic on the specified target with correction on the last joint angle
    '''
    arm_q = arm_q.copy()
    arm_euler = arm_chain.links[0].origin_orientation

    ef_init_matrix = arm_chain.forward_kinematics([0] * len(arm_chain.links))[:3, :3] # end effector initial rotation when arm has all 0 joint angles
    base_init_direction, l2r_init_direction = [0, 0, 1], R.from_euler('xyz', arm_euler).apply([0, -1, 0])
    gripper_init_matrix = R.align_vectors([base_init_direction, l2r_init_direction], [*get_gripper_basis_directions(gripper_type)])[0].as_matrix() # gripper initial rotation

    ef_target_matrix = R.from_quat(gripper_quat[[1, 2, 3, 0]]).as_matrix() @ gripper_init_matrix.T @ ef_init_matrix # end effector target rotation for given gripper state
    ef_curr_matrix = arm_chain.forward_kinematics(arm_q) # end effector current rotation from current joint angles

    correct_rotvec = R.from_matrix(ef_curr_matrix[:3, :3].T @ ef_target_matrix).as_rotvec() # rotation vector for last joint angle correction (NOTE: ideally should be 0, 0, theta)
    
    arm_q[-1] += correct_rotvec[-1]
    if arm_q[-1] < -np.pi * 2:
        arm_q[-1] += np.pi * 2
    elif arm_q[-1] > np.pi * 2:
        arm_q[-1] -= np.pi * 2

    return arm_q
    

def check_inverse_kinematics_success(arm_chain, arm_q, gripper_type, gripper_quat, eps=1e-3, verbose=False):

    arm_q = arm_q.copy()
    arm_euler = arm_chain.links[0].origin_orientation

    ef_init_matrix = arm_chain.forward_kinematics([0] * len(arm_chain.links))[:3, :3] # end effector initial rotation when arm has all 0 joint angles
    base_init_direction, l2r_init_direction = [0, 0, 1], R.from_euler('xyz', arm_euler).apply([0, -1, 0])
    gripper_init_matrix = R.align_vectors([base_init_direction, l2r_init_direction], [*get_gripper_basis_directions(gripper_type)])[0].as_matrix() # gripper initial rotation

    ef_target_matrix = R.from_quat(gripper_quat[[1, 2, 3, 0]]).as_matrix() @ gripper_init_matrix.T @ ef_init_matrix # end effector target rotation for given gripper state
    ef_curr_matrix = arm_chain.forward_kinematics(arm_q) # end effector current rotation from current joint angles

    correct_rotvec = R.from_matrix(ef_curr_matrix[:3, :3].T @ ef_target_matrix).as_rotvec() # rotation vector for last joint angle correction (NOTE: ideally should be 0, 0, theta)
    deviation_norm = np.linalg.norm(correct_rotvec[:2])
    
    if verbose:
        print('IK rotation deviation: {:.4f}'.format(deviation_norm) + f', Success: {deviation_norm < eps}')

    return deviation_norm < eps


def get_arm_path_from_gripper_path(gripper_path, gripper_type, arm_chain, arm_q_init, optimizer='L-BFGS-B'):
    arm_path_local = []
    arm_q = arm_q_init.copy() if arm_q_init is not None else None # full
    for qm in gripper_path:
        gripper_pos = qm[:3]
        gripper_rot = R.from_euler('xyz', qm[3:])
        gripper_ori = gripper_rot.apply([0, 0, 1])
        gripper_quat = gripper_rot.as_quat()[[3, 0, 1, 2]]

        arm_q, ik_success = arm_chain.inverse_kinematics(target_position=gripper_pos, target_orientation=gripper_ori, orientation_mode='Z', n_restart=3, initial_position=arm_q, optimizer=optimizer)
        arm_q = inverse_kinematics_correction(arm_chain, arm_q, gripper_type, gripper_quat)

        if not ik_success: # IK not fully checked for every step in the path during planning
            print('inverse kinematics failed')
        arm_path_local.append(arm_q)
    return arm_path_local
