import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import trimesh
import os

from assets.transform import get_transform_matrix_quat
from plan_path.pyplanners.rrt_connect import rrt_connect
from plan_path.pyplanners.rrt_star import rrt_star
from plan_path.pyplanners.smoothing import smooth_path
from plan_path.pyplanners.utils import compute_path_cost
from plan_robot.util_arm import get_arm_chain, get_gripper_pos_quat_from_arm_q
from plan_robot.geometry import load_arm_meshes, load_gripper_meshes, transform_arm_meshes, transform_gripper_meshes


def find_max_distance_index(path, distance_fn):
    max_distance = 0
    max_index = None
    
    for i in range(len(path) - 1):
        distance = distance_fn(path[i + 1], path[i])
        if distance > max_distance:
            max_distance = distance
            max_index = i
    
    return max_index


def interpolate_angle(start_angle, end_angle, bidirectional):
    if bidirectional: # if joint angle can take [-2pi, 2pi]
        angular_distance = (end_angle - start_angle) % (2 * np.pi)
        if angular_distance > np.pi:
            angular_distance -= 2 * np.pi
        interpolated_angle = (start_angle + angular_distance / 2) % (2 * np.pi)
    else:
        interpolated_angle = (start_angle + end_angle) / 2

    return interpolated_angle


def interpolate_angles(start_angle, end_angle, num_steps, bidirectional):
    if bidirectional: # if joint angle can take [-2pi, 2pi]
        angular_distance = (end_angle - start_angle) % (2 * np.pi)
        if angular_distance > np.pi:
            angular_distance -= 2 * np.pi
        step_size = angular_distance / num_steps
        interpolated_angles = []
        current_angle = start_angle
        for _ in range(num_steps + 1):  # including the start and end angles
            interpolated_angles.append(current_angle)
            current_angle = (current_angle + step_size) % (2 * np.pi)
    else:
        interpolated_angles = np.linspace(start_angle, end_angle, num_steps + 1)
    return np.array(interpolated_angles)


def interpolate_q(start_q, end_q):
    assert len(start_q) == 7 and len(end_q) == 7
    interpolated_q = []
    for i in range(len(start_q)):
        interpolated_q.append(interpolate_angle(start_q[i], end_q[i], bidirectional=i in [0, 2, 4, 6])) # hardcoded for xarm7
    return np.array(interpolated_q)


def interpolate_qs(start_q, end_q, num_steps):
    assert len(start_q) == 7 and len(end_q) == 7
    interpolated_qs = np.zeros((num_steps + 1, len(start_q)))
    for i in range(len(start_q)):
        interpolated_qs[:, i] = interpolate_angles(start_q[i], end_q[i], num_steps, bidirectional=i in [0, 2, 4, 6]) # hardcoded for xarm7
    return np.array(interpolated_qs)


def interpolate_arm_path(original_path, distance_fn, length):
    interpolated_path = original_path.copy()
    
    while len(interpolated_path) < length:
        max_index = find_max_distance_index(interpolated_path, distance_fn)
        
        start_q = interpolated_path[max_index]
        end_q = interpolated_path[max_index + 1]
        interpolated_q = interpolate_q(start_q, end_q)
        interpolated_path.insert(max_index + 1, interpolated_q)
    
    return interpolated_path


class ArmMotionPlanner:

    def __init__(self, base_pos, base_euler, scale, gripper_type):
        self.scale = scale
        self.arm_chain = get_arm_chain(base_pos, base_euler, scale)
        # self.step_size = 5 * scale
        self.step_size = 0.2 * scale
        self.min_num_steps = 10
        self.min_path_len = 300
        asset_folder = os.path.join(project_base_dir, 'assets')
        self.arm_meshes = load_arm_meshes(asset_folder, visual=False, convex=True)
        self.gripper_type = gripper_type
        self.gripper_meshes = load_gripper_meshes(gripper_type, asset_folder, visual=False)

    def transform_meshes(self, q, open_ratio, move_mesh, move_transform):

        # calculate arm transform
        q_full = self.arm_chain.active_to_full(q, initial_position=[0] * len(self.arm_chain.links))
        arm_meshes_i = transform_arm_meshes(self.arm_meshes, self.arm_chain, q_full, scale=self.scale)

        # calculate gripper transform
        gripper_pos, gripper_quat = get_gripper_pos_quat_from_arm_q(self.arm_chain, q_full, self.gripper_type)
        pose = np.eye(4) # TODO: check pose
        gripper_meshes_i = transform_gripper_meshes(self.gripper_type, self.gripper_meshes, gripper_pos, gripper_quat, self.scale, pose, open_ratio)

        # calculate move part transform
        if move_mesh is None:
            move_mesh_i = None
        else:
            gripper_matrix = get_transform_matrix_quat(gripper_pos, gripper_quat)
            move_matrix = np.matmul(gripper_matrix, move_transform)
            move_mesh_i = move_mesh.copy()
            move_mesh_i.apply_transform(move_matrix)

        return arm_meshes_i, gripper_meshes_i, move_mesh_i

    def visualize_meshes(self, q, open_ratio, move_mesh=None, move_transform=None, still_meshes=None):
        arm_meshes_i, gripper_meshes_i, move_mesh_i = self.transform_meshes(q, open_ratio, move_mesh, move_transform)
        all_meshes = [*arm_meshes_i.values(), *gripper_meshes_i.values()]
        if move_mesh_i is not None:
            all_meshes.append(move_mesh_i)
        if still_meshes is not None:
            all_meshes.extend(still_meshes)
        trimesh.Scene(all_meshes).show()

    def get_fns(self, move_mesh, move_transform, still_meshes, open_ratio, verbose=False):

        part_col_manager = trimesh.collision.CollisionManager()
        for idx, mesh in enumerate(still_meshes):
            part_col_manager.add_object(f'part_still_{idx}', mesh)

        def distance_fn(q1, q2):
            dist_sum = 0.0
            fk1 = self.arm_chain.forward_kinematics_active(q1, full_kinematics=True)
            fk2 = self.arm_chain.forward_kinematics_active(q2, full_kinematics=True)
            # NOTE: use all joints or only the endeffector joint
            # for i in range(len(fk1)):
            #     dist_sum += np.linalg.norm(fk1[i][:3, 3] - fk2[i][:3, 3])
            # dist_sum += np.linalg.norm(fk1[-1][:3, 3] - fk2[-1][:3, 3])
            if move_transform is None:
                dist_sum += np.linalg.norm(fk1[-1][:3, 3] - fk2[-1][:3, 3])
            else:
                part_fk1 = np.matmul(fk1[-1], move_transform)
                part_fk2 = np.matmul(fk2[-1], move_transform)
                dist_sum += np.linalg.norm(part_fk1[:3, 3] - part_fk2[:3, 3])
            return dist_sum

        def collision_fn(q):

            # transform arm and gripper meshes
            arm_meshes_i, gripper_meshes_i, move_mesh_i = self.transform_meshes(q, open_ratio, move_mesh, move_transform)

            # check collision between arm and ground
            for name, mesh in arm_meshes_i.items():
                if name == 'linkbase': continue
                if mesh.vertices[:, 2].min() < 0:
                    if verbose:
                        print('collision detected: arm and ground')
                    return True

            # check collision between gripper and ground
            for mesh in gripper_meshes_i.values():
                if mesh.vertices[:, 2].min() < 0:
                    if verbose: 
                        print('collision detected: gripper and ground')
                    return True

            # check collision between move part and ground
            if move_mesh_i is not None:
                if move_mesh_i.vertices[:, 2].min() < -0.05:
                    if verbose:
                        print('collision detected: move part and ground')
                    return True

            # check collision between arm and grippers
            arm_col_manager = trimesh.collision.CollisionManager()
            for name, mesh in arm_meshes_i.items():
                if name != 'link7':
                    arm_col_manager.add_object(name, mesh)

            gripper_col_manager = trimesh.collision.CollisionManager()
            for name, mesh in gripper_meshes_i.items():
                gripper_col_manager.add_object(name, mesh)

            if gripper_col_manager.in_collision_other(arm_col_manager):
                if verbose:
                    print('collision detected: arm and gripper')
                return True

            # check collision between arm and parts
            arm_col_manager.add_object('link7', arm_meshes_i['link7'])
            if arm_col_manager.in_collision_other(part_col_manager):
                if verbose:
                    print('collision detected: arm and part')
                return True
            
            # check collision between arm and move part
            if move_mesh_i is not None:
                if arm_col_manager.in_collision_single(move_mesh_i):
                    if verbose:
                        print('collision detected: arm and move part')
                    return True
            
            # check collision between gripper and parts
            if gripper_col_manager.in_collision_other(part_col_manager):
                if verbose:
                    print('collision detected: gripper and part')
                return True

            # check collision between move part and parts
            if move_mesh_i is not None:
                if part_col_manager.in_collision_single(move_mesh_i):
                    if verbose:
                        print('collision detected: move part and part')
                    return True

            # check arm self-intersection
            _, objs_in_collision = arm_col_manager.in_collision_internal(return_names=True)
            for obj_pair in objs_in_collision:
                if not self.arm_chain.check_neighboring_links(obj_pair[0], obj_pair[1]):
                    if verbose:
                        print('collision detected: arm self-intersection')
                    return True

            if verbose:
                print('no collision detected')

            return False

        def sample_fn():
            while True:
                q = self.arm_chain.sample_joint_angles_active()
                # ef_matrix = self.arm_chain.forward_kinematics_active(q)
                # ef_pos = ef_matrix[:3, 3]
                # if ef_pos[0] < -20 or ef_pos[0] > 30:
                #     continue
                # if ef_pos[1] < -20 or ef_pos[1] > 30:
                #     continue
                # if ef_pos[2] < 0 or ef_pos[2] > 40:
                #     continue
                if not collision_fn(q):
                    return q

        def extend_fn(q1, q2):
            q_dist = distance_fn(q1, q2)
            num_steps = int(np.ceil(q_dist / self.step_size))
            if num_steps < self.min_num_steps: num_steps = self.min_num_steps
            interpolated_qs = interpolate_qs(q1, q2, num_steps)
            for i in range(num_steps):
                yield interpolated_qs[i + 1]
        
        return distance_fn, sample_fn, extend_fn, collision_fn

    def plan_with_grasp(self, start, goal, move_mesh, move_transform, still_meshes, open_ratio, total_rrt_trials=5, max_rrt_iter=20, max_smooth_iter=1000, verbose=False):
        '''
        Plan arm reaching with moving part in held and a single gripper open ratio
        ----------------------------------
        start: start q of arm (active)
        goal: goal q of arm (active)
        move_mesh: mesh to move
        still_meshes: dict of still meshes {name: mesh}
        open_ratio: open ratio of gripper
        '''
        distance_fn, sample_fn, extend_fn, collision_fn = self.get_fns(move_mesh, move_transform, still_meshes, open_ratio, verbose=False)

        # print('visualize start')
        # self.visualize_meshes(start, open_ratio, move_mesh, move_transform, still_meshes)
        # print('visualize goal')
        # self.visualize_meshes(goal, open_ratio, move_mesh, move_transform, still_meshes)

        if collision_fn(start):
            if verbose:
                print('start is in collision')
            return None
        if collision_fn(goal):
            if verbose:
                print('goal is in collision')
            return None
        
        paths = []
        costs = []
        for i in range(total_rrt_trials):
            path = rrt_connect(start, goal, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=max_rrt_iter)
            cost = compute_path_cost(path, cost_fn=distance_fn)
            if path is None:
                cost = np.inf
            else:
                paths.append(path)
                costs.append(cost)
            if verbose:
                print('RRT trial', i, 'cost', cost)

        path = paths[np.argmin(costs)]
        path.insert(0, start)

        if verbose:
            print(f'path planned (len: {len(path)})')

        # smooth path
        path = smooth_path(path, extend_fn, collision_fn, distance_fn, None, sample_fn, verbose=verbose, max_iterations=max_smooth_iter)
        path.insert(0, start)

        in_collision = False
        for q in path:
            if collision_fn(q):
                in_collision = True

        # interpolate path
        path = interpolate_arm_path(path, distance_fn, self.min_path_len)

        if verbose:
            print(f'smoothing path completed (len: {len(path)}, collision: {in_collision})')

        path = [self.arm_chain.active_to_full(q, initial_position=[0] * len(self.arm_chain.links)) for q in path]
        return path

    def plan_without_grasp(self, start, goal, part_meshes, open_ratio_init, total_rrt_trials=5, max_rrt_iter=20, max_smooth_iter=1000, verbose=False):
        '''
        Plan arm retracting with no part in held and multiple gripper open ratios
        '''
        for open_ratio in np.linspace(open_ratio_init, 1.0, 4)[1:]:

            path = self.plan_with_grasp(start, goal, move_mesh=None, move_transform=None, still_meshes=part_meshes, open_ratio=open_ratio, 
                total_rrt_trials=total_rrt_trials, max_rrt_iter=max_rrt_iter, max_smooth_iter=max_smooth_iter, verbose=verbose)

            if path is not None:
                return path, open_ratio

        return None, None
