import numpy as np
from scipy.spatial.transform import Rotation as R
from ikpy.chain import Chain as IKPyChain
from .inverse_kinematics import inverse_kinematic_optimization


class Chain(IKPyChain):

    def forward_kinematics_active(self, angles, **kwargs):
        """Computes the forward kinematics on the active joints of the chain

        Parameters
        ----------
        angles: numpy.array
            The angles of the active joints of the chain

        Returns
        -------
        numpy.array:
            The transformation matrix of the end effector
        """
        return self.forward_kinematics(self.active_to_full(angles, initial_position=[0] * len(self.links)), **kwargs)

    def check_ori_success(self, actual_q, target, orientation_mode):

        ef_actual_matrix = self.forward_kinematics(actual_q)[:3, :3]
        if orientation_mode == 'X':
            actual_ori = ef_actual_matrix[:3, 0]
            target_ori = target[:3, 0]
        elif orientation_mode == 'Y':
            actual_ori = ef_actual_matrix[:3, 1]
            target_ori = target[:3, 1]
        elif orientation_mode == 'Z':
            actual_ori = ef_actual_matrix[:3, 2]
            target_ori = target[:3, 2]
        else:
            return True # does not support other modes
        
        return np.dot(actual_ori, target_ori) > 0.99

    def inverse_kinematics_frame(self, target, initial_position=None, n_restart=3, **kwargs):
        """Computes the inverse kinematic on the specified target

        Parameters
        ----------
        target: numpy.array
            The frame target of the inverse kinematic, in meters. It must be 4x4 transformation matrix
        initial_position: numpy.array
            Optional : the initial position of each joint of the chain. Defaults to 0 for each joint
        kwargs: See ikpy.inverse_kinematics.inverse_kinematic_optimization

        Returns
        -------
        list:
            The list of the positions of each joint according to the target. Note : Inactive joints are in the list.
        """
        # Checks on input
        target = np.array(target)
        if target.shape != (4, 4):
            raise ValueError("Your target must be a 4x4 transformation matrix")

        if initial_position is None:
            initial_position = [0] * len(self.links)

        success = False
        n_optimized = 0
        assert n_restart >= 1 or n_restart == -1 # -1 means infinite looping
        while not success and ((n_optimized < n_restart and n_restart >= 1) or n_restart == -1):
            solution, success = inverse_kinematic_optimization(self, target, starting_nodes_angles=initial_position, **kwargs)
            success = self.check_ori_success(solution, target, kwargs['orientation_mode'])
            n_optimized += 1
            initial_position = self.sample_joint_angles()
        return solution, success

    def sample_joint_angles(self):

        joint_angles = []

        for link in self.links:
            if link.joint_type != 'fixed':
                joint_angle = np.random.uniform(link.bounds[0], link.bounds[1])
                joint_angles.append(joint_angle)
            else:
                joint_angles.append(0)

        return np.array(joint_angles)

    def sample_joint_angles_active(self):

        joint_angles = []

        for link in self.links:
            if link.joint_type != 'fixed':
                joint_angle = np.random.uniform(link.bounds[0], link.bounds[1])
                joint_angles.append(joint_angle)

        return np.array(joint_angles)

    def check_neighboring_links(self, linka_name, linkb_name):

        all_link_names = [link.name for link in self.links]
        linka_idx = all_link_names.index(linka_name)
        linkb_idx = all_link_names.index(linkb_name)

        if linka_idx == linkb_idx - 1 or linka_idx == linkb_idx + 1:
            return True
        else:
            return False
