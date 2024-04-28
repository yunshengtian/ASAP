import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import redmax_py as redmax
import os
from argparse import ArgumentParser
from tqdm import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R

from utils.renderer import SimRenderer


def arr_to_str(arr):
    return ' '.join([str(x) for x in arr])


def create_robotiq_gripper_xml(pos=[0, 0, 5], quat=[1, 0, 0, 0], scale=1):
    string = f'''
<redmax model="robotiq_2f140_description">
<option integrator="BDF1" timestep="1e-3" gravity="0. 0. 1e-12"/>

<ground pos="0 0 0" normal="0 0 1"/>

<robot>
    <link name="robotiq_base">
        <joint name="robotiq_base" type="fixed" pos="{arr_to_str(pos)}" quat="{arr_to_str(quat)}"/>
        <body name= "robotiq_base" type = "mesh" filename = "robotiq_140/visual/robotiq_base_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        <link name="robotiq_left_outer_knuckle">
            <joint name = "robotiq_left_outer_knuckle" type="revolute" pos="0 {-3.0601 * scale} {5.4905 * scale}" quat="0.41040502 0.91190335 0.0 0.0" axis="-1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_outer_knuckle" type = "mesh" filename = "robotiq_140/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_left_outer_finger">
                <joint name = "robotiq_left_outer_finger" type="fixed" pos="0 {1.821998610742 * scale} {2.60018192872234 * scale}" quat="1.0 0.0 0.0 0.0" axis="1.0 0.0 0.0"/>
                <body name= "robotiq_left_outer_finger" type = "mesh" filename = "robotiq_140/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                <link name="robotiq_left_inner_finger">
                    <joint name = "robotiq_left_inner_finger" type="revolute" pos="0 {8.17554015893473 * scale} {-2.82203446692936 * scale}" quat="0.93501321 -0.35461287 0.0 0.0" axis="1.0 0.0 0.0"/>
                    <body name= "robotiq_left_inner_finger" type = "mesh" filename = "robotiq_140/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    <link name="robotiq_left_pad">
                        <joint name = "robotiq_left_pad" type="fixed" pos="0 {3.8 * scale} {-2.3 * scale}" quat="0.0 0.0 0.70710678 0.70710678" axis="1.0 0.0 0.0"/>
                        <body name= "robotiq_left_pad" type = "mesh" filename = "robotiq_140/visual/pad_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    </link>
                </link>
            </link>
        </link>
        <link name="robotiq_left_inner_knuckle">
            <joint name = "robotiq_left_inner_knuckle" type="revolute" pos="0 {-1.27 * scale} {6.142 * scale}" quat="0.41040502 0.91190335 0.0 0.0" axis="1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_inner_knuckle" type = "mesh" filename = "robotiq_140/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        </link>
        <link name="robotiq_right_outer_knuckle">
            <joint name = "robotiq_right_outer_knuckle" type="revolute" pos="0 {3.0601 * scale} {5.4905 * scale}" quat="0.0 0.0 0.91190335 0.41040502" axis="1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_right_outer_knuckle" type = "mesh" filename = "robotiq_140/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_right_outer_knuckle">
                <joint name = "robotiq_right_outer_finger" type="fixed" pos="0 {1.821998610742 * scale} {2.60018192872234 * scale}" quat="1.0 0.0 0.0 0.0" axis="1.0 0.0 0.0"/>
                <body name= "robotiq_right_outer_finger" type = "mesh" filename = "robotiq_140/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                <link name="robotiq_right_inner_finger">
                    <joint name = "robotiq_right_inner_finger" type="revolute" pos="0 {8.17554015893473 * scale} {-2.82203446692936 * scale}" quat="0.93501321 -0.35461287 0.0 0.0" axis="1.0 0.0 0.0"/>
                    <body name= "robotiq_right_inner_finger" type = "mesh" filename = "robotiq_140/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    <link name="robotiq_right_pad">
                        <joint name = "robotiq_right_pad" type="fixed" pos="0 {3.8 * scale} {-2.3 * scale}" quat="0.0 0.0 0.70710678 0.70710678" axis="1.0 0.0 0.0"/>
                        <body name= "robotiq_right_pad" type = "mesh" filename = "robotiq_140/visual/pad_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    </link>
                </link>
            </link>
        </link>
        <link name="robotiq_right_inner_knuckle">
            <joint name = "robotiq_right_inner_knuckle" type="revolute" pos="0 {1.27 * scale} {6.142 * scale}" quat="0.0 0.0 -0.91190335 -0.41040502" axis="1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_right_inner_knuckle" type = "mesh" filename = "robotiq_140/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        </link>
    </link>
</robot>

<variable>
    <endeffector joint="robotiq_left_outer_knuckle" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_left_outer_finger" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_left_inner_knuckle" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_left_inner_finger" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_right_outer_knuckle" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_right_outer_finger" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_right_inner_knuckle" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_right_inner_finger" pos="0 0 0" radius="{0.8 * scale}"/>

    <endeffector joint="robotiq_left_inner_finger" pos="0 5.3 -2.3" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_right_inner_finger" pos="0 5.3 -2.3" radius="{0.8 * scale}"/>
</variable>
</redmax>
    '''
    return string


def get_body_mesh(sim, body_name):
    vertices = sim.get_body_vertices(body_name, world_frame=True).T
    faces = sim.get_body_faces(body_name).T
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pos', type=float, nargs=3, default=[0, 0, 0], help='position of the model')
    parser.add_argument('--quat', type=float, nargs=4, default=[1, 0, 0, 0], help='quaternion of the model')
    parser.add_argument('--scale', type=float, default=1, help='scale of the model')
    parser.add_argument('--close', type=int, default=0, help='how much to close/open the gripper (0-255)')
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=None, help='camera lookat')
    parser.add_argument('--camera-pos', type=float, nargs=3, default=None, help='camera position')
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')
    xml_string = create_robotiq_gripper_xml(pos=args.pos, quat=args.quat, scale=args.scale)

    sim = redmax.Simulation(xml_string, asset_folder)
    if args.camera_lookat is not None:
        sim.viewer_options.camera_lookat = args.camera_lookat
    if args.camera_pos is not None:
        sim.viewer_options.camera_pos = args.camera_pos

    ratio = args.close / 255
    
    sim.set_joint_q_init('robotiq_left_outer_knuckle', [0.8757 * ratio])
    sim.set_joint_q_init('robotiq_left_inner_finger', [0.8757 * ratio])
    sim.set_joint_q_init('robotiq_left_inner_knuckle', [-0.8757 * ratio])

    sim.set_joint_q_init('robotiq_right_outer_knuckle', [-0.8757 * ratio])
    sim.set_joint_q_init('robotiq_right_inner_finger', [0.8757 * ratio])
    sim.set_joint_q_init('robotiq_right_inner_knuckle', [-0.8757 * ratio])

    sim.reset(backward_flag=False)

    variables = sim.get_variables()
    variables = np.array(variables).reshape(-1, 3)
    print(variables)
    
    SimRenderer.replay(sim, record=False)
