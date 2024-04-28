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
    return f'''
<redmax model="robotiq_2f85_description">
<option integrator="BDF1" timestep="1e-3" gravity="0. 0. 1e-12"/>

<default>
    <joint lim_stiffness="1e5" damping="1e4"/>
    <general_SDF_contact kn="1e5" kt="5e3" mu="1.0" damping="1e3"/>
</default>

<ground pos="0 0 0" normal="0 0 1"/>

<robot>
    <link name="robotiq_base">
        <joint name="robotiq_base" type="fixed" pos="{arr_to_str(pos)}" quat="{arr_to_str(quat)}"/>
        <body name= "robotiq_base" type = "mesh" filename = "robotiq_85/visual/robotiq_base_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        <link name="robotiq_left_outer_knuckle">
            <joint name = "robotiq_left_outer_knuckle" type="revolute" pos="{3.06011444260539 * scale} 0.0 {6.27920162695395 * scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 -1.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_outer_knuckle" type = "mesh" filename = "robotiq_85/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_left_outer_finger">
                <joint name="robotiq_left_outer_finger" type="fixed" pos="{3.16910442266543 * scale} 0.0 {-0.193396375724605 * scale}" quat="1.0 0.0 0.0 0.0"/>
                <body name= "robotiq_left_outer_finger" type = "mesh" filename = "robotiq_85/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
        <link name="robotiq_left_inner_knuckle">
            <joint name = "robotiq_left_inner_knuckle" type="revolute" pos="{1.27000000001501 * scale} 0.0 {6.93074999999639 * scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 -1.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_inner_knuckle" type = "mesh" filename = "robotiq_85/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_left_inner_finger">
                <joint name = "robotiq_left_inner_finger" type="revolute" pos="{3.4585310861294003 * scale} 0.0 {4.5497019381797505 * scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 -1.0 0.0" lim="-0.8757 0.0"/>
                <body name= "robotiq_left_inner_finger" type = "mesh" filename = "robotiq_85/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
        <link name="robotiq_right_outer_knuckle">
            <joint name = "robotiq_right_outer_knuckle" type="revolute" pos="{-3.06011444260539 * scale} 0.0 {6.27920162695395 * scale}" quat="0.0 0.0 0.0 1.0" axis="0.0 -1.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_right_outer_knuckle" type = "mesh" filename = "robotiq_85/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_right_outer_finger">
                <joint name="robotiq_right_outer_finger" type="fixed" pos="{3.16910442266543 * scale} 0.0 {-0.193396375724605 * scale}" quat="1.0 0.0 0.0 0.0"/>
                <body name= "robotiq_right_outer_finger" type = "mesh" filename = "robotiq_85/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
        <link name="robotiq_right_inner_knuckle">
            <joint name = "robotiq_right_inner_knuckle" type="revolute" pos="{-1.27000000001501 * scale} 0.0 {6.93074999999639 * scale}" quat="0.0 0.0 0.0 1.0" axis="0.0 1.0 0.0" lim="-0.8757 0.0"/>
            <body name= "robotiq_right_inner_knuckle" type = "mesh" filename = "robotiq_85/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_right_inner_finger">
                <joint name = "robotiq_right_inner_finger" type="revolute" pos="{3.4585310861294003 * scale} 0.0 {4.5497019381797505 * scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 1.0 0.0" lim="0.0 0.8757" damping="0.0"/>
                <body name= "robotiq_right_inner_finger" type = "mesh" filename = "robotiq_85/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{scale} {scale} {scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
    </link>
</robot>

<variable>
    <endeffector joint="robotiq_base" pos="0 0 0" radius="{0.8 * scale}"/>

    <endeffector joint="robotiq_left_outer_knuckle" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_left_outer_finger" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_left_inner_knuckle" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_left_inner_finger" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_right_outer_knuckle" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_right_outer_finger" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_right_inner_knuckle" pos="0 0 0" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_right_inner_finger" pos="0 0 0" radius="{0.8 * scale}"/>

    <endeffector joint="robotiq_left_inner_finger" pos="{-0.8 * scale} 0 {3.5 * scale}" radius="{0.1 * scale}"/>
    <endeffector joint="robotiq_right_inner_finger" pos="{-0.8 * scale} 0 {3.5 * scale}" radius="{0.1 * scale}"/>

    <endeffector joint="robotiq_left_outer_finger" pos="{0.3 * scale} 0 {4.745 * scale}" radius="{0.8 * scale}"/>
    <endeffector joint="robotiq_right_outer_finger" pos="{0.3 * scale} 0 {4.745 * scale}" radius="{0.8 * scale}"/>
</variable>
</redmax>
    '''


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
    sim.set_joint_q_init('robotiq_left_inner_knuckle', [0.8757 * ratio])
    sim.set_joint_q_init('robotiq_left_inner_finger', [-0.8757 * ratio])

    sim.set_joint_q_init('robotiq_right_outer_knuckle', [0.8757 * ratio])
    sim.set_joint_q_init('robotiq_right_inner_knuckle', [-0.8757 * ratio])
    sim.set_joint_q_init('robotiq_right_inner_finger', [0.8757 * ratio])

    sim.reset(backward_flag=False)

    variables = sim.get_variables()
    variables = np.array(variables).reshape(-1, 3)
    
    SimRenderer.replay(sim, record=False)
