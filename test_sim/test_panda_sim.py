'''
Test gripper simulation by specifying pos, quat, scale, and open ratio of the gripper
'''

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


def create_panda_gripper_xml(pos=[0, 0, 5], quat=[1, 0, 0, 0], scale=1):
    return f'''
<redmax model="panda_gripper">

<option integrator="BDF1" timestep="1e-3" gravity="0. 0. 1e-12"/>

<default>
    <joint lim_stiffness="1e2" damping="1"/>
    <ground_contact kn="1e6" kt="1e3" mu="0.8" damping="5e1"/>
    <general_SDF_contact kn="1e5" kt="1e3" mu="0.1" damping="1"/>
</default>

<ground pos="0 0 0" normal="0 0 1"/>

<robot>
    <link name="panda_hand">
        <joint name="panda_hand" type="free3d" pos="{arr_to_str(pos)}" quat="{arr_to_str(quat)}"/>
        <body name="panda_hand" type="mesh" scale="{scale} {scale} {scale}" filename="panda/visual/hand.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
        <link name="panda_leftfinger">
            <joint name="panda_leftfinger" type="prismatic" axis="0 1 0" pos="0 0 {5.84 * scale}" quat="1 0 0 0" lim="0.0 {4 * scale}"/>
            <body name="panda_leftfinger" type="mesh" scale="{scale} {scale} {scale}" filename="panda/visual/finger.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
        </link>
        <link name="panda_rightfinger">
            <joint name="panda_rightfinger" type="prismatic" axis="0 -1 0" pos="0 0 {5.84 * scale}" quat="1 0 0 0" lim="0.0 {4 * scale}"/>
            <body name="panda_rightfinger" type="mesh" scale="{scale} {scale} {scale}" filename="panda/visual/finger.obj" pos="0 0 0" quat="0 0 0 1" transform_type="OBJ_TO_JOINT"/>
        </link>
    </link>
</robot>

<actuator>
    <motor joint="panda_hand" ctrl="force" ctrl_range="-1 1"/>
    <motor joint="panda_leftfinger" ctrl="force" ctrl_range="-1 1"/>
    <motor joint="panda_rightfinger" ctrl="force" ctrl_range="-1 1"/>
</actuator>

<variable>
    <endeffector joint="panda_hand" pos="0 0 0" radius="{1 * scale}"/>
    <endeffector joint="panda_hand" pos="0 0 {10.4 * scale}" radius="{0.2 * scale}"/>
    <endeffector joint="panda_leftfinger" pos="0 0 {(10.4 - 5.84) * scale}" radius="{0.2 * scale}"/>
    <endeffector joint="panda_rightfinger" pos="0 0 {(10.4 - 5.84) * scale}" radius="{0.2 * scale}"/>
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
    parser.add_argument('--pos', type=float, nargs=3, default=[0, 0, 5], help='position of the model')
    parser.add_argument('--quat', type=float, nargs=4, default=[1, 0, 0, 0], help='quaternion of the model')
    parser.add_argument('--scale', type=float, default=1, help='scale of the model')
    parser.add_argument('--open-ratio', type=float, default=0.0, help='open ratio of the gripper')
    parser.add_argument('--steps', type=int, default=1, help='number of simulation steps')
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=None, help='camera lookat')
    parser.add_argument('--camera-pos', type=float, nargs=3, default=None, help='camera position')
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')
    xml_string = create_panda_gripper_xml(pos=args.pos, quat=args.quat, scale=args.scale)

    sim = redmax.Simulation(xml_string, asset_folder)
    sim.set_joint_q_init('panda_leftfinger', np.array([args.open_ratio * 4 * args.scale]))
    sim.set_joint_q_init('panda_rightfinger', np.array([args.open_ratio * 4 * args.scale]))
    sim.reset(backward_flag=False)
    if args.camera_lookat is not None:
        sim.viewer_options.camera_lookat = args.camera_lookat
    if args.camera_pos is not None:
        sim.viewer_options.camera_pos = args.camera_pos

    hand_mesh = get_body_mesh(sim, 'panda_hand')
    leftfinger_mesh = get_body_mesh(sim, 'panda_leftfinger')
    rightfinger_mesh = get_body_mesh(sim, 'panda_rightfinger')
    print('hand mesh extent:', hand_mesh.extents)
    print('leftfinger mesh extent:', leftfinger_mesh.extents)
    print('rightfinger mesh extent:', rightfinger_mesh.extents)
    
    for i in tqdm(range(args.steps)):
        sim.set_u(np.zeros(sim.ndof_u))
        sim.forward(1, verbose=False, test_derivatives=False)
        q = sim.get_q()
        qdot = sim.get_qdot()
        variables = sim.get_variables()

    variables = np.array(variables).reshape(-1, 3)
    print('hand origin:', variables[0])
    print('grasp center:', variables[1])
    print('leftfinger grasp point:', variables[2])
    print('rightfinger grasp point:', variables[3]) 
    
    SimRenderer.replay(sim, record=False)
