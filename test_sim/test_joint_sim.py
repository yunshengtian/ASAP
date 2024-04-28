import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import redmax_py as redmax
import json
from argparse import ArgumentParser
from tqdm import tqdm
import time

from assets.load import load_pos_quat_dict, load_part_ids
from assets.color import get_color
from utils.renderer import SimRenderer


def arr_to_str(arr):
    return ' '.join([str(x) for x in arr])


def get_xml_string(assembly_dir, fixed, transform, body_type, sdf_dx, gravity, friction, ground_z):
    all_part_ids = load_part_ids(assembly_dir)
    color_map = get_color(all_part_ids)
    pos_dict, quat_dict = load_pos_quat_dict(assembly_dir, transform=transform)
    joint_type = 'fixed' if fixed else 'free3d-exp'
    body_type = body_type.upper()
    string = f'''
<redmax model="assemble">
<option integrator="BDF1" timestep="1e-3" gravity="0. 0. {-gravity}"/>

<ground pos="0 0 {ground_z}" normal="0 0 1"/>
<default>
    <ground_contact kn="1e6" kt="1e3" mu="0.8" damping="5e1"/>
    <general_{body_type}_contact kn="1e5" kt="1e3" mu="{friction}" damping="0"/>
</default>
'''
    for part_id in ['0', '1']:
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{arr_to_str(pos_dict[part_id])}" quat="{arr_to_str(quat_dict[part_id])}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="{body_type}" filename="{assembly_dir}/{part_id}.obj" BVH_mesh_filename="{assembly_dir}/{part_id}.obj" pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" dx="{sdf_dx}" mu="0" rgba="{arr_to_str(color_map[part_id])}"/>
    </link>
</robot>
'''
    string += f'''
<contact>
'''
    for part_id in ['0', '1']:
        string += f'''
    <ground_contact body="part{part_id}"/>
'''
    string += f'''
    <general_{body_type}_contact general_body="part0" {body_type}_body="part1"/>
    <general_{body_type}_contact general_body="part1" {body_type}_body="part0"/>
'''
    string += f'''
</contact>
</redmax>
'''
    return string


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--id', type=str, required=True, help='assembly id (e.g. 00000)')
    parser.add_argument('--dir', type=str, default='joint_assembly', help='directory storing all assemblies')
    parser.add_argument('--steps', type=int, default=1000, help='number of simulation steps')
    parser.add_argument('--fixed', default=False, action='store_true', help='whether parts can move')
    parser.add_argument('--transform', type=str, default='final', choices=['final', 'initial'], help='transform type')
    parser.add_argument('--body-type', type=str, default='sdf', choices=['bvh', 'sdf'], help='simulation type of body')
    parser.add_argument('--sdf-dx', type=float, default=0.05, help='grid resolution of SDF')
    parser.add_argument('--gravity', type=float, default=1e-12, help='magnitude of gravity')
    parser.add_argument('--friction', type=float, default=0.0, help='friction coefficient between parts')
    parser.add_argument('--ground-z', type=float, default=0.0, help='z coordinate of ground')
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')
    assembly_dir = os.path.join(asset_folder, args.dir, args.id)

    model_string = get_xml_string(
        assembly_dir=assembly_dir,
        fixed=args.fixed,
        transform=args.transform,
        body_type=args.body_type,
        sdf_dx=args.sdf_dx,
        gravity=args.gravity,
        friction=args.friction,
        ground_z=args.ground_z,
    )

    t_start = time.time()

    sim = redmax.Simulation(model_string, asset_folder)
    sim.reset()

    for i in tqdm(range(args.steps)):
        sim.forward(1, verbose=True)
    
    t_end = time.time()

    print('Time = ', t_end - t_start)

    SimRenderer.replay(sim, record=False)
