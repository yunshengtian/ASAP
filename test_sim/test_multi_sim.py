import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import redmax_py as redmax
import json
from argparse import ArgumentParser
from tqdm import tqdm

from assets.load import load_pos_quat_dict, load_part_ids
from assets.color import get_color
from utils.renderer import SimRenderer


def arr_to_str(arr):
    return ' '.join([str(x) for x in arr])


def get_xml_string(assembly_dir, part_ids, base_part, fixed, transform, body_type, sdf_dx, sdf_res, gravity, friction, ground_z):
    all_part_ids = load_part_ids(assembly_dir)
    color_map = get_color(all_part_ids)
    pos_dict, quat_dict = load_pos_quat_dict(assembly_dir, transform=transform)
    joint_type = 'fixed' if fixed else 'free3d-exp'
    body_type = body_type.upper()
    sdf_args = 'load_sdf="true" save_sdf="true"' if body_type == 'SDF' else ''
    string = f'''
<redmax model="assemble">
<option integrator="BDF1" timestep="1e-3" gravity="0. 0. {-gravity}"/>

<ground pos="0 0 {ground_z}" normal="0 0 1"/>
<default>
    <ground_contact kn="1e6" kt="1e3" mu="0.8" damping="5e1"/>
    <general_{body_type}_contact kn="1e5" kt="1e3" mu="{friction}" damping="0"/>
</default>
'''
    for part_id in part_ids:
        if base_part is not None and part_id == base_part:
            part_color = np.array([0.4, 0.4, 0.4, 1.0])
            curr_joint_type = 'fixed'
        else:
            part_color = color_map[part_id]
            curr_joint_type = joint_type
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{curr_joint_type}" axis="0. 0. 0." pos="{arr_to_str(pos_dict[part_id])}" quat="{arr_to_str(quat_dict[part_id])}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="{body_type}" filename="{assembly_dir}/{part_id}.obj" {sdf_args} pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" dx="{sdf_dx}" res="{sdf_res}" mu="0" rgba="{arr_to_str(part_color)}"/>
    </link>
</robot>
'''
    string += f'''
<contact>
'''
    for part_id in part_ids:
        if part_id == base_part:
            continue
        string += f'''
    <ground_contact body="part{part_id}"/>
'''
    for i in range(len(part_ids)):
        for j in range(i + 1, len(part_ids)):
            string += f'''
    <general_{body_type}_contact general_body="part{part_ids[i]}" {body_type}_body="part{part_ids[j]}"/>
    <general_{body_type}_contact general_body="part{part_ids[j]}" {body_type}_body="part{part_ids[i]}"/>
'''
    string += f'''
</contact>
</redmax>
'''
    return string


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--id', type=str, required=True, help='assembly id (e.g. 00000)')
    parser.add_argument('--dir', type=str, default='multi_assembly', help='directory storing all assemblies')
    parser.add_argument('--part-ids', type=str, nargs='+', default=None, help='which parts to simulate (default: all)')
    parser.add_argument('--base-part', type=str, default=None)
    parser.add_argument('--steps', type=int, default=1000, help='number of simulation steps')
    parser.add_argument('--fixed', default=False, action='store_true', help='whether parts can move')
    parser.add_argument('--transform', type=str, default='final', choices=['final', 'initial'], help='transform type')
    parser.add_argument('--body-type', type=str, default='sdf', choices=['bvh', 'sdf'], help='simulation type of body')
    parser.add_argument('--sdf-dx', type=float, default=0.05, help='grid resolution of SDF')
    parser.add_argument('--sdf-res', type=int, default=20, help='grid resolution of SDF')
    parser.add_argument('--gravity', type=float, default=1e-12, help='magnitude of gravity')
    parser.add_argument('--friction', type=float, default=0.0, help='friction coefficient between parts')
    parser.add_argument('--ground-z', type=float, default=0.0, help='z coordinate of ground')
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=[-1, 1, 0], help='camera lookat')
    parser.add_argument('--camera-pos', type=float, nargs=3, default=[1.25, -1.5, 1.5], help='camera position')
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')
    assembly_dir = os.path.join(asset_folder, args.dir, args.id)

    if args.part_ids is None:
        part_ids = load_part_ids(assembly_dir)
    else:
        part_ids = args.part_ids

    print('[test_multi_sim] n_part:', len(part_ids))

    model_string = get_xml_string(
        assembly_dir=assembly_dir,
        part_ids=part_ids,
        base_part=args.base_part,
        fixed=args.fixed,
        transform=args.transform,
        body_type=args.body_type,
        sdf_dx=args.sdf_dx,
        sdf_res=args.sdf_res,
        gravity=args.gravity,
        friction=args.friction,
        ground_z=args.ground_z,
    )

    sim = redmax.Simulation(model_string, asset_folder)
    sim.reset()
    sim.viewer_options.camera_pos = args.camera_pos
    sim.viewer_options.camera_lookat = args.camera_lookat

    for i in tqdm(range(args.steps)):
        sim.forward(1, verbose=True)
        
    SimRenderer.replay(sim, record=False)
