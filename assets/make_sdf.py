import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import redmax_py as redmax
import json
from argparse import ArgumentParser

from assets.load import load_pos_quat_dict, load_part_ids
from assets.save import clear_saved_sdfs


def arr_to_str(arr):
    return ' '.join([str(x) for x in arr])


def get_xml_string(assembly_dir, part_ids, sdf_dx):
    pos_dict, quat_dict = load_pos_quat_dict(assembly_dir)
    joint_type = 'fixed'
    string = f'''
<redmax model="assemble">
<option integrator="BDF1" timestep="1e-3" gravity="0. 0. 1e-12"/>
'''
    for part_id in part_ids:
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{arr_to_str(pos_dict[part_id])}" quat="{arr_to_str(quat_dict[part_id])}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="SDF" filename="{assembly_dir}/{part_id}.obj" save_sdf="true" pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" dx="{sdf_dx}" mu="0"/>
    </link>
</robot>
'''
    string += f'''
</redmax>
'''
    return string


def make_sdf(asset_folder, assembly_dir, sdf_dx):
    part_ids = load_part_ids(assembly_dir)
    model_string = get_xml_string(
        assembly_dir=assembly_dir,
        part_ids=part_ids,
        sdf_dx=sdf_dx,
    )
    redmax.Simulation(model_string, asset_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--id', type=str, required=True, help='assembly id (e.g. 00000)')
    parser.add_argument('--dir', type=str, default='multi_assembly', help='directory storing all assemblies')
    parser.add_argument('--sdf-dx', type=float, default=0.05, help='grid resolution of SDF')
    parser.add_argument('--clear', default=False, action='store_true')
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')
    assembly_dir = os.path.join(asset_folder, args.dir, args.id)
    
    if args.clear:
        clear_saved_sdfs(assembly_dir)
    else:
        make_sdf(asset_folder, assembly_dir, args.sdf_dx)
