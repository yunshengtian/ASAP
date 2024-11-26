import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import redmax_py as redmax
import os

from utils.renderer import SimRenderer
from assets.load import load_pos_quat_dict, load_part_ids
from assets.color import get_color
from assets.transform import get_pos_quat_from_pose
from plan_robot.util_grasp import get_gripper_finger_states, get_gripper_base_name, get_gripper_path_from_part_path


def arr_to_str(arr):
    return ' '.join([str(x) for x in arr])


def create_gripper_with_assembly_posed_xml(assembly_dir, move_id, still_ids, removed_ids, pose=None, gripper_type=None, gripper_pos=[0, 0, 5], gripper_quat=[1, 0, 0, 0], gripper_scale=1):
    part_ids = [move_id] + still_ids + removed_ids
    all_part_ids = load_part_ids(assembly_dir)
    color_map = get_color(all_part_ids)
    pos_dict_final, quat_dict_final = load_pos_quat_dict(assembly_dir, transform='final')
    pos_dict_initial, quat_dict_initial = load_pos_quat_dict(assembly_dir, transform='initial')
    string = f'''
<redmax model="gripper">
<option integrator="BDF1" timestep="1e-3" gravity="0. 0. 1e-12"/>
<ground pos="0 0 0" normal="0 0 1"/>
'''
    for part_id in part_ids:
        joint_type = 'free3d-exp' if part_id == move_id else 'fixed'
        if part_id in removed_ids:
            pos, quat = pos_dict_initial[part_id], quat_dict_initial[part_id]
            if pos is None or quat is None:
                continue
        else:
            pos, quat = get_pos_quat_from_pose(pos_dict_final[part_id], quat_dict_final[part_id], pose)
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{arr_to_str(pos)}" quat="{arr_to_str(quat)}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="mesh" filename="{assembly_dir}/{part_id}.obj" pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" mu="0" rgba="{arr_to_str(color_map[part_id])}"/>
    </link>
</robot>
'''
    gripper_pos, gripper_quat = get_pos_quat_from_pose(gripper_pos, gripper_quat, pose)
    if gripper_type == 'panda':
        string += f'''
<robot>
    <link name="panda_hand">
        <joint name="panda_hand" type="free3d-exp" pos="{arr_to_str(gripper_pos)}" quat="{arr_to_str(gripper_quat)}"/>
        <body name="panda_hand" type="mesh" scale="{gripper_scale} {gripper_scale} {gripper_scale}" filename="panda/visual/hand.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
        <link name="panda_leftfinger">
            <joint name="panda_leftfinger" type="prismatic" axis="0 1 0" pos="0 0 {5.84 * gripper_scale}" quat="1 0 0 0" lim="0.0 {4 * gripper_scale}"/>
            <body name="panda_leftfinger" type="mesh" scale="{gripper_scale} {gripper_scale} {gripper_scale}" filename="panda/visual/finger.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
        </link>
        <link name="panda_rightfinger">
            <joint name="panda_rightfinger" type="prismatic" axis="0 -1 0" pos="0 0 {5.84 * gripper_scale}" quat="1 0 0 0" lim="0.0 {4 * gripper_scale}"/>
            <body name="panda_rightfinger" type="mesh" scale="{gripper_scale} {gripper_scale} {gripper_scale}" filename="panda/visual/finger.obj" pos="0 0 0" quat="0 0 0 1" transform_type="OBJ_TO_JOINT"/>
        </link>
    </link>
</robot>
'''
    elif gripper_type == 'robotiq_85':
        string += f'''
<robot>
    <link name="robotiq_base">
        <joint name="robotiq_base" type="free3d-exp" pos="{arr_to_str(gripper_pos)}" quat="{arr_to_str(gripper_quat)}"/>
        <body name= "robotiq_base" type = "mesh" filename = "robotiq_85/visual/robotiq_base_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        <link name="robotiq_left_outer_knuckle">
            <joint name = "robotiq_left_outer_knuckle" type="revolute" pos="{3.06011444260539 * gripper_scale} 0.0 {6.27920162695395 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 -1.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_outer_knuckle" type = "mesh" filename = "robotiq_85/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_left_outer_finger">
                <joint name="robotiq_left_outer_finger" type="fixed" pos="{3.16910442266543 * gripper_scale} 0.0 {-0.193396375724605 * gripper_scale}" quat="1.0 0.0 0.0 0.0"/>
                <body name= "robotiq_left_outer_finger" type = "mesh" filename = "robotiq_85/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
        <link name="robotiq_left_inner_knuckle">
            <joint name = "robotiq_left_inner_knuckle" type="revolute" pos="{1.27000000001501 * gripper_scale} 0.0 {6.93074999999639 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 -1.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_inner_knuckle" type = "mesh" filename = "robotiq_85/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_left_inner_finger">
                <joint name = "robotiq_left_inner_finger" type="revolute" pos="{3.4585310861294003 * gripper_scale} 0.0 {4.5497019381797505 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 -1.0 0.0" lim="-0.8757 0.0"/>
                <body name= "robotiq_left_inner_finger" type = "mesh" filename = "robotiq_85/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
        <link name="robotiq_right_outer_knuckle">
            <joint name = "robotiq_right_outer_knuckle" type="revolute" pos="{-3.06011444260539 * gripper_scale} 0.0 {6.27920162695395 * gripper_scale}" quat="0.0 0.0 0.0 1.0" axis="0.0 -1.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_right_outer_knuckle" type = "mesh" filename = "robotiq_85/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_right_outer_finger">
                <joint name="robotiq_right_outer_finger" type="fixed" pos="{3.16910442266543 * gripper_scale} 0.0 {-0.193396375724605 * gripper_scale}" quat="1.0 0.0 0.0 0.0"/>
                <body name= "robotiq_right_outer_finger" type = "mesh" filename = "robotiq_85/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
        <link name="robotiq_right_inner_knuckle">
            <joint name = "robotiq_right_inner_knuckle" type="revolute" pos="{-1.27000000001501 * gripper_scale} 0.0 {6.93074999999639 * gripper_scale}" quat="0.0 0.0 0.0 1.0" axis="0.0 1.0 0.0" lim="-0.8757 0.0"/>
            <body name= "robotiq_right_inner_knuckle" type = "mesh" filename = "robotiq_85/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_right_inner_finger">
                <joint name = "robotiq_right_inner_finger" type="revolute" pos="{3.4585310861294003 * gripper_scale} 0.0 {4.5497019381797505 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 1.0 0.0" lim="0.0 0.8757" damping="0.0"/>
                <body name= "robotiq_right_inner_finger" type = "mesh" filename = "robotiq_85/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            </link>
        </link>
    </link>
</robot>
'''
    elif gripper_type == 'robotiq-140':
        string += f'''
<robot>
    <link name="robotiq_base">
        <joint name="robotiq_base" type="free3d-exp" pos="{arr_to_str(gripper_pos)}" quat="{arr_to_str(gripper_quat)}"/>
        <body name= "robotiq_base" type = "mesh" filename = "robotiq_140/visual/robotiq_base_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        <link name="robotiq_left_outer_knuckle">
            <joint name = "robotiq_left_outer_knuckle" type="revolute" pos="0 {-3.0601 * gripper_scale} {5.4905 * gripper_scale}" quat="0.41040502 0.91190335 0.0 0.0" axis="-1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_outer_knuckle" type = "mesh" filename = "robotiq_140/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_left_outer_finger">
                <joint name = "robotiq_left_outer_finger" type="fixed" pos="0 {1.821998610742 * gripper_scale} {2.60018192872234 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="1.0 0.0 0.0"/>
                <body name= "robotiq_left_outer_finger" type = "mesh" filename = "robotiq_140/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                <link name="robotiq_left_inner_finger">
                    <joint name = "robotiq_left_inner_finger" type="revolute" pos="0 {8.17554015893473 * gripper_scale} {-2.82203446692936 * gripper_scale}" quat="0.93501321 -0.35461287 0.0 0.0" axis="1.0 0.0 0.0"/>
                    <body name= "robotiq_left_inner_finger" type = "mesh" filename = "robotiq_140/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    <link name="robotiq_left_pad">
                        <joint name = "robotiq_left_pad" type="fixed" pos="0 {3.8 * gripper_scale} {-2.3 * gripper_scale}" quat="0.0 0.0 0.70710678 0.70710678" axis="1.0 0.0 0.0"/>
                        <body name= "robotiq_left_pad" type = "mesh" filename = "robotiq_140/visual/pad_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    </link>
                </link>
            </link>
        </link>
        <link name="robotiq_left_inner_knuckle">
            <joint name = "robotiq_left_inner_knuckle" type="revolute" pos="0 {-1.27 * gripper_scale} {6.142 * gripper_scale}" quat="0.41040502 0.91190335 0.0 0.0" axis="1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_left_inner_knuckle" type = "mesh" filename = "robotiq_140/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        </link>
        <link name="robotiq_right_outer_knuckle">
            <joint name = "robotiq_right_outer_knuckle" type="revolute" pos="0 {3.0601 * gripper_scale} {5.4905 * gripper_scale}" quat="0.0 0.0 0.91190335 0.41040502" axis="1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_right_outer_knuckle" type = "mesh" filename = "robotiq_140/visual/outer_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
            <link name="robotiq_right_outer_knuckle">
                <joint name = "robotiq_right_outer_finger" type="fixed" pos="0 {1.821998610742 * gripper_scale} {2.60018192872234 * gripper_scale}" quat="1.0 0.0 0.0 0.0" axis="1.0 0.0 0.0"/>
                <body name= "robotiq_right_outer_finger" type = "mesh" filename = "robotiq_140/visual/outer_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                <link name="robotiq_right_inner_finger">
                    <joint name = "robotiq_right_inner_finger" type="revolute" pos="0 {8.17554015893473 * gripper_scale} {-2.82203446692936 * gripper_scale}" quat="0.93501321 -0.35461287 0.0 0.0" axis="1.0 0.0 0.0"/>
                    <body name= "robotiq_right_inner_finger" type = "mesh" filename = "robotiq_140/visual/inner_finger_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    <link name="robotiq_right_pad">
                        <joint name = "robotiq_right_pad" type="fixed" pos="0 {3.8 * gripper_scale} {-2.3 * gripper_scale}" quat="0.0 0.0 0.70710678 0.70710678" axis="1.0 0.0 0.0"/>
                        <body name= "robotiq_right_pad" type = "mesh" filename = "robotiq_140/visual/pad_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
                    </link>
                </link>
            </link>
        </link>
        <link name="robotiq_right_inner_knuckle">
            <joint name = "robotiq_right_inner_knuckle" type="revolute" pos="0 {1.27 * gripper_scale} {6.142 * gripper_scale}" quat="0.0 0.0 -0.91190335 -0.41040502" axis="1.0 0.0 0.0" lim="0.0 0.8757"/>
            <body name= "robotiq_right_inner_knuckle" type = "mesh" filename = "robotiq_140/visual/inner_knuckle_fine.obj" pos = "0 0 0" quat = "1 0 0 0" scale = "{gripper_scale} {gripper_scale} {gripper_scale}" transform_type="OBJ_TO_JOINT" rgba = "0.1 0.1 0.1 1.0"/>
        </link>
    </link>
</robot>
'''
    else:
        raise NotImplementedError
    string += '''
</redmax>
'''
    return string


def render_path_with_grasp(asset_folder, assembly_dir, move_id, still_ids, removed_ids, pose, part_path, gripper_type, gripper_scale, grasp, 
    camera_lookat=None, camera_pos=None, body_color_map=None, reverse=False, render=True, record_path=None, make_video=False):
    if part_path is None:
        print('no path found')
        return

    gripper_pos, gripper_quat = grasp.pos, grasp.quat

    xml_string = create_gripper_with_assembly_posed_xml(
        assembly_dir=assembly_dir, move_id=move_id, still_ids=still_ids, removed_ids=removed_ids, pose=pose,
        gripper_type=gripper_type, gripper_pos=gripper_pos, gripper_quat=gripper_quat, gripper_scale=gripper_scale)
    sim = redmax.Simulation(xml_string, asset_folder)
    if camera_lookat is not None:
        sim.viewer_options.camera_lookat = camera_lookat
    if camera_pos is not None:
        sim.viewer_options.camera_pos = camera_pos

    finger_states = get_gripper_finger_states(gripper_type, grasp.open_ratio, gripper_scale)
    for finger_name, finger_state in finger_states.items():
        sim.set_joint_q_init(finger_name, np.array(finger_state))
    sim.reset(backward_flag=False)

    if body_color_map is not None:
        sim.set_body_color_map(body_color_map)

    # get gripper path
    gripper_path = get_gripper_path_from_part_path(part_path, gripper_pos, gripper_quat)
    
    # transform from global coordinate to local coordinate
    part_path_local = [sim.get_joint_q_from_qm(f'part{move_id}', qm) for qm in part_path]
    gripper_base_name = get_gripper_base_name(gripper_type)
    gripper_path_local = [sim.get_joint_q_from_qm(gripper_base_name, qm) for qm in gripper_path]

    states = [np.concatenate([part_state, gripper_state, np.concatenate(list(finger_states.values()))]) for part_state, gripper_state in zip(part_path_local, gripper_path_local)]
    if reverse:
        states = states[::-1]
    sim.set_state_his(states, [np.zeros_like(states[0]) for _ in range(len(states))])

    if render:
        SimRenderer.replay(sim, record=record_path is not None, record_path=record_path, make_video=make_video)

    return sim.export_replay_matrices()
