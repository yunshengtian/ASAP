import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import redmax_py as redmax
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import ikpy.utils.plot as plot_utils

from assets.load import load_pos_quat_dict, load_part_ids
from assets.color import get_color
from plan_robot.util_arm import get_arm_chain, get_arm_pos, get_arm_euler
from utils.renderer import SimRenderer


def arr_to_str(arr):
    return ' '.join([str(x) for x in arr])


def get_arm_assembly_xml(arm_pos, arm_euler, arm_scale, assembly_dir, part_ids):
    all_part_ids = load_part_ids(assembly_dir)
    color_map = get_color(all_part_ids)
    pos_dict, quat_dict = load_pos_quat_dict(assembly_dir, transform='final')
    arm_quat = R.from_euler('xyz', arm_euler).as_quat()[[3, 0, 1, 2]]
    string = f'''
    <redmax model="xarm7">
    <option integrator="BDF1" timestep="5e-3" gravity="0. 0. -980."/>
    
    <ground pos="0 0 0" normal="0 0 1"/>

    <robot>
        <link name="linkbase">
            <joint name="linkbase" type="fixed" pos="{arr_to_str(arm_pos)}" quat="{arr_to_str(arm_quat)}"/>
            <body name= "linkbase" type = "abstract" pos = "{-2.1131 * arm_scale} {-0.16302 * arm_scale} {5.6488 * arm_scale}" quat = "0.41928822390350623 -0.3384325829202692 0.5661965601674424 0.6237645608467303" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "885.5600000000001" inertia = "16772.46795287213 33528.24905872843 38202.28298839946" rgba = "0.8 0.8 0.8 1.0">
                <visual mesh = "xarm7/visual/linkbase_smooth.obj" pos = "{4.2037215880446706 * arm_scale} {-4.303182668106523 * arm_scale} {-0.4604913739852395 * arm_scale}" quat = "-0.4192882239035062 -0.3384325829202692 0.5661965601674422 0.6237645608467302"/>
                <collision contacts = "xarm7/contacts/linkbase.txt" pos = "{4.2037215880446706 * arm_scale} {-4.303182668106523 * arm_scale} {-0.4604913739852395 * arm_scale}" quat = "-0.4192882239035062 -0.3384325829202692 0.5661965601674422 0.6237645608467302"/>
            </body>
            <link name="joint1">
                <joint name = "joint1" type="revolute" pos="0.0 0.0 {26.700000000000003 * arm_scale}" quat="1.0 0.0 0.0 0.0" axis="0.0 0.0 1.0" lim="-6.28318530718 6.28318530718" damping="0.0"/>
                <body name= "link1" type = "abstract" pos = "{-0.42142 * arm_scale} {2.8209999999999997 * arm_scale} {-0.87788 * arm_scale}" quat = "0.6918679191340069 0.0005252148724432689 -0.6060703296327886 0.39242484906198305" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "426.03000000000003" inertia = "8235.113114059062 13775.66035044562 14455.126535495323" rgba = "0.9 0.9 0.9 1.0">
                    <visual mesh = "xarm7/visual/link1_smooth.obj" pos = "{-0.8114216775604369 * arm_scale} {-2.5981972214308087 * arm_scale} {1.2236319587744608 * arm_scale}" quat = "0.6918679191340068 -0.0005252148724432888 0.6060703296327885 -0.392424849061983"/>
                    <collision contacts = "xarm7/contacts/link1_vhacd.txt" pos = "{-0.8114216775604369 * arm_scale} {-2.5981972214308087 * arm_scale} {1.2236319587744608 * arm_scale}" quat = "0.6918679191340068 -0.0005252148724432888 0.6060703296327885 -0.392424849061983"/>
                </body>
                <link name="joint2">
                    <joint name = "joint2" type="revolute" pos="0.0 0.0 0.0" quat="0.7071054825112364 -0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-2.059 2.0944" damping="0.0"/>
                    <body name= "link2" type = "abstract" pos = "{-0.0033178 * arm_scale} {-12.849 * arm_scale} {2.6337 * arm_scale}" quat = "0.6337905179370223 -0.3150527964362565 0.6307030926183779 -0.3182215011472822" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "560.9499999999999" inertia = "9808.04401005623 31159.849558318598 31915.106431625172" rgba = "0.8 0.8 0.8 1.0">
                        <visual mesh = "xarm7/visual/link2_smooth.obj" pos = "{-8.711764384019158 * arm_scale} {9.804940501796535 * arm_scale} {-0.03861050843999272 * arm_scale}" quat = "0.6337905179370222 0.3150527964362564 -0.6307030926183778 0.31822150114728215"/>
                        <collision contacts = "xarm7/contacts/link2_vhacd.txt" pos = "{-8.711764384019158 * arm_scale} {9.804940501796535 * arm_scale} {-0.03861050843999272 * arm_scale}" quat = "0.6337905179370222 0.3150527964362564 -0.6307030926183778 0.31822150114728215"/>
                    </body>
                    <link name="joint3">
                        <joint name = "joint3" type="revolute" pos="0.0 {-29.299999999999997 * arm_scale} 0.0" quat="0.7071054825112364 0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-6.28318530718 6.28318530718" damping="0.0"/>
                        <body name= "link3" type = "abstract" pos = "{4.223 * arm_scale} {-2.3258 * arm_scale} {-0.9667399999999999 * arm_scale}" quat = "-0.24066027491433598 0.8530839593824141 -0.23989343952005776 0.39595647235247505" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "444.63000000000005" inertia = "7804.745076690154 11912.598616264928 13322.65630704492" rgba = "0.9 0.9 0.9 1.0">
                            <visual mesh = "xarm7/visual/link3_smooth.obj" pos = "{-3.266493961637458 * arm_scale} {-1.4456637070979548 * arm_scale} {-3.379013837226153 * arm_scale}" quat = "0.24066027491433598 0.8530839593824141 -0.23989343952005787 0.39595647235247505"/>
                            <collision contacts = "xarm7/contacts/link3_vhacd.txt" pos = "{-3.266493961637458 * arm_scale} {-1.4456637070979548 * arm_scale} {-3.379013837226153 * arm_scale}" quat = "0.24066027491433598 0.8530839593824141 -0.23989343952005787 0.39595647235247505"/>
                        </body>
                        <link name="joint4">
                            <joint name = "joint4" type="revolute" pos="{5.25 * arm_scale} 0.0 0.0" quat="0.7071054825112364 0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-0.19198 3.927" damping="0.0"/>
                            <body name= "link4" type = "abstract" pos = "{6.7148 * arm_scale} {-10.732 * arm_scale} {2.4479 * arm_scale}" quat = "0.6707722696010192 0.47605887893032944 0.012741086884645855 -0.5685685278230704" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "523.8699999999999" inertia = "8944.094777271237 28270.539922454394 28898.36530027436" rgba = "0.8 0.8 0.8 1.0">
                                <visual mesh = "xarm7/visual/link4_smooth.obj" pos = "{-9.059983366567941 * arm_scale} {-7.802235142560355 * arm_scale} {-4.826842200415131 * arm_scale}" quat = "0.6707722696010191 -0.47605887893032933 -0.012741086884645859 0.5685685278230704"/>
                                <collision contacts = "xarm7/contacts/link4_vhacd.txt" pos = "{-9.059983366567941 * arm_scale} {-7.802235142560355 * arm_scale} {-4.826842200415131 * arm_scale}" quat = "0.6707722696010191 -0.47605887893032933 -0.012741086884645859 0.5685685278230704"/>
                            </body>
                            <link name="joint5">
                                <joint name = "joint5" type="revolute" pos="{7.75 * arm_scale} {-34.25 * arm_scale} 0.0" quat="0.7071054825112364 0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-6.28318530718 6.28318530718" damping="0.0"/>
                                <body name= "link5" type = "abstract" pos = "{-0.023397 * arm_scale} {3.6705 * arm_scale} {-8.0064 * arm_scale}" quat = "0.6892055355098681 -0.16046408568558085 0.698228216539248 -0.10827910535309972" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "185.54000000000002" inertia = "2471.2608713476575 9886.134859618787 9955.304269033553" rgba = "0.9 0.9 0.9 1.0">
                                    <visual mesh = "xarm7/visual/link5_smooth.obj" pos = "{-6.057144261834268 * arm_scale} {-6.378684331187784 * arm_scale} {-0.4460361240938076 * arm_scale}" quat = "-0.6892055355098679 -0.16046408568558085 0.6982282165392479 -0.1082791053530997"/>
                                    <collision contacts = "xarm7/contacts/link5_vhacd.txt" pos = "{-6.057144261834268 * arm_scale} {-6.378684331187784 * arm_scale} {-0.4460361240938076 * arm_scale}" quat = "-0.6892055355098679 -0.16046408568558085 0.6982282165392479 -0.1082791053530997"/>
                                </body>
                                <link name="joint6">
                                    <joint name = "joint6" type="revolute" pos="0.0 0.0 0.0" quat="0.7071054825112364 0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-1.69297 3.14159265359" damping="0.0"/>
                                    <body name= "link6" type = "abstract" pos = "{5.8911 * arm_scale} {2.8469 * arm_scale} {0.68428 * arm_scale}" quat = "0.9529732922502667 0.01599291734637919 0.1692539717525629 0.250876909855057" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "313.44" inertia = "3867.077886736355 7688.706404074782 8278.915709188861" rgba = "1.0 1.0 1.0 1.0">
                                        <visual mesh = "xarm7/visual/link6_smooth.obj" pos = "{-5.973444004856806 * arm_scale} {0.21893372810568068 * arm_scale} {-2.747393798118139 * arm_scale}" quat = "0.9529732922502666 -0.01599291734637919 -0.16925397175256282 -0.250876909855057"/>
                                        <collision contacts = "xarm7/contacts/link6_vhacd.txt" pos = "{-5.973444004856806 * arm_scale} {0.21893372810568068 * arm_scale} {-2.747393798118139 * arm_scale}" quat = "0.9529732922502666 -0.01599291734637919 -0.16925397175256282 -0.250876909855057"/>
                                    </body>
                                    <link name="joint7">
                                        <joint name = "joint7" type="revolute" pos="{7.6 * arm_scale} {9.700000000000001 * arm_scale} 0.0" quat="0.7071054825112364 -0.7071080798594735 0.0 0.0" axis="0.0 0.0 1.0" lim="-6.28318530718 6.28318530718" damping="0.0"/>
                                        <body name= "link7" type = "abstract" pos = "{-0.0015846 * arm_scale} {-0.46376999999999996 * arm_scale} {-1.2705 * arm_scale}" quat = "-0.0051369063433354505 0.7078680662792562 -0.706304495783441 -0.005511095298183751" scale="{arm_scale} {arm_scale} {arm_scale}" mass = "314.68" inertia = "1192.0774998754055 1698.502197488278 2603.520302636317" rgba = "0.753 0.753 0.753 1.0">
                                            <visual mesh = "xarm7/visual/link7_smooth.obj" pos = "{-0.4828448608210097 * arm_scale} {-0.0019607575065535687 * arm_scale} {-1.2633734086428683 * arm_scale}" quat = "0.0051369063433354505 0.7078680662792562 -0.706304495783441 -0.005511095298183751"/>
                                            <collision contacts = "xarm7/contacts/link7_vhacd.txt" pos = "{-0.4828448608210097 * arm_scale} {-0.0019607575065535687 * arm_scale} {-1.2633734086428683 * arm_scale}" quat = "0.0051369063433354505 0.7078680662792562 -0.706304495783441 -0.005511095298183751"/>
                                        </body>
                                    </link>
                                </link>
                            </link>
                        </link>
                    </link>
                </link>
            </link>
        </link>
    </robot>
'''
    for part_id in part_ids:
        part_color = color_map[part_id]
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="fixed" axis="0. 0. 0." pos="{arr_to_str(pos_dict[part_id])}" quat="{arr_to_str(quat_dict[part_id])}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="mesh" filename="{assembly_dir}/{part_id}.obj" pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" mu="0" rgba="{arr_to_str(part_color)}"/>
    </link>
</robot>
'''
    string += f'''
    <variable>
        <endeffector joint="joint7" pos="0 0 0" radius="{1 * arm_scale}"/>
    </variable>
</redmax>
    '''
    return string


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--target-pos', type=float, nargs='+', default=[0, 0, 10], help='target position')
    parser.add_argument('--target-ori', type=float, nargs='+', default=[0, 0, -1], help='target orientation')
    parser.add_argument('--gripper-pos', type=float, nargs='+', default=[0, 0, 0], help='gripper position')
    parser.add_argument('--arm-scale', type=float, default=0.4, help='arm scale')
    parser.add_argument('--optimizer', type=str, default='L-BFGS-B', help='optimizer')
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=None, help='camera lookat')
    parser.add_argument('--camera-pos', type=float, nargs=3, default=None, help='camera position')
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')

    arm_pos = get_arm_pos(gripper_pos=args.gripper_pos, gripper_ori=[0, 0, -1], arm_scale=args.arm_scale, center=np.zeros(3))
    arm_euler = get_arm_euler(arm_pos=arm_pos, center=np.zeros(3))

    assembly_dir = args.assembly_dir
    part_ids = load_part_ids(assembly_dir)

    xml_string = get_arm_assembly_xml(arm_pos=arm_pos, arm_euler=arm_euler, arm_scale=args.arm_scale, assembly_dir=assembly_dir, part_ids=part_ids)

    sim = redmax.Simulation(xml_string, asset_folder)
    if args.camera_lookat is not None:
        sim.viewer_options.camera_lookat = args.camera_lookat
    if args.camera_pos is not None:
        sim.viewer_options.camera_pos = args.camera_pos

    chain = get_arm_chain(base_pos=arm_pos, base_euler=arm_euler, scale=args.arm_scale)
    q, ik_success = chain.inverse_kinematics(target_position=args.target_pos, target_orientation=args.target_ori, orientation_mode='Z', n_restart=3, optimizer=args.optimizer)
    print('IK success:', ik_success)

    sim.set_q_init(q[1:])
    sim.reset()

    print('Endeffector position:', sim.get_variables())

    SimRenderer.replay(sim, record=False)
