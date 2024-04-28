mesh_type = 'collision'
body_type = 'SDF'
dx = 1
scale = 1

string = f'''
<redmax model="panda">

<option integrator="BDF1" timestep="1e-3" gravity="0. 0. 1e-12"/>

<default>
    <joint lim_stiffness="1e2" damping="1"/>
    <ground_contact kn="1e6" kt="1e3" mu="0.8" damping="5e1"/>
    <general_{body_type}_contact kn="1e5" kt="1e3" mu="0.1" damping="1"/>
</default>

<robot>
    <link name="panda_link0">
        <joint name="panda_joint0" type="fixed" pos="0 0 0" quat="1 0 0 0"/>
        <body name="panda_link0" type="{body_type}" dx="{dx}" scale="{scale} {scale} {scale}" filename="{mesh_type}/link0.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
        <link name="panda_link1">
            <joint name="panda_joint1" type="revolute" axis="0 0 1" pos="0 0 {33.3 * scale}" quat="1 0 0 0" lim="-2.9671 2.9671"/>
            <body name="panda_link1" type="{body_type}" dx="{dx}" scale="{scale} {scale} {scale}" filename="{mesh_type}/link1.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
            <link name="panda_link2">
                <joint name="panda_joint2" type="revolute" axis="0 0 1" pos="0 0 0" quat="-0.7071068 0.7071068 0 0" lim="-1.8326 1.8326"/>
                <body name="panda_link2" type="{body_type}" dx="{dx}" scale="{scale} {scale} {scale}" filename="{mesh_type}/link2.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
                <link name="panda_link3">
                    <joint name="panda_joint3" type="revolute" axis="0 0 1" pos="0 {-31.6 * scale} 0" quat="0.7071068 0.7071068 0 0" lim="-2.9671 2.9671"/>
                    <body name="panda_link3" type="{body_type}" dx="{dx}" scale="{scale} {scale} {scale}" filename="{mesh_type}/link3.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
                    <link name="panda_link4">
                        <joint name="panda_joint4" type="revolute" axis="0 0 1" pos="{8.25 * scale} 0 0" quat="0.7071068 0.7071068 0 0" lim="-3.1416 3.1416"/>
                        <body name="panda_link4" type="{body_type}" dx="{dx}" scale="{scale} {scale} {scale}" filename="{mesh_type}/link4.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
                        <link name="panda_link5">
                            <joint name="panda_joint5" type="revolute" axis="0 0 1" pos="{-8.25 * scale} {3.84 * scale} 0" quat="-0.7071068 0.7071068 0 0" lim="-2.9671 2.9671"/>
                            <body name="panda_link5" type="{body_type}" dx="{dx}" scale="{scale} {scale} {scale}" filename="{mesh_type}/link5.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
                            <link name="panda_link6">
                                <joint name="panda_joint6" type="revolute" axis="0 0 1" pos="0 0 0" quat="0.7071068 0.7071068 0 0" lim="-0.0873 3.8223"/>
                                <body name="panda_link6" type="{body_type}" dx="{dx}" scale="{scale} {scale} {scale}" filename="{mesh_type}/link6.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
                                <link name="panda_link7">
                                    <joint name="panda_joint7" type="revolute" axis="0 0 1" pos="{8.8 * scale} 0 0" quat="0.7071068 0.7071068 0 0" lim="-2.9671 2.9671"/>
                                    <body name="panda_link7" type="{body_type}" dx="{dx}" scale="{scale} {scale} {scale}" filename="{mesh_type}/link7.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
                                    <link name="panda_link8">
                                        <joint name="panda_joint8" type="fixed" pos="0 0 {10.7 * scale}" quat="1 0 0 0"/>
                                        <body name="panda_link8" type="sphere" radius="0.01" pos="0 0 0" quat="1 0 0 0"/>
                                        <link name="panda_hand">
                                            <joint name="panda_hand_joint" type="fixed" pos="0 0 0" quat="0.9238795 0 0 -0.3826834"/>
                                            <body name="panda_hand" type="{body_type}" dx="{dx}" scale="{scale} {scale} {scale}" filename="{mesh_type}/hand.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
                                            <link name="panda_leftfinger">
                                                <joint name="panda_finger_joint1" type="prismatic" axis="0 1 0" pos="0 0 {5.84 * scale}" quat="1 0 0 0" lim="0.0 {4 * scale}"/>
                                                <body name="panda_leftfinger" type="{body_type}" dx="{dx}" scale="{scale} {scale} {scale}" filename="{mesh_type}/finger.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT"/>
                                            </link>
                                            <link name="panda_rightfinger">
                                                <joint name="panda_finger_joint2" type="prismatic" axis="0 -1 0" pos="0 0 {5.84 * scale}" quat="1 0 0 0" lim="0.0 {4 * scale}"/>
                                                <body name="panda_rightfinger" type="{body_type}" dx="{dx}" scale="{scale} {scale} {scale}" filename="{mesh_type}/finger.obj" pos="0 0 0" quat="0 0 0 1" transform_type="OBJ_TO_JOINT"/>
                                            </link>
                                        </link>
                                    </link>
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

# assume no self-contact for now

string += '''
<actuator>'''

joint_names = [f'panda_joint{i}' for i in range(1, 8)] + ['panda_finger_joint1', 'panda_finger_joint2']
for joint_name in joint_names:
    string += f'''
    <motor joint="{joint_name}" ctrl="force" ctrl_range="-1 1"/>'''

string += '''
</actuator>'''

string += '''
</redmax>
'''

with open('panda.xml', 'w') as fp:
    fp.write(string)