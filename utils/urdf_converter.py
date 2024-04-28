from argparse import ArgumentParser
import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import xml.etree.ElementTree as ET
from utils.common import *
import shutil
import trimesh
from scipy.spatial.transform import Rotation
from trimesh import inertia as trimesh_inertia

def str_to_value(str):
    items = str.split(' ')
    if len(items) == 1:
        try:
            value = float(items[0])
        except:
            value = items[0]
        return value
    else:
        return np.array([float(item) for item in items])

def xml_to_dict(node):
    params = {}
    for child in node:
        child_params = xml_to_dict(child)
        params[child.tag] = child_params

    for key in node.attrib:
        params[key] = str_to_value(node.attrib[key])

    return params

class Body:
    def __init__(self):
        self.name = None
        self.joint_name = None
        self.exported = False
        self.params = {}
        self.children = []

    def parse_urdf(self, node):
        self.params = xml_to_dict(node)

        self.name = self.params['name']

        ''' Part 1: body frame mass inertia '''

        ''' mass from kg to g '''
        if export_unit == 'cm-g':
            self.params['inertial']['mass']['value'] = self.params['inertial']['mass']['value'] * 1000.
        else:
            self.params['inertial']['mass']['value'] = self.params['inertial']['mass']['value'] 

        ''' convert xyz to pos '''
        if export_unit == 'cm-g':
            self.params['inertial']['origin']['pos'] = self.params['inertial']['origin']['xyz'] * 100.
        else:
            self.params['inertial']['origin']['pos'] = self.params['inertial']['origin']['xyz']

        ''' compute the quaternion of the body frame '''
        rot_joint_inertial = Rotation.from_euler('xyz', self.params['inertial']['origin']['rpy'], degrees = False)
        
        # step 1: find the principal axis of the inertia tensor
        # if I is the original inertia tensor, I0 is the diagonal matrix from eigen decomposition and R is the rotational matrix from eigen decomposition
        # then we have I * R = R * I0 -> I = R * I0 * R^T
        # so I0 is the inertia tensor after rotation R, the new body frame is defined by rotation R from original body frame
        # reference: https://en.wikipedia.org/wiki/Moment_of_inertia#Principal_axes
        inertia = np.zeros((3, 3))
        inertia[0][0] = self.params['inertial']['inertia']['ixx']
        inertia[0][1], inertia[1][0] = self.params['inertial']['inertia']['ixy'], self.params['inertial']['inertia']['ixy']
        inertia[0][2], inertia[2][0] = self.params['inertial']['inertia']['ixz'], self.params['inertial']['inertia']['ixz']
        inertia[1][1] = self.params['inertial']['inertia']['iyy']
        inertia[1][2], inertia[2][1] = self.params['inertial']['inertia']['iyz'], self.params['inertial']['inertia']['iyz']
        inertia[2][2] = self.params['inertial']['inertia']['izz']
        if export_unit == 'cm-g':
            inertia = inertia * 10000000.

        components, principal_axis = trimesh_inertia.principal_axis(inertia)
        # check if principal_axis are determinant = 1 (correct rotation)
        if np.linalg.det(principal_axis) < 0.:
            principal_axis = -principal_axis
        R_inertial_body = Rotation.from_matrix(principal_axis.T)
    
        # step 2: compute the rotation matrix from the joint frame to the new body frame
        self.params['inertial']['inertia'] = components
        rot_joint_body = rot_joint_inertial * R_inertial_body
        self.params['inertial']['origin']['quat'] = rot_joint_body.as_quat()

        E_joint_body = np.identity(4)
        E_joint_body[0:3, 0:3] = rot_joint_body.as_matrix()
        E_joint_body[0:3, 3] = self.params['inertial']['origin']['pos']

        E_body_joint = np.linalg.inv(E_joint_body)

        ''' Part 2: body frame visual mesh '''
        E_joint_visual = np.identity(4)
        if export_unit == 'cm-g':
            E_joint_visual[0:3, 3] = self.params['visual']['origin']['xyz'] * 100.
        else:
            E_joint_visual[0:3, 3] = self.params['visual']['origin']['xyz']
        rot_joint_visual = Rotation.from_euler('xyz', self.params['visual']['origin']['rpy'], degrees = False)
        E_joint_visual[0:3, 0:3] = rot_joint_visual.as_matrix()
        
        E_body_visual = E_body_joint @ E_joint_visual
        
        self.params['visual']['origin']['quat'] = Rotation.from_matrix(E_body_visual[0:3, 0:3]).as_quat()
        self.params['visual']['origin']['pos'] = E_body_visual[0:3, 3]

        ''' Part 3: body frame contact mesh '''
        E_joint_collision = np.identity(4)
        if export_unit == 'cm-g':
            E_joint_collision[0:3, 3] = self.params['collision']['origin']['xyz'] * 100.
        else:
            E_joint_collision[0:3, 3] = self.params['collision']['origin']['xyz']
        rot_joint_collision = Rotation.from_euler('xyz', self.params['collision']['origin']['rpy'], degrees = False)
        E_joint_collision[0:3, 0:3] = rot_joint_collision.as_matrix()

        E_body_collision = E_body_joint @ E_joint_collision
        
        self.params['collision']['origin']['quat'] = Rotation.from_matrix(E_body_collision[0:3, 0:3]).as_quat()
        self.params['collision']['origin']['pos'] = E_body_collision[0:3, 3]

class Joint:
    def __init__(self):
        self.name = None
        self.body_name = None
        self.params = {}
    
    def parse_urdf(self, node):
        self.params = xml_to_dict(node)
        
        self.name = self.params['name']

        # convert rpy to quaternion
        rot = Rotation.from_euler('xyz', self.params['origin']['rpy'], degrees = False)
        self.params['origin']['quat'] = rot.as_quat()
        # convert xyz to pos
        if export_unit == 'cm-g':
            self.params['origin']['pos'] = self.params['origin']['xyz'] * 100.
        else:
            self.params['origin']['pos'] = self.params['origin']['xyz']

def convert_from_urdf_to_redmax(urdf_path: str, redmax_path: str) -> bool:
    urdf_folder = os.path.dirname(urdf_path)
    redmax_folder = os.path.dirname(redmax_path)
    redmax_mesh_folder = os.path.join(redmax_folder, 'visual')
    redmax_contacts_folder = os.path.join(redmax_folder, 'contacts')

    # check if target folder exists
    if os.path.exists(redmax_folder):
        print_error(r'target path exists: {redmax_folder}')
    
    # # create folders
    os.makedirs(redmax_folder)
    os.makedirs(redmax_mesh_folder)
    os.makedirs(redmax_contacts_folder)

    # load xml of urdf
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    config_name = root.attrib.get('name', 'robot')

    # parse bodies and joints and create links
    bodies = {}
    joints = {}
    for child in root:
        if child.tag == 'link':
            body = Body()
            body.parse_urdf(child)
            bodies[child.attrib['name']] = body
        elif child.tag == 'joint':
            joint = Joint()
            joint.parse_urdf(child)
            joints[child.attrib['name']] = joint

    # parse connection
    for (_, joint) in joints.items():
        bodies[joint.params['parent']['link']].children.append(joint.params['child']['link'])
        bodies[joint.params['child']['link']].joint_name = joint.name
    
    # collect visual meshes and copy to redmax_mesh_folder
    visual_mesh_paths = set()
    for (_, body) in bodies.items():
        visual_mesh_paths.add(body.params['visual']['geometry']['mesh']['filename'])
    for visual_mesh_pash in visual_mesh_paths:
        visual_mesh = trimesh.load_mesh(os.path.join(urdf_folder, visual_mesh_pash), file_type = 'obj')
        if export_unit == 'cm-g':
            visual_mesh.vertices = visual_mesh.vertices * 100.
        mesh_filename = os.path.basename(visual_mesh_pash)
        with open(os.path.join(redmax_mesh_folder, mesh_filename), 'w', encoding = 'utf-8') as f:
            visual_mesh.export(f, file_type = 'obj')
        # shutil.copyfile(os.path.join(urdf_folder, visual_mesh_pash), os.path.join(redmax_mesh_folder, mesh_filename))
    
    # collect collision meshes and create contacts.txt in redmax_contacts_folder
    collision_mesh_paths = set()
    for (_, body) in bodies.items():
        collision_mesh_paths.add(body.params['collision']['geometry']['mesh']['filename'])
    for collision_mesh_path in collision_mesh_paths:
        mesh_filename = os.path.basename(collision_mesh_path)
        redmax_contacts_filename = mesh_filename.split('.')[0] + '.txt'
        collision_mesh = trimesh.load_mesh(os.path.join(urdf_folder, collision_mesh_path), file_type = 'obj')
        contact_fp = open(os.path.join(redmax_contacts_folder, redmax_contacts_filename), 'w')
        contact_fp.write('{}\n'.format(collision_mesh.vertices.shape[0]))
        if export_unit == 'cm-g':
            for i in range(collision_mesh.vertices.shape[0]):
                contact_fp.write('{} {} {}\n'.format(collision_mesh.vertices[i][0] * 100., collision_mesh.vertices[i][1] * 100., collision_mesh.vertices[i][2] * 100.))
        else:
            for i in range(collision_mesh.vertices.shape[0]):
                contact_fp.write('{} {} {}\n'.format(collision_mesh.vertices[i][0], collision_mesh.vertices[i][1], collision_mesh.vertices[i][2]))
        contact_fp.close()

    # find root(s) and export xml
    def export(body, fp, indentation):
        body.exported = True

        if body.joint_name is None:
            fp.write(indentation + f'<link name="{body.name}">\n')
        else:
            fp.write(indentation + f'<link name="{body.joint_name}">\n')

        # export joint
        if body.joint_name is None:
            fp.write(indentation + '    ' + \
                f'<joint name="{body.name}" '\
                    'type="fixed" '\
                    'pos="0 0 0" '\
                    'quat="1 0 0 0"/>\n')
        elif joints[body.joint_name].params['type'] == 'fixed':
            pos = joints[body.joint_name].params['origin']['pos']
            quat = joints[body.joint_name].params['origin']['quat']
            fp.write(indentation + '    ' + \
                f'<joint name="{body.joint_name}" '\
                    'type="fixed" '\
                    f'pos="{pos[0]} {pos[1]} {pos[2]}" '\
                    f'quat="{quat[3]} {quat[0]} {quat[1]} {quat[2]}"/>\n')
        elif joints[body.joint_name].params['type'] == 'revolute':
            pos = joints[body.joint_name].params['origin']['pos']
            quat = joints[body.joint_name].params['origin']['quat']
            axis = joints[body.joint_name].params['axis']['xyz']
            limit = joints[body.joint_name].params['limit']
            damping = 0.0
            fp.write(indentation + '    ' + \
                f'<joint name = "{body.joint_name}" '\
                    f'type="revolute" '\
                    f'pos="{pos[0]} {pos[1]} {pos[2]}" '\
                    f'quat="{quat[3]} {quat[0]} {quat[1]} {quat[2]}" '\
                    f'axis="{axis[0]} {axis[1]} {axis[2]}" '\
                    f'lim="{limit["lower"]} {limit["upper"]}" '\
                    f'damping="{damping}"/>\n')
        else:
            raise NotImplementedError

        # export body    
        mass = body.params['inertial']['mass']['value']
        inertia = body.params['inertial']['inertia']
        color = body.params['visual']['material']['color']['rgba']
        pos = body.params['inertial']['origin']['pos']
        quat = body.params['inertial']['origin']['quat']
        
        # body part
        fp.write(indentation + '    ' + \
            f'<body name= "{body.name}" '\
                f'type = "abstract" '\
                f'pos = "{pos[0]} {pos[1]} {pos[2]}" '\
                f'quat = "{quat[3]} {quat[0]} {quat[1]} {quat[2]}" '\
                f'mass = "{mass}" '\
                f'inertia = "{inertia[0]} {inertia[1]} {inertia[2]}" '\
                f'rgba = "{color[0]} {color[1]} {color[2]} {color[3]}">\n')

        # visual mesh
        visual_mesh_filename = os.path.basename(body.params['visual']['geometry']['mesh']['filename'])
        mesh_path = os.path.join('visual', visual_mesh_filename)
        visual_pos = body.params['visual']['origin']['pos']
        visual_quat = body.params['visual']['origin']['quat']
        fp.write(indentation + '        ' + \
            f'<visual mesh = "{mesh_path}" '\
                f'pos = "{visual_pos[0]} {visual_pos[1]} {visual_pos[2]}" '\
                f'quat = "{visual_quat[3]} {visual_quat[0]} {visual_quat[1]} {visual_quat[2]}"/>\n')
                
        # contacts
        collision_mesh_filename = os.path.basename(body.params['collision']['geometry']['mesh']['filename'])
        redmax_contacts_filename = collision_mesh_filename.split('.')[0] + '.txt'
        contacts_path = os.path.join('contacts', redmax_contacts_filename)
        collision_pos = body.params['collision']['origin']['pos']
        collision_quat = body.params['collision']['origin']['quat']
        fp.write(indentation + '        ' + \
            f'<collision contacts = "{contacts_path}" '\
                f'pos = "{collision_pos[0]} {collision_pos[1]} {collision_pos[2]}" '\
                f'quat = "{collision_quat[3]} {collision_quat[0]} {collision_quat[1]} {collision_quat[2]}"/>\n')
        
        fp.write(indentation + '    ' + '</body>\n')

        for child_name in body.children:
            export(bodies[child_name], redmax_config_fp, indentation = indentation + '    ')
            
        fp.write(indentation + '</link>\n')

    redmax_config_fp = open(os.path.join(redmax_folder, f'{config_name}.xml'), 'w')
    redmax_config_fp.write(f'<redmax model="{config_name}">\n')
    if export_unit == "cm-g":
        redmax_config_fp.write(f'    <option integrator="BDF1" timestep="5e-3" gravity="0. 0. -980."/>\n')
    else:
        redmax_config_fp.write(f'    <option integrator="BDF1" timestep="5e-3" gravity="0. 0. -9.8"/>\n')
    redmax_config_fp.write('\n    <robot>\n')
    for (_, body) in bodies.items():
        if not body.exported and body.joint_name == None:
            export(body, redmax_config_fp, indentation = '        ')
    redmax_config_fp.write('    </robot>\n')
    redmax_config_fp.write('</redmax>\n')
    redmax_config_fp.close()

parser = ArgumentParser("")
parser.add_argument("--input-urdf-path", type=str, required=True)
parser.add_argument("--export-redmax-path", type=str, required=True)
parser.add_argument("--export-unit", type=str, default="cm-g")
args = parser.parse_args()

export_unit = args.export_unit

convert_from_urdf_to_redmax(args.input_urdf_path, args.export_redmax_path)