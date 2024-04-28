import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import numpy as np
import trimesh
import shutil

from assets.load import load_assembly, get_transform_matrix
from plan_sequence.assets.build_contact_graph import build_contact_graph
from utils.output_grabber import OutputGrabber


def load_assembly_for_subtract(obj_dir, obj_ids, apply_transform=False, revert_transform=False):

    meshes = {}
    assembly = load_assembly(obj_dir, transform='none')
    for obj_id in obj_ids:
        obj_data = assembly[obj_id]
        mesh = obj_data['mesh']
        if apply_transform:
            mesh.apply_transform(get_transform_matrix(obj_data['final_state']))
        if revert_transform:
            mesh.apply_transform(np.linalg.inv(get_transform_matrix(obj_data['final_state'])))
        meshes[obj_id] = mesh

    return meshes


def save_assembly(obj_dir, meshes, temp=False):

    os.makedirs(obj_dir, exist_ok=True)

    for obj_id, mesh in meshes.items():
        save_path = os.path.join(obj_dir, f'{obj_id}.obj') if not temp else os.path.join(obj_dir, f'{obj_id}_temp.obj')
        mesh.export(save_path, header=None, include_color=False)


def copy_config(source_dir, target_dir):

    os.makedirs(target_dir, exist_ok=True)

    source_path = os.path.join(source_dir, 'config.json')
    target_path = os.path.join(target_dir, 'config.json')
    if os.path.exists(source_path):
        shutil.copyfile(source_path, target_path)


def run_mesh_boolean(obj_dir, subtract_pairs, boolean_path):

    assert os.path.exists(boolean_path)

    error_ids = []

    for (part_i, part_j) in subtract_pairs:
        path_i = os.path.join(obj_dir, f'{part_i}.obj')
        path_j = os.path.join(obj_dir, f'{part_j}.obj')
        assert os.path.exists(path_i)
        assert os.path.exists(path_j)

        out = OutputGrabber()
        out.start()
        os.system(f'{boolean_path} subtraction {path_i} {path_j} {path_i}_temp.obj')
        out.stop()

        if 'WARNING' in out.capturedtext:
            error_ids.append(part_i)
            os.system(f'rm {path_i}_temp.obj')
        else:
            os.system(f'mv {path_i}_temp.obj {path_i}')

    return error_ids


def copy_remaining_meshes(source_dir, target_dir, obj_ids):

    for obj_id in obj_ids:
        source_path = os.path.join(source_dir, f'{obj_id}.obj')
        target_path = os.path.join(target_dir, f'{obj_id}.obj')
        assert os.path.exists(source_path)
        shutil.copyfile(source_path, target_path)


def sanity_check_files(obj_dir, obj_ids):
    
    for obj_id in obj_ids:
        obj_path = os.path.join(obj_dir, f'{obj_id}.obj')
        assert os.path.exists(obj_path)

    assert os.path.exists(os.path.join(obj_dir, 'config.json'))


def subtract_mesh(source_dir, target_dir, boolean_path, verbose=False):
    
    # build contact graph
    G = build_contact_graph(source_dir, plot=False)

    # compute a list of subtract pairs
    subtract_pairs = []
    pene_count = {}
    for part_i in G.nodes:
        for part_j in G.neighbors(part_i):
            if G.edges[part_i, part_j]['dist'] < 0:
                if part_i not in pene_count:
                    pene_count[part_i] = 1
                else:
                    pene_count[part_i] += 1
                if part_j not in pene_count:
                    pene_count[part_j] = 1
                else:
                    pene_count[part_j] += 1
    G_temp = G.copy()
    while len(pene_count) > 0:
        part_worst = max(pene_count, key=pene_count.get)
        for part_i in G_temp.neighbors(part_worst):
            if G_temp.edges[part_worst, part_i]['dist'] < 0:
                subtract_pairs.append((part_worst, part_i))
                pene_count[part_i] -= 1
                assert pene_count[part_i] >= 0
                if pene_count[part_i] == 0:
                    pene_count.pop(part_i)
        pene_count.pop(part_worst)
        G_temp.remove_node(part_worst)

    # extract subtract related (all) and applied (main) parts
    subtract_ids_all = set()
    subtract_ids_main = set()
    for part_i, part_j in subtract_pairs:
        subtract_ids_main.add(part_i)
        subtract_ids_all.add(part_i)
        subtract_ids_all.add(part_j)
    other_ids = [part_i for part_i in G.nodes if part_i not in subtract_ids_main]

    if verbose:
        print('Parts to be subtracted:', subtract_ids_main)
    
    # load meshes related to subtraction with transform applied
    meshes = load_assembly_for_subtract(source_dir, subtract_ids_all, apply_transform=True)

    for mesh in meshes.values():
        assert mesh.is_watertight

    # save translated meshes
    save_assembly(target_dir, meshes, temp=False)

    # run mesh boolean
    error_ids = run_mesh_boolean(target_dir, subtract_pairs, boolean_path)

    # copy config
    copy_config(source_dir, target_dir)

    # load meshes subtracted with transform reverted
    meshes_subtracted = load_assembly_for_subtract(target_dir, subtract_ids_main, revert_transform=True)

    # fix normals
    for part_i, mesh in meshes_subtracted.items():

        try:
            trimesh.repair.fix_normals(mesh)
            assert mesh.is_watertight
        except Exception as e:
            error_ids.append(part_i)
            other_ids.append(part_i) # skip this part, has some issues
            if verbose:
                print(f'Error: mesh {part_i} has exception', e)

    meshes_subtracted = {key: val for key, val in meshes_subtracted.items() if key not in error_ids}

    # save untranslated meshes
    save_assembly(target_dir, meshes_subtracted, temp=False)

    # copy other meshes that are not subtracted
    copy_remaining_meshes(source_dir, target_dir, other_ids)

    # sanity check if all objs are written
    sanity_check_files(target_dir, G.nodes)


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--source-dir', type=str, required=True)
    parser.add_argument('--target-dir', type=str, required=True)
    parser.add_argument('--boolean-path', type=str, required=True)
    args = parser.parse_args()

    subtract_mesh(args.source_dir, args.target_dir, args.boolean_path, verbose=True)
