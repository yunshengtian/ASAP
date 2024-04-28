import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import trimesh
from assets.load import load_assembly, load_part_ids
from assets.transform import transform_pt_by_matrix


def get_ground_mesh(z=0, s=10):
    v = np.array([
        [-s, -s, z],
        [-s, s, z],
        [s, -s, z],
        [s, s, z]
    ])
    f = np.array([
        [0, 2, 1],
        [2, 3, 1],
        [1, 3, 2],
        [0, 1, 2],
    ], dtype=int)
    return trimesh.Trimesh(v, f)


def get_combined_mesh(assembly_dir, parts=None):
    if parts is None: parts = load_part_ids(assembly_dir)

    assembly = load_assembly(assembly_dir)
    
    target_meshes = []
    for part_id, part_data in assembly.items():
        if part_id in parts:
            target_meshes.append(part_data['mesh'])

    mesh = trimesh.util.concatenate(target_meshes)
    return mesh


def get_stable_poses(mesh, prob_th=0.9, max_num=3, return_prob=False, shift_center=True):

    assert max_num >= 0
    if max_num == 0: return []

    poses, probs = trimesh.poses.compute_stable_poses(mesh, n_samples=1)

    top_poses = []
    top_probs = []
    prob_cum = 0
    for pose, prob in zip(poses, probs):
        top_poses.append(pose)
        top_probs.append(prob)
        prob_cum += prob
        if prob_cum > prob_th:
            break
        if len(top_poses) >= max_num:
            break

    if shift_center:
        for pose in top_poses:
            pose[:2, 3] -= transform_pt_by_matrix(mesh.centroid, pose)[:2]

    if return_prob:
        return top_poses, top_probs
    else:
        return top_poses


def visualize_stable_poses(mesh, poses):

    print('original pose')
    ground_mesh = get_ground_mesh(z=mesh.vertices.min(axis=0)[2])
    trimesh.Scene([mesh, ground_mesh]).show()

    for i, pose in enumerate(poses):
        print(f'{i+1}/{len(poses)} stable pose')
        mesh_new = mesh.copy()
        mesh_new.apply_transform(pose)
        ground_mesh = get_ground_mesh()
        trimesh.Scene([mesh_new, ground_mesh]).show()


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--part-ids', type=str, nargs='+', default=None)
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    mesh = get_combined_mesh(args.dir, args.part_ids)
    poses, probs = get_stable_poses(mesh, prob_th=1.0, max_num=10, return_prob=True)
    # visualize_stable_poses(mesh, poses)
    # exit()

    from assets.save import save_posed_mesh

    assembly_id = os.path.basename(args.dir)
    for i, (pose, prob) in enumerate(zip(poses, probs)):
        parts = load_part_ids(args.dir) if args.part_ids is None else args.part_ids
        save_posed_mesh(f'pose_{assembly_id}{args.suffix}/{i}', args.dir, parts, pose)
        print(prob)

    # import redmax_py as redmax
    # from utils.renderer import SimRenderer
    # from seq_optim.sim_string import get_stability_sim_string
    # part_ids = [x.replace('.obj', '') for x in load_part_ids(args.dir)]
    # asset_folder = os.path.join(project_base_dir, 'assets')
    # sim_string = get_stability_sim_string(args.dir, parts_fix=[], parts_move=part_ids, pose=poses[0])
    # sim = redmax.Simulation(sim_string, asset_folder)
    # sim.reset()
    # for _ in range(1000):
    #     sim.forward(1)
    # SimRenderer.replay(sim, record=False)
