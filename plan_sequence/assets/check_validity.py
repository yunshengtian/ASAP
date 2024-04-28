import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import matplotlib
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import trimesh

from plan_sequence.feasibility_check import check_given_connection_assemblable
from plan_sequence.physics_planner import CONTACT_EPS, get_distance_all_bodies
from assets.load import load_part_ids


def check_validity(assembly_dir, num_proc=1, contact_eps=CONTACT_EPS, collision_th=0.01, save_graph_path=None, save_fig_path=None, skip_draw=False, save_sdf=False, verbose=False):
    violations = set()

    asset_folder = os.path.join(project_base_dir, './assets')

    G = nx.Graph()
    distance = get_distance_all_bodies(asset_folder, assembly_dir, save_sdf=save_sdf)
    good_pairs = []

    if verbose:
        print('Computed distance')

    # check contact and penetration
    for (part_i, part_j), dist in distance.items():
        if not G.has_node(part_i): G.add_node(part_i, color='gray')
        if not G.has_node(part_j): G.add_node(part_j, color='gray')
        if dist < contact_eps:
            G.add_edge(part_i, part_j, color='black') # good
            if dist < -collision_th:
                G.edges[part_i, part_j]['color'] = 'blue' # penetrated
                G.edges[part_i, part_j]['dist'] = dist
                violations.add('penetration')
            else:
                good_pairs.append([part_i, part_j])
            G.nodes[part_i]['color'] = 'green' # good
            G.nodes[part_j]['color'] = 'green' # good

    if verbose:
        print('Checked contact and penetration')

    # check assemblable
    _, failures = check_given_connection_assemblable(asset_folder, assembly_dir, good_pairs, bidirection=True, num_proc=num_proc, save_sdf=save_sdf, debug=0)
    for (part_i, part_j) in failures:
        if (part_j, part_i) in failures:
            G.edges[part_i, part_j]['color'] = 'red' # unassemblable
            violations.add('unassemblable')

    if verbose:
        print('Checked assemblability')

    # check thickness
    min_thickness = 0.05
    parts = load_part_ids(assembly_dir)
    for part_i in parts:
        mesh = trimesh.load_mesh(f'{assembly_dir}/{part_i}.obj', process=False, maintain_order=True)
        _, bbox = trimesh.bounds.oriented_bounds(mesh)
        if bbox.min() < min_thickness:
            G.nodes[part_i]['color'] = 'red' # thin
            violations.add('thin part')

    if verbose:
        print('Checked thickness')

    node_colors = list(nx.get_node_attributes(G, 'color').values())
    if 'gray' in node_colors:
        violations.add('disconnected part')

    if save_graph_path is not None:
        save_dir = os.path.dirname(os.path.abspath(save_graph_path))
        os.makedirs(save_dir, exist_ok=True)
        if not save_graph_path.endswith('.pkl'):
            save_graph_path += '.pkl'
        with open(save_graph_path, 'wb') as fp:
            pickle.dump(G, fp)
    
    if not skip_draw:
        if save_fig_path is not None:
            matplotlib.use('agg')
        
        from networkx.drawing.nx_agraph import graphviz_layout
        edge_colors = list(nx.get_edge_attributes(G, 'color').values())
        pos = graphviz_layout(G)
        nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        edge_labels = {edge: str(round(G.edges[edge]['dist'],2)) for edge in G.edges if 'dist' in G.edges[edge]}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        if save_fig_path is None:
            plt.show()
        else:
            save_dir = os.path.dirname(os.path.abspath(save_fig_path))
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_fig_path)

    return sorted(list(violations))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='assembly dir')
    parser.add_argument('--num-proc', type=int, default=1)
    parser.add_argument('--contact-eps', type=float, default=CONTACT_EPS)
    parser.add_argument('--collision-th', type=float, default=0.01)
    parser.add_argument('--save-graph-path', type=str, default=None)
    parser.add_argument('--save-fig-path', type=str, default=None)
    parser.add_argument('--skip-draw', default=False, action='store_true')
    parser.add_argument('--disable-save-sdf', default=False, action='store_true')
    args = parser.parse_args()

    violations = check_validity(args.dir, args.num_proc, args.contact_eps, args.collision_th, args.save_graph_path, args.save_fig_path, args.skip_draw, not args.disable_save_sdf, verbose=True)
    print(f'violations: {violations}')
