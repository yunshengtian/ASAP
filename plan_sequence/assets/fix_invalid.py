import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import shutil
import pickle
import json


def save_new_parts(source_dir, target_dir, parts):

    os.makedirs(target_dir, exist_ok=True)
    for file_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        if file_name.endswith('.obj'):
            part_id = file_name.replace('.obj', '')
            if part_id in parts:
                shutil.copyfile(source_path, target_path)

        elif file_name == 'config.json':

            with open(source_path, 'r') as fp:
                config = json.load(fp)
            new_config = {key: val for key, val in config.items() if key in parts}
            with open(target_path, 'w') as fp:
                json.dump(new_config, fp)


def fix_invalid_graph(G):

    # remove thin parts
    bad_parts = set()
    for part_i in G.nodes:
        if G.nodes[part_i]['color'] == 'red':
            bad_parts.add(part_i)
    for part_i in bad_parts:
        G.remove_node(part_i)
    if len(G.nodes) < 3:
        return None

    # remove most unassemblable parts
    bad_count = {}
    for part_i, part_j in G.edges:
        if G.edges[part_i, part_j]['color'] == 'red':
            if part_i not in bad_count:
                bad_count[part_i] = 1
            else:
                bad_count[part_i] += 1
            if part_j not in bad_count:
                bad_count[part_j] = 1
            else:
                bad_count[part_j] += 1
    while len(bad_count) > 0:
        part_worst = max(bad_count, key=bad_count.get)
        for part_i in G.neighbors(part_worst):
            if G.edges[part_worst, part_i]['color'] == 'red':
                bad_count[part_i] -= 1
                assert bad_count[part_i] >= 0
                if bad_count[part_i] == 0:
                    bad_count.pop(part_i)
        bad_count.pop(part_worst)
        G.remove_node(part_worst)
    if len(G.nodes) < 3:
        return None

    # take the biggest subgraph
    Gs = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gs[0])
    if len(G.nodes) < 3:
        return None

    return G


def fix_invalid_assembly(assembly_dir_src, assembly_dir_tgt, save_dir_src, save_dir_tgt, skip_draw=False, verbose=False):

    assembly_id = os.path.basename(assembly_dir_src)
    graph_dir_src = os.path.join(save_dir_src, 'graph')
    graph_dir_tgt = os.path.join(save_dir_tgt, 'graph')
    graph_path_src = os.path.join(graph_dir_src, f'{assembly_id}.pkl')
    graph_path_tgt = os.path.join(graph_dir_tgt, f'{assembly_id}.pkl')
    fig_dir_tgt = os.path.join(save_dir_tgt, 'fig')
    fig_path_tgt = os.path.join(fig_dir_tgt, assembly_id)

    with open(graph_path_src, 'rb') as fp:
        G = pickle.load(fp)

    G_new = fix_invalid_graph(G)

    success, n_parts, penetration = False, None, None

    if G_new is not None:
        os.makedirs(assembly_dir_tgt, exist_ok=True)
        os.makedirs(graph_dir_tgt, exist_ok=True)
        os.makedirs(fig_dir_tgt, exist_ok=True)

        # save new assembly
        save_new_parts(assembly_dir_src, assembly_dir_tgt, list(G_new.nodes))

        # save new graph
        with open(graph_path_tgt, 'wb') as fp:
            pickle.dump(G_new, fp)

        # save new figure
        if not skip_draw:
            matplotlib.use('agg')

            from networkx.drawing.nx_agraph import graphviz_layout
            node_colors = list(nx.get_node_attributes(G_new, 'color').values())
            edge_colors = list(nx.get_edge_attributes(G_new, 'color').values())
            pos = graphviz_layout(G_new)
            nx.draw(G_new, pos, node_color=node_colors, edge_color=edge_colors, with_labels=True)
            edge_labels = {edge: str(round(G_new.edges[edge]['dist'],2)) for edge in G_new.edges if 'dist' in G_new.edges[edge]}
            nx.draw_networkx_edge_labels(G_new, pos, edge_labels=edge_labels)
            plt.savefig(fig_path_tgt)

        # compute stats
        success = True
        n_parts = len(G_new.nodes)
        for edge in G_new.edges:
            if G_new.edges[edge]['color'] == 'blue':
                if penetration is None:
                    penetration = G_new.edges[edge]['dist']
                else:
                    penetration = min(G_new.edges[edge]['dist'], penetration)

    if verbose:
        print(f'Success: {success}, n_parts: {n_parts}, penetration: {penetration}')

    return success, n_parts, penetration


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dir-src', type=str, required=True, help='source assembly dir')
    parser.add_argument('--dir-tgt', type=str, required=True, help='target assembly dir')
    parser.add_argument('--save-dir-src', type=str, required=True)
    parser.add_argument('--save-dir-tgt', type=str, required=True)
    parser.add_argument('--skip-draw', default=False, action='store_true')
    args = parser.parse_args()

    fix_invalid_assembly(args.dir_src, args.dir_tgt, args.save_dir_src, args.save_dir_tgt, args.skip_draw, verbose=True)
