import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import networkx as nx
import matplotlib.pyplot as plt

from plan_sequence.physics_planner import CONTACT_EPS, get_distance_all_bodies


def build_contact_graph(obj_dir, plot=False):
    asset_folder = os.path.join(project_base_dir, './assets')

    G = nx.Graph()
    distance = get_distance_all_bodies(asset_folder, obj_dir)

    # check contact and penetration
    for (part_i, part_j), dist in distance.items():
        if not G.has_node(part_i): G.add_node(part_i, color='gray')
        if not G.has_node(part_j): G.add_node(part_j, color='gray')
        if dist < CONTACT_EPS:
            color = 'blue' if dist < 0 else 'black'
            G.add_edge(part_i, part_j, color=color, dist=dist)
            G.nodes[part_i]['color'] = 'green' # good
            G.nodes[part_j]['color'] = 'green' # good

    if plot:
        from networkx.drawing.nx_agraph import graphviz_layout
        node_colors = list(nx.get_node_attributes(G, 'color').values())
        edge_colors = list(nx.get_edge_attributes(G, 'color').values())
        pos = graphviz_layout(G)
        nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        edge_labels = {edge: str(round(G.edges[edge]['dist'],2)) for edge in G.edges if G.edges[edge]['dist'] < 0}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
    
    return G


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    build_contact_graph(args.dir, plot=True)
