import os
from urllib.request import urlretrieve
import numpy as np
import torch
from torch_geometric.utils import from_networkx

from plan_sequence.feasibility_check import get_contact_graph

from .base import Generator
from .network.model import ModelGraph
from .network.data import AssemblyDataGraph
from .network.args import get_args, curr_dir


def nx_to_pyg_graph(nx_graph):
    subgraph = nx_graph
    for node in subgraph:
        # Add additional feature for the number of edges connected to this node
        edge_count = torch.tensor(len(subgraph.edges(node))).float()
        subgraph.nodes[node]["edge_count"] = edge_count
        # Also add the file name
        subgraph.nodes[node]["file"] = node
    # Convert to a pytorch geometric graph
    pyg_subgraph = from_networkx(subgraph)
    # Randomly rotate the graph points if requested and only with training data
    # pyg_subgraph.x = AssemblyDataGraph.random_rotate_points(pyg_subgraph.x)
    return pyg_subgraph


class LearningBasedGenerator(Generator):
    '''
    Generate candidate parts by learned network
    '''
    model_ckpt = 'pretrained_model'

    def __init__(self, asset_folder, assembly_dir, base_part=None, save_sdf=False):
        super().__init__(asset_folder, assembly_dir, base_part=base_part, save_sdf=save_sdf)
        args = get_args()
        # from pathlib import Path
        # contact_graph = AssemblyDataGraph.load_contact_graph(Path(assembly_dir))
        contact_graph = get_contact_graph(self.asset_folder, self.assembly_dir, save_sdf=save_sdf)
        self.nx_graph = AssemblyDataGraph.load_assembly_graph(assembly_dir, max_pc_size=args.max_pc_size, contact_graph=contact_graph)
        self.model = ModelGraph(args)
        model_path = os.path.join(curr_dir, f'{self.model_ckpt}.pt')
        if not os.path.exists(model_path):
            urlretrieve('https://people.csail.mit.edu/yunsheng/ASAP/pretrained_model.pt', model_path)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
        self.model.eval()
    
    def generate_candidate_part(self, assembled_parts):
        assembled_parts = self._remove_base_part(assembled_parts)

        eps = 0.0
        if np.random.random() < eps:
            ordered_parts = np.random.permutation(np.array(assembled_parts, dtype=object))
        else:
            node_list = [p for p in assembled_parts]
            nx_subgraph = self.nx_graph.subgraph(node_list)
            pyg_subgraph = nx_to_pyg_graph(nx_subgraph)
            with torch.no_grad():
                logits = self.model.forward_logits(pyg_subgraph)
            probs = torch.sigmoid(logits).squeeze().detach().numpy()
            ordered_parts = np.array(pyg_subgraph.file, dtype=str)[np.argsort(probs)][::-1]

        return ordered_parts

