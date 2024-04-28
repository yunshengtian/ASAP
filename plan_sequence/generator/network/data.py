import os
from pathlib import Path
import json
import copy
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import trimesh
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.readwrite import node_link_graph
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split

import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

from assets import load


class AssemblyDataBase(Dataset):
    def __init__(self, args, shuffle_dataset=True, split="train"):
        self.max_pc_size = args.max_pc_size #1000
        self.max_num_parts = args.max_num_parts #20
        self.seed = args.seed
        self.random_rotate = args.random_rotate
        self.split = split
        self.val_fraction = args.val_fraction
        label_path = os.path.join(args.data_dir, args.data_file)
        with open(label_path, 'r') as fp:
            all_data = json.load(fp)	

        # Get the split we want
        self.dataset = self.get_split(all_data['data'])

        # Limit the number of data samples to load
        if args.data_limit > 0:
            self.dataset = self.dataset[:args.data_limit]

    def __getitem__(self, idx):
        # To be implemented by the child class
        pass

    def __len__(self):
        return len(self.dataset)
    
    def get_split(self, all_samples):
        """Get the train/test split"""
        if self.split != "all":
            train_ratio = 1.0 - self.val_fraction * 2
            validation_ratio = self.val_fraction
            test_ratio = self.val_fraction

            train_samples, testval_samples = train_test_split(all_samples, test_size=1.0 - train_ratio)
            val_samples, test_samples = train_test_split(testval_samples, test_size=test_ratio/(test_ratio + validation_ratio))

        if self.split == "train":
            split_samples = train_samples
        elif self.split == "val" or self.split == "valid" or self.split == "validation":
            split_samples = val_samples
        elif self.split == "test":
            split_samples = test_samples
        elif self.split == "all":
            split_samples = all_samples
        else:
            raise Exception("Unknown split name")
        return split_samples


class AssemblyData(AssemblyDataBase):
    def __init__(self, args, shuffle_dataset=True, split="train"):
        super().__init__(args, shuffle_dataset, split)
        self.data_path = os.path.join(args.data_dir,  'part_pc')
        
    def __getitem__(self, index):
        mesh_id = self.dataset[index]['assembly-id']
        parts = sorted(self.dataset[index]["parts-included"])
        part_removed = self.dataset[index]["part-removed"]

        removal_index = parts.index(part_removed)
        part_pc_size = self.max_pc_size // len(parts)
        part_pointcloud = []
        counter = 0
        for part_id in parts :
            # (num_points, xyz) i.e. (2048, 3)
            cur_part_pc = np.load(os.path.join(self.data_path, mesh_id, str(part_id.split('.')[0])+'.npy'))
            assert cur_part_pc.shape[-1] == 3
            # (num_points, xyza) i.e. (2048, 4)
            # where the 4 is xyz followed by a counter for each part index
            cur_part_pc = np.hstack([cur_part_pc, counter * np.ones((cur_part_pc.shape[0], 1))])
            # Downsampling so the full assembly has self.max_pc_size total points
            chosen_idx = np.random.choice(cur_part_pc.shape[0], part_pc_size, replace=False)
            # (part_pc_size, xyza) e.g. (125, 4)
            cur_part_pc = cur_part_pc[chosen_idx]
            part_pointcloud.append(cur_part_pc)
            counter += 1
        
        # Combine the individual parts into a single point cloud
        # (self.max_pc_size, 4)
        part_pointcloud = np.vstack(part_pointcloud)
        # TODO: Randomly rotate the pointcloud xyz if self.random_rotate and self.split == train 
        if part_pointcloud.shape[0] < self.max_pc_size: part_pointcloud = np.vstack([part_pointcloud, np.zeros( (self.max_pc_size - part_pointcloud.shape[0], 4))])
        
        assert part_pointcloud.shape[0] == self.max_pc_size

        part_pointcloud = torch.tensor(part_pointcloud)
        gt = F.one_hot(torch.tensor(removal_index).long(), self.max_num_parts )

        return part_pointcloud, gt


class AssemblyDataGraph(AssemblyDataBase):

    def __init__(self, args, shuffle_dataset=True, split="train"):
        super().__init__(args, shuffle_dataset, split)
        self.data_path = os.path.join(args.data_dir,  "assembly_obj")
        # Preload the full graphs used to generate the subgraphs
        self.graphs_full, failed_graphs = self.load_assembly_graphs()
        # Remove failed graphs from self.dataset
        if len(failed_graphs) > 0:
            assembly_ids = set(failed_graphs)
            # Keep only the assemblies not listed in assembly_ids
            self.dataset = AssemblyDataGraph.remove_assemblies_from_dataset(self.dataset, assembly_ids)

    @staticmethod
    def remove_assemblies_from_dataset(dataset, assembly_ids):
        """Remove any subassembly from the dataset from any of the given assembly ids"""
        dataset[:] = [x for x in dataset if not x["assembly-id"] in assembly_ids]
        return dataset

    @staticmethod
    def get_assembly_dirs(input_dir):
        """Get a list of assembly directories"""
        if not isinstance(input_dir, Path):
            input_dir = Path(input_dir)
        # Looking for e.g. 00014
        pattern = "[0-9][0-9][0-9][0-9][0-9]"
        return [f for f in input_dir.glob(pattern)]
    
    @staticmethod
    def remove_assemblies_not_in_dataset(dataset, assembly_dirs):
        """Remove any assemblies that aren't in the dataset"""
        assembly_ids_in_dataset = {x["assembly-id"] for x in dataset}
        # print(f"Dataset contains {len(dataset)} samples from {len(assembly_ids_in_dataset)} assemblies")
        assembly_dirs[:] = [x for x in assembly_dirs if x.stem in assembly_ids_in_dataset]
        return assembly_dirs

    def load_assembly_graphs(self):
        """Load a directory of assemblies into a graph"""
        # TODO: Currently we load all the data once for each split we have :(
        assembly_dirs = AssemblyDataGraph.get_assembly_dirs(self.data_path)
        # We don't want to waste time loading any graphs that aren't in the dataset
        # so filter them out. Useful when --data_limit is set
        assembly_dirs = AssemblyDataGraph.remove_assemblies_not_in_dataset(self.dataset, assembly_dirs)
        # First load the full graphs without any parts removed
        graphs_full = {}
        # Keep track of the graphs that failed to remove them later
        failed_graphs = []
        print(f"Loading {len(assembly_dirs)} full assemblies for {len(self.dataset)} subgraphs in the {self.split} split...")
        pbar = tqdm(assembly_dirs)
        for assembly_dir in pbar:
            pbar.set_description(assembly_dir.stem)
            try:
                graph = AssemblyDataGraph.load_assembly_graph(assembly_dir, self.max_pc_size)
            except Exception as ex:
                print(f"Failed to load assembly {assembly_dir.stem} into a graph", ex)
                failed_graphs.append(assembly_dir.stem)
                continue
            graphs_full[assembly_dir.stem] = graph
        return graphs_full, failed_graphs

    @staticmethod
    def load_assembly_graph(assembly_dir, max_pc_size, contact_graph=None):
        """Load a single assembly into a graph"""
        if not isinstance(assembly_dir, Path):
            assembly_dir = Path(assembly_dir)
        # Load the meshes
        assembly = load.load_assembly(assembly_dir)
        name_to_mesh = {part_id: part_data['mesh'] for part_id, part_data in assembly.items()}
        # Load the contact graph
        if contact_graph is None: # NOTE: differ from the baseline code
            g = AssemblyDataGraph.load_contact_graph(assembly_dir)
        else:
            g = contact_graph.copy()
        # Add a point cloud to each node in the graph
        pcs = []
        for node in g.nodes:
            mesh = name_to_mesh[node]
            # ------------------
            # Part point cloud
            pc, _ = trimesh.sample.sample_surface(mesh, max_pc_size)
            # Convert to float for downstream NN
            pc = torch.from_numpy(pc).float()
            pcs.append(pc)
            g.nodes[node]["x"] = pc
            # ------------------
            # Part volume
            assert mesh.is_watertight
            g.nodes[node]["volume"] = torch.tensor(mesh.volume).float()
            # ------------------
            # Part center of mass
            g.nodes[node]["center_mass"] = torch.tensor(mesh.center_mass).float()

        # Combine all the point clouds and work out the center and scale
        # at the assembly level
        pcs = torch.vstack(pcs)
        center, scale = AssemblyDataGraph.get_center_and_scale(pcs)
        # Apply the scale to each individual part
        # so the assembly should be at the origin, rather than the individual parts
        for node in g.nodes:
            g.nodes[node]["x"][..., :3] -= center
            g.nodes[node]["x"][..., :3] *= scale
            g.nodes[node]["volume"] *= scale
            g.nodes[node]["center_mass"] -= center
            g.nodes[node]["center_mass"] *= scale
            # Distance from the assembly center to the part center of mass
            g.nodes[node]["distance"] = torch.linalg.norm(g.nodes[node]["center_mass"])
        return g

    @staticmethod
    def load_contact_graph(assembly_dir):
        """Load the contact graph stored in networkx node link json format"""
        contact_graph_file = assembly_dir / "contact_graph.json"
        assert contact_graph_file.exists()
        with open(contact_graph_file, 'r') as fp:
            contact_graph_dict = json.load(fp)
        g = node_link_graph(contact_graph_dict)
        return g

    @staticmethod
    def get_center_and_scale(pts):
        """Get the center and scale of a point cloud"""
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        bbox = torch.tensor([[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]])
        diag = bbox[1] - bbox[0]
        scale = 2.0 / max(diag[0], diag[1], diag[2])
        center = 0.5 * (bbox[0] + bbox[1])
        return center, scale

    @staticmethod
    def get_random_rotation():
        """Get a random rotation in 45 degree increments along the canonical axes"""
        axes = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        angles = [0.0, 90.0, 180.0, 270.0]
        axis = random.choice(axes)
        angle_radians = np.radians(random.choice(angles))
        return Rotation.from_rotvec(angle_radians * axis)

    @staticmethod
    def random_rotate_points(pc, rotation=None):
        """
        Randomly rotate a batch of pointclouds
        of shape (num_pointclouds, num_points, 3)
        """
        if rotation is None:
            rotation = AssemblyDataGraph.get_random_rotation()
        Rmat = torch.tensor(rotation.as_matrix()).float()
        Rmat = Rmat.to(pc.device)
        orig_size = pc.size()
        return torch.mm(pc.view(-1, 3), Rmat).view(orig_size)

    def __getitem__(self, index):
        """
        Constructs a subgraph from the full (preloaded) assembly contact graph
        based on which parts have been removed

        Returns a pytorch geometric graph with the following attributes:
        - x: Per node point clouds with shape (num_nodes, max_pc_size, 3)
        - y: Per node binary labels indicating if this part should be removed,
                with shape (num_nodes)
        - file: A list of original file names for each part as integers
                e.g. [0, 1, ...]
        """
        meta_data = self.dataset[index]
        # Get the full graph from which we derive the data sample from
        assembly_id = meta_data["assembly-id"]
        graph_full = self.graphs_full[assembly_id]
        # No we remove all the parts that aren't in the sub-assembly
        # we do this by removing each node
        parts_included = set(meta_data["parts-included"])
        # Remove zero padding
        parts_included = set(str(Path(s).stem) for s in parts_included)
        parts_all = set(graph_full.nodes)
        # Use set difference to find the parts we want to remove
        parts_difference = parts_all - parts_included
        # Remove the parts that don't exist in the sub-assembly
        # to form a subgraph
        subgraph = copy.deepcopy(graph_full)
        for part in parts_difference:
            subgraph.remove_node(part)
        # Add the labels to the graph
        part_removed = str(Path(meta_data['part-removed']).stem)
        positive_label_count = 0
        for node in subgraph:
            label = float(part_removed == node)
            subgraph.nodes[node]["y"] = label
            if label == 1:
                positive_label_count += 1
            # Sanity check that we don't have any incorrect nodes
            assert node not in parts_difference
            assert node in parts_included
            # Add additional feature for the number of edges connected to this node
            edge_count = torch.tensor(len(subgraph.edges(node))).float()
            subgraph.nodes[node]["edge_count"] = edge_count
            # Also add the file name
            subgraph.nodes[node]["file"] = node
        # Check we have exactly one positive label
        assert positive_label_count == 1
        # Convert to a pytorch geometric graph
        pyg_subgraph = from_networkx(subgraph)
        # Randomly rotate the graph points if requested and only with training data
        if self.random_rotate and self.split == "train":
            pyg_subgraph.x = AssemblyDataGraph.random_rotate_points(pyg_subgraph.x)
        return pyg_subgraph
    
    @staticmethod
    def collate_fn(batch):
        return Batch.from_data_list(batch)
    
    def plot(self, index):
        """Debug functionality to plot the point clouds in a graph at a given index"""
        graph = self[index]
        pts = graph.x
        labels = graph.y
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        colors = sns.color_palette("husl", len(labels))
        # Draw each point cloud with a unique color
        for i, node_pts in enumerate(pts):
            color = colors[i]
            ax.scatter(node_pts[:, 0], node_pts[:, 1], node_pts[:, 2], c=color)
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
        ax.set_title(self.dataset[index]["assembly-id"])
        plt.show()
        