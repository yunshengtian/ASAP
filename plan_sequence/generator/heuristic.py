import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import numpy as np
import trimesh

from plan_sequence.feasibility_check import get_contact_graph
from assets.load import load_assembly
from .base import Generator


class HeuristicsGenerator(Generator):
    '''
    Generate candidate parts by heuristics
    '''
    def __init__(self, asset_folder, assembly_dir, base_part=None, save_sdf=False):
        super().__init__(asset_folder, assembly_dir, base_part=base_part, save_sdf=save_sdf)
        self.contact_graph = get_contact_graph(self.asset_folder, self.assembly_dir, save_sdf=save_sdf)
        assembly = load_assembly(self.assembly_dir)
        self.mesh_dict = {part_id: part_data['mesh'] for part_id, part_data in assembly.items()}

    def heuristic_func(self, part):
        raise NotImplementedError

    def generate_candidate_part(self, assembled_parts):
        assembled_parts = self._remove_base_part(assembled_parts)
        
        eps = 0.0
        if np.random.random() < eps:
            sorted_parts = np.random.permutation(np.array(assembled_parts, dtype=object))
        else:
            sorted_parts = sorted(np.random.permutation(np.array(assembled_parts, dtype=object)), key=self.heuristic_func)
        for part in sorted_parts:
            yield part


class HeuristicsVolumeGenerator(HeuristicsGenerator):

    def heuristic_func(self, part):
        return self.mesh_dict[part].volume


class HeuristicsOutsidenessGenerator(HeuristicsGenerator):

    def get_center(self, parts):
        meshes = [self.mesh_dict[part] for part in parts]
        centroid = trimesh.Scene(meshes).centroid
        return centroid

    def heuristic_func(self, part, center):
        return -np.linalg.norm(self.mesh_dict[part].center_mass - center)

    def generate_candidate_part(self, assembled_parts):
        assembled_parts = self._remove_base_part(assembled_parts)
        
        center = self.get_center(assembled_parts)
        sorted_parts = sorted(np.random.permutation(np.array(assembled_parts, dtype=object)), key=lambda x: self.heuristic_func(x, center))
        for part in sorted_parts:
            yield part
