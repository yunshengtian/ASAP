import numpy as np

from .base import SequencePlanner


class DFSSequencePlanner(SequencePlanner):

    G_path = None

    def _reset(self):
        G0 = self.parts.copy()
        self.G_path = [G0]

    def _select_node(self, tree):
        G = self.G_path[-1]
        if tree.out_degree(tuple(G)) < len(G): # current node has unexplored child
            return G
        else:
            self.G_path.pop()
            return self._select_node(tree)
    
    def _update_tree(self, tree, parts_parent, parts_child, n_eval, sim_info):
        super()._update_tree(tree, parts_parent, parts_child, n_eval, sim_info)
        if sim_info['feasible'] and len(parts_child) > 2:
            self.G_path.append(parts_child) # expand tree vertically
        else:
            self.G_path[-1] = parts_parent # expand tree horizontally
