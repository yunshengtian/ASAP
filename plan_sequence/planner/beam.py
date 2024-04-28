import numpy as np

from .base import SequencePlanner


class BeamSearchSequencePlanner(SequencePlanner):

    beams = None
    beam_width = None
    n_candidates = None
    curr_beam_idx = None

    beam_width_init = 1
    n_candidate_init = 1

    def _reset(self):
        self.beams = [[] for _ in range(len(self.parts) - 1)]
        self.beam_width = self.beam_width_init
        self.n_candidates = [self.n_candidate_init for _ in range(len(self.parts) - 1)]
        self.curr_beam_idx = 0

        G0 = self.parts.copy()
        self.beams[0].append(G0)

    def _select_node(self, tree):

        # explore the current beam
        if len(self.beams[self.curr_beam_idx + 1]) < self.beam_width:
        
            for G in self.beams[self.curr_beam_idx]:
                n_child = tree.out_degree(tuple(G))
                if n_child < min(self.n_candidates[self.curr_beam_idx], len(G)):
                    return G

            # too few candidates, increase n_candidate
            curr_n_parts = len(self.parts) - self.curr_beam_idx
            if self.n_candidates[self.curr_beam_idx] < curr_n_parts:
                self.n_candidates[self.curr_beam_idx] += 1
                return self._select_node(tree)
            else: # NOTE: failed in all children
                self.beam_width += 1
                self.curr_beam_idx = 0
                return self._select_node(tree)

        # move to the next beam
        self.curr_beam_idx += 1

        def _score_function(parts): # for selecting top-k candidates for moving beam
            n_gripper = tree.nodes[tuple(parts)]['n_gripper']
            if n_gripper is None:
                return len(self.parts) + 1
            else:
                return n_gripper

        # print(f'[DEBUG] next_beam {self.beams[self.curr_beam_idx]}, beam_width: {self.beam_width}, n_candidate: {self.n_candidates[self.curr_beam_idx - 1]}')

        self.beams[self.curr_beam_idx] = sorted(np.random.permutation(self.beams[self.curr_beam_idx]).tolist(), key=_score_function)[:min(self.beam_width, len(self.beams[self.curr_beam_idx]))]

        if self.curr_beam_idx == len(self.parts) - 2: # fully explored to the last beam, enlarge beam and reset
            self.beam_width += 1
            self.curr_beam_idx = 0

        return self._select_node(tree)

    def _update_tree(self, tree, parts_parent, parts_child, n_eval, sim_info):
        super()._update_tree(tree, parts_parent, parts_child, n_eval, sim_info)

        if sim_info['feasible'] and \
            (not self._check_fully_explored(tree, parts_child) or len(parts_child) == 2) and \
            parts_child not in self.beams[self.curr_beam_idx + 1]: # add feasible child to the next beam
            self.beams[self.curr_beam_idx + 1].append(parts_child)
