import numpy as np

from .base import Generator


class RandomGenerator(Generator):
    '''
    Generate candidate parts by random permutation
    '''
    def generate_candidate_part(self, assembled_parts):
        assembled_parts = self._remove_base_part(assembled_parts)

        candidate_parts = np.random.permutation(assembled_parts)
        for part in candidate_parts:
            yield part
