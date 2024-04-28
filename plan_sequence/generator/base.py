class Generator:
    '''
    Candidate part generator
    '''
    def __init__(self, asset_folder, assembly_dir, base_part=None, save_sdf=False):
        self.asset_folder = asset_folder
        self.assembly_dir = assembly_dir
        self.base_part = base_part
        self.save_sdf = save_sdf

    def _remove_base_part(self, assembled_parts):
        if self.base_part is not None:
            assembled_parts = [part for part in assembled_parts if part != self.base_part]
        return assembled_parts

    def generate_candidate_part(self, assembled_parts):
        '''
        Generate the next candidate part to disassemble
        Input:
            assembled_parts: parts that are still assembled but need to be disassembled
        '''
        raise NotImplementedError
