import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np
import redmax_py as redmax
import os
from argparse import ArgumentParser
from tqdm import tqdm

from utils.renderer import SimRenderer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='box_stack', help='name of xml model')
    parser.add_argument('--steps', type=int, default=1000, help='number of simulation steps')
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=None, help='camera lookat')
    parser.add_argument('--camera-pos', type=float, nargs=3, default=None, help='camera position')
    args = parser.parse_args()

    asset_folder = os.path.join(project_base_dir, './assets')
    model_path = os.path.join(asset_folder, args.model)
    if not model_path.endswith('.xml'): model_path += '.xml'

    sim = redmax.Simulation(model_path)
    sim.reset(backward_flag=False)
    if args.camera_lookat is not None:
        sim.viewer_options.camera_lookat = args.camera_lookat
    if args.camera_pos is not None:
        sim.viewer_options.camera_pos = args.camera_pos

    for i in tqdm(range(args.steps)):
        sim.set_u(np.zeros(sim.ndof_u))
        sim.forward(1, verbose=False, test_derivatives=False)
        q = sim.get_q()
        qdot = sim.get_qdot()
        variables = sim.get_variables()
        
    SimRenderer.replay(sim, record=False)
