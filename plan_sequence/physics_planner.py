import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from time import time
import numpy as np
import redmax_py as redmax
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from assets.load import load_assembly, load_part_ids
from assets.transform import transform_pt_by_matrix
from plan_path.run_connect import ConnectPathPlanner
from plan_sequence.sim_string import get_path_sim_string, get_stability_sim_string, get_contact_sim_string
from utils.renderer import SimRenderer


# Parameters for physics-based planner

FORCE_MAG = 1e2
FRAME_SKIP = 100
MAX_TIME = 60
CONTACT_EPS = 1e-1
POS_FAR_THRESHOLD = 0.2
POS_NEAR_THRESHOLD = 0.05
NEAR_STEP = 100
MAX_STEP = 1000
CHECK_FREQ = 20
COL_TH_PATH = 0.01
COL_TH_STABLE = 0.00
MIN_SEP = 0.5
DEBUG_SIM = False


class State:

    def __init__(self, q, qdot):
        self.q = q
        self.qdot = qdot

    def __repr__(self):
        return f'[State object at {hex(id(self))}]'


class MultiPartPathPlanner:

    frame_skip = FRAME_SKIP
    max_time = MAX_TIME
    contact_eps = CONTACT_EPS
    col_th_base = COL_TH_PATH
    min_sep = MIN_SEP

    def __init__(self, asset_folder, assembly_dir, parts_fix, part_move, parts_removed=[], pose=None, force_mag=FORCE_MAG, save_sdf=False,
        camera_pos=None, camera_lookat=None):
        model_string = get_contact_sim_string(assembly_dir, parts_fix + [part_move] + parts_removed, save_sdf=save_sdf)
        sim = redmax.Simulation(model_string, asset_folder)
        col_th_dict = self._compute_col_th_dict(sim, parts_fix + parts_removed, part_move)

        model_string = get_path_sim_string(assembly_dir, parts_fix, part_move, parts_removed=parts_removed, 
            save_sdf=save_sdf, pose=pose, col_th=col_th_dict)
        self.sim = redmax.Simulation(model_string, asset_folder)
        if camera_pos is not None: self.sim.viewer_options.camera_pos = camera_pos
        if camera_lookat is not None: self.sim.viewer_options.camera_lookat = camera_lookat
        self.parts_fix = parts_fix
        self.part_move = part_move
        self.parts_removed = parts_removed
        self.pose = pose
        self.force_mag = force_mag

        # connect path planner
        self.connect_path_planner = ConnectPathPlanner(assembly_dir, min_sep=self.min_sep)
        
    def _compute_col_th_dict(self, sim, parts_fix, part_move):
        col_th_dict = {}
        for part_fix in parts_fix:
            d_mf = sim.get_body_distance(f'part{part_move}', f'part{part_fix}')
            d_fm = sim.get_body_distance(f'part{part_fix}', f'part{part_move}')
            col_th_dict[part_fix] = max(0, max(-d_mf, -d_fm)) + self.col_th_base
        col_th_dict[part_move] = max(col_th_dict.values())
        return col_th_dict

    def get_state(self):
        q = self.sim.get_joint_qm(f'part{self.part_move}')
        qdot = self.sim.get_joint_qmdot(f'part{self.part_move}')
        return State(q, qdot)

    def set_state(self, state):
        qm = state.q
        self.sim.set_joint_qm(f'part{self.part_move}', qm)
        self.sim.zero_joint_qdot(f'part{self.part_move}')
        self.sim.update_robot()

    def apply_action(self, action):
        assert len(action) == 3
        action = action * self.force_mag
        force = np.concatenate([np.zeros(3), action])
        self.sim.set_body_external_force(f'part{self.part_move}', force)

    def is_disassembled(self, min_sep=None):
        in_contact = False
        if min_sep is None: min_sep = self.min_sep
        for part_fix in self.parts_fix: # if any part in contact, then not fully disassembled
            in_contact = in_contact or self.sim.body_in_contact(f'part{part_fix}', f'part{self.part_move}', min_sep)
        return not in_contact # if all movable parts are not in contact with fixed parts, then fully disassembled
    
    def check_success(self, action, return_path=False, min_sep=None):

        self.sim.reset()
        self.apply_action(action)

        t_start = time()
        step = 0
        path = []

        while True:

            self.set_state(self.get_state())
            state = self.get_state()
            last_qdot = state.qdot[:3]
            path.append(state.q)

            for _ in range(self.frame_skip):
                self.sim.forward(1, verbose=False)
                new_state = self.get_state()
                path.append(new_state.q)

                t_plan = time() - t_start
                if t_plan > self.max_time:
                    if return_path:
                        return False, path
                    else:
                        return False

            if self.is_disassembled(min_sep):
                break

            qdot = new_state.qdot[:3] # measure translation qdot only
            qdotdot = (qdot - last_qdot) / self.sim.options.h / self.frame_skip
            # if self.pose is not None:
            #     qdotdot = np.dot(qdotdot, self.pose[:3, :3].T) # revert to local frame
            # qdotdot = np.dot(qdotdot, action) 

            if np.linalg.norm(qdotdot) < 0.01 * self.force_mag:
                if return_path:
                    return False, path
                else:
                    return False

            step += 1

        if return_path:
            return True, path
        else:
            return True

    def plan_path(self, action, rotation=False):
        success, path = self.check_success(action, return_path=True)
        # if success:
        #     connect_path = self.connect_path_planner.plan(self.part_move, self.parts_fix, self.parts_removed, 
        #         rotation=rotation, final_state=path[-1])
        #     if connect_path is not None:
        #         path += connect_path
        return success, path

    def render(self, path=None, reverse=False, record_path=None, make_video=False):
        q_his, qdot_his = self.sim.get_q_his(), self.sim.get_qdot_his()
        if path is not None:
            # assume path is global coordinate
            path = [self.sim.get_joint_q_from_qm(f'part{self.part_move}', qm) for qm in path]
        if reverse:
            path = q_his[::-1] if path is None else path[::-1]
            self.sim.set_state_his(path, [np.zeros(6) for _ in range(len(path))])
        else:
            if path is not None:
                self.sim.set_state_his(path, [np.zeros(6) for _ in range(len(path))])
        SimRenderer.replay(self.sim, record=record_path is not None, record_path=record_path, make_video=make_video)
        self.sim.set_state_his(q_his, qdot_his)
        return self.sim.export_replay_matrices()


class StabilityChecker:

    contact_eps = CONTACT_EPS
    pos_far_threshold = POS_FAR_THRESHOLD
    pos_near_threshold = POS_NEAR_THRESHOLD
    near_step = NEAR_STEP

    def __init__(self, allow_gap):
        self.sim = None
        self.parts_move = None
        self.parts_fix = None
        self.G = None
        self.pos_his_map = None
        self.dist_his_map = None
        self.n_step = 0
        if allow_gap:
            self.pos_far_threshold = np.inf

    def get_part_pos(self, part):
        return self.sim.get_joint_qm(f'part{part}')[:3]

    def derive_contact_graph(self):
        G = nx.Graph()
        for i in range(len(self.parts_move)):
            G.add_node(self.parts_move[i])
            for j in range(i + 1, len(self.parts_move)):
                in_contact = self.sim.body_in_contact(f'part{self.parts_move[i]}', f'part{self.parts_move[j]}', self.contact_eps)
                if in_contact:
                    G.add_edge(self.parts_move[i], self.parts_move[j])
            for part_fix in self.parts_fix:
                in_contact = self.sim.body_in_contact(f'part{self.parts_move[i]}', f'part{part_fix}', self.contact_eps)
                if in_contact:
                    G.add_edge(self.parts_move[i], part_fix)
        return G

    def update_sim(self, sim, parts_move, parts_fix):
        self.sim = sim
        self.parts_move = parts_move
        self.parts_fix = parts_fix
        self.G = self.derive_contact_graph()
        if self.pos_his_map is None:
            self.pos_his_map = {part: [self.get_part_pos(part)] for part in self.parts_move}
        if self.dist_his_map is None:
            self.dist_his_map = {part: [0.0] for part in self.parts_move}

    def update_status(self):
        for part_move in self.parts_move:
            pos = self.get_part_pos(part_move)
            self.pos_his_map[part_move].append(pos)
            self.dist_his_map[part_move].append(np.linalg.norm(pos - self.pos_his_map[part_move][0]))
        self.n_step += 1

    def check_disconnected_parts(self):
        parts_disconnected = []
        for part_move in self.parts_move:
            if self.G.degree(part_move) == 0:
                parts_disconnected.append(part_move)
        return parts_disconnected
    
    def check_fallen_parts(self, group=True):
        parts_fallen = []
        for part_move in self.parts_move:
            if self.dist_his_map[part_move][-1] > self.pos_far_threshold: # check distance
                parts_fallen.append(part_move)
                continue
            for part_other in self.G.neighbors(part_move): # check connectivity
                in_contact = self.sim.body_in_contact(f'part{part_other}', f'part{part_move}', self.contact_eps)
                if not in_contact:
                    parts_fallen.append(part_move)
                    break
        if group and len(parts_fallen) > 1:
            parts_fallen_grouped = []
            G = self.derive_contact_graph()
            G = G.subgraph([x for x in parts_fallen])
            for G_sub in nx.connected_components(G):
                G_sub = list(G_sub)
                if len(G_sub) > 1:
                    G_sub_sorted = sorted(G_sub, key=lambda x: self.sim.get_body_mass(f'part{x}'), reverse=True)
                    parts_fallen_grouped.append(G_sub_sorted[0])
                else:
                    parts_fallen_grouped.append(G_sub[0])
            return parts_fallen_grouped
        else:
            return parts_fallen

    def check_stable_parts(self):
        parts_stable = []
        if self.n_step >= self.near_step:
            for part_move in self.parts_move:
                dist_interval = self.dist_his_map[part_move][self.n_step - self.near_step:self.n_step]
                if np.max(dist_interval) - np.min(dist_interval) < self.pos_near_threshold:
                    parts_stable.append(part_move)
        return parts_stable

    def plot_his(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, len(self.parts_move), figsize=(3 * len(self.parts_move), 3 * 3))
        axis_names = ['X', 'Y', 'Z']
        for i, part_id in enumerate(self.parts_move):
            for j in range(3):
                axes[j][i].plot(list(range(len(self.pos_his_map[part_id]))), np.array(self.pos_his_map[part_id])[:, j].round(3))
                if i == 0:
                    axes[j][i].set_ylabel(f'{axis_names[j]}')
                if j == 0:
                    axes[j][i].set_title(f'Part {part_id}')
            axes[3][i].plot(list(range(len(self.dist_his_map[part_id]))), np.array(self.dist_his_map[part_id]).round(3))
            axes[3][i].set_ylabel(f'Dist')
            axes[3][i].set_xlabel('Time step')
        plt.tight_layout()
        plt.show()


class MultiPartStabilityPlanner:

    max_step = MAX_STEP
    col_th = COL_TH_STABLE

    def __init__(self, asset_folder, assembly_dir, parts_fix, parts_move, pose=None, save_sdf=False, allow_gap=False):
        model_string = get_stability_sim_string(assembly_dir, parts_fix, parts_move, pose=pose, save_sdf=save_sdf, col_th=self.col_th)
        self.sim = redmax.Simulation(model_string, asset_folder)
        self.parts_fix = parts_fix.copy()
        self.parts_move = parts_move.copy()
        self.allow_gap = allow_gap

    def check_success(self, max_step=MAX_STEP, timeout=None):

        t_start = time()

        # initialize sim and stability checker
        self.sim.reset()
        checker = StabilityChecker(self.allow_gap)
        checker.update_sim(self.sim, self.parts_move, self.parts_fix)

        # check initial connectivity
        parts_disconnected = checker.check_disconnected_parts()
        if len(parts_disconnected) > 0:
            return False, parts_disconnected

        # iterate until max step
        iterator = tqdm(range(max_step)) if DEBUG_SIM else range(max_step)
        for i in iterator:

            # simulate and update status
            self.sim.forward(1, verbose=DEBUG_SIM)
            checker.update_status()

            if (i + 1) % CHECK_FREQ == 0:
                # check fallen parts
                parts_fall = checker.check_fallen_parts()
                if len(parts_fall) > 0:
                    # checker.plot_his()
                    # self.render()
                    return False, parts_fall

            if timeout is not None and time() - t_start > timeout:
                return False, None

        # checker.plot_his()
        # self.render()

        return True, None

    def render(self, record_path=None, make_video=False):
        SimRenderer.replay(self.sim, record=record_path is not None, record_path=record_path, make_video=make_video)


class MultiPartNoForceStabilityPlanner(MultiPartStabilityPlanner):

    def __init__(self, asset_folder, assembly_dir, parts, save_sdf=False, allow_gap=False):
        model_string = get_stability_sim_string(assembly_dir, [], parts, gravity=False, save_sdf=save_sdf, col_th=self.col_th)
        self.sim = redmax.Simulation(model_string, asset_folder)
        self.parts_fix = []
        self.parts_move = parts.copy()
        self.allow_gap = allow_gap

    def check_success(self, max_step=MAX_STEP, timeout=None):

        t_start = time()

        # initialize sim and stability checker
        self.sim.reset()
        checker = StabilityChecker(self.allow_gap)
        checker.update_sim(self.sim, self.parts_move, self.parts_fix)

        # check initial connectivity
        parts_fall = checker.check_disconnected_parts()

        # iterate until max step
        iterator = tqdm(range(max_step)) if DEBUG_SIM else range(max_step)
        for i in iterator:

            # simulate and update status
            self.sim.forward(1, verbose=DEBUG_SIM)
            checker.update_status()

            if (i + 1) % CHECK_FREQ == 0:
                # check fallen parts
                parts_fall_i = checker.check_fallen_parts()
                for part_fall in parts_fall_i:
                    if part_fall not in parts_fall:
                        parts_fall.append(part_fall)

            if timeout is not None and time() - t_start > timeout:
                return False, nx.Graph()

        # get result graph
        for part_fall in parts_fall:
            self.parts_move.remove(part_fall)
        checker.update_sim(self.sim, self.parts_move, self.parts_fix)

        success = (len(parts_fall) == 0)
        return success, checker.G


class MultiPartAdaptiveStabilityPlanner(MultiPartStabilityPlanner):

    def __init__(self, asset_folder, assembly_dir, parts_fix, parts_move, pose=None, save_sdf=False, allow_gap=False):
        self.asset_folder = asset_folder
        self.assembly_dir = assembly_dir
        self.parts_fix = parts_fix.copy()
        self.parts_move = parts_move.copy()
        self.pose = pose
        self.save_sdf = save_sdf
        self.allow_gap = allow_gap

    def check_success(self, max_step=MAX_STEP, timeout=None):

        t_start = time()

        # initialize sim and stability checker
        model_string = get_stability_sim_string(self.assembly_dir, self.parts_fix, self.parts_move, gravity=True, 
            save_sdf=self.save_sdf, pose=self.pose, col_th=self.col_th)
        self.sim = redmax.Simulation(model_string, self.asset_folder)
        self.sim.reset()
        checker = StabilityChecker(self.allow_gap)
        checker.update_sim(self.sim, self.parts_move, self.parts_fix)

        parts_stable_all = []

        # check initial connectivity
        parts_disconnected = checker.check_disconnected_parts()
        if len(parts_disconnected) > 0:
            return False, parts_disconnected, parts_stable_all

        # iterate until max step
        iterator = tqdm(range(max_step)) if DEBUG_SIM else range(max_step)
        for i in iterator:

            # simulate and update status
            self.sim.forward(1, verbose=DEBUG_SIM)
            checker.update_status()

            if (i + 1) % CHECK_FREQ == 0:
                # check fallen parts
                parts_fall = checker.check_fallen_parts()
                if len(parts_fall) > 0:
                    return False, parts_fall, parts_stable_all

                # check stable parts
                parts_stable = checker.check_stable_parts()
                if len(parts_stable) > 0:
                    parts_stable_all.extend(parts_stable)

                    # fix stable parts
                    self.parts_fix.extend(parts_stable)
                    for part_stable in parts_stable:
                        self.parts_move.remove(part_stable)
                    
                    # re-initialize simulation
                    mat_dict = self.sim.get_body_E0j_map()
                    mat_dict = {key.replace('part', ''): val for key, val in mat_dict.items()}
                    q_map = self.sim.get_q_map()
                    qdot_map = self.sim.get_qdot_map()

                    model_string = get_stability_sim_string(self.assembly_dir, self.parts_fix, self.parts_move, gravity=True, 
                        save_sdf=self.save_sdf, pose=self.pose, mat_dict=mat_dict, col_th=self.col_th)
                    self.sim = redmax.Simulation(model_string, self.asset_folder)
                    self.sim.reset()
                    self.sim.set_q_map(q_map)
                    self.sim.set_qdot_map(qdot_map)

                    # self.render()

                    # update checker with new sim
                    checker.update_sim(self.sim, self.parts_move, self.parts_fix)

            if timeout is not None and time() - t_start > timeout:
                return False, None, None # NOTE

        return True, None, parts_stable_all


def plot_stability_curve(pos_list_map):
    fig, axes = plt.subplots(3, len(pos_list_map), figsize=(3 * len(pos_list_map), 2 * 3))
    axis_names = ['X', 'Y', 'Z']
    for i, part_id in enumerate(pos_list_map.keys()):
        for j in range(3):
            axes[j][i].plot(list(range(len(pos_list_map[part_id]))), np.array(pos_list_map[part_id])[:, j].round(3))
            if i == 0:
                axes[j][i].set_ylabel(f'{axis_names[j]}')
            if j == 0:
                axes[j][i].set_title(f'Part {part_id}')
            if j == 2:
                axes[j][i].set_xlabel('Time step')
    plt.tight_layout()
    plt.show()


def get_contact_graph(asset_folder, assembly_dir, parts=None, contact_eps=CONTACT_EPS, save_sdf=False):
    '''
    Get contact graph for assembly
    '''
    if parts is None: parts = load_part_ids(assembly_dir)

    model_string = get_contact_sim_string(assembly_dir, parts, save_sdf=save_sdf)
    sim = redmax.Simulation(model_string, asset_folder)

    G = nx.Graph()
    for i in range(len(parts)):
        G.add_node(parts[i])
        for j in range(i + 1, len(parts)):
            in_contact = sim.body_in_contact(f'part{parts[i]}', f'part{parts[j]}', contact_eps)
            if in_contact:
                G.add_edge(parts[i], parts[j])
    return G


def get_distance_all_bodies(asset_folder, assembly_dir, parts=None, save_sdf=False):
    if parts is None: parts = load_part_ids(assembly_dir)

    model_string = get_contact_sim_string(assembly_dir, parts, save_sdf=save_sdf)
    sim = redmax.Simulation(model_string, asset_folder)

    distance = {}
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            dist_ij = sim.get_body_distance(f'part{parts[i]}', f'part{parts[j]}')
            dist_ji = sim.get_body_distance(f'part{parts[j]}', f'part{parts[i]}')
            distance[(parts[i], parts[j])] = min(dist_ij, dist_ji)

    return distance


def get_body_mass(asset_folder, assembly_dir, parts=None, save_sdf=False):
    if parts is None: parts = load_part_ids(assembly_dir)

    model_string = get_contact_sim_string(assembly_dir, parts, save_sdf=save_sdf)
    sim = redmax.Simulation(model_string, asset_folder)

    mass_dict = {}
    for part in parts:
        mass_dict[part] = sim.get_body_mass(f'part{part}')

    return mass_dict
