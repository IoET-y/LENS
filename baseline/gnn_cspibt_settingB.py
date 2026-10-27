# ====================================================================
# File: gnn_cspibt_pogema_benchmark_setting_b.py
# Desc: GNN-Guided CS-PIBT Adapted for Setting B (Unknown Map / Joint Exploration)
#       - Map is initialized as empty (optimistic).
#       - Map is discovered online via Agent FOV (Strict Alignment with LPSS).
#       - Heuristics (BD) & GNN Inference are re-run every step on the evolving map.
# ====================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time
import argparse
import yaml
import json
import pandas as pd
from collections import defaultdict, deque
from tqdm import tqdm
import logging
import heapq
from typing import Dict, List, Tuple, Optional, Set
import queue  # For BFS

# --- PyTorch Geometric ---
try:
    import torch_geometric.nn as pyg_nn
    import torch_geometric.utils as pyg_utils
    from torch_geometric.data import Data
except ImportError:
    logging.error("torch_geometric not found. Please install it.")
    exit(1)

# --- Pogema Imports ---
try:
    from pogema import GridConfig
    from pogema_toolbox.registry import ToolboxRegistry
    from pogema_toolbox.create_env import Environment
    from create_env import create_eval_env
except ImportError as e:
    logging.error(f"Error: Pogema or pogema_toolbox import failed: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================================================================
# 1. Constants and Mappings
# ====================================================================
POGEMA_ACTIONS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
POGEMA_ACTION_NAMES = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
POGEMA_STAY = 0
CS_PIBT_DELTAS = np.array([[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]], dtype=int)
NUM_CS_PIBT_ACTIONS = 5
delta_to_pogema_action_idx = {v: k for k, v in POGEMA_ACTIONS.items()}
FREE_CELL = 0
OBSTACLE_CELL = 1
INVALID_BD_VALUE = 999999

# ====================================================================
# 2. Model Definition (Preserved from Original)
# ====================================================================

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, linear_dim, in_channels, out_channels, relu_type):
        super(CustomConv, self).__init__(aggr='add')
        self.lin = nn.Linear(linear_dim, out_channels)
        self.lin_self = nn.Linear(linear_dim, out_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.conv_self = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.relu_type = relu_type

    def forward(self, x, bd_pred, edge_index):
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        conv_out = self.conv_self(x)
        flattened_conv = torch.flatten(conv_out, start_dim=1)
        bd_pred_device = bd_pred.to(flattened_conv.device, dtype=flattened_conv.dtype)
        
        if bd_pred.ndim > 1 and bd_pred.shape[0] > 0 and bd_pred.shape[1] > 0:
            combined_features = torch.hstack([flattened_conv, bd_pred_device])
        else:
            combined_features = flattened_conv

        if self.relu_type != "relu":
            activated_features = F.leaky_relu(combined_features)
        else:
            activated_features = F.relu(combined_features)

        self_x = self.lin_self(activated_features)
        
        if self.relu_type != "relu":
             x_neighbors_pre_lin = F.leaky_relu(combined_features)
        else:
             x_neighbors_pre_lin = F.relu(combined_features)

        x_neighbors = self.lin(x_neighbors_pre_lin)
        propagated_neighbors = self.propagate(edge_index, x=x_neighbors)
        self_and_propogated = self_x + propagated_neighbors
        return self_and_propogated

    def message(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out

class GNNStack(nn.Module):
    def __init__(self, linear_dim, in_channels, hidden_dim, output_dim, relu_type, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.relu_type = relu_type
        self.convs = nn.ModuleList([self.build_conv_model(linear_dim, in_channels, hidden_dim, True)])
        for _ in range(3):
            self.convs.append(self.build_conv_model(linear_dim, hidden_dim, hidden_dim, False))
        self.lns = nn.ModuleList()
        for _ in range(5):
             self.lns.append(nn.LayerNorm(hidden_dim))
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim)
        )
        if task not in ['node', 'graph']: raise RuntimeError('Unknown task.')
        self.dropout = 0.25
        self.num_layers = 4
        if self.relu_type == "leaky_relu": self.activation = F.leaky_relu
        else: self.activation = F.relu

    def build_conv_model(self, linear_dim, in_channels, hidden_dim, is_first_layer):
        if is_first_layer: return CustomConv(linear_dim, in_channels, hidden_dim, self.relu_type)
        else: return pyg_nn.SAGEConv(in_channels, hidden_dim)

    def forward(self, data):
        x, edge_index, batch, bd_pred = data.x, data.edge_index, data.batch, data.bd_pred
        if x.shape[0] == 0:
             emb = torch.empty((0, args.gnn_hidden_dim), device=x.device)
             output = torch.empty((0, NUM_CS_PIBT_ACTIONS), device=x.device)
             return emb, output

        x = self.convs[0](x, bd_pred, edge_index)
        emb = x
        for i in range(1, self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i < self.num_layers:
                 if i < len(self.lns): x = self.lns[i](x)
        x = self.post_mp(x)
        return emb, F.log_softmax(x, dim=1)

# ====================================================================
# 3. Input Feature Creation & Helpers
# ====================================================================

def normalize_graph_data(data, k, edge_normalize="k", bd_normalize="center"):
    """Modifies data in place"""
    if edge_normalize == "k": data.edge_attr /= k
    else: raise KeyError("Invalid edge normalization method: {}".format(edge_normalize))
    bd_grid = data.x
    center = bd_grid[:, 1, k, k].unsqueeze(1).unsqueeze(2)
    bd_grid[:, 1, :, :] -= center
    bd_grid[:, 1, :, :] *= (1 - bd_grid[:, 0, :, :])
    bd_grid[:, 1, :, :] /= (2*k)
    bd_grid[:, 1, :, :] = torch.clamp(bd_grid[:, 1, :, :], min=-1.0, max=1.0)
    data.x = bd_grid
    return data

def compute_bd_for_pogema(grid_map_padded: np.ndarray, goals_padded: np.ndarray) -> np.ndarray:
    """ 
    Computes BFS distance on the CURRENTLY KNOWN map. 
    NOTE: In Setting B, 'grid_map_padded' is the persistent_known_map.
    0 = Free/Unknown (Optimistic), 1 = Obstacle.
    """
    num_agents = goals_padded.shape[0]
    h_pad, w_pad = grid_map_padded.shape
    bd = np.full((num_agents, h_pad, w_pad), INVALID_BD_VALUE, dtype=np.int32)
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for i in range(num_agents):
        goal_r, goal_c = goals_padded[i]
        # Even if goal is in unknown area, we attempt to plan to it.
        # Check bounds only.
        if not (0 <= goal_r < h_pad and 0 <= goal_c < w_pad):
            continue 
        
        # If goal is on a KNOWN obstacle, we can't reach it, but BFS handles it naturally.
        if grid_map_padded[goal_r, goal_c]:
            continue

        q = queue.Queue()
        q.put((goal_r, goal_c))
        bd[i, goal_r, goal_c] = 0
        visited = np.zeros_like(grid_map_padded, dtype=bool)
        visited[goal_r, goal_c] = True

        while not q.empty():
            r, c = q.get()
            current_dist = bd[i, r, c]
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h_pad and 0 <= nc < w_pad and \
                   not grid_map_padded[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    bd[i, nr, nc] = current_dist + 1
                    q.put((nr, nc))
    return bd

def create_data_object(cur_locs_padded, bd, grid_map_padded, k, m, goals_padded, labels=np.array([])):
    pos_list = cur_locs_padded; bd_list = bd; grid = grid_map_padded; num_agents = len(pos_list)
    num_layers = 3
    if num_agents == 0:
        D = 2 * k + 1
        return Data(x=torch.empty((0, num_layers, D, D)), edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 2)), bd_pred=torch.empty((0, 5)), batch=torch.empty(0, dtype=torch.long))

    range_num_agents = np.arange(num_agents)
    rowLocs = pos_list[:,0][:, None]; colLocs = pos_list[:,1][:, None]
    D = 2 * k + 1
    
    # Clip indices to stay within map bounds (since it's padded map, this should be safe)
    # But in Setting B, we pass the persistent map directly (no extra padding logic in main loop)
    # So we must clip.
    x_mesh, y_mesh = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1), indexing='ij')
    x_mesh_abs = np.clip(x_mesh[None, :, :] + rowLocs[:, None, None], 0, grid.shape[0] - 1)
    y_mesh_abs = np.clip(y_mesh[None, :, :] + colLocs[:, None, None], 0, grid.shape[1] - 1)

    if x_mesh_abs.ndim == 4: x_mesh_abs = np.squeeze(x_mesh_abs, axis=1)
    if y_mesh_abs.ndim == 4: y_mesh_abs = np.squeeze(y_mesh_abs, axis=1)

    # Extract slices
    grid_slices = grid[x_mesh_abs, y_mesh_abs]
    if grid_slices.ndim == 4: grid_slices = np.squeeze(grid_slices, axis=1)
    
    bd_slices = bd_list[range_num_agents[:,None,None], x_mesh_abs, y_mesh_abs]

    # Agent Position Slice
    temp_agent_map = np.zeros_like(grid, dtype=bool)
    valid_mask = (pos_list[:, 0] >= 0) & (pos_list[:, 0] < grid.shape[0]) & \
                 (pos_list[:, 1] >= 0) & (pos_list[:, 1] < grid.shape[1])
    if np.any(valid_mask):
        temp_agent_map[pos_list[valid_mask, 0], pos_list[valid_mask, 1]] = True
    
    agent_pos_slices = temp_agent_map[x_mesh_abs, y_mesh_abs].astype(np.float32)
    if agent_pos_slices.ndim == 4: agent_pos_slices = np.squeeze(agent_pos_slices, axis=1)

    # Stack Features
    node_features = np.stack([grid_slices.astype(np.float32), bd_slices.astype(np.float32), agent_pos_slices], axis=1)

    # Edges
    deltas = pos_list[:, None, :] - pos_list[None, :, :]
    dists_sq = np.sum(deltas**2, axis=2).astype(np.float32)
    fov_mask = np.any(np.abs(deltas) > k, axis=2)
    dists_sq[fov_mask] = np.inf
    np.fill_diagonal(dists_sq, np.inf)
    
    closest_indices = np.argsort(dists_sq, axis=1)[:, :m]
    source = np.repeat(np.arange(num_agents), m)
    target = closest_indices.flatten()
    valid_edge = dists_sq[source, target] != np.inf
    
    edge_index = np.stack([source[valid_edge], target[valid_edge]])
    edge_attr = deltas[edge_index[0], edge_index[1]].astype(np.float32)

    # BD Pred (Local Gradient)
    bd_pred_arr = np.zeros((num_agents, 5), dtype=np.float32)
    if num_agents > 0:
        r, c = pos_list[:, 0], pos_list[:, 1]
        for i in range(num_agents):
            if 0 < r[i] < grid.shape[0]-1 and 0 < c[i] < grid.shape[1]-1:
                # Extract 3x3 around agent
                slice_3x3 = bd_list[i, r[i]-1:r[i]+2, c[i]-1:c[i]+2]
                # Mapping: Center, Right, Down, Up, Left
                # indices: (1,1), (1,2), (2,1), (0,1), (1,0)
                vals = np.array([slice_3x3[1,1], slice_3x3[1,2], slice_3x3[2,1], slice_3x3[0,1], slice_3x3[1,0]])
                
                # Softmax-like one-hot for min direction
                min_val = np.min(vals)
                if min_val == INVALID_BD_VALUE:
                    bd_pred_arr[i] = 0 # No valid gradient
                else:
                    bd_pred_arr[i] = (vals == min_val).astype(np.float32)

    data = Data(x=torch.from_numpy(node_features), edge_index=torch.from_numpy(edge_index).long(),
                edge_attr=torch.from_numpy(edge_attr), bd_pred=torch.from_numpy(bd_pred_arr),
                y=torch.from_numpy(labels).long(), batch=torch.zeros(num_agents, dtype=torch.long))
    data.num_nodes = num_agents
    return data

def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def convertProbsToPreferences(probs, conversion_type="sampled"):
    if isinstance(probs, torch.Tensor): probs = probs.cpu().numpy()
    num_agents, num_actions = probs.shape
    if num_agents == 0: return np.empty((0, num_actions), dtype=int)
    probs_tensor = torch.tensor(probs, dtype=torch.float32)
    # Simple sort for preferences: Higher prob -> Lower index (earlier in preference)
    preferences = torch.argsort(-probs_tensor, dim=1) 
    return preferences.numpy()

# ====================================================================
# 4. Map & Stat Helpers
# ====================================================================
def get_clustered_positions(map_data, num_agents, seed=42):
    """
    Clusters agent start positions in a randomly chosen corner of the map
    and target positions in the diagonally opposite corner.
    """
    rng = np.random.default_rng(seed)
    free_cells = []

    def is_free(char):
        return char in {'.', '0', '@', '$', '!'} or ('a' <= char <= 'z') or ('A' <= char <= 'Z')

    if isinstance(map_data, str):
        lines = map_data.split()
        map_height = len(lines)
        map_width = max(len(line) for line in lines) if lines else 0
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if is_free(char):
                    free_cells.append([r, c])
    elif isinstance(map_data, list):
        map_height = len(map_data)
        map_width = max(len(row) for row in map_data) if map_data else 0
        for r, row in enumerate(map_data):
            for c, val in enumerate(row):
                if val == 0 or is_free(str(val)):
                    free_cells.append([r, c])

    map_size = max(map_height, map_width)

    if len(free_cells) < num_agents * 2:
        raise ValueError(f"Not enough free cells for {num_agents} agents (found: {len(free_cells)}).")

    corners = [
        lambda p: p[0] + p[1],   # Top-Left
        lambda p: p[0] - p[1],   # Top-Right
        lambda p: -p[0] + p[1],  # Bottom-Left
        lambda p: -p[0] - p[1],  # Bottom-Right
    ]

    start_corner_idx = rng.integers(0, 4)
    goal_corner_idx = 3 - start_corner_idx

    free_cells.sort(key=corners[start_corner_idx])
    agents_xy = free_cells[:num_agents]

    remaining_cells = [p for p in free_cells if p not in agents_xy]
    remaining_cells.sort(key=corners[goal_corner_idx])
    targets_xy = remaining_cells[:num_agents]

    rng.shuffle(targets_xy)

    return agents_xy, targets_xy, map_size, map_width, map_height

def load_and_register_maps(maps_dir):
    """ Loads and registers map configurations. """
    maps_path = Path(maps_dir); map_count = 0; registered_maps = set()
    if not maps_path.is_dir(): logging.warning(f"Maps directory not found: {maps_dir}"); return False, []
    all_map_names = []
    for maps_file in maps_path.rglob('maps.yaml'): 
        try:
            with open(maps_file, 'r') as f: maps_data = yaml.safe_load(f)
            if maps_data:
                 maps_data = {str(k): v for k, v in maps_data.items()}
                 new_maps = {k: v for k, v in maps_data.items() if k not in registered_maps}
                 if new_maps:
                     ToolboxRegistry.register_maps(new_maps)
                     logging.info(f"Registered {len(new_maps)} new maps from: {maps_file.name}")
                     registered_maps.update(new_maps.keys())
                     all_map_names.extend(list(new_maps.keys()))
                     map_count += len(new_maps)
        except Exception as e: logging.error(f"Failed to load/register {maps_file}: {e}")
    if map_count == 0: logging.warning(f"No new maps registered from {maps_dir}."); return False, []
    return True, sorted(list(registered_maps))

def filter_maps_by_pattern(all_map_names, pattern):
    if pattern == "*" or not pattern: return all_map_names
    if pattern.startswith("*") and pattern.endswith("*"):
        substring = pattern[1:-1]; return [name for name in all_map_names if substring in name]
    elif pattern.startswith("*"):
        suffix = pattern[1:]; return [name for name in all_map_names if name.endswith(suffix)]
    elif pattern.endswith("*"):
        prefix = pattern[:-1]; return [name for name in all_map_names if name.startswith(prefix)]
    else: return [name for name in all_map_names if name == pattern]

def calculate_overall_stats(results, args):
    """Simple aggregation for summary."""
    if not results: return {}
    success_count = sum(1 for r in results if r['success'])
    total = len(results)
    
    valid_soc = [r['sum_of_costs'] for r in results if r['success']]
    valid_mk = [r['makespan'] for r in results if r['success']]
    times = [r['computation_time_sec'] for r in results]
    
    return {
        "overall_success_rate": success_count / total if total else 0,
        "avg_soc": np.mean(valid_soc) if valid_soc else 0,
        "avg_makespan": np.mean(valid_mk) if valid_mk else 0,
        "avg_runtime": np.mean(times) if times else 0
    }

# ====================================================================
# 5. CS-PIBT Logic (Core Logic Preserved)
# ====================================================================
def updatePriorities(prev, at_goal):
    p = prev.copy(); p[(prev<=0)&at_goal] -= 1; p[(prev>0)&at_goal]=0
    p[~at_goal] = np.maximum(p[~at_goal], 0) + 1; return p

def pibtRecursive(grid, aid, prefs, planned, move_d, occ_n, occ_e, locs, loc_to_ag, start, limit):
    if limit > 0 and (time.time()-start) > limit: return False
    curr = tuple(locs[aid]); h, w = grid.shape
    for aidx in prefs[aid]:
        d = CS_PIBT_DELTAS[aidx]; next_l = (curr[0]+d[0], curr[1]+d[1])
        if not (0<=next_l[0]<h and 0<=next_l[1]<w) or grid[next_l]: continue
        if occ_n[next_l] or occ_e.get((next_l[0], next_l[1], curr[0], curr[1])): continue
        move_d[aid] = d; planned[aid] = True; occ_n[next_l] = True; occ_e[(curr[0], curr[1], next_l[0], next_l[1])] = True
        conf_id = loc_to_ag[next_l]; resolved = True
        if conf_id != -1 and conf_id != aid and not planned[conf_id]:
            resolved = pibtRecursive(grid, conf_id, prefs, planned, move_d, occ_n, occ_e, locs, loc_to_ag, start, limit)
        if resolved: return True
        planned[aid] = False; occ_n[next_l] = False; occ_e[(curr[0], curr[1], next_l[0], next_l[1])] = False
        move_d[aid] = [0,0]
        if limit > 0 and (time.time()-start) > limit: return False
    return False

def pibt(grid, prefs, locs, prios, limit=-1.0):
    start = time.time(); num = len(prios); order = np.argsort(-prios)
    deltas = np.zeros((num, 2), dtype=int); planned = np.zeros(num, dtype=bool)
    occ_n = np.zeros(grid.shape, dtype=bool); occ_e = defaultdict(bool)
    loc_to_ag = np.full(grid.shape, -1, dtype=int)
    
    valid = (locs[:,0]>=0) & (locs[:,0]<grid.shape[0]) & (locs[:,1]>=0) & (locs[:,1]<grid.shape[1])
    if np.any(valid):
        loc_to_ag[locs[valid,0], locs[valid,1]] = np.where(valid)[0]
    
    success = True
    for i in order:
        if not planned[i]:
            if not pibtRecursive(grid, i, prefs, planned, deltas, occ_n, occ_e, locs, loc_to_ag, start, limit):
                success = False; deltas[i] = [0,0]; planned[i] = True
    return deltas, success

# ====================================================================
# 6. Setting B: Map Update Logic (ALIGNED WITH LPSS_Direct.py)
# ====================================================================

def update_persistent_map(current_obs_list, current_agent_positions, known_map, obs_rad):
    """
    Updates the global `known_map` based on current agent observations.
    Aligned with LPSS_Direct.py logic.
    known_map: 0 = Free/Unknown (Optimistic), 1 = Obstacle.
    """
    map_h, map_w = known_map.shape
    window_size = obs_rad * 2 + 1
    
    for i, (r, c) in enumerate(current_agent_positions):
        # Safety checks for valid observation
        if i >= len(current_obs_list) or current_obs_list[i] is None: continue
        
        # In Pogema, obs is typically a list/dict. We look for 'obstacles' grid.
        fov = current_obs_list[i].get("obstacles") 
        if fov is None: continue
        
        # Calculate top-left of FOV in global coords
        tl_r, tl_c = r - obs_rad, c - obs_rad
        
        # Iterate through the FOV window
        for fr in range(window_size):
            for fc in range(window_size):
                gr, gc = tl_r + fr, tl_c + fc
                
                # Check boundaries
                if 0 <= gr < map_h and 0 <= gc < map_w:
                    # Logic: 
                    # If we see an obstacle (fov=1), mark it as 1.
                    # If we see free space (fov=0), mark it as 0.
                    # This overwrites any previous state (including 'unknown' which we treat as 0).
                    # This handles dynamic discovery.
                    
                    if fov[fr, fc] == 1:
                        known_map[gr, gc] = 1 # Obstacle
                    else:
                        known_map[gr, gc] = 0 # Free

# ====================================================================
# 7. Setting B: Simulation Loop (MODIFIED)
# ====================================================================

def run_gnn_cspibt_simulation_setting_b(
    env, model, device, k_padding, m_neighbors,
    max_episode_steps=256, verbose=False, time_limit_pibt_step=-1.0,
):
    sim_start_time = time.time()
    model.eval()

    num_agents = env.grid_config.num_agents
    # Ground truth map (ONLY used for initializing array size and checking bounds)
    gt_map = env.unwrapped.grid.get_obstacles().astype(bool)
    map_h, map_w = gt_map.shape
    
    # 记录初始目标和位置，用于统计验证
    agents_global_goals = np.array(env.get_targets_xy(), dtype=int)
    agents_start_pos = np.array(env.get_agents_xy(), dtype=int)

    # --- SETTING B INITIALIZATION ---
    # 0 = Free, 1 = Obstacle. Initialize as empty (Optimistic).
    persistent_known_map = np.zeros((map_h, map_w), dtype=bool) 

    agents_active = np.array([True] * num_agents)
    # Track paths for SoC calculation and visualization
    current_pos = agents_start_pos.copy()
    executed_paths = {i: [tuple(current_pos[i])] for i in range(num_agents)}
    
    total_steps = 0
    success = True
    error_messages = []
    truncated = {i: False for i in range(num_agents)}
    
    # Init Priorities based on Euclidean
    initial_distances = np.array([heuristic(tuple(pos), tuple(goal)) for pos, goal in zip(current_pos, agents_global_goals)], dtype=float)
    agent_priorities = initial_distances / (np.max(initial_distances) + 1e-6)
    prev_map_hash = None
    bd = compute_bd_for_pogema(persistent_known_map, agents_global_goals) # 初始计算
    
    while np.any(agents_active) and total_steps < max_episode_steps:
        # Get Current Positions
        current_pos = np.array(env.get_agents_xy(), dtype=int)
        active_idx = np.where(agents_active)[0]
        
        # --- 1. Re-compute Heuristics (BD) based on PERSISTENT MAP ---
        #bd = compute_bd_for_pogema(persistent_known_map, agents_global_goals)
        # --- 1. Re-compute Heuristics (BD) ---
        # 优化：检查地图是否发生变化
        current_map_hash = hash(persistent_known_map.tobytes())
        
        if current_map_hash != prev_map_hash:
            # 只有地图变了才重算，这符合逻辑，且稍微快一点
            # 但在探索初期，地图几乎每步都变，所以提升有限
            bd = compute_bd_for_pogema(persistent_known_map, agents_global_goals)
            prev_map_hash = current_map_hash
        else:
            # 地图没变，直接复用上一步的 bd
            pass

        # --- 2. GNN Inference ---
        try:
            with torch.no_grad():
                data = create_data_object(
                    current_pos, bd, persistent_known_map, 
                    k_padding, m_neighbors, agents_global_goals
                )
                data = normalize_graph_data(data, k_padding)
                data = data.to(device)
                
                _, log_probs = model(data)
                probs = torch.exp(log_probs).cpu().numpy()
                
            # --- 3. Mask Invalid Actions (Against KNOWN map) ---
            mask = np.zeros_like(probs, dtype=bool)
            for i in active_idx:
                r, c = current_pos[i]
                for aidx in range(5):
                    dr, dc = CS_PIBT_DELTAS[aidx]
                    nr, nc = r+dr, c+dc
                    # Check against KNOWN map bounds and obstacles
                    if not (0<=nr<map_h and 0<=nc<map_w and not persistent_known_map[nr, nc]):
                        mask[i, aidx] = True
            
            probs[mask] = 1e-9
            row_sums = probs.sum(axis=1, keepdims=True)
            probs = probs / np.where(row_sums < 1e-9, 1.0, row_sums)
            
            prefs = convertProbsToPreferences(probs, "sampled")
            
        except Exception as e:
            logging.error(f"GNN Error at step {total_steps}: {e}")
            error_messages.append(f"GNN_ERR_S{total_steps}")
            success = False; break

        # Update Priorities
        at_goal = np.array([tuple(pos) == tuple(goal) for pos, goal in zip(current_pos, agents_global_goals)], dtype=bool)
        agent_priorities = updatePriorities(agent_priorities, at_goal)

        # --- 4. PIBT ---
        deltas, p_succ = pibt(persistent_known_map, prefs, current_pos, agent_priorities, time_limit_pibt_step)
        
        # --- 5. Prepare Actions ---
        actions_for_env = []
        for i in range(num_agents):
            if agents_active[i]:
                act = delta_to_pogema_action_idx.get(tuple(deltas[i]), 0)
                actions_for_env.append(act)
            else: 
                actions_for_env.append(0)

        # --- 6. Environment Step ---
        obs_list, _, term, trunc_dict, _ = env.step(actions_for_env)
        total_steps += 1
        new_pos = np.array(env.get_agents_xy(), dtype=int)

        # --- 7. UPDATE GLOBAL MAP (Mapping Phase) ---
        update_persistent_map(obs_list, new_pos, persistent_known_map, env.grid_config.obs_radius)

        # --- 8. State Update & Path Tracking ---
        for i in range(num_agents):
            if agents_active[i]:
                executed_paths[i].append(tuple(new_pos[i]))
                if term[i]: 
                    agents_active[i] = False
                elif trunc_dict[i]: 
                    agents_active[i] = False
                    truncated[i] = True
                    success = False # Partial failure
                    error_messages.append(f"A{i}_Truncated")

        if verbose: logging.debug(f"Step {total_steps}: Active {len(np.where(agents_active)[0])}")

    # --- Stats Calculation ---
    sim_duration = time.time() - sim_start_time
    num_reached = 0
    all_finished = True
    valid_costs = []
    
    for i in range(num_agents):
        final_p = executed_paths[i][-1]
        goal_p = tuple(agents_global_goals[i])
        # Success criteria: At goal, inactive, and not truncated
        if final_p == goal_p and not agents_active[i] and not truncated[i]:
            num_reached += 1
            valid_costs.append(len(executed_paths[i]) - 1)
        else:
            all_finished = False

    # Return dictionary compatible with benchmark.py stats function
    return {
        "success": success and all_finished,
        "makespan": total_steps,
        "sum_of_costs": sum(valid_costs) if valid_costs else 0,
        "individual_costs": {i: len(p)-1 for i, p in executed_paths.items()},
        "executed_paths_global": executed_paths, # Needed for detailed debugging if necessary
        "num_agents_reached_target": num_reached,
        "num_agents_at_start": num_agents,
        "computation_time_sec": sim_duration,
        "error_summary": "; ".join(error_messages) if error_messages else "No Errors"
    }



# ====================================================================
# 8. Main Entry Point
# ====================================================================
def calculate_overall_stats(detailed_results_list, args_config):
    """ 计算所有测试的全局统计数据 """
    if not detailed_results_list:
        return {"error": "No detailed results to calculate overall stats."}

    stats = {}
    num_agents_set = set()
    all_trial_isrs = [] # Individual Success Rates
    successful_overall_runs_count = 0
    successful_socs = []
    successful_makespans = []
    all_durations = []
    maps_actually_tested_set = set()

    for r_dict in detailed_results_list:
        num_agents_set.add(r_dict.get('num_agents_at_start', r_dict.get('num_agents', 0)))
        maps_actually_tested_set.add(r_dict.get('map_name', 'unknown'))

        num_at_start = r_dict.get('num_agents_at_start', 0)
        num_reached = r_dict.get('num_agents_reached_target', 0)
        
        # 计算个体成功率 (ISR)
        isr_trial = (num_reached / num_at_start) if num_at_start > 0 else (1.0 if r_dict.get('success', False) else 0.0)
        all_trial_isrs.append(isr_trial)

        if r_dict.get('success', False):
            successful_overall_runs_count += 1
            if 'sum_of_costs' in r_dict: successful_socs.append(r_dict['sum_of_costs'])
            if 'makespan' in r_dict: successful_makespans.append(r_dict['makespan'])

        if 'computation_time_sec' in r_dict: all_durations.append(r_dict['computation_time_sec'])

    total_runs_attempted = len(detailed_results_list)

    stats['num_agents_configurations_tested'] = sorted(list(n for n in num_agents_set if n > 0))
    stats['avg_isr'] = np.mean(all_trial_isrs) if all_trial_isrs else 0.0
    stats['overall_success_rate'] = (successful_overall_runs_count / total_runs_attempted) if total_runs_attempted > 0 else 0.0
    stats['avg_soc_on_overall_success'] = np.mean(successful_socs) if successful_socs else 0.0
    stats['avg_makespan_on_overall_success'] = np.mean(successful_makespans) if successful_makespans else 0.0
    stats['avg_duration_s_per_run'] = np.mean(all_durations) if all_durations else 0.0
    stats['maps_tested_count'] = len(maps_actually_tested_set)
    stats['runs_attempted_total'] = total_runs_attempted

    return stats
# ====================================================================
# 8. Main Benchmark Execution Logic (Restored Statistics & Animation)
# ====================================================================

def run_benchmark_main(args):
    # Setup Directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    animations_dir = output_dir / "animations"
    animations_dir.mkdir(exist_ok=True)
    animations_fail_dir = output_dir / "animations_fail"
    animations_fail_dir.mkdir(exist_ok=True)

    # Setup Device
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")

    # Load Maps
    maps_registered_ok, all_available_map_names = load_and_register_maps(args.map_config_dir)
    if not maps_registered_ok and not all_available_map_names: exit(1)
    maps_to_run_on = filter_maps_by_pattern(all_available_map_names, args.map_pattern)
    
    if args.map_limit:
        maps_to_run_on = maps_to_run_on[:args.map_limit]

    logging.info(f"Setting B (Unknown Map) Benchmark.")
    logging.info(f"Maps: {len(maps_to_run_on)}, Agents: {args.agent_counts}")

    # Load Model
    try:
        model_path = Path(args.model_path)
        logging.info(f"Loading model from {model_path}...")
        loaded_object = torch.load(model_path, map_location=device)
        state_dict = loaded_object.get('model_state_dict', loaded_object) if isinstance(loaded_object, dict) else loaded_object.state_dict()
        
        # Instantiate Model (Assuming standard parameters or extracted from args)
        D = 2 * args.k + 1
        linear_dim = (D - 2)**2 * 3 + 5 
        model = GNNStack(linear_dim, 3, args.gnn_hidden_dim, NUM_CS_PIBT_ACTIONS, args.gnn_relu_type)
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
    except Exception as e:
        logging.error(f"Model Load Failed: {e}")
        exit(1)

    # --- Run Loop ---
    all_run_results_detailed = []
    summary_results_per_scenario = []

    for map_name in tqdm(maps_to_run_on, desc="Maps"):
        for num_agents in args.agent_counts:
            tqdm.write(f"\n--- Map: {map_name}, Agents: {num_agents} ---")
            
            scenario_results = []
            scenario_succeeded_once = False

            for trial in range(args.num_trials):
                if args.stop_scenario_on_first_success and scenario_succeeded_once:
                    continue

                seed = args.seed + trial
                try:
                    if args.clustered_positions:
                        map_data = ToolboxRegistry.get_maps()[map_name]
                        agents_xy, targets_xy, map_size, map_width, map_height = get_clustered_positions(map_data, num_agents, seed=seed)
                        env = create_eval_env(
                            Environment(
                                map_name=map_name, num_agents=num_agents, obs_radius=args.obs_radius,
                                observation_type="POMAPF", on_target="finish", collision_system="soft",
                                max_episode_steps=args.max_steps, seed=seed, with_animation=True,
                                agents_xy=agents_xy, targets_xy=targets_xy,
                                size=map_size, width=map_width, height=map_height
                            )
                        )
                    else:
                        env = create_eval_env(
                            Environment(
                                map_name=map_name, num_agents=num_agents,
                                obs_radius=args.obs_radius, max_episode_steps=args.max_steps,
                                seed=seed, observation_type="POMAPF",
                                collision_system="soft", with_animation=True
                            )
                        )
                except Exception as e:
                    logging.error(f"Env create failed for {map_name}: {e}")
                    continue

                # Run Simulation (Setting B)
                try:
                    res = run_gnn_cspibt_simulation_setting_b(
                        env, model, device, args.k, args.m, 
                        max_episode_steps=args.max_steps, 
                        verbose=args.verbose_simulation,
                        time_limit_pibt_step=args.pibt_time_limit
                    )
                    
                    # Add Metadata
                    res.update({'map_name': map_name, 'num_agents': num_agents, 'trial': trial, 'seed': seed})
                    
                    # --- SAVE ANIMATION ---
                    is_success = res['success']
                    anim_subdir = animations_dir if is_success else animations_fail_dir
                    sanitized_name = map_name.replace("/", "_")
                    anim_path = anim_subdir / f"{sanitized_name}_A{num_agents}_T{trial}_{'OK' if is_success else 'FAIL'}.svg"
                    try:
                        env.save_animation(str(anim_path))
                        # tqdm.write(f"Animation saved: {anim_path.name}")
                    except Exception as anim_e:
                        logging.warning(f"Animation save failed: {anim_e}")

                    # Store results
                    scenario_results.append(res)
                    all_run_results_detailed.append(res)
                    
                    if is_success: scenario_succeeded_once = True
                    
                    tqdm.write(f"  Trial {trial}: Success={res['success']}, Steps={res['makespan']}, SoC={res['sum_of_costs']}, Time={res['computation_time_sec']:.2f}s")

                except Exception as e:
                    logging.error(f"Simulation failed: {e}", exc_info=True)
                finally:
                    env.close()

            # Scenario Summary
            if scenario_results:
                successes = [r for r in scenario_results if r['success']]
                sr = len(successes) / len(scenario_results)
                avg_soc = np.mean([r['sum_of_costs'] for r in successes]) if successes else 0
                avg_makespan = np.mean([r['makespan'] for r in successes]) if successes else 0
                
                summary_results_per_scenario.append({
                    'map': map_name, 'agents': num_agents, 
                    'success_rate': sr, 'avg_soc': avg_soc, 'avg_makespan': avg_makespan,
                    'trials': len(scenario_results)
                })

    # --- Save Final Results ---
    overall_stats = calculate_overall_stats(all_run_results_detailed, args)
    
    logging.info("\n--- Overall Setting B Benchmark Statistics ---")
    if "error" in overall_stats:
        logging.error(overall_stats["error"])
    else:
        print(f"  Overall System Success Rate  : {overall_stats.get('overall_success_rate', 0):.3f}")
        print(f"  Avg. Sum of Costs (SoC)      : {overall_stats.get('avg_soc', 0):.2f} (on overall success)")
        print(f"  Avg. Makespan                : {overall_stats.get('avg_makespan', 0):.2f} (on overall success)")
        print(f"  Avg. Computation Time (s)    : {overall_stats.get('avg_runtime', 0):.3f} (per run)")

    # --- Save Files ---
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    json_path = output_dir / f"setting_b_full_results_{int(time.time())}.json"
    with open(json_path, 'w') as f:
        output_data = {
            "overall_stats": overall_stats,
            "detailed_results": all_run_results_detailed
        }
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)
    
    logging.info(f"Detailed results saved to {json_path}")

    if summary_results_per_scenario:
        df = pd.DataFrame(summary_results_per_scenario)
        csv_path = output_dir / f"setting_b_summary_{int(time.time())}.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Scenario summary saved to {csv_path}")


if __name__ == "__main__":
    # Standard Config
    parser = argparse.ArgumentParser(description="GNN CS-PIBT Setting B Benchmark")
    parser.add_argument("--map_config_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="benchmark_output_setting_b")
    parser.add_argument("--map_pattern", type=str, default="*")
    parser.add_argument("--map_limit", type=int, default=None)
    parser.add_argument("--agent_counts", type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop_scenario_on_first_success", action='store_true')
    parser.add_argument("--clustered_positions", action='store_true', help="Use clustered corner start/target positions.")

    # Model Config
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--gnn_hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_relu_type", type=str, default='relu')

    # Sim Config
    parser.add_argument("--max_steps", type=int, default=128)
    parser.add_argument("--obs_radius", type=int, default=5)
    parser.add_argument("--pibt_time_limit", type=float, default=-1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose_simulation", action="store_true")

    args = parser.parse_args()
    
    # Setup Logging
    if args.verbose_simulation: logging.getLogger().setLevel(logging.DEBUG)
    
    run_benchmark_main(args)