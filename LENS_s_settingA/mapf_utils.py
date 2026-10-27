# mapf_utils.py
import numpy as np
import logging
from collections import deque
from pogema_toolbox.create_env import Environment
from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.results_holder import ResultsHolder

# --- Constants (Ensure consistency across all files!) ---
OBS_RADIUS = 5
OBS_H = OBS_W = OBS_RADIUS * 2 + 1
OBS_WINDOW_SIZE = (OBS_H, OBS_W)

# PFG Channel Definitions (Spatial Only for CNN)
CH_IDX_PFG_SPATIAL = {
    "obstacles": 0,         # From observation
    "other_agents": 1,      # From observation
    "self_pos": 2,          # Generated (one-hot)
    "target_hotspot": 3,    # From observation (if available, e.g., POMAPF)
}
PFG_SPATIAL_INPUT_CHANNELS = len(CH_IDX_PFG_SPATIAL)  # = 4

# PFG Non-Spatial Feature Definitions (for late fusion)
PFG_NON_SPATIAL_FEATURES = ["wait_time_norm", "target_vec_y", "target_vec_x"]
PFG_NON_SPATIAL_DIM = len(PFG_NON_SPATIAL_FEATURES)  # = 3
PFG_OUTPUT_CHANNELS = 1  # Potential Field

# ADN Input Channel Definitions
CH_IDX_ADN_OBS = {
    "other_agents": 0,
    "self_pos": 1,
    "wait_time": 2,
    "target_vec_y": 3,
    "target_vec_x": 4,
}

ADN_OBS_INPUT_CHANNELS = 5
ADN_TOTAL_INPUT_CHANNELS = ADN_OBS_INPUT_CHANNELS + 1
ADN_OBS_NAME_TO_PFG_SPATIAL_IDX = {
    "other_agents": CH_IDX_PFG_SPATIAL["other_agents"],
    "self_pos": CH_IDX_PFG_SPATIAL["self_pos"],
}
PFG_SPATIAL_IDXS_FOR_ADN_OBS = [ADN_OBS_NAME_TO_PFG_SPATIAL_IDX[name] for name in ["other_agents", "self_pos"]]

# Potential Field Constants
OBSTACLE_POTENTIAL = np.inf
TARGET_POTENTIAL = 0.0
MAX_POTENTIAL_DISTANCE = 128
POTENTIAL_NORM_VALUE = 10.0
AGENT_REPULSION_STRENGTH = 0.0

# Action Definitions
ACTION_DELTAS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
NUM_ACTIONS = len(ACTION_DELTAS)
ACTION_STAY = 0

# Wait time normalization
MAX_WAIT_TIME_NORM = 50

# Critical Event Sampling Parameters
PROXIMITY_THRESHOLD_MANHATTAN = 1
WAIT_STEPS_THRESHOLD = 15
DIRECTION_CHANGE_DOT_PRODUCT_THRESHOLD = -1
LOCAL_DENSITY_RADIUS = 5
LOCAL_DENSITY_THRESHOLD = 0.7
SAMPLE_WINDOW_AROUND_EVENT = 0


def precompute_tdh(obstacle_map: np.ndarray) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
    """
    Precomputes True Distance Heuristics (TDH) for all pairs of non-obstacle cells.
    Returns a dictionary: tdh[start_pos][end_pos] = distance.
    """
    height, width = obstacle_map.shape
    tdh = {}
    free_cells = []
    for r in range(height):
        for c in range(width):
            if obstacle_map[r, c] == 0:
                free_cells.append((r, c))

    for start_node in free_cells:
        tdh[start_node] = {}
        q = deque([(start_node, 0)])
        visited = {start_node}
        tdh[start_node][start_node] = 0

        while q:
            curr, dist = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                neighbor = (nr, nc)
                if 0 <= nr < height and 0 <= nc < width and obstacle_map[nr, nc] == 0 and neighbor not in visited:
                    visited.add(neighbor)
                    tdh[start_node][neighbor] = dist + 1
                    q.append((neighbor, dist + 1))
    logging.info(f"TDH precomputation complete for map shape {obstacle_map.shape}. Found distances for {len(free_cells)} start cells.")
    return tdh


def simulate_complete_paths(env, inference_agent, max_steps=256):
    results_holder = ResultsHolder()
    obs, _ = env.reset(seed=env.grid_config.seed)
    start_positions = env.get_agents_xy()
    goal_positions = env.get_targets_xy()
    num_agents = len(start_positions)
    expert_paths = [[] for _ in range(num_agents)]
    success = False
    while True:
        actions = inference_agent.act(obs)
        obs, rew, terminated, truncated, infos = env.step(actions)

        results_holder.after_step(infos)
        for i in range(num_agents):
            expert_paths[i].append(actions[i])
        if all(terminated):
            success = True
            break
        elif all(truncated):
            break
    return start_positions, goal_positions, expert_paths, success


def extract_patch(global_map, center_xy, patch_shape, padding_value):
    """Extracts patch, handles boundaries."""
    H_global, W_global = global_map.shape
    H_patch, W_patch = patch_shape
    center_r, center_c = center_xy
    pad_top = max(0, H_patch // 2 - center_r)
    pad_bottom = max(0, (center_r + (H_patch - H_patch // 2)) - H_global)
    pad_left = max(0, W_patch // 2 - center_c)
    pad_right = max(0, (center_c + (W_patch - W_patch // 2)) - W_global)
    start_r = max(0, center_r - H_patch // 2 + pad_top)
    end_r = min(H_global, center_r + (H_patch - H_patch // 2) - pad_bottom)
    start_c = max(0, center_c - W_patch // 2 + pad_left)
    end_c = min(W_global, center_c + (W_patch - W_patch // 2) - pad_right)
    extracted = global_map[start_r:end_r, start_c:end_c]
    padded_patch = np.pad(extracted, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=padding_value)
    if padded_patch.shape != patch_shape:
        final_patch = np.full(patch_shape, padding_value, dtype=global_map.dtype)
        h_min, w_min = min(H_patch, padded_patch.shape[0]), min(W_patch, padded_patch.shape[1])
        final_patch[:h_min, :w_min] = padded_patch[:h_min, :w_min]
        return final_patch
    return padded_patch


def create_pfg_spatial_input_tensor_sim(agent_obs_channels):
    """Creates ONLY the SPATIAL PFG input channels."""
    input_tensor = np.zeros((PFG_SPATIAL_INPUT_CHANNELS, OBS_H, OBS_W), dtype=np.float32)
    center_r, center_c = OBS_H // 2, OBS_W // 2
    raw_obs_ch_count = agent_obs_channels.shape[0]
    idx_obs = CH_IDX_PFG_SPATIAL.get("obstacles")
    idx_agents = CH_IDX_PFG_SPATIAL.get("other_agents")
    idx_self = CH_IDX_PFG_SPATIAL.get("self_pos")
    idx_target = CH_IDX_PFG_SPATIAL.get("target_hotspot")
    target_channel_idx_in_obs = 2

    if idx_obs is not None and 0 < raw_obs_ch_count:
        input_tensor[idx_obs] = agent_obs_channels[0].astype(np.float32)
    if idx_agents is not None and 1 < raw_obs_ch_count:
        input_tensor[idx_agents] = agent_obs_channels[1].astype(np.float32)
    if idx_self is not None:
        input_tensor[idx_self, center_r, center_c] = 1.0
    if idx_target is not None:
        if target_channel_idx_in_obs < raw_obs_ch_count:
            target_mask = agent_obs_channels[target_channel_idx_in_obs].astype(np.float32)
            if idx_obs is not None and idx_obs in CH_IDX_PFG_SPATIAL.values():
                obstacle_mask = input_tensor[idx_obs] > 0.5
                target_mask[obstacle_mask] = 0
            input_tensor[idx_target] = target_mask
    return input_tensor


def get_non_spatial_features(global_agent_xy, global_goal_xy, wait_time):
    """Calculates non-spatial features."""
    wait_time_norm = min(wait_time / MAX_WAIT_TIME_NORM, 1.0) if MAX_WAIT_TIME_NORM > 0 else 0.0
    relative_target_r = global_goal_xy[0] - global_agent_xy[0]
    relative_target_c = global_goal_xy[1] - global_agent_xy[1]
    norm_dy, norm_dx = 0.0, 0.0
    if not (relative_target_r == 0 and relative_target_c == 0):
        magnitude = np.sqrt(relative_target_r**2 + relative_target_c**2)
        epsilon = 1e-6
        norm_dy = relative_target_r / (magnitude + epsilon)
        norm_dx = relative_target_c / (magnitude + epsilon)
    return np.array([wait_time_norm, norm_dy, norm_dx], dtype=np.float32)


def bfs_distance_map(obstacle_map, goal_xy):
    """Calculates BFS distances."""
    H, W = obstacle_map.shape
    dist_map = np.full((H, W), np.inf, dtype=np.float32)
    r_goal, c_goal = goal_xy
    if not (0 <= r_goal < H and 0 <= c_goal < W):
        return dist_map
    if obstacle_map[r_goal, c_goal] == 1:
        return dist_map
    dist_map[r_goal, c_goal] = 0
    queue = deque([(r_goal, c_goal)])
    visited = np.zeros((H, W), dtype=bool)
    visited[r_goal, c_goal] = True
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        r, c = queue.popleft()
        current_dist = dist_map[r, c]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and obstacle_map[nr, nc] == 0:
                visited[nr, nc] = True
                dist_map[nr, nc] = current_dist + 1
                queue.append((nr, nc))
    return dist_map


def normalize_potential_patch(potential_patch_raw, obstacle_mask_patch, agent_other_agents_patch):
    """Normalizes potential field patch to [0, 10] scale."""
    potential_with_repulsion = potential_patch_raw.copy()
    other_agent_locations = (agent_other_agents_patch == 1)
    finite_mask_initial = np.isfinite(potential_with_repulsion)
    repulsion_mask = other_agent_locations & finite_mask_initial
    potential_with_repulsion[repulsion_mask] += AGENT_REPULSION_STRENGTH
    clamped_dist = np.minimum(potential_with_repulsion, MAX_POTENTIAL_DISTANCE)
    finite_mask = np.isfinite(clamped_dist)
    normalized_patch = np.full_like(potential_patch_raw, POTENTIAL_NORM_VALUE, dtype=np.float32)
    if np.any(finite_mask):
        normalized_patch[finite_mask] = np.log1p(clamped_dist[finite_mask])
    unreachable_or_obstacle_mask = (obstacle_mask_patch == 1) | ~finite_mask
    normalized_patch[unreachable_or_obstacle_mask] = POTENTIAL_NORM_VALUE
    valid_mask = ~unreachable_or_obstacle_mask
    if np.any(valid_mask):
        min_val = np.min(normalized_patch[valid_mask])
        normalized_patch[valid_mask] -= min_val
        max_val_after_shift = np.max(normalized_patch[valid_mask])
        if max_val_after_shift > 0:
            scale_factor = (POTENTIAL_NORM_VALUE * 0.9) / max_val_after_shift
            normalized_patch[valid_mask] *= scale_factor
    normalized_patch[unreachable_or_obstacle_mask] = POTENTIAL_NORM_VALUE
    return normalized_patch
