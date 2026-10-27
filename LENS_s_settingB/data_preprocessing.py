# data_preprocessing.py
import sys
import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque, defaultdict
import logging
import warnings
import argparse
import time
import random

# --- Seed ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# --- LaCAM3 Solver Import ---
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    build_path = os.path.join(current_script_dir, 'lacam3', 'build')
    if build_path not in sys.path:
        sys.path.append(build_path)
    import lacam3_solver
    logging.info("✅ Successfully imported 'lacam3_solver' module.")
except ImportError:
    logging.error("❌ FATAL: Could not import 'lacam3_solver'. Ensure 'make' was run in 'lacam3/build'.")
    sys.exit(1)

# --- POGEMA Imports ---
try:
    from pogema_toolbox.create_env import Environment
    from pogema_toolbox.registry import ToolboxRegistry
    from create_env import create_eval_env
except ImportError as e:
    logging.error(f"Failed to import POGEMA or related modules: {e}")
    sys.exit(1)

# --- Import from mapf_utils ---
try:
    from mapf_utils import (
        OBS_RADIUS, OBS_H, OBS_W, OBS_WINDOW_SIZE, ACTION_DELTAS, NUM_ACTIONS,
        PFG_SPATIAL_INPUT_CHANNELS, CH_IDX_PFG_SPATIAL,
        PFG_NON_SPATIAL_DIM, PFG_NON_SPATIAL_FEATURES,
        POTENTIAL_NORM_VALUE, AGENT_REPULSION_STRENGTH, MAX_WAIT_TIME_NORM,
        bfs_distance_map, extract_patch, normalize_potential_patch,
        create_pfg_spatial_input_tensor_sim, get_non_spatial_features
    )
except ImportError as e:
    logging.error(f"Failed to import constants/functions from mapf_utils: {e}. Check mapf_utils.py.")
    sys.exit(1)


# --- Helper Functions ---
def load_and_register_maps(maps_dir):
    """ Loads and registers map configurations. """
    maps_path = Path(maps_dir); map_count = 0; registered_maps = set()
    if not maps_path.is_dir(): logging.warning(f"Maps directory not found: {maps_dir}"); return False
    for maps_file in maps_path.rglob('maps.yaml'):
        try:
            with open(maps_file, 'r') as f: maps = yaml.safe_load(f)
            if maps:
                new_maps = {k: v for k, v in maps.items() if k not in registered_maps}
                if new_maps:
                    ToolboxRegistry.register_maps(new_maps)
                    logging.info(f"Registered {len(new_maps)} new maps from: {maps_file}")
                    registered_maps.update(new_maps.keys())
                    map_count += len(new_maps)
        except Exception as e: logging.error(f"Failed to load/register {maps_file}: {e}")
    if map_count == 0: logging.warning(f"No new maps registered from {maps_dir}."); return False
    logging.info(f"Total new maps registered: {map_count}")
    return True

def detect_goal_arrival(current_positions, goal_positions, agent_idx):
    return tuple(current_positions[agent_idx]) == tuple(goal_positions[agent_idx])


# --- LaCAM Wrapper Function ---
def solve_with_lacam(env, timeout_seconds=60.0) -> list:
    """
    Uses LaCAM3 to generate an optimal/expert path for all agents.
    Returns: expert_actions_sequences (list of list of action indices)
    """
    obstacles_bool = env.unwrapped.grid.get_obstacles().astype(bool)
    map_str_list = ["".join(["@" if cell else "." for cell in row]) for row in obstacles_bool]
    
    starts_xy = [tuple(p) for p in env.get_agents_xy()]
    goals_xy = [tuple(p) for p in env.get_targets_xy()]
    num_agents = len(starts_xy)

    try:
        solution_paths_from_cpp = lacam3_solver.solve(
            map_str=map_str_list,
            starts_xy=starts_xy,
            goals_xy=goals_xy,
            num_agents=num_agents,
            timeout_ms=int(timeout_seconds * 1000),
            fast_mode=True
        )
        
        if not solution_paths_from_cpp:
            logging.error("LaCAM failed to find a solution or timed out.")
            return None
            
        # Convert coordinates to discrete action indices
        expert_actions = []
        for i in range(num_agents):
            actions = []
            for t in range(len(solution_paths_from_cpp) - 1):
                p1 = solution_paths_from_cpp[t][i]
                p2 = solution_paths_from_cpp[t+1][i]
                dr, dc = p2[0] - p1[0], p2[1] - p1[1]
                
                act_idx = 0  # Default to stay (0)
                for act, (ddr, ddc) in ACTION_DELTAS.items():
                    if dr == ddr and dc == ddc:
                        act_idx = act
                        break
                actions.append(act_idx)
            expert_actions.append(actions)
            
        return expert_actions
    except Exception as e:
        logging.error(f"LaCAM solver encountered an exception: {e}", exc_info=True)
        return None


# --- Main Data Preparation Function (LaCAM + Targeted Sampling) ---
def prepare_agent_centric_dataset_lacam(
    map_names, grid_search_agents, output_dir, obs_radius=OBS_RADIUS,
    max_episode_steps=512, random_seed=42):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"--- Starting Dataset Preparation (LaCAM Expert, Targeted Sampling) ---")

    processed_scenarios = 0; skipped_scenarios = 0; total_samples_saved = 0

    for num_agents_requested in grid_search_agents:
        for map_idx, map_name in enumerate(map_names):
            instance_id = f"{map_name}_agents{num_agents_requested}"
            logging.info(f"--- Processing scenario: {instance_id} ---")
            scenario_start_time = time.time()

            full_trajectory_data = defaultdict(dict)
            pos_history = []
            agent_reached_goal_history = []

            try:
                instance_seed = random_seed
                expert_env_cfg = Environment(
                    map_name=map_name, num_agents=num_agents_requested, obs_radius=obs_radius,
                    observation_type="MAPF", seed=instance_seed, max_episode_steps=max_episode_steps,
                    on_target="nothing", collision_system="soft"
                )
                expert_env = create_eval_env(expert_env_cfg)
                _, _ = expert_env.reset(seed=instance_seed)
                actual_num_agents = expert_env.grid_config.num_agents

                global_obstacles = expert_env.unwrapped.grid.get_obstacles().astype(np.int_)
                map_H, map_W = global_obstacles.shape
                global_starts_xy = expert_env.get_agents_xy()
                global_goals_xy = expert_env.get_targets_xy()

                # --- 1. Run LaCAM Engine ---
                logging.info(f"Simulating MAPF episode using LaCAM for {instance_id}...")
                expert_actions_sequences = solve_with_lacam(expert_env, timeout_seconds=120.0)
                
                if not expert_actions_sequences:
                    logging.warning(f"LaCAM failed for {instance_id}. Skipping scenario.")
                    skipped_scenarios += 1
                    expert_env.close()
                    continue

                path_lengths = [len(seq) for seq in expert_actions_sequences if seq]
                max_path_len = max(path_lengths) if path_lengths else 0
                logging.info(f"LaCAM simulation successful. Max path length: {max_path_len}")

                # Calculate BFS Distance maps for target potentials
                global_potential_maps_bfs = np.stack([
                    bfs_distance_map(expert_env.unwrapped.grid.get_obstacles().astype(np.int_), goal_pos)
                    for goal_pos in global_goals_xy
                ], axis=0)

                # --- 2. Replay & Collect Full Trajectory Data ---
                logging.info("Replaying simulation to collect environmental observations...")
                current_obs_list, _ = expert_env.reset(seed=instance_seed)
                current_global_pos = list(global_starts_xy)
                agent_wait_times = np.zeros(actual_num_agents, dtype=int)

                pos_history = [list(global_starts_xy)]
                initial_reached_goal = np.array([detect_goal_arrival(global_starts_xy, global_goals_xy, i) for i in range(actual_num_agents)])
                agent_reached_goal_history = [initial_reached_goal]

                full_trajectory_data[0] = {
                    'obs': current_obs_list, 'pos': pos_history[0], 'wait': list(agent_wait_times),
                    'goal': list(global_goals_xy), 'action': [-1] * actual_num_agents
                }

                for t in range(max_path_len):
                    current_actions_for_step = [expert_actions_sequences[ag][t] if t < len(expert_actions_sequences[ag]) else 0 for ag in range(actual_num_agents)]

                    pos_before_step = list(current_global_pos)
                    next_obs_list, _, terminated, truncated, infos = expert_env.step(current_actions_for_step)

                    current_obs_list = next_obs_list
                    current_global_pos = list(expert_env.get_agents_xy())
                    pos_history.append(current_global_pos)

                    # Update wait times
                    for i in range(actual_num_agents):
                        if tuple(pos_before_step[i]) == tuple(current_global_pos[i]):
                            if not (terminated[i] or truncated[i]): agent_wait_times[i] += 1
                        else: agent_wait_times[i] = 0
                    
                    current_reached_goal = np.array([detect_goal_arrival(current_global_pos, global_goals_xy, i) for i in range(actual_num_agents)])
                    agent_reached_goal_history.append(current_reached_goal)

                    full_trajectory_data[t+1] = {
                        'obs': current_obs_list, 'pos': current_global_pos, 'wait': list(agent_wait_times),
                        'goal': list(global_goals_xy), 'action': current_actions_for_step
                    }
                    
                    if all(terminated) or all(truncated):
                        break
                
                actual_trajectory_length = len(full_trajectory_data)
                expert_env.close()

                # --- 3. TARGETED SAMPLING (Initialization & Terminal Approach) ---
                logging.info("Applying targeted event-based sampling (Initialization & Terminal approach)...")
                sampled_steps = set()
                
                # Find T_reach for each agent
                arrival_times = {}
                for i in range(actual_num_agents):
                    t_reach = actual_trajectory_length - 1
                    for t in range(actual_trajectory_length):
                        if agent_reached_goal_history[t][i]:
                            t_reach = t
                            break
                    arrival_times[i] = t_reach

                for i in range(actual_num_agents):
                    # Phase 1: Initialization Configuration (t=0)
                    sampled_steps.add((0, i))
                    
                    # Phase 2: Terminal Approach (within 5 steps of the goal)
                    t_reach = arrival_times[i]
                    for t in range(max(0, t_reach - 5), t_reach + 1):
                        if t < actual_trajectory_length:
                            sampled_steps.add((t, i))

                num_sampled = len(sampled_steps)
                logging.info(f"Targeted sampling selected {num_sampled} critical samples out of {actual_trajectory_length * actual_num_agents} total states.")

                if num_sampled == 0:
                    logging.warning(f"No valid samples for {instance_id}. Skipping.")
                    skipped_scenarios += 1
                    continue

                # --- 4. Format and Extract Data ---
                final_pfg_spatial_inputs = []
                final_pfg_nonspatial_wait = []
                final_pfg_nonspatial_tvec_y = []
                final_pfg_nonspatial_tvec_x = []
                final_pfg_targets = []
                final_expert_actions_output = []
                final_metadata = []

                global_other_agents_maps = {} 
                for t_map in range(actual_trajectory_length):
                    agent_map = np.zeros((map_H, map_W), dtype=np.int8)
                    for ag_idx_map, ag_pos_map in enumerate(full_trajectory_data[t_map]['pos']):
                        if 0 <= ag_pos_map[0] < map_H and 0 <= ag_pos_map[1] < map_W:
                            agent_map[ag_pos_map[0], ag_pos_map[1]] = 1
                    global_other_agents_maps[t_map] = agent_map
                
                logging.info(f"Processing and formatting {num_sampled} selected samples...")
                
                for (t, i) in sorted(list(sampled_steps)):
                    # Prevent redundant saving after t_reach except the exact step it arrived
                    if t > arrival_times[i]:
                        continue

                    state_data = full_trajectory_data[t]
                    agent_obs_raw = state_data['obs'][i]
                    agent_pos = state_data['pos'][i]
                    agent_goal = state_data['goal'][i]
                    agent_wait = state_data['wait'][i]
                    
                    agent_target_action = 0 
                    if i < len(expert_actions_sequences) and t < len(expert_actions_sequences[i]):
                         agent_target_action = expert_actions_sequences[i][t]

                    expected_obs_shape = (OBS_H, OBS_W)
                    raw_obs_channels = None
                    if isinstance(agent_obs_raw, dict):
                        obs_ch = agent_obs_raw.get("obstacles")
                        agents_ch = agent_obs_raw.get("agents")
                        target_ch = agent_obs_raw.get("target")
                        channels_to_stack = []
                        channels_to_stack.append(obs_ch if obs_ch is not None and obs_ch.shape == expected_obs_shape else np.zeros(expected_obs_shape, dtype=np.float32))
                        channels_to_stack.append(agents_ch if agents_ch is not None and agents_ch.shape == expected_obs_shape else np.zeros(expected_obs_shape, dtype=np.float32))
                        if "target_hotspot" in CH_IDX_PFG_SPATIAL:
                             channels_to_stack.append(target_ch if target_ch is not None and target_ch.shape == expected_obs_shape else np.zeros(expected_obs_shape, dtype=np.float32))
                        if channels_to_stack: raw_obs_channels = np.stack(channels_to_stack, axis=0).astype(np.float32)

                    if raw_obs_channels is None:
                        num_fallback_channels = max(CH_IDX_PFG_SPATIAL.values()) + 1 if CH_IDX_PFG_SPATIAL else 0
                        raw_obs_channels = np.zeros((num_fallback_channels, OBS_H, OBS_W), dtype=np.float32)

                    pfg_spatial_input = create_pfg_spatial_input_tensor_sim(raw_obs_channels)
                    wait_norm, target_dy, target_dx = get_non_spatial_features(agent_pos, agent_goal, agent_wait)
                    
                    bfs_dist_patch_raw = extract_patch(global_potential_maps_bfs[i], agent_pos, OBS_WINDOW_SIZE, padding_value=np.inf)
                    obstacle_patch_raw = extract_patch(global_obstacles, agent_pos, OBS_WINDOW_SIZE, padding_value=1)
                    
                    other_agents_map_t_for_patch = global_other_agents_maps.get(t, np.zeros((map_H, map_W), dtype=np.int8)).copy()
                    if 0 <= agent_pos[0] < map_H and 0 <= agent_pos[1] < map_W:
                         other_agents_map_t_for_patch[agent_pos[0], agent_pos[1]] = 0
                    other_agents_patch_t = extract_patch(other_agents_map_t_for_patch, agent_pos, OBS_WINDOW_SIZE, padding_value=0)
                    pfg_target = normalize_potential_patch(bfs_dist_patch_raw, obstacle_patch_raw, other_agents_patch_t)

                    final_pfg_spatial_inputs.append(pfg_spatial_input)
                    final_pfg_nonspatial_wait.append(wait_norm)
                    final_pfg_nonspatial_tvec_y.append(target_dy)
                    final_pfg_nonspatial_tvec_x.append(target_dx)
                    final_pfg_targets.append(pfg_target)
                    final_expert_actions_output.append(agent_target_action)
                    final_metadata.append({
                        "agent_index": i, "time_step": t, "global_pos_xy": agent_pos, 
                        "global_goal_xy": agent_goal, "wait_time": agent_wait, 
                        "reached_goal_at_t": agent_reached_goal_history[t][i]
                    })
                
                # --- 5. Save Output ---
                if not final_pfg_spatial_inputs:
                    skipped_scenarios += 1
                    continue

                output_path = output_dir / f"{instance_id}_pfg_only.npz"
                np.savez_compressed(
                    output_path,
                    pfg_spatial_input_tensors=np.stack(final_pfg_spatial_inputs, axis=0),
                    pfg_nonspatial_wait_time=np.array(final_pfg_nonspatial_wait, dtype=np.float32),
                    pfg_nonspatial_target_vec_y=np.array(final_pfg_nonspatial_tvec_y, dtype=np.float32),
                    pfg_nonspatial_target_vec_x=np.array(final_pfg_nonspatial_tvec_x, dtype=np.float32),
                    pfg_target_potentials=np.stack(final_pfg_targets, axis=0),
                    expert_actions=np.array(final_expert_actions_output, dtype=np.int32),
                    map_name=map_name, num_agents=actual_num_agents, map_shape=global_obstacles.shape,
                    obs_radius=obs_radius, agent_metadata=np.array(final_metadata, dtype=object)
                )
                total_samples_saved += len(final_pfg_spatial_inputs)
                logging.info(f"Saved {len(final_pfg_spatial_inputs)} targeted PFG samples for {instance_id}. Time: {time.time() - scenario_start_time:.2f}s")
                processed_scenarios += 1
            
            except Exception as e:
                logging.error(f"Failed to process instance {instance_id}. Error: {e}", exc_info=True)
                skipped_scenarios += 1

    logging.info("--- Data Preparation (LaCAM + Targeted Sampling) Summary ---")
    logging.info(f"Total scenarios processed successfully: {processed_scenarios}")
    logging.info(f"Total scenarios skipped: {skipped_scenarios}")
    logging.info(f"Total PFG samples saved: {total_samples_saved}")
    logging.info(f"Datasets saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAPF Agent-Centric Dataset Preparation (LaCAM Expert, Initialization + Terminal Sampling)")
    parser.add_argument('--map-config-dir', type=str, required=True, help="Directory containing Pogema map YAML configurations.")
    parser.add_argument('--map-pattern', type=str, required=True, help="Pattern for map names (e.g., 'maze-seed-???')")
    parser.add_argument('--map-indices', type=str, default="0:12", help="Range of map indices to process (e.g., '0:50')")
    parser.add_argument('--agent-counts', type=str, default="8,24", help="Comma-separated list of agent counts (e.g., '8,16,32')")
    parser.add_argument('--output-dir', type=str, default="dataset_pfg_lacam", help="Directory to save the processed .npz datasets.")
    parser.add_argument('--obs-radius', type=int, default=OBS_RADIUS, help="Agent observation radius.")
    parser.add_argument('--max-steps', type=int, default=512, help="Maximum steps per episode during simulation.")
    parser.add_argument('--seed', type=int, default=42, help="Base random seed for environment generation.")

    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    logging.info("--- Starting Data Preprocessing Script (LaCAM Version) ---")

    map_dir = Path(args.map_config_dir)
    output_dir = Path(args.output_dir)
    agent_counts_list = [int(x) for x in args.agent_counts.split(',') if x]
    
    try:
        map_range = args.map_indices.split(':')
        start_idx, end_idx = map(int, map_range)
        if start_idx >= end_idx: raise ValueError("Start index must be less than end index")
    except (ValueError, IndexError):
        logging.error("Invalid format for --map-indices. Use 'start:end' (e.g., '0:10')."); sys.exit(1)

    map_names_list = []
    if '?' in args.map_pattern:
        num_digits = args.map_pattern.count('?')
        name_prefix = args.map_pattern.split('?')[0]
        name_suffix_parts = args.map_pattern.split('?' * num_digits)
        name_suffix = name_suffix_parts[1] if len(name_suffix_parts) > 1 else ""
        map_names_list = [f"{name_prefix}{str(i).zfill(num_digits)}{name_suffix}" for i in range(start_idx, end_idx)]
    elif start_idx == 0 and end_idx == 1:
        map_names_list = [args.map_pattern]
    else:
        logging.error("Cannot generate map names. Use '?' in map-pattern for ranges, or specify a single map with indices 0:1."); sys.exit(1)

    if not load_and_register_maps(args.map_config_dir):
        logging.warning("Map registration failed or found no new maps. Ensure maps exist and paths are correct.")

    prepare_agent_centric_dataset_lacam(
        map_names=map_names_list, grid_search_agents=agent_counts_list, output_dir=args.output_dir,
        obs_radius=args.obs_radius, max_episode_steps=args.max_steps, random_seed=seed
    )
    
    logging.info("--- Script Finished ---")