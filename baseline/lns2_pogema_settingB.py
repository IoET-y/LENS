# ====================================================================
# File: lns2_pogema_benchmark_setting_b.py
# Desc: LNS2-RL Benchmark Adapted for Setting B (Unknown Map / Joint Exploration)
#       - Map is initialized as empty (optimistic).
#       - Map is discovered online via Agent FOV.
#       - LNS2 is forced to re-plan on the evolving map, triggering heuristic re-computation.
# ====================================================================

import os
import torch
import numpy as np
import time
import argparse
import yaml
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# --- Pogema Imports ---
try:
    from pogema_toolbox.registry import ToolboxRegistry
    from pogema_toolbox.create_env import Environment
    from create_env import create_eval_env
except ImportError:
    logging.error("Pogema import failed. Please install pogema and pogema_toolbox.")
    sys.exit(1)

# --- LNS2 Imports ---
try:
    from LNS2_RL.lns2.build import my_lns2
    from LNS2_RL.model import Model
    from LNS2_RL.alg_parameters import NetParameters, EnvParameters
    from lns2_adapter import PogemaLNS2Env, LNS2_TO_POGEMA
except ImportError as e:
    logging.error(f"LNS2 Imports failed: {e}. Ensure LNS2_RL and lns2_adapter are in path.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================================================================
# 1. Map Loading & Helper Functions
# ====================================================================

def load_and_register_maps(maps_dir):
    """ Loads and registers map configurations. """
    maps_path = Path(maps_dir)
    map_count = 0
    registered_maps = set()
    if not maps_path.is_dir():
        logging.warning(f"Maps directory not found: {maps_dir}")
        return False, []
    
    all_map_names = []
    for maps_file in maps_path.rglob('maps.yaml'): # Search recursively
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
        except Exception as e:
            logging.error(f"Failed to load/register {maps_file}: {e}")
            
    if map_count == 0:
        logging.warning(f"No new maps registered from {maps_dir}.")
        return False, []
    
    logging.info(f"Total unique maps registered: {len(registered_maps)}")
    return True, sorted(list(registered_maps))

def filter_maps_by_pattern(all_map_names, pattern):
    """ Filters map names based on a simple wildcard pattern. """
    if pattern == "*" or not pattern: return all_map_names
    if pattern.startswith("*") and pattern.endswith("*"):
        substring = pattern[1:-1]; return [name for name in all_map_names if substring in name]
    elif pattern.startswith("*"):
        suffix = pattern[1:]; return [name for name in all_map_names if name.endswith(suffix)]
    elif pattern.endswith("*"):
        prefix = pattern[:-1]; return [name for name in all_map_names if name.startswith(prefix)]
    else: return [name for name in all_map_names if name == pattern]

def calculate_overall_stats(detailed_results_list, args_config):
    """Calculates overall statistics from all trial results."""
    if not detailed_results_list:
        return {"error": "No detailed results to calculate overall stats."}

    stats = {}
    num_agents_set = set()
    all_trial_isrs = []
    successful_overall_runs_count = 0
    successful_socs = []
    successful_makespans = []
    all_durations = []
    maps_actually_tested_set = set()

    for r_dict in detailed_results_list:
        num_agents_set.add(r_dict.get('num_agents_at_start', 0))
        maps_actually_tested_set.add(r_dict['map_name'])

        num_at_start = r_dict.get('num_agents_at_start', 0)
        num_reached = r_dict.get('num_agents_reached_target', 0)
        isr_trial = (num_reached / num_at_start) if num_at_start > 0 else (1.0 if r_dict.get('success', False) else 0.0)
        all_trial_isrs.append(isr_trial)

        if r_dict.get('success', False):
            successful_overall_runs_count += 1
            if 'sum_of_costs' in r_dict and r_dict['sum_of_costs'] >= 0: successful_socs.append(r_dict['sum_of_costs'])
            if 'makespan' in r_dict and r_dict['makespan'] >= 0: successful_makespans.append(r_dict['makespan'])

        if 'computation_time_sec' in r_dict: all_durations.append(r_dict['computation_time_sec'])

    total_runs_attempted = len(detailed_results_list)
    
    stats['num_agents_configurations_tested'] = sorted(list(n for n in num_agents_set if n > 0))
    stats['avg_isr'] = np.mean(all_trial_isrs) if all_trial_isrs else 0.0
    stats['overall_success_rate'] = (successful_overall_runs_count / total_runs_attempted) if total_runs_attempted > 0 else 0.0
    stats['avg_soc_on_overall_success'] = np.mean(successful_socs) if successful_socs else float('nan')
    stats['avg_makespan_on_overall_success'] = np.mean(successful_makespans) if successful_makespans else float('nan')
    stats['avg_duration_s_per_run'] = np.mean(all_durations) if all_durations else float('nan')
    stats['maps_tested_count'] = len(maps_actually_tested_set)
    stats['runs_attempted_total'] = total_runs_attempted

    return stats

# ====================================================================
# 2. LNS2 Solver Logic
# ====================================================================

def solve_with_lns2(raw_grid, agents_xy, targets_xy, model, device, max_iters, time_limit):
    """
    Wrapper for LNS2 CPP and Python Adapter.
    Args:
        raw_grid: The CURRENTLY KNOWN map (0=Free, 1=Obs). 
                  In Setting B, this is 'persistent_known_map'.
    """
    num_agents = len(agents_xy)
    H, W = raw_grid.shape
    safe_dim = max(H, W) + 2
    trash_can_pos = (safe_dim - 1, safe_dim - 1)

    # Prepare map for CPP (0=Free, -1=Obs)
    # Note: raw_grid is 0(Free)/1(Obs)
    map_for_cpp = np.full((safe_dim, safe_dim), -1, dtype=int)
    map_for_cpp[:H, :W] = -raw_grid 
    map_for_cpp[trash_can_pos] = 0 
    
    # Initialize LNS2 (This re-computes PP and initializes Sipps)
    # This step implicitly includes re-computing heuristics inside LNS2 initialization if implemented there,
    # or in PogemaLNS2Env wrapper below.
    lns2_cpp = my_lns2.MyLns2(42, map_for_cpp.tolist(), agents_xy, targets_xy, num_agents, safe_dim)
    lns2_cpp.init_pp()
    
    # Initialize Adapter (This triggers _recalc_heuristic_map based on raw_grid)
    env = PogemaLNS2Env(raw_grid, agents_xy, targets_xy, lns2_cpp, safe_dim=safe_dim)
    
    start_time = time.time()
    iter_count = 0
    HISTORY_LEN = NetParameters.TIME_DEPT
    
    while iter_count < max_iters and (time.time() - start_time) < time_limit:
        subset_size = min(num_agents//2, 8) 
        local_agents = np.random.choice(num_agents, subset_size, replace=False).tolist()
        
        env.reset_for_planning(local_agents)
        
        obs_tensor = np.zeros((1, subset_size, NetParameters.NUM_CHANNEL * HISTORY_LEN, env.fov_size, env.fov_size), dtype=np.float32)
        vec_tensor = np.zeros((1, subset_size, NetParameters.VECTOR_LEN), dtype=np.float32)
        hidden_state_np = np.zeros((1, subset_size, 2, NetParameters.NET_SIZE), dtype=np.float32)
        
        obs_history = []
        for i in range(subset_size):
            o, v = env.observe(i)
            hist_buffer = [np.zeros_like(o) for _ in range(HISTORY_LEN - 1)]
            hist_buffer.append(o)
            obs_history.append(hist_buffer)
            
            vec_tensor[0, i, :3] = v
            vec_tensor[0, i, 3] = (env.sipp_coll_pair_num - len(env.new_collision_pairs)) / (env.sipp_coll_pair_num + 1)
            vec_tensor[0, i, 4] = env.time_step / (env.episode_len + 1e-5)
            vec_tensor[0, i, 5] = env.time_step / (env.sipps_max_len + 1e-5)
            vec_tensor[0, i, 6] = 0.0 

        current_paths = {ag: [env.world.local_agents_poss[i]] for i, ag in enumerate(local_agents)}
        
        done = False
        while not done:
            num_on_goal = 0
            for i in range(subset_size):
                if env.world.local_agents_poss[i] == env.goal_list[local_agents[i]]:
                    num_on_goal += 1
            
            for i in range(subset_size):
                obs_tensor[0, i] = np.concatenate(obs_history[i], axis=0)
                vec_tensor[0, i, 6] = num_on_goal / subset_size

            valid_actions = [env.list_next_valid_actions(i) for i in range(subset_size)]
            
            h_in = torch.from_numpy(hidden_state_np[:, :, 0, :].reshape(-1, NetParameters.NET_SIZE)).to(device)
            c_in = torch.from_numpy(hidden_state_np[:, :, 1, :].reshape(-1, NetParameters.NET_SIZE)).to(device)
            
            actions, _, _, next_hidden_tuple, _ = model.step(
                obs_tensor[0], vec_tensor[0], valid_actions, (h_in, c_in), subset_size
            )
            
            done = env.joint_step(actions)
            
            if not done:
                hidden_state_np[0, :, 0, :] = next_hidden_tuple[0].detach().cpu().numpy()
                hidden_state_np[0, :, 1, :] = next_hidden_tuple[1].detach().cpu().numpy()
                for i in range(subset_size):
                    current_paths[local_agents[i]].append(env.world.local_agents_poss[i])
                    o, v = env.observe(i)
                    obs_history[i].pop(0)
                    obs_history[i].append(o)
                    vec_tensor[0, i, :3] = v
                    vec_tensor[0, i, 3] = (env.sipp_coll_pair_num - len(env.new_collision_pairs)) / (env.sipp_coll_pair_num + 1)
                    vec_tensor[0, i, 4] = env.time_step / (env.episode_len + 1e-5)
                    vec_tensor[0, i, 5] = env.time_step / (env.sipps_max_len + 1e-5)
                    vec_tensor[0, i, 7] = actions[i] 
        
        for i, ag in enumerate(local_agents):
            path = current_paths[ag]
            goal = env.goal_list[ag]
            try:
                idx = len(path) - 1 - path[::-1].index(goal)
                truncated = path[:idx+1]
            except ValueError:
                truncated = path
            env.paths[ag] = truncated + [trash_can_pos] * 10
            
        try:
            env.lns2_model.vector_path = env.paths
        except Exception: pass
        
        iter_count += 1
        
    clean_paths = []
    for p in env.paths:
        try:
            cut = p.index(trash_can_pos)
            clean_paths.append(p[:cut])
        except ValueError:
            clean_paths.append(p)
    return clean_paths

# ====================================================================
# 3. Setting B: Map Update Logic
# ====================================================================

def update_persistent_map(current_obs_list, current_agent_positions, known_map, obs_rad):
    """
    Updates the global `known_map` based on current agent observations (FOV).
    This logic is strictly aligned with LPSS_Direct.py.
    known_map: 0 = Free/Unknown (Optimistic), 1 = Obstacle.
    """
    map_h, map_w = known_map.shape
    window_size = obs_rad * 2 + 1
    
    for i, (r, c) in enumerate(current_agent_positions):
        # Safety check
        if i >= len(current_obs_list) or current_obs_list[i] is None: continue
        
        fov = current_obs_list[i].get("obstacles") 
        if fov is None: continue
        
        tl_r, tl_c = r - obs_rad, c - obs_rad
        
        for fr in range(window_size):
            for fc in range(window_size):
                gr, gc = tl_r + fr, tl_c + fc
                
                if 0 <= gr < map_h and 0 <= gc < map_w:
                    # Mark observed status. 
                    # If fov is 1 (obstacle), set global to 1.
                    # If fov is 0 (free), set global to 0.
                    # This updates knowledge dynamically.
                    if fov[fr, fc] == 1:
                        known_map[gr, gc] = 1 
                    else:
                        known_map[gr, gc] = 0

# ====================================================================
# 4. Simulation Loop (Setting B)
# ====================================================================

def run_lns2_simulation_setting_b(env, model, device, args):
    """
    Simulates LNS2-RL in Unknown Map Setting.
    Re-plans every REPLAN_INTERVAL steps using the CURRENTLY KNOWN map.
    """
    sim_start_time = time.time()
    
    # Ground Truth info (ONLY for array sizing and goal checking)
    raw_grid_gt = env.unwrapped.grid.get_obstacles().astype(int)
    map_h, map_w = raw_grid_gt.shape
    targets_xy = env.get_targets_xy()
    
    if hasattr(env, 'grid_config'):
        num_agents = env.grid_config.num_agents
    else:
        num_agents = len(env.get_agents_xy())

    # --- INITIALIZE PERSISTENT MAP (All Free/Unknown) ---
    # 0 = Free, 1 = Obstacle
    persistent_known_map = np.zeros((map_h, map_w), dtype=int)

    total_steps = 0
    done = False
    REPLAN_INTERVAL = 8 # Replanning frequency
    
    current_planned_paths = []
    
    active_agents = [True] * num_agents
    agent_costs = [0] * num_agents
    finished_successfully = [False] * num_agents
    
    # --- Initial Observation ---
    # We need to populate map before first plan. 
    # Since env.reset() returns obs but create_eval_env might hide it, 
    # we assume we can get it via a no-op or access env.
    # To be safe and fair, we assume start map is empty, plan is Euclidean, 
    # then first step happens, then map updates.
    # OR: If possible, execute map update immediately if obs is available.
    # Here we stick to loop logic: First plan on empty map (optimistic).
    
    while not done and total_steps < args.max_steps:
        # --- 1. Replan Check (On Known Map) ---
        if total_steps % REPLAN_INTERVAL == 0 or not current_planned_paths:
            current_agents_xy = env.get_agents_xy()
            
            if not any(active_agents): break

            iters = args.lns_iters if total_steps == 0 else max(10, args.lns_iters // 2)
            t_limit = args.planning_time_limit if total_steps == 0 else 1.0
            
            # [CRITICAL]: Plan using PERSISTENT_KNOWN_MAP, not Ground Truth
            try:
                current_planned_paths = solve_with_lns2(
                    persistent_known_map, # <-- The partial map
                    current_agents_xy, 
                    targets_xy, 
                    model, device, 
                    max_iters=iters, time_limit=t_limit
                )
            except Exception as e:
                logging.error(f"LNS2 Plan Failed at step {total_steps}: {e}")
                # Fallback: empty paths will trigger stay
                current_planned_paths = [[] for _ in range(num_agents)]
        
        # --- 2. Action Calculation ---
        actions = []
        current_agents_xy = env.get_agents_xy()
        step_in_interval = total_steps % REPLAN_INTERVAL
        
        for i in range(num_agents):
            if not active_agents[i]:
                actions.append(0)
                continue

            # Check validity of planned path against known obstacles?
            # LNS2 planned it on known map, so it should be valid w.r.t known map.
            # But new obstacles might have appeared since plan start if interval > 1.
            # Here we follow plan. If it hits unseen wall, POGEMA handles collision.
            
            # Get path for agent
            path = current_planned_paths[i] if i < len(current_planned_paths) else []
            curr_real = current_agents_xy[i]
            goal_real = targets_xy[i]
            act = 0 
            
            # Goal Grabber
            dist = abs(curr_real[0] - goal_real[0]) + abs(curr_real[1] - goal_real[1])
            if dist == 1:
                dr, dc = goal_real[0] - curr_real[0], goal_real[1] - curr_real[1]
                nx, ny = curr_real[0] + dr, curr_real[1] + dc
                # Check against GT for instant success? No, simply check if valid move.
                # Just emit action.
                if dr == -1 and dc == 0: act = 1
                elif dr == 1 and dc == 0: act = 2
                elif dr == 0 and dc == -1: act = 3
                elif dr == 0 and dc == 1: act = 4
                actions.append(act)
                continue

            # Path Following
            if step_in_interval + 1 < len(path):
                target_pos = path[step_in_interval + 1]
                dr, dc = target_pos[0] - curr_real[0], target_pos[1] - curr_real[1]
                if dr == -1 and dc == 0: act = 1
                elif dr == 1 and dc == 0: act = 2
                elif dr == 0 and dc == -1: act = 3
                elif dr == 0 and dc == 1: act = 4
                else: act = 0
            else:
                act = 0
            
            actions.append(act)
        
        # --- 3. Step Environment & Get Obs ---
        # [CRITICAL]: Must capture observations for map update
        obs_list, _, terminated, truncated, _ = env.step(actions)
        total_steps += 1
        
        # --- 4. Update Global Map (Discovery) ---
        new_agents_xy = env.get_agents_xy()
        update_persistent_map(obs_list, new_agents_xy, persistent_known_map, env.grid_config.obs_radius)
        
        # --- 5. Statistics Update ---
        for i in range(num_agents):
            if active_agents[i]:
                agent_costs[i] += 1
                if terminated[i]:
                    active_agents[i] = False
                    finished_successfully[i] = True
                elif truncated[i]:
                    active_agents[i] = False
        
        done = all(terminated) or all(truncated) or (not any(active_agents))
        
    sim_duration = time.time() - sim_start_time
    num_reached = sum(finished_successfully)
    is_overall_success = (num_reached == num_agents)
    valid_costs = [agent_costs[i] for i in range(num_agents) if finished_successfully[i]]
    sum_of_costs = sum(valid_costs) if valid_costs else 0
    
    return {
        "success": is_overall_success,
        "makespan": total_steps,
        "sum_of_costs": sum_of_costs,
        "computation_time_sec": sim_duration,
        "num_agents_at_start": num_agents,
        "num_agents_reached_target": num_reached,
        "map_name": env.grid_config.map_name if hasattr(env, 'grid_config') else "unknown_map"
    }

# ====================================================================
# 5. Main Benchmark Loop
# ====================================================================

def run_benchmark_main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")

    # Load Maps
    maps_registered_ok, all_available_map_names = load_and_register_maps(args.map_config_dir)
    if not maps_registered_ok and not all_available_map_names: sys.exit(1)
    
    maps_to_run_on = filter_maps_by_pattern(all_available_map_names, args.map_pattern)
    if not maps_to_run_on: logging.error(f"No maps matching '{args.map_pattern}'."); sys.exit(1)
    
    if args.map_limit and 0 < args.map_limit < len(maps_to_run_on):
        maps_to_run_on = maps_to_run_on[:args.map_limit]
        logging.info(f"Limiting to first {args.map_limit} maps.")

    # Load Model
    if not os.path.exists(args.model_path):
        logging.error(f"Model not found: {args.model_path}")
        sys.exit(1)
        
    logging.info(f"Loading LNS2-RL model from {args.model_path}")
    model_wrapper = Model(0, device, True)
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model' in checkpoint:
        model_wrapper.network.load_state_dict(checkpoint['model'])
    else:
        model_wrapper.network.load_state_dict(checkpoint)
    model_wrapper.network.eval()

    # Loop Scenarios
    all_results = []
    summary_results = []
    
    total_scenarios = len(maps_to_run_on) * len(args.agent_counts)
    pbar = tqdm(total=total_scenarios, desc="Scenarios")
    
    for map_name in maps_to_run_on:
        for num_agents in args.agent_counts:
            pbar.set_description(f"Map: {map_name}, A: {num_agents}")
            
            scenario_results = []
            succeeded_once = False
            
            for trial_idx in range(args.num_trials):
                if args.stop_scenario_on_first_success and succeeded_once:
                    break
                
                seed = args.seed + trial_idx
                
                # Create Environment
                env_config = Environment(
                    map_name=map_name,
                    num_agents=num_agents,
                    seed=seed,
                    max_episode_steps=args.max_steps,
                    observation_type="POMAPF", 
                    collision_system="soft", with_animation=True
                )
                env = create_eval_env(env_config)

                # [CRITICAL]: Call SETTING B Simulation
                res = run_lns2_simulation_setting_b(env, model_wrapper, device, args)
                
                # Save animation only if needed (slows down)
                # env.save_animation("lns2_setting_b.svg")

                res["trial"] = trial_idx + 1
                res["seed"] = seed
                res["map_name"] = map_name
                res["num_agents"] = num_agents
                
                scenario_results.append(res)
                all_results.append(res)
                
                if res["success"]:
                    succeeded_once = True
                    
                env.close()
                
            # Aggregate Scenario Stats
            if scenario_results:
                success_count = sum(1 for r in scenario_results if r['success'])
                total_run = len(scenario_results)
                sr = (success_count / total_run) * 100.0
                
                valid_makespans = [r['makespan'] for r in scenario_results if r['success']]
                avg_mk = np.mean(valid_makespans) if valid_makespans else float('nan')
                avg_time = np.mean([r['computation_time_sec'] for r in scenario_results])
                
                summary_results.append({
                    "map": map_name, "agents": num_agents,
                    "SR": sr, "Avg_Makespan": avg_mk, "Avg_Time": avg_time
                })
                
                tqdm.write(f"  Result (Sett.B): SR={sr:.1f}%, Time={avg_time:.2f}s")

            pbar.update(1)
            
    pbar.close()
    
    # Save Results
    stats = calculate_overall_stats(all_results, args)
    logging.info("\n--- Overall Statistics (Setting B) ---")
    print(json.dumps(stats, indent=2))
    
    time_str = time.strftime('%Y%m%d_%H%M%S')
    pd.DataFrame(all_results).to_csv(output_dir / f"lns2_setting_b_detailed_{time_str}.csv", index=False)
    pd.DataFrame(summary_results).to_csv(output_dir / f"lns2_setting_b_summary_{time_str}.csv", index=False)
    logging.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LNS2-RL Setting B Benchmark")
    
    parser.add_argument("--map_config_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="benchmark_output_lns2_setting_b")
    parser.add_argument("--map_pattern", type=str, default="*")
    parser.add_argument("--map_limit", type=int, default=None)
    parser.add_argument("--agent_counts", type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop_scenario_on_first_success", action='store_true')
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_steps", type=int, default=128)
    parser.add_argument("--lns_iters", type=int, default=30)
    parser.add_argument("--planning_time_limit", type=int, default=0.1)
    
    args = parser.parse_args()
    
    run_benchmark_main(args)