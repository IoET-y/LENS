# mapf_gpt_test.py
import torch
import yaml
import numpy as np
from pathlib import Path
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

# --- POGEMA and Expert Imports ---
try:
    from pogema_toolbox.create_env import Environment
    from pogema_toolbox.registry import ToolboxRegistry
    from create_env import create_eval_env
    from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
    from mapf_utils import simulate_complete_paths
except ImportError as e:
    logging.error(f"Failed to import POGEMA or related modules: {e}")
    exit(1)

# --- Helper Function ---
def load_and_register_maps(maps_dir):
    """ Loads and registers map configurations. """
    maps_path = Path(maps_dir)
    map_count = 0
    registered_maps = set()
    if not maps_path.is_dir():
        logging.warning(f"Maps directory not found: {maps_dir}")
        return False
    for maps_file in maps_path.rglob('maps.yaml'):
        try:
            with open(maps_file, 'r') as f:
                maps = yaml.safe_load(f)
            if maps:
                new_maps = {k: v for k, v in maps.items() if k not in registered_maps}
                if new_maps:
                    ToolboxRegistry.register_maps(new_maps)
                    logging.info(f"Registered {len(new_maps)} new maps from: {maps_file}")
                    registered_maps.update(new_maps.keys())
                    map_count += len(new_maps)
                else:
                    logging.debug(f"No new maps to register in {maps_file}")
            else:
                logging.warning(f"No maps found or empty file: {maps_file}")
        except Exception as e:
            logging.error(f"Failed to load/register {maps_file}: {e}")
    if map_count == 0:
        logging.warning(f"No new maps registered from {maps_dir}.")
        return False
    logging.info(f"Total new maps registered: {map_count}")
    return True

# --- Main MAPF-GPT Performance Testing Function ---
def test_mapf_gpt_performance(
    map_names, grid_search_agents,
    max_episode_steps=1024, random_seed=42, obs_radius=5,
    gpt_model_path='weights/model-85M.pt'
):
    """
    Tests the MAPF-GPT model across a grid of scenarios and reports performance metrics.
    """
    logging.info(f"--- Starting MAPF-GPT Performance Test ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Initialize Expert Agent ---
    try:
        gpt_cfg = MAPFGPTInferenceConfig(path_to_weights=gpt_model_path, device=device)
        inference_agent = MAPFGPTInference(gpt_cfg)
        logging.info(f"Initialized MAPF-GPT expert from {gpt_model_path} on device '{device}'")
    except Exception as e:
        logging.error(f"Failed to initialize MAPF-GPT expert: {e}. Cannot proceed.")
        return

    # --- Results Storage ---
    results = []
    total_scenarios = len(map_names) * len(grid_search_agents)
    processed_scenarios = 0

    for num_agents_requested in grid_search_agents:
        for i in range(8):
            for map_idx, map_name in enumerate(map_names):
                instance_id = f"{map_name}_agents{num_agents_requested}"
                processed_scenarios += 1
                logging.info(f"--- Testing scenario [{processed_scenarios}/{total_scenarios}]: {instance_id} ---")
                
                run_metrics = {
                    'map_name': map_name,
                    'agents_requested': num_agents_requested,
                    'overall_success': False,
                    'individual_success_rate': 0.0,
                    'soc': np.nan,
                    'makespan': np.nan,
                    'computation_time': np.nan
                }
    
                env = None
                try:
                    # Use a consistent seed for each specific scenario instance
                    instance_seed = random_seed + map_idx
    
                    # --- Environment Setup ---
                    env_cfg = Environment(
                        map_name=map_name, num_agents=num_agents_requested, obs_radius=obs_radius,
                        observation_type="MAPF", seed=instance_seed, max_episode_steps=max_episode_steps,
                        on_target="nothing", # Use 'nothing' for simulation to get full paths
                        collision_system="soft"
                    )
                    env = create_eval_env(env_cfg)
                    _, _ = env.reset(seed=instance_seed)
                    actual_num_agents = env.grid_config.num_agents
                    
                    if actual_num_agents == 0:
                        logging.warning(f"Scenario {instance_id} has 0 agents. Skipping.")
                        results.append(run_metrics)
                        continue
    
                    # --- Run Simulation ---
                    inference_agent.reset_states()
                    start_time = time.time()
                    
                    # simulate_complete_paths runs the expert until completion or timeout
                    starts, goals, actions, overall_success = simulate_complete_paths(
                        env, inference_agent, max_steps=max_episode_steps
                    )
                    
                    run_metrics['computation_time'] = time.time() - start_time
                    run_metrics['overall_success'] = overall_success
    
                    # --- Calculate Metrics for the Run ---
                    final_pos = env.get_agents_xy()
                    num_successful_agents = sum(1 for i in range(actual_num_agents) if tuple(final_pos[i]) == tuple(goals[i]))
                    run_metrics['individual_success_rate'] = num_successful_agents / actual_num_agents if actual_num_agents > 0 else 0
    
                    if overall_success:
                        # SoC and Makespan are only valid if the whole system succeeds
                        path_lengths = [len(p) for p in actions if p]
                        if path_lengths:
                            run_metrics['soc'] = sum(path_lengths)
                            run_metrics['makespan'] = max(path_lengths)
                        else: # All agents started at their goals
                            run_metrics['soc'] = 0
                            run_metrics['makespan'] = 0
                        logging.info(f"SCENARIO SUCCESS: SoC={run_metrics['soc']}, Makespan={run_metrics['makespan']}, Time={run_metrics['computation_time']:.3f}s")
                    else:
                        logging.warning(f"SCENARIO FAILED. Individual Success: {run_metrics['individual_success_rate']:.2%}, Time={run_metrics['computation_time']:.3f}s")
    
                    results.append(run_metrics)
    
                except Exception as e:
                    logging.error(f"Failed to process instance {instance_id}. Error: {e}", exc_info=True)
                    results.append(run_metrics) # Append failed run with default values
                finally:
                    if env is not None:
                        env.close()

    # --- Aggregate and Print Final Results ---
    logging.info("--- MAPF-GPT Performance Test Summary ---")
    if not results:
        logging.warning("No scenarios were tested.")
        return

    # Extract data for analysis
    individual_success_rates = np.array([r['individual_success_rate'] for r in results])
    overall_success_flags = np.array([r['overall_success'] for r in results])
    computation_times = np.array([r['computation_time'] for r in results])

    # Filter for successful runs for SoC and Makespan calculation
    successful_runs = [r for r in results if r['overall_success']]
    soc_values = np.array([r['soc'] for r in successful_runs if not np.isnan(r['soc'])])
    makespan_values = np.array([r['makespan'] for r in successful_runs if not np.isnan(r['makespan'])])

    # Calculate final metrics
    avg_ind_success = np.mean(individual_success_rates)
    overall_sys_success = np.mean(overall_success_flags)
    avg_comp_time = np.mean(computation_times)
    
    avg_soc = np.mean(soc_values) if len(soc_values) > 0 else 0
    std_soc = np.std(soc_values) if len(soc_values) > 0 else 0
    
    avg_makespan = np.mean(makespan_values) if len(makespan_values) > 0 else 0
    std_makespan = np.std(makespan_values) if len(makespan_values) > 0 else 0

    # Print formatted results
    print("\n" + "="*50)
    print(" " * 15 + "PERFORMANCE METRICS")
    print("="*50)
    print(f"  Avg. Individual Success Rate : {avg_ind_success:.3f}")
    print(f"  Overall System Success Rate  : {overall_sys_success:.3f}")
    print(f"  Avg. Sum of Costs (SoC)      : {avg_soc:.2f} (on overall success)")
    print(f"  Std. Dev. of SoC             : {std_soc:.2f} (on overall success)")
    print(f"  Avg. Makespan                : {avg_makespan:.2f} (on overall success)")
    print(f"  Std. Dev. of Makespan        : {std_makespan:.2f} (on overall success)")
    print(f"  Avg. Computation Time (s)    : {avg_comp_time:.3f} (per run)")
    print("="*50)
    print(f"  Total Scenarios Tested: {len(results)}")
    print(f"  Overall Success Count : {len(successful_runs)}")
    print("="*50 + "\n")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run performance tests for a MAPF-GPT model.")
    parser.add_argument('--map-config-dir', type=str, required=True, help="Directory containing Pogema map YAML configurations.")
    parser.add_argument('--map-pattern', type=str, required=True, help="Pattern for map names (e.g., 'maze-seed-???')")
    parser.add_argument('--map-indices', type=str, default="0:10", help="Range of map indices to process (e.g., '0:50')")
    parser.add_argument('--agent-counts', type=str, default="8,16,32,64", help="Comma-separated list of agent counts (e.g., '8,16,32')")
    parser.add_argument('--gpt-weights', type=str, default='weights/model-6M.pt', help="Path to MAPF-GPT model weights.")
    parser.add_argument('--max-steps', type=int, default=1024, help="Maximum steps per episode during simulation.")
    parser.add_argument('--obs-radius', type=int, default=5, help="Agent observation radius.")
    parser.add_argument('--seed', type=int, default=42, help="Base random seed for environment generation.")
    
    args = parser.parse_args()

    # Override global seed if provided in args
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    logging.info("--- Starting MAPF-GPT Test Script ---")
    logging.info(f"Runtime Arguments: {args}")

    # Parse arguments
    agent_counts_list = [int(x) for x in args.agent_counts.split(',') if x]
    try:
        map_range = args.map_indices.split(':')
        start_idx, end_idx = map(int, map_range)
        if start_idx >= end_idx: raise ValueError("Start index must be less than end index")
    except (ValueError, IndexError):
        logging.error("Invalid format for --map-indices. Use 'start:end' (e.g., '0:10').")
        exit(1)

    # Generate map names from pattern and range
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
        logging.error("Cannot generate map names. Use '?' in map-pattern for ranges, or specify a single map with indices 0:1.")
        exit(1)

    if not map_names_list:
        logging.error(f"No map names generated for pattern '{args.map_pattern}' and indices {start_idx}:{end_idx}.")
        exit(1)
    
    logging.info(f"Targeting {len(map_names_list)} maps (Indices {start_idx}-{end_idx-1}), Agent Counts: {agent_counts_list}")

    # Load and register maps from the specified directory
    if not load_and_register_maps(args.map_config_dir):
        logging.warning("Map registration failed or found no new maps. Continuing with any pre-registered maps.")

    # Run the performance test
    test_mapf_gpt_performance(
        map_names=map_names_list,
        grid_search_agents=agent_counts_list,
        max_episode_steps=args.max_steps,
        random_seed=seed,
        obs_radius=args.obs_radius,
        gpt_model_path=args.gpt_weights
    )

    logging.info("--- Script Finished ---")