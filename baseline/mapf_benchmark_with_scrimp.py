# mapf_benchmark_with_scrimp.py

import torch
import numpy as np
from pathlib import Path
import time
import argparse
import yaml
import json
import pandas as pd
from tqdm import tqdm
import logging

# --- Pogema Imports ---
try:
    from pogema import GridConfig
    from pogema_toolbox.registry import ToolboxRegistry
    # Assuming create_env.py is provided by the user and works as intended
    from pogema_toolbox.create_env import Environment

    from create_env import create_eval_env
except ImportError as e:
    logging.error(f"Error: Pogema or pogema_toolbox import failed: {e}")
    exit(1)

# --- SCRIMP Model and Simulation Imports ---
from scrimp_model.model import Model
from mapf_simulation_with_scrimp import run_scrimp_simulation
from scrimp_model.alg_parameters import EnvParameters

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default values for command-line arguments
DEFAULT_MAX_EPISODE_STEPS = 512
DEFAULT_NUM_TRIALS = 10

# --- Helper functions: load_and_register_maps & filter_maps_by_pattern ---
# (These are copied from your provided code for consistency)
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
            with open(maps_file, 'r') as f:
                maps_data = yaml.safe_load(f)
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
    if pattern == "*" or not pattern:
        return all_map_names
    if pattern.startswith("*") and pattern.endswith("*"):
        substring = pattern[1:-1]
        return [name for name in all_map_names if substring in name]
    elif pattern.startswith("*"):
        suffix = pattern[1:]
        return [name for name in all_map_names if name.endswith(suffix)]
    elif pattern.endswith("*"):
        prefix = pattern[:-1]
        return [name for name in all_map_names if name.startswith(prefix)]
    else: # Exact match
        return [name for name in all_map_names if name == pattern]

# --- Results Aggregation ---
def calculate_overall_stats(detailed_results_list):
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
        num_agents_set.add(r_dict.get('num_agents_at_start', r_dict.get('num_agents', 0)))
        maps_actually_tested_set.add(r_dict['map_name'])

        num_at_start = r_dict.get('num_agents_at_start', 0)
        num_reached = r_dict.get('num_agents_reached_target', 0)
        if num_at_start > 0:
            isr_trial = num_reached / num_at_start
            all_trial_isrs.append(isr_trial)
        elif r_dict.get('success', False):
            all_trial_isrs.append(1.0)
            
        if r_dict.get('success', False):
            successful_overall_runs_count += 1
            if 'sum_of_costs' in r_dict and r_dict['sum_of_costs'] is not None:
                successful_socs.append(r_dict['sum_of_costs'])
            if 'makespan' in r_dict and r_dict['makespan'] is not None:
                successful_makespans.append(r_dict['makespan'])
        
        if 'computation_time_sec' in r_dict:
            all_durations.append(r_dict['computation_time_sec'])

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

# --- Main Benchmark Runner ---
def run_benchmark_main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")

    # --- Load SCRIMP Model ---
    try:
        model = Model(0, device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.network.load_state_dict(checkpoint['model'])
        model.network.eval() # Set model to evaluation mode
        logging.info(f"Successfully loaded SCRIMP model from: {args.model_path}")
    except Exception as e:
        logging.error(f"Failed to load SCRIMP model: {e}", exc_info=True)
        return

    # --- Load and Filter Maps ---
    maps_registered_ok, all_available_map_names = load_and_register_maps(args.map_config_dir)
    if not maps_registered_ok and not all_available_map_names:
        logging.error("Map registration failed and no maps available. Exiting.")
        return
    
    maps_to_run_on = filter_maps_by_pattern(all_available_map_names, args.map_pattern)
    if not maps_to_run_on:
        logging.error(f"No maps found matching pattern '{args.map_pattern}'. Exiting.")
        return

    if args.map_limit is not None and args.map_limit > 0:
        maps_to_run_on = maps_to_run_on[:args.map_limit]
        logging.info(f"Limiting benchmark to the first {args.map_limit} matching maps.")
    
    logging.info(f"Will run benchmark on {len(maps_to_run_on)} maps and agent counts: {args.agent_counts}.")

    # --- Main Loop ---
    all_run_results_detailed = []
    summary_results_per_scenario = []
    total_scenarios = len(maps_to_run_on) * len(args.agent_counts)
    benchmark_start_time = time.time()

    with tqdm(total=total_scenarios, desc="Scenarios") as scenario_pbar:
        for map_name in maps_to_run_on:
            for num_agents in args.agent_counts:
                scenario_pbar.set_description(f"Scenario: {map_name}, A:{num_agents}")
                tqdm.write(f"\n--- Running Scenario: Map='{map_name}', Agents={num_agents} ---")
                
                scenario_trial_results = []
                for trial_idx in range(args.num_trials):
                    current_seed = (args.seed + trial_idx) if args.seed is not None else int(time.time() * 1000)
                    
                    env_grid_config = Environment(
                    map_name=map_name, num_agents=num_agents, obs_radius=args.obs_radius,
                    observation_type="MAPF", on_target="finish", collision_system="soft",
                    max_episode_steps=args.max_steps, seed=current_seed, with_animation=True
                    )
                    try:
                        env = create_eval_env(config=env_grid_config)
                    except Exception as e:
                        logging.error(f"Failed to create env for trial {trial_idx+1}: {e}", exc_info=True)
                        continue
                    
                    trial_start_time = time.time()
                    try:
                        sim_result_dict = run_scrimp_simulation(env, model, device, args.max_steps)
                    except Exception as e:
                        logging.error(f"Error during SCRIMP simulation for trial {trial_idx+1}: {e}", exc_info=True)
                        sim_result_dict = {"success": False, "error": str(e)}
                    finally:
                        env.close()

                    trial_duration = time.time() - trial_start_time
                    
                    sim_result_dict.update({
                        "map_name": map_name, "num_agents": num_agents, "trial": trial_idx + 1,
                        "computation_time_sec": trial_duration, "seed_used": current_seed,
                        "num_agents_at_start": num_agents
                    })
                    
                    all_run_results_detailed.append(sim_result_dict)
                    scenario_trial_results.append(sim_result_dict)
                    
                    tqdm.write(
                        f"  Trial {trial_idx+1}/{args.num_trials}: "
                        f"Success={sim_result_dict.get('success', False)}, "
                        f"Makespan={sim_result_dict.get('makespan', -1)}, "
                        f"SoC={sim_result_dict.get('sum_of_costs', -1)}, "
                        f"ISR={sim_result_dict.get('num_agents_reached_target',0)}/{num_agents}, "
                        f"Time={trial_duration:.2f}s"
                    )
                
                if scenario_trial_results:
                    success_count = sum(1 for r in scenario_trial_results if r.get('success', False))
                    valid_runs = [r for r in scenario_trial_results if r.get('success', False)]
                    summary_results_per_scenario.append({
                        'map': map_name, 'agents': num_agents, 'num_trials': len(scenario_trial_results),
                        'success_rate_perc': (success_count / len(scenario_trial_results)) * 100.0,
                        'avg_makespan_on_success': np.mean([r['makespan'] for r in valid_runs]) if valid_runs else float('nan'),
                        'avg_sum_of_costs_on_success': np.mean([r['sum_of_costs'] for r in valid_runs]) if valid_runs else float('nan'),
                        'avg_trial_computation_time_sec': np.mean([r['computation_time_sec'] for r in scenario_trial_results])
                    })
                
                scenario_pbar.update(1)

    benchmark_duration = time.time() - benchmark_start_time
    logging.info(f"\n--- Benchmark Completed in {benchmark_duration:.2f} seconds ---")

    # --- Final Reporting ---
    overall_stats = calculate_overall_stats(all_run_results_detailed)
    logging.info("\n--- Overall Benchmark Statistics (SCRIMP) ---")
    print(yaml.dump(overall_stats, sort_keys=False, indent=2))
    # --- Final Reporting (UPDATED) ---
    overall_stats_results = calculate_overall_stats(all_run_results_detailed)
    logging.info("\n--- Overall Benchmark Statistics (SCRIMP) ---")
    if "error" in overall_stats_results:
        logging.warning(overall_stats_results["error"])
    else:
        print(f"  Agent Configurations Tested  : {overall_stats_results['num_agents_configurations_tested']}")
        print(f"  Maps Tested Count            : {overall_stats_results['maps_tested_count']}")
        print(f"  Total Runs Attempted         : {overall_stats_results['runs_attempted_total']}")
        print(f"  Overall System Success Rate  : {overall_stats_results['overall_success_rate']:.3f}")
        print(f"  Avg. Individual Success Rate : {overall_stats_results['avg_isr']:.3f}")
        print(f"  Avg. Sum of Costs (SoC)      : {overall_stats_results['avg_soc_on_overall_success']:.2f} (on overall success)")
        print(f"  Avg. Computation Time (s)    : {overall_stats_results['avg_duration_s_per_run']:.3f} (per run)")
        print(f"  Avg. Makespan                : {overall_stats_results['avg_makespan_on_overall_success']:.2f} (on overall success)")

    # if summary_results_per_scenario:
    #     df_summary = pd.DataFrame(summary_results_per_scenario)
        
    #     print(df_summary.to_string(index=False, float_format="%.2f"))

    #     time_str = time.strftime('%Y%m%d_%H%M%S')
    #     summary_csv_path = output_dir / f"scrimp_benchmark_summary_{time_str}.csv"
    #     full_json_path = output_dir / f"scrimp_benchmark_full_{time_str}.json"
        
    #     try:
    #         df_summary.to_csv(summary_csv_path, index=False, float_format="%.3f")
    #         logging.info(f"\nPer-scenario summary saved to {summary_csv_path}")

    #         output_data = {
    #             'args': vars(args), 'overall_statistics': overall_stats,
    #             'summary_per_scenario': df_summary.to_dict(orient='records'),
    #             'detailed_trial_results': all_run_results_detailed
    #         }
    #         class NumpyEncoder(json.JSONEncoder):
    #             def default(self, obj):
    #                 if isinstance(obj, np.integer): return int(obj)
    #                 if isinstance(obj, np.floating): return float(obj)
    #                 if isinstance(obj, np.ndarray): return obj.tolist()
    #                 if isinstance(obj, Path): return str(obj)
    #                 return super(NumpyEncoder, self).default(obj)
    #         with open(full_json_path, 'w') as f:
    #             json.dump(output_data, f, indent=2, cls=NumpyEncoder)
    #         logging.info(f"Full benchmark data saved to {full_json_path}")
    #     except Exception as e:
    #         logging.error(f"Error saving results: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SCRIMP model on Pogema MAPF benchmark.")
    
    # Paths & Model
    parser.add_argument("--model_path", type=str, required=True, help="Path to SCRIMP's net_checkpoint.pkl file.")
    parser.add_argument("--map_config_dir", type=str, required=True, help="Directory containing Pogema's maps.yaml files.")
    parser.add_argument("--output_dir", type=str, default="benchmark_output_pogema_scrimp", help="Directory to save results and logs.")
    
    # Scenario Selection
    parser.add_argument("--map_pattern", type=str, default="*", help="Pattern to match map names (e.g., 'scen-random*').")
    parser.add_argument("--map_limit", type=int, default=None, help="Limit the number of maps to test.")
    parser.add_argument("--agent_counts", type=int, nargs='+', default=[8, 16, 32], help="List of agent counts to test.")
    parser.add_argument("--num_trials", type=int, default=DEFAULT_NUM_TRIALS, help="Number of simulation runs per scenario.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility. Trial index is added to this.")

    # Simulation & Execution Parameters
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS, help="Max steps per episode.")
    parser.add_argument("--obs_radius", type=int, default=EnvParameters.FOV_SIZE // 2, help=f"Pogema observation radius. Should match SCRIMP's training setting, which is {EnvParameters.FOV_SIZE // 2}.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for PyTorch inference.")
    
    args = parser.parse_args()
    
    if args.obs_radius != EnvParameters.FOV_SIZE // 2:
        logging.warning(f"Provided --obs_radius ({args.obs_radius}) does not match SCRIMP's default expected radius ({EnvParameters.FOV_SIZE // 2}). This may lead to poor performance.")

    run_benchmark_main(args)