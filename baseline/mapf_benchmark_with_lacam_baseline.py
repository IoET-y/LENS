# mapf_benchmark_with_lacam.py
import torch
import numpy as np
from pathlib import Path
import time
import argparse
import yaml
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import logging

# Pogema Imports
try:
    from pogema import GridConfig
    from pogema_toolbox.registry import ToolboxRegistry
    from pogema_toolbox.create_env import Environment
    from create_env import create_eval_env #
except ImportError as e:
    logging.error(f"Error: Pogema or pogema_toolbox import failed: {e}")
    exit(1)

from mapf_simulation_with_lacam_Baseline import run_mapf_simulation #

SIM_OBS_RADIUS = 5 # Default observation radius for agents
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default values for command-line arguments
DEFAULT_MAX_EPISODE_STEPS = 256
DEFAULT_NUM_TRIALS = 2         # Max attempts for a scenario if stop_on_first_success is true
DEFAULT_OBS_RADIUS_ARG = SIM_OBS_RADIUS




# --- Helper functions load_and_register_maps & filter_maps_by_pattern ---
# These functions remain unchanged from your provided code.
# For brevity, they are omitted here but assumed to be present.
def load_and_register_maps(maps_dir):
    """ Loads and registers map configurations. """
    maps_path = Path(maps_dir); map_count = 0; registered_maps = set()
    if not maps_path.is_dir(): logging.warning(f"Maps directory not found: {maps_dir}"); return False, []
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
                 else: logging.debug(f"No new maps to register in {maps_file.name}")
            else: logging.warning(f"No maps found or empty file: {maps_file}")
        except Exception as e: logging.error(f"Failed to load/register {maps_file}: {e}")
    if map_count == 0: logging.warning(f"No new maps registered from {maps_dir}."); return False, []
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
        num_agents_set.add(r_dict.get('num_agents_at_start', r_dict.get('num_agents', 0))) # num_agents for scale
        maps_actually_tested_set.add(r_dict['map_name'])

        # For ISR
        num_at_start = r_dict.get('num_agents_at_start', 0)
        num_reached = r_dict.get('num_agents_reached_target', 0)
        if num_at_start > 0:
            isr_trial = num_reached / num_at_start
            all_trial_isrs.append(isr_trial)
        elif r_dict.get('success', False): # If overall success, all agents succeeded
             all_trial_isrs.append(1.0)
        # else: if no agent data and not overall success, ISR for this trial is effectively 0 or undefined.

        # For overall success based metrics
        if r_dict.get('success', False):
            if r_dict['success']:
                successful_overall_runs_count += 1
            if 'sum_of_costs' in r_dict:
                 successful_socs.append(r_dict['sum_of_costs'])
            if 'makespan' in r_dict:
                 successful_makespans.append(r_dict['makespan'])
        
        if 'computation_time_sec' in r_dict:
            all_durations.append(r_dict['computation_time_sec'])

    # 1. Task Scale Metrics
    stats['num_agents_configurations_tested'] = sorted(list(n for n in num_agents_set if n > 0))

    # 2. Task Success Metrics
    stats['avg_isr'] = np.mean(all_trial_isrs) if all_trial_isrs else 0.0
    total_runs_attempted = len(detailed_results_list)
    stats['overall_success_rate'] = (successful_overall_runs_count / args.map_limit) if args.map_limit > 0 else 0.0
    stats['failure_rate'] = 1.0 - stats['overall_success_rate']

    # 3. Path Quality Metrics
    stats['avg_soc_on_overall_success'] = np.mean(successful_socs) if successful_socs else float('nan')
    stats['std_soc_on_overall_success'] = np.std(successful_socs) if successful_socs else float('nan')

    # 4. Task Concurrency Metrics
    stats['avg_makespan_on_overall_success'] = np.mean(successful_makespans) if successful_makespans else float('nan')
    stats['std_makespan_on_overall_success'] = np.std(successful_makespans) if successful_makespans else float('nan')

    # 5. Computational Efficiency
    stats['avg_duration_s_per_run'] = np.mean(all_durations) if all_durations else float('nan')

    # 6. Experimental Coverage
    stats['maps_tested_count'] = len(maps_actually_tested_set)
    stats['runs_attempted_total'] = total_runs_attempted
    
    return stats


def run_benchmark_main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")

    maps_registered_ok, all_available_map_names = load_and_register_maps(args.map_config_dir)
    if not maps_registered_ok and not all_available_map_names: # Fallback if dir invalid
        logging.error("Map registration failed and no maps available. Exiting.")
        return
    
    maps_to_run_on = filter_maps_by_pattern(all_available_map_names, args.map_pattern)
    if not maps_to_run_on:
        logging.error(f"No maps found matching pattern '{args.map_pattern}' from registered maps. Exiting.")
        return

    if args.map_limit is not None and args.map_limit > 0 and args.map_limit < len(maps_to_run_on):
        maps_to_run_on = maps_to_run_on[:args.map_limit]
        logging.info(f"Limiting benchmark to the first {args.map_limit} matching maps.")
    
    logging.info(f"Will run benchmark on {len(maps_to_run_on)} maps and agent counts: {args.agent_counts}.")
    if args.stop_scenario_on_first_success:
        logging.info(f"For each scenario, will stop after the first successful trial (max {args.num_trials} attempts).")


    all_run_results_detailed = [] 
    summary_results_per_scenario = [] 

    total_scenarios_to_configure = len(maps_to_run_on) * len(args.agent_counts)
    current_scenario_config_count = 0
    # TQDM for scenarios
    scenario_pbar = tqdm(total=total_scenarios_to_configure, desc="Scenarios Progress")

    benchmark_overall_start_time = time.time()

    for map_name in maps_to_run_on:
        for num_agents in args.agent_counts:
            current_scenario_config_count +=1
            scenario_pbar.set_description(f"Scenario: {map_name}, A:{num_agents}")
            scenario_pbar.update(0) # Refresh display for description
            
            tqdm.write(f"\n--- Scenario {current_scenario_config_count}/{total_scenarios_to_configure}: Map='{map_name}', Agents={num_agents} ---")
            
            scenario_trial_results = []
            scenario_succeeded_once = False

            for trial_idx in range(args.num_trials):
                # tqdm.write(f"  Trial {trial_idx + 1}/{args.num_trials}")

                # Implement early exit for scenario if requested and already succeeded
                if args.stop_scenario_on_first_success and scenario_succeeded_once:
                    tqdm.write(f"    Skipping trial {trial_idx + 1} as scenario already succeeded and stop_on_first_success is True.")
                    # Add a placeholder or note if needed for consistent detailed results length
                    # For now, we just don't run and it won't appear in this scenario's trials.
                    # This means num_trials in summary might reflect actual runs for this scenario.
                    break 
                
                current_seed = (args.seed-1 + trial_idx) if args.seed is not None else int(time.time() * 1000) % (2**32) # Ensure different seed if base is None

                env_grid_config = Environment(
                    map_name=map_name, num_agents=num_agents, obs_radius=args.obs_radius,
                    observation_type="POMAPF", on_target="finish", collision_system="soft",
                    max_episode_steps=args.max_steps, seed=current_seed, with_animation=True
                )
                try:
                    env = create_eval_env(config=env_grid_config) #
                except Exception as e:
                    logging.error(f"Failed to create environment for map {map_name}, agents {num_agents}, trial {trial_idx+1}: {e}", exc_info=True)
                    sim_result_dict = {"success": False, "makespan": args.max_steps, "sum_of_costs": -1,
                                       "individual_costs": {}, "error": f"Env creation failed: {e}",
                                       "map_name": map_name, "num_agents": num_agents, "trial": trial_idx + 1,
                                       "num_agents_at_start": num_agents, "num_agents_reached_target":0, # For consistent stats
                                       "computation_time_sec": 0, "seed_used": current_seed}
                    scenario_trial_results.append(sim_result_dict)
                    all_run_results_detailed.append(sim_result_dict)
                    continue

                trial_start_time = time.time()
                sim_result_dict = {}
                try:
                    current_trial_viz_dir = output_dir / "cbs_input_visualizations" / f"{map_name}_A{num_agents}_T{trial_idx+1}"

                    sim_result_dict = run_mapf_simulation(
                        env, 100,max_episode_steps=512
                    ) #

                    
                    # Animation saving (path construction improved)
                    animation_dir = output_dir / "animations" 
                    animation_dir_fail = output_dir / "animationfail" 
                    animation_dir_fail.mkdir(parents=True, exist_ok=True)

                    animation_dir.mkdir(parents=True, exist_ok=True)
                    sanitized_map_name = map_name.replace("/", "_").replace("\\", "_") # Sanitize map name for filename
                    animation_file_name = f"{sanitized_map_name}_A{num_agents}_T{trial_idx+1}.svg"
                    animation_file_name_false = f"{sanitized_map_name}_A{num_agents}.svg"
                    animation_path = animation_dir / animation_file_name
                    animation_path_fail = animation_dir_fail / animation_file_name_false

                    # try:
                    #     env.save_animation(str(animation_path)) 
                    #     logging.info(f"Animation saved to {animation_path}")  
                    # except Exception as e_anim:
                    #     logging.error(f"Failed to save animation for {map_name}, trial {trial_idx+1}: {e_anim}")

                except Exception as e:
                    logging.error(f"Error during simulation for map {map_name}, A:{num_agents}, T:{trial_idx+1}: {e}", exc_info=True)
                    sim_result_dict = {"success": False, "makespan": args.max_steps, "sum_of_costs": -1, 
                                       "error": str(e), "num_agents_at_start": num_agents, "num_agents_reached_target":0}
                finally:
                    if 'env' in locals() and env is not None:
                        try: env.close()
                        except Exception as e_close: logging.error(f"Error closing env: {e_close}")
                
                trial_computation_time = time.time() - trial_start_time
                
                # Ensure consistent keys for all_run_results_detailed
                sim_result_dict["map_name"] = map_name
                sim_result_dict["num_agents"] = num_agents # This is the requested N
                sim_result_dict["trial"] = trial_idx + 1
                sim_result_dict["computation_time_sec"] = trial_computation_time
                sim_result_dict["seed_used"] = current_seed
                # Ensure num_agents_at_start and num_agents_reached_target are present
                if "num_agents_at_start" not in sim_result_dict:
                    sim_result_dict["num_agents_at_start"] = num_agents # Assume requested N if not returned
                if "num_agents_reached_target" not in sim_result_dict:
                    sim_result_dict["num_agents_reached_target"] = 0 # Assume 0 if error before it's set

                scenario_trial_results.append(sim_result_dict)
                all_run_results_detailed.append(sim_result_dict)
                
                if sim_result_dict.get('success', False):
                    scenario_succeeded_once = False # Mark success for early exit logic
                try:
                    if scenario_succeeded_once == True:
                        env.save_animation(str(animation_path)) 
                        logging.info(f"Animation saved to {animation_path}")  
                    else:
                        env.save_animation(str(animation_path_fail)) 
                        logging.info(f"Animation fail case saved to {animation_path_fail}")  
                        
                except Exception as e_anim:
                    logging.error(f"Failed to save animation for {map_name}, trial {trial_idx+1}: {e_anim}")
                #tqdm.write(f"    Trial {trial_idx + 1} Result: Success={sim_result_dict.get('success', False)}, "
                           # f"Makespan={sim_result_dict.get('makespan', -1)}, SoC={sim_result_dict.get('sum_of_costs', -1)}, "
                           # f"Time={trial_computation_time:.2f}s, ISR={sim_result_dict.get('num_agents_reached_target',0)}/{sim_result_dict.get('num_agents_at_start',num_agents)}")
                if sim_result_dict.get("error_summary") and not sim_result_dict.get('success', False) :
                    tqdm.write(f"      Error Summary: {sim_result_dict['error_summary']}")
                elif sim_result_dict.get("error") and not sim_result_dict.get('success', False):
                     tqdm.write(f"      Error: {sim_result_dict['error']}")
            
            # Aggregate results for this (map, num_agents) scenario
            if scenario_trial_results:
                # Number of actual trials run for this scenario (could be < args.num_trials due to early exit)
                actual_trials_for_scenario = len(scenario_trial_results)
                success_count_sc = sum(1 for r_sc in scenario_trial_results if r_sc.get('success', False))
                success_rate_sc = (success_count_sc / actual_trials_for_scenario) * 100.0 if actual_trials_for_scenario > 0 else 0.0
                
                valid_runs_sc = [r_sc for r_sc in scenario_trial_results if r_sc.get('success', False)]
                avg_makespan_sc = np.mean([r_sc['makespan'] for r_sc in valid_runs_sc]) if valid_runs_sc else float('nan')
                avg_soc_sc = np.mean([r_sc['sum_of_costs'] for r_sc in valid_runs_sc]) if valid_runs_sc else float('nan')
                
                comp_times_this_scenario = [r_sc.get('computation_time_sec') for r_sc in scenario_trial_results]
                avg_comp_time_sc = np.mean([t for t in comp_times_this_scenario if t is not None]) if comp_times_this_scenario else float('nan')
                
                num_errors_sc = sum(1 for r_sc in scenario_trial_results if not r_sc.get('success', False) and (r_sc.get('error') or r_sc.get('error_summary')))

                summary_results_per_scenario.append({
                    'map': map_name, 'agents': num_agents, 
                    'trials_run_for_scenario': actual_trials_for_scenario, # Renamed from num_trials
                    'seed_base': args.seed, 'success_rate_perc': success_rate_sc,
                    'avg_makespan_on_success': avg_makespan_sc, 'avg_sum_of_costs_on_success': avg_soc_sc,
                    'avg_trial_computation_time_sec': avg_comp_time_sc, 'num_errors_in_trials': num_errors_sc
                })
                #tqdm.write(f"  Scenario Summary (Map: {map_name}, Agents: {num_agents}, Ran {actual_trials_for_scenario} trial(s)): "
                           # f"SR: {success_rate_sc:.2f}%, Avg Makespan(S): {avg_makespan_sc:.2f}, "
                           # f"Avg SoC(S): {avg_soc_sc:.2f}, Avg Time: {avg_comp_time_sc:.2f}s, Errors: {num_errors_sc}")
            scenario_pbar.update(1) # scenario_pbar updated here
    scenario_pbar.close()


    benchmark_overall_duration = time.time() - benchmark_overall_start_time
    logging.info(f"\n\n--- Benchmark Completed in {benchmark_overall_duration:.2f} seconds ---")

    # --- Overall Statistics Calculation ---
    overall_stats_results = calculate_overall_stats(all_run_results_detailed, args)
    logging.info("\n--- Overall Benchmark Statistics ---")
    if "error" in overall_stats_results:
        logging.warning(overall_stats_results["error"])
    else:
        print(f"  Agent Configurations Tested  : {overall_stats_results['num_agents_configurations_tested']}")
        print(f"  Maps Tested Count            : {overall_stats_results['maps_tested_count']}")
        print(f"  Total Runs Attempted         : {overall_stats_results['runs_attempted_total']}")
        print(f"  Avg. Individual Success Rate : {overall_stats_results['avg_isr']:.3f}")
        print(f"  Overall System Success Rate  : {overall_stats_results['overall_success_rate']:.3f}")
        print(f"  Overall System Failure Rate  : {overall_stats_results['failure_rate']:.3f}")
        print(f"  Avg. Sum of Costs (SoC)      : {overall_stats_results['avg_soc_on_overall_success']:.2f} (on overall success)")
        print(f"  Std. Dev. of SoC             : {overall_stats_results['std_soc_on_overall_success']:.2f} (on overall success)")
        print(f"  Avg. Makespan                : {overall_stats_results['avg_makespan_on_overall_success']:.2f} (on overall success)")
        print(f"  Std. Dev. of Makespan        : {overall_stats_results['std_makespan_on_overall_success']:.2f} (on overall success)")
        print(f"  Avg. Computation Time (s)    : {overall_stats_results['avg_duration_s_per_run']:.3f} (per run)")


    if summary_results_per_scenario:
        df_summary = pd.DataFrame(summary_results_per_scenario)
        # Update column name for clarity due to early exit
        cols_summary = ['map', 'agents', 'trials_run_for_scenario', 'seed_base', 'success_rate_perc', 
                        'avg_makespan_on_success', 'avg_sum_of_costs_on_success',
                        'avg_trial_computation_time_sec', 'num_errors_in_trials']
        df_summary = df_summary.reindex(columns=cols_summary) 

        logging.info("\n--- Aggregated Per-Scenario Results ---")
        print(df_summary.to_string(index=False, float_format="%.2f"))

        time_str = time.strftime('%Y%m%d_%H%M%S')
        summary_csv_path = output_dir / f"benchmark_summary_per_scenario_{time_str}.csv"
        summary_json_path = output_dir / f"benchmark_full_data_{time_str}.json"
        try:
            df_summary.to_csv(summary_csv_path, index=False, float_format="%.3f")
            logging.info(f"\nPer-scenario summary results saved to {summary_csv_path}")
            
            benchmark_output_data = {
                'args': vars(args), 
                'overall_statistics': overall_stats_results, # Add new overall stats
                'summary_per_scenario': df_summary.to_dict(orient='records'),
                'detailed_trial_results': all_run_results_detailed 
            }
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, Path): return str(obj)
                    return super(NumpyEncoder, self).default(obj)
            with open(summary_json_path, 'w') as f:
                json.dump(benchmark_output_data, f, indent=2, cls=NumpyEncoder)
            logging.info(f"Full benchmark data (args, overall_stats, per-scenario_summary, details) saved to {summary_json_path}")

        except Exception as e:
            logging.error(f"Error saving results: {e}", exc_info=True)
    else:
        logging.warning("No scenarios were successfully summarized (summary_results_per_scenario is empty).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAPF Solver Benchmarking Script with Pogema")
    
    # Paths & Scenario Selection
    parser.add_argument("--map_config_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="benchmark_output_pogema_lacam")
    parser.add_argument("--map_pattern", type=str, default="*")
    parser.add_argument("--map_limit", type=int, default=None)
    parser.add_argument("--agent_counts", type=int, nargs='+', default=[16])
    parser.add_argument("--num_trials", type=int, default=DEFAULT_NUM_TRIALS, 
                        help="Max simulation runs per scenario. If --stop_on_first_success, loop breaks earlier.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Base random seed. Trial index is added. If None, uses time-based seed for variability.")
    # START NEW ARGUMENT FOR EARLY EXIT
    parser.add_argument("--stop_scenario_on_first_success", action="store_false",
                        help="For a given (map, agent_count) scenario, stop running further trials after the first success.")
    # END NEW ARGUMENT

    # Simulation Parameters
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--obs_radius", type=int, default=DEFAULT_OBS_RADIUS_ARG)


    # Execution & CBS strategy & pattering solving
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--verbose_simulation", action="store_true")

    args = parser.parse_args()



    if args.verbose_simulation:
        logging.getLogger().setLevel(logging.DEBUG) 

    run_benchmark_main(args)