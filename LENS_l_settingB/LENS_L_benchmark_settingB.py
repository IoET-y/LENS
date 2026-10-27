# LENS_L_benchmark_settingB.py
import argparse
import json
import logging
from pathlib import Path
import time
import yaml

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from pogema import GridConfig
    from pogema_toolbox.create_env import Environment
    from pogema_toolbox.registry import ToolboxRegistry
    from create_env import create_eval_env
except ImportError as e:
    logging.error(f"Error: Pogema or pogema_toolbox import failed: {e}")
    raise SystemExit(1)

from LENS_l import run_mapf_simulation
from unet_model_new import UNetPotentialField

SIM_OBS_RADIUS = 5
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_and_register_maps(maps_dir):
    """Loads and registers map configurations."""
    maps_path = Path(maps_dir)
    map_count = 0
    registered_maps = set()
    if not maps_path.is_dir():
        logging.warning(f"Maps directory not found: {maps_dir}")
        return False, []
    all_map_names = []
    for maps_file in maps_path.rglob("maps.yaml"):
        try:
            with open(maps_file, "r") as f:
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
    """Filters map names based on a simple wildcard pattern."""
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
    else:
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
        num_agents_set.add(r_dict.get("num_agents_at_start", r_dict.get("num_agents", 0)))
        maps_actually_tested_set.add(r_dict["map_name"])

        num_at_start = r_dict.get("num_agents_at_start", 0)
        num_reached = r_dict.get("num_agents_reached_target", 0)
        if num_at_start > 0:
            all_trial_isrs.append(num_reached / num_at_start)
        elif r_dict.get("success", False):
            all_trial_isrs.append(1.0)

        if r_dict.get("success", False):
            successful_overall_runs_count += 1
            if "sum_of_costs" in r_dict:
                successful_socs.append(r_dict["sum_of_costs"])
            if "makespan" in r_dict:
                successful_makespans.append(r_dict["makespan"])

        if "computation_time_sec" in r_dict:
            all_durations.append(r_dict["computation_time_sec"])

    stats["num_agents_configurations_tested"] = sorted(list(n for n in num_agents_set if n > 0))
    stats["avg_isr"] = float(np.mean(all_trial_isrs)) if all_trial_isrs else 0.0

    total_runs_attempted = len(detailed_results_list)
    stats["overall_success_rate"] = (
        (successful_overall_runs_count / total_runs_attempted) if total_runs_attempted > 0 else 0.0
    )
    stats["failure_rate"] = 1.0 - stats["overall_success_rate"]

    stats["avg_soc_on_overall_success"] = float(np.mean(successful_socs)) if successful_socs else float("nan")
    stats["std_soc_on_overall_success"] = float(np.std(successful_socs)) if successful_socs else float("nan")

    stats["avg_makespan_on_overall_success"] = (
        float(np.mean(successful_makespans)) if successful_makespans else float("nan")
    )
    stats["std_makespan_on_overall_success"] = (
        float(np.std(successful_makespans)) if successful_makespans else float("nan")
    )

    stats["avg_duration_s_per_run"] = float(np.mean(all_durations)) if all_durations else float("nan")
    stats["maps_tested_count"] = len(maps_actually_tested_set)
    stats["runs_attempted_total"] = total_runs_attempted

    return stats


def _load_unet_checkpoint(unet_path: Path, args, device: torch.device) -> UNetPotentialField:
    ckpt = torch.load(unet_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    init_chn = args.unet_init_channels
    if "inc.double_conv.0.weight" in state_dict:
        ckpt_init_chn = int(state_dict["inc.double_conv.0.weight"].shape[0])
        if ckpt_init_chn != init_chn:
            logging.info(
                f"Auto-detected U-Net base channels={ckpt_init_chn} from checkpoint (overriding {init_chn})."
            )
            init_chn = ckpt_init_chn

    bilinear = args.unet_bilinear
    if "up1.up.weight" in state_dict and bilinear:
        bilinear = False
    elif "up1.up.weight" not in state_dict and not bilinear and any(k.startswith("up1.") for k in state_dict):
        bilinear = True

    model = UNetPotentialField(
        n_spatial_channels=args.unet_spatial_channels,
        n_non_spatial_features=args.unet_non_spatial_features,
        n_classes_out=1,
        bilinear_upsample=bilinear,
        init_channel=init_chn,
    )
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logging.warning(f"U-Net loading: Missing keys: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"U-Net loading: Unexpected keys: {unexpected_keys}")
    model.to(device).eval()
    logging.info(f"U-Net model loaded from {unet_path} ({init_chn} base channels) -> {device}")
    return model


def run_benchmark_main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logging.info(f"Using device: {device} | Preset: {args.preset}")

    maps_registered_ok, all_available_map_names = load_and_register_maps(args.map_config_dir)
    if not maps_registered_ok and not all_available_map_names:
        logging.error("Map registration failed and no maps available. Exiting.")
        return

    maps_to_run_on = filter_maps_by_pattern(all_available_map_names, args.map_pattern)
    if not maps_to_run_on:
        logging.error(f"No maps found matching pattern '{args.map_pattern}' from registered maps. Exiting.")
        return

    if args.map_limit is not None and 0 < args.map_limit < len(maps_to_run_on):
        maps_to_run_on = maps_to_run_on[: args.map_limit]
        logging.info(f"Limiting benchmark to the first {args.map_limit} matching maps.")

    logging.info(f"Will run benchmark on {len(maps_to_run_on)} maps and agent counts: {args.agent_counts}.")

    unet_path = Path(args.model_path)
    try:
        loaded_unet_model = _load_unet_checkpoint(unet_path, args, device)
    except FileNotFoundError:
        logging.error(f"U-Net model file not found: {unet_path}. Exiting.")
        return
    except Exception as e:
        logging.error(f"Error loading U-Net model from {unet_path}: {e}. Exiting.", exc_info=True)
        return

    all_run_results_detailed = []
    summary_results_per_scenario = []

    total_scenarios_to_configure = len(maps_to_run_on) * len(args.agent_counts)
    current_scenario_config_count = 0
    scenario_pbar = tqdm(total=total_scenarios_to_configure, desc="Scenarios Progress")
    benchmark_overall_start_time = time.time()

    for map_name in maps_to_run_on:
        for num_agents in args.agent_counts:
            current_scenario_config_count += 1
            scenario_pbar.set_description(f"Scenario: {map_name}, A:{num_agents}")
            scenario_pbar.update(0)

            tqdm.write(
                f"\n--- Scenario {current_scenario_config_count}/{total_scenarios_to_configure}: "
                f"Map='{map_name}', Agents={num_agents} ---"
            )

            scenario_trial_results = []
            scenario_succeeded_once = False

            for trial_idx in range(args.num_trials):
                tqdm.write(f"  Trial {trial_idx + 1}/{args.num_trials}")

                if args.stop_scenario_on_first_success and scenario_succeeded_once:
                    tqdm.write(
                        f"    Skipping trial {trial_idx + 1} as scenario already succeeded."
                    )
                    break

                current_seed = (args.seed + trial_idx) if args.seed is not None else 42

                env_grid_config = Environment(
                    map_name=map_name,
                    num_agents=num_agents,
                    obs_radius=args.obs_radius,
                    observation_type="POMAPF",
                    on_target="finish",
                    collision_system="soft",
                    max_episode_steps=args.max_steps,
                    seed=current_seed,
                    with_animation=args.save_animations,
                )
                try:
                    env = create_eval_env(config=env_grid_config)
                except Exception as e:
                    logging.error(
                        f"Failed to create environment for map {map_name}, agents {num_agents}, trial {trial_idx + 1}: {e}",
                        exc_info=True,
                    )
                    sim_result_dict = {
                        "success": False,
                        "makespan": args.max_steps,
                        "sum_of_costs": -1,
                        "individual_costs": {},
                        "error": f"Env creation failed: {e}",
                        "map_name": map_name,
                        "num_agents": num_agents,
                        "trial": trial_idx + 1,
                        "num_agents_at_start": num_agents,
                        "num_agents_reached_target": 0,
                        "computation_time_sec": 0,
                        "seed_used": current_seed,
                    }
                    scenario_trial_results.append(sim_result_dict)
                    all_run_results_detailed.append(sim_result_dict)
                    continue

                trial_start_time = time.time()
                sim_result_dict = {}
                try:
                    current_trial_viz_dir = (
                        output_dir / "cbs_input_visualizations" / f"{map_name}_A{num_agents}_T{trial_idx + 1}"
                    )

                    sim_result_dict = run_mapf_simulation(
                        env,
                        loaded_unet_model,
                        device,
                        max_episode_steps=args.max_steps,
                        preset=args.preset,
                        local_plan_horizon=args.plan_horizon,
                        n_exec_steps=args.n_exec_steps,
                        cbs_max_iterations=args.cbs_iterations,
                        verbose=args.verbose_simulation,
                        visualization_output_dir=current_trial_viz_dir,
                        pattering_history_len=args.pattering_hist_len,
                        pattering_unique_pos_threshold=args.pattering_unique_pos,
                        pattering_astar_bonus_horizon=args.pattering_astar_bonus,
                        max_consecutive_cbs_fails_for_intervention=args.max_cbs_fails_intervention,
                        sub_cbs_max_iterations_multiplier=args.sub_cbs_iter_multiplier,
                        cbs_time_limit_s=args.cbs_time_limit_s,
                        large_group_threshold=args.large_group_threshold,
                    )

                    if args.save_animations:
                        anim_subdir = "animations" if sim_result_dict.get("success", False) else "animationfail"
                        anim_dir = output_dir / anim_subdir
                        anim_dir.mkdir(parents=True, exist_ok=True)
                        sanitized_map_name = map_name.replace("/", "_").replace("\\", "_")
                        anim_path = anim_dir / f"{sanitized_map_name}_A{num_agents}_T{trial_idx + 1}.svg"
                        try:
                            env.save_animation(str(anim_path))
                            logging.info(f"Animation saved to {anim_path}")
                        except Exception as e_anim:
                            logging.error(f"Failed to save animation: {e_anim}")

                except Exception as e:
                    logging.error(
                        f"Error during simulation for map {map_name}, A:{num_agents}, T:{trial_idx + 1}: {e}",
                        exc_info=True,
                    )
                    sim_result_dict = {
                        "success": False,
                        "makespan": args.max_steps,
                        "sum_of_costs": -1,
                        "error": str(e),
                        "num_agents_at_start": num_agents,
                        "num_agents_reached_target": 0,
                    }
                finally:
                    if "env" in locals() and env is not None:
                        try:
                            env.close()
                        except Exception as e_close:
                            logging.error(f"Error closing env: {e_close}")

                trial_computation_time = time.time() - trial_start_time

                sim_result_dict["map_name"] = map_name
                sim_result_dict["num_agents"] = num_agents
                sim_result_dict["trial"] = trial_idx + 1
                sim_result_dict["computation_time_sec"] = trial_computation_time
                sim_result_dict["seed_used"] = current_seed
                sim_result_dict.setdefault("num_agents_at_start", num_agents)
                sim_result_dict.setdefault("num_agents_reached_target", 0)

                scenario_trial_results.append(sim_result_dict)
                all_run_results_detailed.append(sim_result_dict)

                if sim_result_dict.get("success", False):
                    scenario_succeeded_once = True

                tqdm.write(
                    f"    Trial {trial_idx + 1} Result: Success={sim_result_dict.get('success', False)}, "
                    f"Makespan={sim_result_dict.get('makespan', -1)}, SoC={sim_result_dict.get('sum_of_costs', -1)}, "
                    f"Time={trial_computation_time:.2f}s, "
                    f"ISR={sim_result_dict.get('num_agents_reached_target', 0)}/{sim_result_dict.get('num_agents_at_start', num_agents)}"
                )
                if sim_result_dict.get("error_summary") and not sim_result_dict.get("success", False):
                    tqdm.write(f"      Error Summary: {sim_result_dict['error_summary']}")
                elif sim_result_dict.get("error") and not sim_result_dict.get("success", False):
                    tqdm.write(f"      Error: {sim_result_dict['error']}")

            if scenario_trial_results:
                actual_trials_for_scenario = len(scenario_trial_results)
                success_count_sc = sum(1 for r_sc in scenario_trial_results if r_sc.get("success", False))
                success_rate_sc = (
                    (success_count_sc / actual_trials_for_scenario) * 100.0
                    if actual_trials_for_scenario > 0
                    else 0.0
                )

                valid_runs_sc = [r_sc for r_sc in scenario_trial_results if r_sc.get("success", False)]
                avg_makespan_sc = (
                    float(np.mean([r_sc["makespan"] for r_sc in valid_runs_sc]))
                    if valid_runs_sc
                    else float("nan")
                )
                avg_soc_sc = (
                    float(np.mean([r_sc["sum_of_costs"] for r_sc in valid_runs_sc]))
                    if valid_runs_sc
                    else float("nan")
                )

                comp_times_this_scenario = [
                    r_sc.get("computation_time_sec") for r_sc in scenario_trial_results
                ]
                avg_comp_time_sc = (
                    float(np.mean([t for t in comp_times_this_scenario if t is not None]))
                    if comp_times_this_scenario
                    else float("nan")
                )

                num_errors_sc = sum(
                    1
                    for r_sc in scenario_trial_results
                    if not r_sc.get("success", False) and (r_sc.get("error") or r_sc.get("error_summary"))
                )

                summary_results_per_scenario.append(
                    {
                        "map": map_name,
                        "agents": num_agents,
                        "trials_run_for_scenario": actual_trials_for_scenario,
                        "seed_base": args.seed,
                        "success_rate_perc": success_rate_sc,
                        "avg_makespan_on_success": avg_makespan_sc,
                        "avg_sum_of_costs_on_success": avg_soc_sc,
                        "avg_trial_computation_time_sec": avg_comp_time_sc,
                        "num_errors_in_trials": num_errors_sc,
                        "plan_horizon_k": args.plan_horizon,
                        "exec_steps_n": args.n_exec_steps,
                        "cbs_time_limit_s": args.cbs_time_limit_s,
                        "cbs_iterations": args.cbs_iterations,
                    }
                )
                tqdm.write(
                    f"  Scenario Summary (Map: {map_name}, Agents: {num_agents}, Ran {actual_trials_for_scenario} trial(s)): "
                    f"SR: {success_rate_sc:.2f}%, Avg Makespan(S): {avg_makespan_sc:.2f}, "
                    f"Avg SoC(S): {avg_soc_sc:.2f}, Avg Time: {avg_comp_time_sc:.2f}s, Errors: {num_errors_sc}"
                )
            scenario_pbar.update(1)
    scenario_pbar.close()

    benchmark_overall_duration = time.time() - benchmark_overall_start_time
    logging.info(f"\n\n--- Benchmark Completed in {benchmark_overall_duration:.2f} seconds ---")

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
        print(
            f"  Avg. Sum of Costs (SoC)      : {overall_stats_results['avg_soc_on_overall_success']:.2f} (on overall success)"
        )
        print(
            f"  Std. Dev. of SoC             : {overall_stats_results['std_soc_on_overall_success']:.2f} (on overall success)"
        )
        print(
            f"  Avg. Makespan                : {overall_stats_results['avg_makespan_on_overall_success']:.2f} (on overall success)"
        )
        print(
            f"  Std. Dev. of Makespan        : {overall_stats_results['std_makespan_on_overall_success']:.2f} (on overall success)"
        )
        print(f"  Avg. Computation Time (s)    : {overall_stats_results['avg_duration_s_per_run']:.3f} (per run)")

    if summary_results_per_scenario:
        df_summary = pd.DataFrame(summary_results_per_scenario)
        cols_summary = [
            "map",
            "agents",
            "trials_run_for_scenario",
            "seed_base",
            "success_rate_perc",
            "avg_makespan_on_success",
            "avg_sum_of_costs_on_success",
            "avg_trial_computation_time_sec",
            "num_errors_in_trials",
            "plan_horizon_k",
            "exec_steps_n",
            "cbs_time_limit_s",
            "cbs_iterations",
        ]
        df_summary = df_summary.reindex(columns=cols_summary)

        logging.info("\n--- Aggregated Per-Scenario Results ---")
        print(df_summary.to_string(index=False, float_format="%.2f"))

        time_str = time.strftime("%Y%m%d_%H%M%S")
        summary_csv_path = output_dir / f"benchmark_summary_per_scenario_{time_str}.csv"
        summary_json_path = output_dir / f"benchmark_full_data_{time_str}.json"
        try:
            df_summary.to_csv(summary_csv_path, index=False, float_format="%.3f")
            logging.info(f"\nPer-scenario summary results saved to {summary_csv_path}")

            benchmark_output_data = {
                "args": vars(args),
                "overall_statistics": overall_stats_results,
                "summary_per_scenario": df_summary.to_dict(orient="records"),
                "detailed_trial_results": all_run_results_detailed,
            }

            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, Path):
                        return str(obj)
                    return super().default(obj)

            with open(summary_json_path, "w") as f:
                json.dump(benchmark_output_data, f, indent=2, cls=NumpyEncoder)
            logging.info(
                f"Full benchmark data (args, overall_stats, per-scenario_summary, details) saved to {summary_json_path}"
            )
        except Exception as e:
            logging.error(f"Error saving results: {e}", exc_info=True)
    else:
        logging.warning("No scenarios were successfully summarized.")


if __name__ == "__main__":
    DEFAULT_MODEL_PATH = "./ckpt/unet_potential_field_64ch_best_new_ddp.pth"

    parser = argparse.ArgumentParser(description="LENS-L MAPF Benchmarking Script (Setting B)")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["paper", "original"],
        default="paper",
        help="'paper' uses the hyperparameters from Table 13 of the paper; 'original' uses the legacy script settings.",
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--map_config_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="benchmark_output_pogema_v2")
    parser.add_argument("--map_pattern", type=str, default="*")
    parser.add_argument("--map_limit", type=int, default=None)
    parser.add_argument("--agent_counts", type=int, nargs="+", default=[16])
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stop_scenario_on_first_success",
        action="store_false",
        help="Stop running further trials for a scenario after the first success.",
    )
    parser.add_argument("--save_animations", action="store_true", help="Save SVG animations for episodes.")

    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--obs_radius", type=int, default=SIM_OBS_RADIUS)
    parser.add_argument("--plan_horizon", "-k", type=int, default=None, dest="plan_horizon")
    parser.add_argument("--n_exec_steps", "-n", type=int, default=None, dest="n_exec_steps")
    parser.add_argument("--cbs_iterations", type=int, default=None)

    parser.add_argument("--unet_init_channels", type=int, default=None)
    parser.add_argument("--unet_spatial_channels", type=int, default=4)
    parser.add_argument("--unet_non_spatial_features", type=int, default=2)
    parser.add_argument("--unet_bilinear", action="store_true")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--verbose_simulation", action="store_true")
    parser.add_argument("--use_dynamic_cbs_map", action="store_true")
    parser.add_argument("--cbs_map_expansion_range", type=int, default=10)

    parser.add_argument("--pattering_hist_len", type=int, default=None)
    parser.add_argument("--pattering_unique_pos", type=int, default=None)
    parser.add_argument("--pattering_astar_bonus", type=int, default=30)
    parser.add_argument("--max_cbs_fails_intervention", type=int, default=1)
    parser.add_argument("--sub_cbs_iter_multiplier", type=float, default=1.5)
    parser.add_argument("--cbs_time_limit_s", type=float, default=None)
    parser.add_argument("--large_group_threshold", type=int, default=None)

    args = parser.parse_args()

    if args.preset == "original":
        if args.plan_horizon is None:
            args.plan_horizon = 10
        if args.n_exec_steps is None:
            args.n_exec_steps = 6
        if args.cbs_iterations is None:
            args.cbs_iterations = 120
        if args.unet_init_channels is None:
            args.unet_init_channels = 512
        if args.pattering_hist_len is None:
            args.pattering_hist_len = 12
        if args.pattering_unique_pos is None:
            args.pattering_unique_pos = 2
        if args.cbs_time_limit_s is None:
            args.cbs_time_limit_s = 50.0
        if args.large_group_threshold is None:
            args.large_group_threshold = 16
    else:
        # Paper Table 13 & Appendix F.1 defaults
        if args.plan_horizon is None:
            args.plan_horizon = 10
        if args.n_exec_steps is None:
            args.n_exec_steps = 6
        if args.cbs_iterations is None:
            args.cbs_iterations = 100
        if args.unet_init_channels is None:
            args.unet_init_channels = 64
        if args.pattering_hist_len is None:
            args.pattering_hist_len = 10
        if args.pattering_unique_pos is None:
            args.pattering_unique_pos = 3
        if args.cbs_time_limit_s is None:
            args.cbs_time_limit_s = 30.0
        if args.large_group_threshold is None:
            args.large_group_threshold = 12

    if args.n_exec_steps > args.plan_horizon:
        logging.warning(
            f"n_exec_steps ({args.n_exec_steps}) > plan_horizon ({args.plan_horizon}). Setting n_exec_steps = plan_horizon."
        )
        args.n_exec_steps = args.plan_horizon
    if args.n_exec_steps <= 0:
        logging.warning(f"n_exec_steps ({args.n_exec_steps}) must be positive. Setting to 1.")
        args.n_exec_steps = 1

    if args.verbose_simulation:
        logging.getLogger().setLevel(logging.DEBUG)

    run_benchmark_main(args)
