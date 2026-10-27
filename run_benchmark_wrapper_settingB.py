#!/usr/bin/env python3
"""
Unified Benchmark Wrapper for LENS (Setting B: Completely Unknown Environment).
Automatically routes small/medium POGEMA maps (Random, Mazes, Warehouse, Puzzle)
to LENS_s_settingB and large MovingAI city maps (City*, Paris*, Berlin*, Boston*)
to LENS_l_settingB.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

SMALL_MAP_PREFIXES = (
    "validation-random",
    "validation-mazes",
    "wfi_warehouse",
    "puzzle",
    "random",
    "maze",
)
LARGE_MAP_PREFIXES = ("City", "Paris", "Berlin", "Boston")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--map_pattern", type=str, default="*")
    parser.add_argument("--map_config_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    known_args, unknown_args = parser.parse_known_args()
    map_pattern = known_args.map_pattern

    if any(map_pattern.startswith(p) for p in LARGE_MAP_PREFIXES):
        target_subfolder = "LENS_l_settingB"
        target_script = "LENS_L_benchmark_settingB.py"
    elif any(map_pattern.startswith(p) for p in SMALL_MAP_PREFIXES):
        target_subfolder = "LENS_s_settingB"
        target_script = "LENS_S_benchmark_settingB.py"
    else:
        print(
            f"[Info] Map pattern '{map_pattern}' not explicitly matched; defaulting to LENS_s_settingB."
        )
        target_subfolder = "LENS_s_settingB"
        target_script = "LENS_S_benchmark_settingB.py"

    root_dir = Path(__file__).resolve().parent
    work_dir = root_dir / target_subfolder

    if not work_dir.exists():
        print(f"[Error] Target directory {work_dir} does not exist.")
        sys.exit(1)

    cmd = [sys.executable, target_script]
    if known_args.map_pattern:
        cmd.extend(["--map_pattern", known_args.map_pattern])
    if known_args.map_config_dir:
        abs_map_dir = os.path.abspath(known_args.map_config_dir)
        cmd.extend(["--map_config_dir", abs_map_dir])
    if known_args.model_path:
        abs_model_path = os.path.abspath(known_args.model_path)
        cmd.extend(["--model_path", abs_model_path])
    if known_args.output_dir:
        abs_output_dir = os.path.abspath(known_args.output_dir)
        cmd.extend(["--output_dir", abs_output_dir])

    cmd.extend(unknown_args)

    print("=" * 60)
    print(f"Routing to [{target_subfolder}/{target_script}] (Setting B)")
    print(f"Working directory: {work_dir}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    try:
        subprocess.run(cmd, cwd=str(work_dir), check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Benchmark execution failed with return code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
