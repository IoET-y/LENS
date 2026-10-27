# LENS: Learning to Navigate with Active Search for Partially Observable MAPF in Unknown Environments

[![TMLR](https://img.shields.io/badge/TMLR-06%2F2026-blue)](https://openreview.net/forum?id=DuNU6ZN5GG)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of the paper:

> **LENS: Learning to Navigate with Active Search for Partially Observable MAPF in Unknown Environments**  
> Di Yang, Yuheng Li, Yanhai Xiong (College of William & Mary)  
> *Transactions on Machine Learning Research (TMLR)*, 06/2026.  
> [OpenReview (`id=DuNU6ZN5GG`)](https://openreview.net/forum?id=DuNU6ZN5GG)

---

## Overview

**LENS** (**LE**arning to **N**avigate with Active **S**earch) is a hybrid neural-search framework for **Partially Observable Multi-Agent Path Finding (PO-MAPF)** in a-priori unknown environments. Rather than relying solely on reactive local policies or expensive global re-planning, LENS integrates:

1. **Spatial Guidance via U-Net**: A 5-channel ego-centric $11 \times 11$ local Field-of-View (FOV) representation (Static Obstacles, Other Agents, Agent's Goal, Local Path History, and Action Feasibility Mask) is processed by a U-Net trained via imitation learning on CBS demonstrations to predict an action-preference distribution over $\{ \text{STAY}, \text{UP}, \text{DOWN}, \text{LEFT}, \text{RIGHT} \}$.
2. **Proactive Horizon Rollout & Cascade Local CBS**: Each agent decodes a short-horizon trajectory ($H=5$) from its U-Net predictions. When trajectory conflicts arise within local conflict clusters, a bounded **Local Conflict-Based Search (CBS)** solver resolves multi-agent contention, falling back gracefully to one-step Conflict-Avoiding Resolution when node/time limits are reached.
3. **Active Frontier & Region-Graph Exploration (LENS-L)**: For large-scale, non-convex, and maze-like environments where local lookahead can be trapped in cul-de-sacs, **LENS-L** maintains a shared or ego-centric occupancy map, constructs a topological **Region Graph**, detects frontiers via vectorized morphological operations, and assigns temporary exploration sub-goals using utility-based frontier evaluation, topological dead-end Tabu penalties, and adaptive deadlock escape.

### Problem Settings

We evaluate two distinct information-sharing regimes:
- **Setting A (Isolated Exploration / Pre-computed Global Guidance)**: Agents operate with local $11 \times 11$ observations (`obs_radius = 5`) and isolated exploration memory while querying pre-computed shortest-path guidance (`BD`).
- **Setting B (Cooperative Exploration in Unknown Maps)**: Agents start with **zero prior knowledge** of the global map (`known_map` initialized as unknown `-1`). As agents move, they fuse their $11 \times 11$ FOV observations into a shared occupancy map and dynamically recompute true shortest-path distances (via Numba-accelerated BFS) and frontier assignments online.

### Model Variants

| Variant | Target Scenarios | Key Mechanisms |
| :--- | :--- | :--- |
| **LENS-S** (`LENS_s_settingA`, `LENS_s_settingB`) | Small-to-medium maps (`mazes`, `random`, `cities_tiles`) | U-Net action preferences + $H$-step greedy/sampled rollout + Cascade Local CBS + 1-step conflict-avoidance fallback |
| **LENS-L** (`LENS_l_settingA`, `LENS_l_settingB`) | Large & complex maps (`warehouse`, `MovingAI_tiles` / Paris) | LENS-S core + Frontier detection + Topological Region Graph (`region_graph.py`) + Dead-end Tabu + Stagnation & Endgame escape |

---

## Repository Structure

```text
LENS/
├── LENS_s_settingA/                 # LENS-S under Setting A
│   ├── unet_model_new.py            # U-Net architecture (5-channel 11x11 input -> 5 actions)
│   ├── mapf_utils.py                # POGEMA action constants & 5-channel FOV feature builder
│   ├── data_preprocessing.py        # CBS expert trajectory generation for U-Net training
│   ├── convert_pfg_npz_to_pt_new.py # Convert .npz expert datasets to PyTorch .pt shards
│   ├── train_unet_new.py            # U-Net training script (supports single-GPU/CPU & DDP)
│   ├── action_sequence_decoder_s.py # H-step action sequence rollout decoder
│   ├── local_cbs_solver_s.py        # Bounded Local CBS low-level A* & high-level solver
│   ├── LENS_s.py                    # Core LENS-S simulation loop
│   └── LENS_S_benchmark_settingA.py # Benchmark runner for LENS-S (Setting A)
├── LENS_s_settingB/                 # LENS-S under Setting B (Unknown Map / Online Exploration)
│   ├── LENS_s.py                    # Core LENS-S simulation loop with online map fusion
│   └── LENS_S_benchmark_settingB.py # Benchmark runner for LENS-S (Setting B)
├── LENS_l_settingA/                 # LENS-L under Setting A
│   ├── region_graph.py              # Topological Region Graph & dead-end analysis
│   ├── action_sequence_decoder_l.py # Horizon rollout decoder with frontier/temp-goal support
│   ├── local_cbs_solver_robust_l.py # Robust Local CBS solver for LENS-L
│   ├── LENS_l_setA.py               # Core LENS-L simulation loop (Setting A)
│   └── LENS_L_benchmark_settingA.py # Benchmark runner for LENS-L (Setting A)
├── LENS_l_settingB/                 # LENS-L under Setting B (Unknown Map / Online Exploration)
│   ├── region_graph.py              # Topological Region Graph & dead-end analysis
│   ├── LENS_l.py                    # Core LENS-L simulation loop (Setting B)
│   └── LENS_L_benchmark_settingB.py # Benchmark runner for LENS-L (Setting B)
├── baseline/                        # Baseline adapters & third-party baselines
│   ├── gnn_cspibt_settingA.py       # GNN-Guided CS-PIBT benchmark (Setting A)
│   ├── gnn_cspibt_settingB.py       # GNN-Guided CS-PIBT benchmark (Setting B)
│   ├── lns2_adapter.py              # POGEMA adapter for LNS2+RL
│   ├── lns2_pogema_settingA.py      # LNS2+RL benchmark (Setting A)
│   ├── lns2_pogema_settingB.py      # LNS2+RL benchmark (Setting B)
│   ├── create_env.py                # POGEMA evaluation environment helper
│   ├── ML-MAPF-with-Search/         # Third-party CS-PIBT / SSIL baseline
│   ├── scrimp_model/                # Third-party SCRIMP baseline
│   └── gpt/                         # Third-party LaCAM-GPT baseline
├── dataset/                         # POGEMA benchmark map configurations (maps.yaml)
│   ├── mazes/
│   ├── random/
│   ├── cities_tiles/
│   ├── warehouse/
│   └── MovingAI_tiles/
├── run_benchmark_wrapper_settingA.py # Unified dispatcher routing maps to LENS-S or LENS-L (Setting A)
└── run_benchmark_wrapper_settingB.py # Unified dispatcher routing maps to LENS-S or LENS-L (Setting B)
```

---

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- `pogema` and `pogema-toolbox`
- `numpy`, `scipy`, `pandas`, `pyyaml`, `tqdm`, `numba`

```bash
pip install torch torchvision numpy scipy pandas pyyaml tqdm numba pogema pogema-toolbox
```

---

## Training the U-Net Guidance Model

LENS trains a lightweight U-Net via supervised imitation learning on expert trajectories generated by Conflict-Based Search (CBS) on small $15 \times 15$ random grid instances.

### Step 1: Generate Expert CBS Trajectories (`.npz`)
```bash
cd LENS_s_settingA
python data_preprocessing.py \
    --num_instances 1000 \
    --map_height 15 \
    --map_width 15 \
    --num_agents 8 \
    --obs_density 0.15 \
    --fov_size 11 \
    --output_dir ./pfg_training_data
```

### Step 2: Convert `.npz` Batches to PyTorch Shards (`.pt`)
```bash
python convert_pfg_npz_to_pt_new.py \
    --npz_dir ./pfg_training_data \
    --pt_dir ./pfg_pt_shards
```

### Step 3: Train the U-Net
`train_unet_new.py` supports both single-GPU/CPU execution and multi-GPU DistributedDataParallel (`torchrun`), as well as `--preset paper` (default, matching TMLR paper specifications) and `--preset original` (matching legacy training script defaults):

```bash
# Single GPU / CPU (Paper preset: init_channel=64, lr=1e-4, weight_decay=1e-5, batch_size=256, epochs=100)
python train_unet_new.py \
    --pt_data_dir ./pfg_pt_shards \
    --preset paper \
    --model_save_path ./unet_best.pth

# Multi-GPU DDP via torchrun
torchrun --nproc_per_node=4 train_unet_new.py \
    --pt_data_dir ./pfg_pt_shards \
    --preset paper \
    --model_save_path ./unet_best.pth
```

> **Note on Checkpoint Compatibility**: All benchmark scripts (`LENS_S_benchmark_*.py` and `LENS_L_benchmark_*.py`) automatically inspect checkpoint tensor shapes on load to infer `init_channel` (`64`, `256`, or `512`) and `bilinear_upsample` (`True`/`False`), so checkpoints trained under either preset load seamlessly.

---

## Running Benchmarks

You can run benchmarks either via the unified root wrappers (`run_benchmark_wrapper_settingA.py` / `run_benchmark_wrapper_settingB.py`), which automatically dispatch small/medium maps (`maze*`, `random*`, `Berlin*`/`Milan*`/`London*`/` Moscow*`) to **LENS-S** and large maps (`warehouse*`, `Paris*`) to **LENS-L**, or by calling the variant-specific scripts directly.

### Unified Wrapper Execution

```bash
# Setting A: Mazes with 8, 16, 32, 64 agents
python run_benchmark_wrapper_settingA.py \
    --map_config_dir ./dataset/mazes \
    --map_pattern "maze*" \
    --agent_counts 8 16 32 64 \
    --max_steps 128 \
    --preset paper

# Setting B (Unknown Map): Warehouse with 32, 64, 128, 256 agents
python run_benchmark_wrapper_settingB.py \
    --map_config_dir ./dataset/warehouse \
    --map_pattern "warehouse*" \
    --agent_counts 32 64 128 256 \
    --max_steps 256 \
    --preset paper
```

### Direct Execution (LENS-S / LENS-L)

```bash
# LENS-S on Random Grids (Setting B)
python LENS_s_settingB/LENS_S_benchmark_settingB.py \
    --map_config_dir ./dataset/random \
    --map_pattern "random*" \
    --agent_counts 8 16 32 64 \
    --max_steps 128 \
    --model_path ./LENS_s_settingB/unet_best.pth \
    --preset paper

# LENS-L on MovingAI Paris Tiles (Setting B)
python LENS_l_settingB/LENS_L_benchmark_settingB.py \
    --map_config_dir ./dataset/MovingAI_tiles \
    --map_pattern "Paris*" \
    --agent_counts 64 128 256 \
    --max_steps 256 \
    --model_path ./LENS_l_settingB/unet_best.pth \
    --preset paper
```

---

## Hyperparameter Presets (`--preset paper` vs. `--preset original`)

To ensure both strict alignment with the TMLR paper description and full reproducibility of legacy experimental scripts, all training and evaluation entrypoints provide a `--preset` flag (`--preset paper` by default, or `--preset original`):

| Component / Hyperparameter | `--preset paper` (Default, TMLR Paper) | `--preset original` (Legacy Script Defaults) |
| :--- | :--- | :--- |
| **U-Net Base Channels (`init_channel`)** | `64` (encoder: `64 -> 128 -> 256`) | `256` (in `train_unet_new.py`) / `512` (in `unet_model_new.py`) |
| **U-Net Learning Rate (`lr`)** | `1e-4` | `5e-3` |
| **U-Net Weight Decay** | `1e-5` | `1e-4` |
| **U-Net Batch Size / Epochs** | `256` / `100` | `512` / `300` |
| **U-Net Early Stopping Patience** | `10` epochs | `30` epochs |
| **Lookahead Horizon (`H`)** | `5` steps | `5` steps |
| **LENS-S CBS Node / Time Limit** | `100` nodes / `0.05` s | `150` nodes / `0.08` s |
| **LENS-L CBS Node / Time Limit** | `100` nodes / `0.05` s | `50` nodes / `0.10` s |
| **LENS-L Frontier Utility Weights** | $\alpha=1.0, \beta=0.7, \gamma=0.2, \delta=0.5$ | $\alpha=1.0, \beta=0.35\text{–}0.4, \gamma=0.2, \delta=0.5$ |
| **LENS-L Stagnation Window ($W$)** | `16` steps | `12` steps |

Any individual hyperparameter can also be overridden explicitly on the command line (e.g., `--cbs_max_nodes 150 --stagnation_window 12`).

---

## Citation

If you find this repository useful in your research, please cite our TMLR paper:

```bibtex
@article{yang2026lens,
  title={LENS: Learning to Navigate with Active Search for Partially Observable MAPF in Unknown Environments},
  author={Yang, Di and Li, Yuheng and Xiong, Yanhai},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://openreview.net/forum?id=DuNU6ZN5GG}
}
```

## License

This project is released under the [MIT License](LICENSE). Third-party baseline implementations in `baseline/` (`ML-MAPF-with-Search`, `scrimp_model`, `gpt`) retain their respective original licenses.
