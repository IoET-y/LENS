
# Baselines Evaluation

This directory contains the Python wrapper and adapter scripts used to evaluate various baseline methods (MAPF-GPT, SCRIMP, LaCAM, GNN-CSPIBT, and LNS2-RL) for comparison against LENS.

## ⚠️ Important Prerequisite: Compilation & Setup

The scripts provided in this directory are **wrappers and adapters**. To successfully run these baselines, **you must first compile and install the official C++/Python implementations** of these algorithms according to their respective original repositories.

Please refer to the official repositories to build the required executables or libraries before running any scripts in this folder, and please download these offical repositories and unzip them in this folder: 

* **MAPF-GPT**: [https://github.com/CognitiveAISystems/MAPF-GPT]
* **SCRIMP**: [https://github.com/marmotlab/SCRIMP]
* **LaCAM**: [https://github.com/Kei18/lacam3]
* **GNN-CSPIBT**: [https://github.com/Rishi-V/ML-MAPF-with-Search]
* **LNS2 / LNS2-RL**: [https://github.com/marmotlab/LNS2-RL]

*(Note: For LNS2-RL, you may need to export your boost library path before running the scripts, e.g., `export LD_LIBRARY_PATH=/workspace/boost_install/lib:$LD_LIBRARY_PATH`)*

---

## 🚀 Running the Baselines

Below are the commands to reproduce the baseline evaluations across different map types (mazes, random, warehouse, City, and big maps). 

### 1. MAPF-GPT
Evaluate MAPF-GPT on large maps (e.g., Berlin).
```bash
python mapf_gpt_test.py \
    --map-config-dir ../MAPF-GPT/eval_configs/06-bigmap \
    --map-pattern "Berlin_1_256_256_256_00" \
    --map-indices 0:1 \
    --agent-counts 512 \
    --gpt-weights weights/model-85M.pt \
    --max-steps 512

```

### 2. SCRIMP

Evaluate SCRIMP on standard mazes and large maps.

**Standard Mazes:**

```bash
python mapf_benchmark_with_scrimp.py \
    --model_path "weights/net_checkpoint.pkl" \
    --map_config_dir "../MAPF-GPT/eval_configs/" \
    --map_pattern "validation-mazes-seed-*" \
    --agent_counts 8 \
    --map_limit 128 \
    --num_trials 1 \
    --device "cuda"

```

**Big Maps (e.g., Berlin):**

```bash
python mapf_benchmark_with_scrimp.py \
    --model_path "weights/net_checkpoint.pkl" \
    --map_config_dir "../MAPF-GPT/eval_configs/" \
    --map_pattern "Berlin_1_256_256_256_00" \
    --agent_counts 512 \
    --map_limit 1 \
    --num_trials 1 \
    --device "cuda" > out_256_bigmap_scrimp.log 2>&1 &

```

### 3. LaCAM Baseline

Evaluate the classical LaCAM method across various scenarios.

**City Maps:**

```bash
python mapf_benchmark_with_lacam_baseline.py \
    --map_config_dir ../MAPF-GPT/eval_configs \
    --map_pattern "City*" \
    --agent_counts 256 \
    --map_limit 128 \
    --num_trials 1 \


```

**Random Maps:**

```bash
python mapf_benchmark_with_lacam_baseline.py \
    --map_config_dir ../MAPF-GPT/eval_configs \
    --map_pattern "validation-random-seed-*" \
    --agent_counts 8 \
    --map_limit 1 \
    --num_trials 128 \


```

**Warehouse:**

```bash
python mapf_benchmark_with_lacam_baseline.py \
    --map_config_dir ../MAPF-GPT/eval_configs \
    --map_pattern "wfi_warehouse*" \
    --agent_counts 192 \
    --map_limit 1 \
    --num_trials 128 \


```

**Puzzles:**

```bash
python mapf_benchmark_with_lacam_baseline.py \
    --map_config_dir ../MAPF-GPT/eval_configs \
    --map_pattern "puzzle-*" \
    --agent_counts 3 \
    --map_limit 16 \
    --num_trials 8

```

### 4. GNN-CSPIBT

We provide scripts for both Setting A and Setting B evaluations.

#### Setting A:

```bash
# Mazes
python gnn_cspibt_settingA.py --map_config_dir ../MAPF-GPT/eval_configs/ --map_pattern "validation-mazes-seed-*" --agent_counts 64 --map_limit 127 --num_trials 1 --model_path ./ML-MAPF-with-Search/model/ssil_model.pt --output_dir ./gnn_cspibt_results

# Random Maps
python gnn_cspibt_settingA.py --map_config_dir ../MAPF-GPT/eval_configs/ --map_pattern "validation-random-seed-*" --agent_counts 64 --map_limit 127 --num_trials 1 --model_path ./ML-MAPF-with-Search/model/ssil_model.pt --output_dir ./gnn_cspibt_results

# Warehouse
python gnn_cspibt_settingA.py --map_config_dir ../MAPF-GPT/eval_configs/ --map_pattern "wfi_warehouse*" --agent_counts 192 --map_limit 1 --num_trials 1 --model_path ./ML-MAPF-with-Search/model/ssil_model.pt --output_dir ./gnn_cspibt_results

# City (Running in background)
nohup python gnn_cspibt_settingA.py --map_config_dir ../MAPF-GPT/eval_configs/ --map_pattern "City*" --agent_counts 256 --map_limit 128 --num_trials 1 --model_path ./ML-MAPF-with-Search/model/ssil_model.pt --output_dir ./gnn_cspibt_results > output_cs-pibt_city.log 2>&1 &

```

#### Setting B :

```bash
# Mazes
python gnn_cspibt_settingB.py --map_config_dir ../MAPF-GPT/eval_configs/ --map_pattern "validation-mazes-seed-*" --agent_counts 64 --map_limit 127 --num_trials 1 --model_path ./ML-MAPF-with-Search/model/ssil_model.pt 
# City
python gnn_cspibt_settingB.py --map_config_dir ../MAPF-GPT/eval_configs/ --map_pattern "City*" --agent_counts 64 --map_limit 1 --num_trials 1 --model_path ./ML-MAPF-with-Search/model/ssil_model.pt 

```

### 5. LNS2-RL

Ensure your `LD_LIBRARY_PATH` is set correctly for Boost before running LNS2-RL.

```bash
export LD_LIBRARY_PATH=/workspace/boost_install/lib:$LD_LIBRARY_PATH

```

#### Setting A:

```bash
# City
python lns2_pogema_settingA.py --map_config_dir ../MAPF-GPT/eval_configs/ --map_pattern "City*" --map_limit 127 --model_path ./LNS2_RL/ckpt/net_checkpoint.pkl --max_steps 512 --agent_counts 256 --num_trials 1 > output_lns2rl_city256.log 2>&1 &

# Warehouse
python lns2_pogema_settingA.py --map_config_dir ../MAPF-GPT/eval_configs/ --map_pattern "wfi_warehouse" --map_limit 1 --model_path ./LNS2_RL/ckpt/net_checkpoint.pkl --agent_counts 32 --num_trials 10

# Mazes
python lns2_pogema_settingA.py --map_config_dir ../MAPF-GPT/eval_configs/ --map_pattern "validation-mazes-seed-*" --model_path ./LNS2_RL/ckpt/net_checkpoint.pkl --agent_counts 8 --num_trials 1 > output_lns2rl_maze16.log 2>&1 &

# Random Maps
python lns2_pogema_settingA.py --map_config_dir ../MAPF-GPT/eval_configs/ --map_pattern "validation-random-seed-*" --model_path ./LNS2_RL/ckpt/net_checkpoint.pkl --agent_counts 32 --num_trials 1 > output_lns2rl_random32.log 2>&1 &

```

#### Setting B :

```bash
# Mazes
python lns2_pogema_settingB.py \
    --map_config_dir ../MAPF-GPT/eval_configs/ \
    --map_pattern "validation-mazes-seed-*" \
    --map_limit 127 \
    --model_path ./LNS2_RL/ckpt/net_checkpoint.pkl \
    --max_steps 128 \
    --agent_counts 8 \
    --num_trials 1 

