
## MAPF-GPT-bench

python mapf_gpt_test.py --map-config-dir ../MAPF-GPT/eval_configs/06-bigmap --map-pattern "Berlin_1_256_256_256_00" --map-indices 0:1 --agent-counts 512 --gpt-weights weights/model-85M.pt --max-steps 512 

## scrimp


python mapf_benchmark_with_scrimp.py \
    --model_path "weights/net_checkpoint.pkl" \
    --map_config_dir "../MAPF-GPT/eval_configs/" \
    --map_pattern "validation-mazes-seed-*" \
    --agent_counts 8 \
    --map_limit 128 \
    --num_trials 1 \
    --device "cuda"

python mapf_benchmark_with_scrimp.py \
    --model_path "weights/net_checkpoint.pkl" \
    --map_config_dir "../MAPF-GPT/eval_configs/" \
    --map_pattern "Berlin_1_256_256_256_00" \
    --agent_counts 512 \
    --map_limit 1 \
    --num_trials 1 \
    --device "cuda" > out_256_bigmap_scrimp.log 2>&1 &


## LACAM baseline:
##    LACAM methods


python mapf_benchmark_with_lacam_baseline.py \
    --map_config_dir ../MAPF-GPT/eval_configs \
    --map_pattern  "City*" \
    --agent_counts 256 \
    --map_limit 128 \
    --num_trials  1 \
    --use_dynamic_cbs_map \
    --cbs_map_expansion_range 0

python mapf_benchmark_with_lacam_baseline.py \
    --map_config_dir ../MAPF-GPT/eval_configs \
    --map_pattern  "validation-random-seed-*" \
    --agent_counts 8 \
    --map_limit 1 \
    --num_trials  128 \
    --use_dynamic_cbs_map \
    --cbs_map_expansion_range 0

python mapf_benchmark_with_lacam_baseline.py \
    --map_config_dir ../MAPF-GPT/eval_configs \
    --map_pattern  "wfi_warehouse*" \
    --agent_counts 192 \
    --map_limit 1 \
    --num_trials  128 \
    --use_dynamic_cbs_map \
    --cbs_map_expansion_range 0

python mapf_benchmark_with_lacam_baseline.py \
    --map_config_dir ../MAPF-GPT/eval_configs \
    --map_pattern  "puzzle-*" \
    --agent_counts 3 \
    --map_limit 16 \
    --num_trials  8 
    







## setting B:
python gnn_cspibt_settingB.py     --map_config_dir ../MAPF-GPT/eval_configs/     --map_pattern "validation-mazes-seed-*"     --agent_counts 64    --map_limit 127     --num_trials 1     --model_path ./ML-MAPF-with-Search/model/ssil_model.pt     --output_dir ./gnn_cspibt_results_settingB_cornercase

python gnn_cspibt_settingB.py     --map_config_dir ../MAPF-GPT/eval_configs/     --map_pattern "City*"     --agent_counts 64     --map_limit 1     --num_trials 1     --model_path ./ML-MAPF-with-Search/model/ssil_model.pt     --output_dir ./gnn_cspibt_results_settingB

## setting A:
python gnn_cspibt_settingA.py     --map_config_dir ../MAPF-GPT/eval_configs/     --map_pattern "validation-mazes-seed-*"     --agent_counts 64     --map_limit 127     --num_trials 1     --model_path ./ML-MAPF-with-Search/model/ssil_model.pt     --output_dir ./gnn_cspibt_results


python gnn_cspibt_settingA.py     --map_config_dir ../MAPF-GPT/eval_configs/     --map_pattern "validation-random-seed-*"     --agent_counts 64     --map_limit 127     --num_trials 1     --model_path ./ML-MAPF-with-Search/model/ssil_model.pt     --output_dir ./gnn_cspibt_results

python gnn_cspibt_settingA.py     --map_config_dir ../MAPF-GPT/eval_configs/     --map_pattern "wfi_warehouse*"     --agent_counts 192     --map_limit 1     --num_trials 1     --model_path ./ML-MAPF-with-Search/model/ssil_model.pt     --output_dir ./gnn_cspibt_results

 nohup python gnn_cspibt_settingA.py     --map_config_dir ../MAPF-GPT/eval_configs/     --map_pattern "City*"     --agent_counts 256     --map_limit 128    --num_trials 1     --model_path ./ML-MAPF-with-Search/model/ssil_model.pt     --output_dir ./gnn_cspibt_results > output_cs-pibt_city.log 2>&1 &

# LNS2-RL


## Setting B:

# LNS2 corner case 

python lns2_pogema_settingB.py \
    --map_config_dir ../MAPF-GPT/eval_configs/ \
    --map_pattern "validation-mazes-seed-*" \
    --map_limit 127 \
    --model_path ./LNS2_RL/ckpt/net_checkpoint.pkl \
    --max_steps 128 \
    --agent_counts 8 \
    --num_trials 1 > ./lns2_settingB/output_maze8.log 2>&1 &



export LD_LIBRARY_PATH=/workspace/boost_install/lib:$LD_LIBRARY_PATH

python lns2_pogema_settingA.py \
    --map_config_dir ../MAPF-GPT/eval_configs/ \
    --map_pattern "City*" \
    --map_limit 127 \
    --model_path ./LNS2_RL/ckpt/net_checkpoint.pkl \
    --max_steps 256 \
    --agent_counts 256 \
    --num_trials 1 > output_lns2rl_city256.log 2>&1 &


python lns2_pogema_settingA.py \
    --map_config_dir ../MAPF-GPT/eval_configs/ \
    --map_pattern "wfi_warehouse" \
    --map_limit 1 \
    --model_path ./LNS2_RL/ckpt/net_checkpoint.pkl \
    --agent_counts 32 \
    --num_trials 10


python lns2_pogema_settingA.py \
    --map_config_dir ../MAPF-GPT/eval_configs/ \
    --map_pattern "validation-mazes-seed-*" \
    --model_path ./LNS2_RL/ckpt/net_checkpoint.pkl \
    --agent_counts 8 \
    --num_trials 1 > output_lns2rl_maze16.log 2>&1 &


python lns2_pogema_settingA.py \
    --map_config_dir ../MAPF-GPT/eval_configs/ \
    --map_pattern "validation-random-seed-*" \
    --model_path ./LNS2_RL/ckpt/net_checkpoint.pkl \
    --agent_counts 32 \
    --num_trials 1 > output_lns2rl_random32.log 2>&1 &




