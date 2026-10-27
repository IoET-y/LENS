# data preprocess =====#
nohup python data_preprocessing.py --map-config-dir ../MAPF-GPT/dataset_configs/10-medium-mazes --map-pattern medium-mazes-seed-???? --output-dir dataset_pfg_v_maze > output_dataset_pfg_v_maze_new.log 2>&1 &

nohup python data_preprocessing.py --map-config-dir ../MAPF-GPT/dataset_configs/12-medium-random --map-pattern training-random-seed-????? --map-indices 640:10639 --output-dir dataset_pfg_v_random > output_dataset_pfg_v_random.log 2>&1 &

nohup python data_preprocessing_random.py --map-config-dir ../MAPF-GPT/dataset_configs/10-medium-mazes --map-pattern medium-mazes-seed-???? --output-dir dataset_pfg_maze_LOCrandom > output_dataset_pfg_v_maze_randomLOC.log 2>&1 &



nohup python data_preprocessing_random.py --map-config-dir ../MAPF-GPT/dataset_configs/12-medium-random --map-pattern training-random-seed-????? --map-indices 640:10639 --output-dir dataset_pfg_random_LOCrandom > output_dataset_pfg_v_random_randomLOC.log 2>&1 &


# Convert data.npz to PT =====#

python convert_pfg_npz_to_pt_new.py --data_dir dataset_pfg_maze_LOCrandom dataset_pfg_random_LOCrandom --out_file new_randonLOC_mixed_dataset.pt


# training unet
nohup python train_unet_new.py >  output_train_pfg_v_maze_256ch.log 2>&1 &



###
python LENS_L_benchmark_settingA.py \
    --map_config_dir ../MAPF-GPT/eval_configs/ \
    --map_pattern "City*" \
    --agent_counts 256 \
    --map_limit 2\
    --num_trials  1


    



