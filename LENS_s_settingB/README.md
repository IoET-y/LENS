# LENS-S (Setting B: Unknown Map)

Implementation of **LENS-S** for small-to-medium scale PO-MAPF benchmarks (`mazes`, `random`, `cities_tiles`) under **Setting B** (a-priori unknown environment where agents fuse local $11 \times 11$ FOV observations into a shared map and update distance heuristics online).

## Usage

```bash
python LENS_S_benchmark_settingB.py \
    --map_config_dir ../dataset/mazes \
    --map_pattern "maze*" \
    --agent_counts 8 16 32 64 \
    --max_steps 128 \
    --model_path ./unet_best.pth \
    --preset paper
```

Use `--preset paper` (default) for the exact hyperparameters reported in the TMLR paper, or `--preset original` for legacy script defaults.
