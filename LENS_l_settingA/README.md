# LENS-L (Setting A)

Implementation of **LENS-L** for large-scale and complex PO-MAPF benchmarks (`warehouse`, `MovingAI_tiles` / Paris) under **Setting A**, combining U-Net action guidance, Cascade Local CBS, frontier exploration, and topological Region Graph (`region_graph.py`) dead-end Tabu reasoning.

## Usage

```bash
python LENS_L_benchmark_settingA.py \
    --map_config_dir ../dataset/warehouse \
    --map_pattern "warehouse*" \
    --agent_counts 32 64 128 256 \
    --max_steps 256 \
    --model_path ./unet_best.pth \
    --preset paper
```

Use `--preset paper` (default) for the exact hyperparameters reported in the TMLR paper, or `--preset original` for legacy script defaults.
