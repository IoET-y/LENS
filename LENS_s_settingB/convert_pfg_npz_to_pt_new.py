# convert_pfg_npz_to_pt_new.py
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def convert_pfg_data(data_dirs, out_file):
    """
    Converts PFG datasets from multiple .npz files located in one or more
    directories into a single .pt file containing (X_spatial, X_nonspatial, Y_target).
    """
    out_file = Path(out_file)

    all_spatial_inputs = []
    all_nonspatial_inputs = []
    all_target_potentials = []

    all_npz_paths = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        print(f"Searching for files in: {data_dir}")
        npz_paths_in_dir = sorted(data_dir.glob("*_pfg_only.npz"))
        all_npz_paths.extend(npz_paths_in_dir)

    if not all_npz_paths:
        print(f"No '*_pfg_only.npz' files found in any of the specified directories: {data_dirs}")
        return

    print(f"\nFound a total of {len(all_npz_paths)} .npz files to process from all directories.")

    for npz_file_path in tqdm(all_npz_paths, desc="Processing PFG .npz files"):
        try:
            data = np.load(npz_file_path)

            pfg_spatial = data["pfg_spatial_input_tensors"]       # (N, C_spatial, H, W)
            pfg_target_vec_y = data["pfg_nonspatial_target_vec_y"]  # (N,)
            pfg_target_vec_x = data["pfg_nonspatial_target_vec_x"]  # (N,)
            pfg_targets = data["pfg_target_potentials"]           # (N, H, W)

            num_samples_in_file = pfg_spatial.shape[0]

            for i in range(num_samples_in_file):
                sample_spatial = pfg_spatial[i].astype(np.float32)
                sample_nonspatial = np.array(
                    [pfg_target_vec_y[i], pfg_target_vec_x[i]], dtype=np.float32
                )
                sample_target = pfg_targets[i].astype(np.float32)[None, :, :]

                all_spatial_inputs.append(torch.from_numpy(sample_spatial))
                all_nonspatial_inputs.append(torch.from_numpy(sample_nonspatial))
                all_target_potentials.append(torch.from_numpy(sample_target))

        except KeyError as e:
            print(f"Warning: KeyError {e} in file {npz_file_path}. Skipping this file.")
            continue
        except Exception as e:
            print(f"Warning: Error processing file {npz_file_path}: {e}. Skipping this file.")
            continue

    if not all_spatial_inputs:
        print("No samples were processed. Output file will not be created.")
        return

    X_spatial = torch.stack(all_spatial_inputs)
    X_nonspatial = torch.stack(all_nonspatial_inputs)
    Y_target = torch.stack(all_target_potentials)

    total_samples = X_spatial.size(0)
    print(f"\nSuccessfully processed all files.")
    print(f"Total samples collected: {total_samples}")
    print(f"  Spatial input tensor shape: {X_spatial.shape}")
    print(f"  Non-spatial input tensor shape: {X_nonspatial.shape}")
    print(f"  Target potential tensor shape: {Y_target.shape}")

    torch.save((X_spatial, X_nonspatial, Y_target), out_file)
    print(f"Saved combined preprocessed PFG dataset to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PFG .npz datasets from one or more directories to a single .pt file."
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing the input .npz PFG dataset files.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="pfg_dataset_combined.pt",
        help="Path to save the output .pt file.",
    )
    args = parser.parse_args()
    convert_pfg_data(args.data_dirs, args.out_file)
