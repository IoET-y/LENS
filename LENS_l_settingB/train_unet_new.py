# train_unet_new.py
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from unet_model_new import UNetPotentialField

# --- Default Configuration (Aligned with Paper Appendix F.1 & G.3, Table 11) ---
DEFAULT_PT_DATA_FILE = Path("new_randonLOC_mixed_dataset.pt")
DEFAULT_MODEL_SAVE_DIR = Path("new_randonLOC_mixed_dataset")

NUM_EPOCHS = 100
BATCH_SIZE = 1024          # Paper Table 11: 1024 (Original DDP script default: 4096)
LEARNING_RATE = 5e-4       # Paper Table 11: 5e-4
VALIDATION_SPLIT = 0.2
RANDOM_SEED_DATASET = 42
USE_BILINEAR_UPSAMPLE = False  # Paper Appendix F.1: Transposed Convolution

SCHEDULER_PATIENCE = 5     # Paper Table 11: Patience 5
SCHEDULER_FACTOR = 0.1     # Paper Table 11: Factor 0.1
SCHEDULER_MIN_LR = 1e-7
SCHEDULER_THRESHOLD = 1e-5

EARLY_STOPPING_PATIENCE = 10  # Paper Table 11: 10 epochs
EARLY_STOPPING_MIN_DELTA = 1e-4
INIT_CHN = 64              # Paper Appendix F.1: 64 -> 128 -> 256 -> 512 (Original script: 256/512)

LAMBDA_GRAD = 0.1          # Paper Table 11: lambda_grad = 0.1


def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters    : {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)\n")


class PFGDataset(Dataset):
    def __init__(self, pt_file_path, transform=None):
        self.pt_file_path = Path(pt_file_path)
        if not self.pt_file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.pt_file_path}")

        if int(os.environ.get("RANK", 0)) == 0:
            print(f"Loading dataset: {self.pt_file_path}...")

        self.X_spatial, self.X_nonspatial, self.Y_target = torch.load(self.pt_file_path, map_location='cpu')

        if int(os.environ.get("RANK", 0)) == 0:
            print("Dataset loaded successfully.")

        if not (self.X_spatial.ndim == 4 and self.X_nonspatial.ndim == 2 and self.Y_target.ndim == 4):
            raise ValueError("Loaded tensors have incorrect dimensions.")
        if not (self.X_spatial.shape[0] == self.X_nonspatial.shape[0] == self.Y_target.shape[0]):
            raise ValueError("Sample counts across loaded tensors do not match.")
        if self.Y_target.shape[1] != 1:
            raise ValueError(f"Target tensor should have 1 channel, but got {self.Y_target.shape[1]}")
        if self.X_nonspatial.shape[1] != 2:
            raise ValueError(f"Expected 2 non-spatial features, but got {self.X_nonspatial.shape[1]}")

        self.n_samples = self.X_spatial.shape[0]
        self.spatial_channels = self.X_spatial.shape[1]
        self.non_spatial_features = self.X_nonspatial.shape[1]
        self.height = self.X_spatial.shape[2]
        self.width = self.X_spatial.shape[3]

        if int(os.environ.get("RANK", 0)) == 0:
            print(f"  Samples             : {self.n_samples}")
            print(f"  Spatial channels    : {self.spatial_channels}")
            print(f"  Non-spatial features: {self.non_spatial_features}")
            print(f"  FOV resolution      : {self.height}x{self.width}")

        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X_spatial[idx], self.X_nonspatial[idx], self.Y_target[idx]


def gradient_loss(prediction, target):
    """Auxiliary Sobel spatial gradient L1 loss (Paper Eq. 3)."""
    sobel_y = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        device=prediction.device, dtype=prediction.dtype
    ).view(1, 1, 3, 3)
    sobel_x = sobel_y.transpose(2, 3)

    pred_grad_x = F.conv2d(prediction, sobel_x, padding=1)
    pred_grad_y = F.conv2d(prediction, sobel_y, padding=1)
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)

    return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)


def train_model(
    rank, is_ddp, device, model, train_loader, val_loader, optimizer, scheduler,
    num_epochs, start_epoch, initial_best_val_loss_saving,
    initial_best_val_loss_early_stop, initial_epochs_no_improve,
    model_save_path, latest_checkpoint_path, lambda_grad
):
    best_val_loss_for_saving = initial_best_val_loss_saving
    best_val_loss_for_early_stop = initial_best_val_loss_early_stop
    epochs_no_improve = initial_epochs_no_improve

    mse_criterion = nn.MSELoss()
    raw_model = model.module if is_ddp else model

    if rank == 0:
        count_parameters(raw_model)
        print(f"LR Scheduler  : ReduceLROnPlateau (patience={SCHEDULER_PATIENCE}, factor={SCHEDULER_FACTOR})")
        print(f"Early Stopping: patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA}")
        print(f"Gradient Loss : lambda_grad={lambda_grad}")

    for epoch in range(start_epoch, num_epochs):
        current_epoch_display = epoch + 1
        epoch_start_time = time.time()

        if is_ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0

        for i, (spatial_inputs, non_spatial_inputs, targets) in enumerate(train_loader):
            spatial_inputs = spatial_inputs.to(device)
            non_spatial_inputs = non_spatial_inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(spatial_inputs, non_spatial_inputs)

            main_loss = mse_criterion(outputs, targets)
            grad_loss = gradient_loss(outputs, targets)
            loss = main_loss + lambda_grad * grad_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * spatial_inputs.size(0)

            if rank == 0 and (i + 1) % max(1, (len(train_loader) // 4)) == 0:
                print(f"  Epoch [{current_epoch_display}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}")

        denom = len(train_loader.sampler) if train_loader.sampler is not None else len(train_loader.dataset)
        epoch_train_loss = running_loss / max(1, denom)

        model.eval()
        val_running_loss = 0.0
        current_val_loss = float('nan')

        if val_loader.dataset and len(val_loader.dataset) > 0:
            with torch.no_grad():
                for spatial_inputs_val, non_spatial_inputs_val, targets_val in val_loader:
                    spatial_inputs_val = spatial_inputs_val.to(device)
                    non_spatial_inputs_val = non_spatial_inputs_val.to(device)
                    targets_val = targets_val.to(device)

                    outputs_val = model(spatial_inputs_val, non_spatial_inputs_val)
                    main_loss_val = mse_criterion(outputs_val, targets_val)
                    grad_loss_val = gradient_loss(outputs_val, targets_val)
                    loss_val = main_loss_val + lambda_grad * grad_loss_val

                    val_running_loss += loss_val.item() * spatial_inputs_val.size(0)

            if is_ddp:
                total_val_loss_tensor = torch.tensor(val_running_loss, device=device)
                dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
                if rank == 0:
                    current_val_loss = total_val_loss_tensor.item() / len(val_loader.dataset)
            else:
                current_val_loss = val_running_loss / len(val_loader.dataset)

        stop_signal = False
        if rank == 0:
            epoch_duration = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"Epoch [{current_epoch_display}/{num_epochs}] ({epoch_duration:.2f}s) - "
                f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {current_val_loss:.4f}, LR: {current_lr:.1e}"
            )

            if not math.isnan(current_val_loss):
                scheduler.step(current_val_loss)

                if current_val_loss < best_val_loss_for_saving:
                    best_val_loss_for_saving = current_val_loss
                    torch.save(raw_model.state_dict(), model_save_path)
                    print(f"  Saved new best model to {model_save_path} (Val Loss: {best_val_loss_for_saving:.4f})")

                if current_val_loss < best_val_loss_for_early_stop - EARLY_STOPPING_MIN_DELTA:
                    best_val_loss_for_early_stop = current_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"  Early stopping counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
                    stop_signal = True

            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss_for_saving': best_val_loss_for_saving,
                'best_val_loss_for_early_stop': best_val_loss_for_early_stop,
                'epochs_no_improve': epochs_no_improve,
            }
            torch.save(checkpoint_state, latest_checkpoint_path)

        if is_ddp:
            stop_tensor = torch.tensor(1 if stop_signal else 0, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_signal = bool(stop_tensor.item())

        if stop_signal:
            break

    if rank == 0:
        print("\nTraining completed.")
        if not math.isnan(best_val_loss_for_saving):
            print(f"Best validation loss: {best_val_loss_for_saving:.4f}")
        print(f"Best model weights saved to: {model_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LENS U-Net Potential Field Estimator")
    parser.add_argument("--preset", type=str, choices=["paper", "original"], default="paper",
                        help="Hyperparameter preset: 'paper' (Table 11: init_channels=64, batch_size=1024) or 'original' (init_channels=256, batch_size=4096).")
    parser.add_argument("--data_file", type=Path, default=DEFAULT_PT_DATA_FILE, help="Path to preprocessed .pt dataset.")
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_MODEL_SAVE_DIR, help="Directory to save model checkpoints.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--init_channels", type=int, default=None)
    parser.add_argument("--lambda_grad", type=float, default=LAMBDA_GRAD)
    parser.add_argument("--bilinear", action="store_true", default=USE_BILINEAR_UPSAMPLE)
    parser.add_argument("--num_workers", type=int, default=6)
    args = parser.parse_args()

    init_chn = args.init_channels if args.init_channels is not None else (256 if args.preset == "original" else INIT_CHN)
    batch_size = args.batch_size if args.batch_size is not None else (4096 if args.preset == "original" else BATCH_SIZE)

    model_save_path = args.save_dir / f"unet_potential_field_{init_chn}ch_best_new_ddp.pth"
    latest_checkpoint_path = args.save_dir / f"unet_potential_field_{init_chn}ch_latest_ddp.pth"

    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Training on {world_size} process(es), device: {device}, preset: {args.preset} (init_channels={init_chn}, batch_size={batch_size})")
        args.save_dir.mkdir(parents=True, exist_ok=True)

    try:
        full_dataset = PFGDataset(pt_file_path=args.data_file)
    except Exception as e:
        if rank == 0:
            print(f"Error loading dataset: {e}")
        if is_ddp:
            dist.destroy_process_group()
        return

    num_samples = len(full_dataset)
    val_size = int(VALIDATION_SPLIT * num_samples)
    train_size = num_samples - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED_DATASET)
    )

    if rank == 0:
        print(f"Training set size  : {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), sampler=val_sampler
    )

    model = UNetPotentialField(
        n_spatial_channels=full_dataset.spatial_channels,
        n_non_spatial_features=full_dataset.non_spatial_features,
        n_classes_out=1,
        bilinear_upsample=args.bilinear,
        init_channel=init_chn
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[device.index])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR, threshold=SCHEDULER_THRESHOLD
    )

    start_epoch = 0
    initial_best_val_loss_saving = float('inf')
    initial_best_val_loss_early_stop = float('inf')
    initial_epochs_no_improve = 0

    if latest_checkpoint_path.exists():
        if rank == 0:
            print(f"Resuming from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        try:
            raw_model = model.module if is_ddp else model
            raw_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if rank == 0:
                initial_best_val_loss_saving = checkpoint['best_val_loss_for_saving']
                initial_best_val_loss_early_stop = checkpoint.get('best_val_loss_for_early_stop', float('inf'))
                initial_epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
                print(f"Checkpoint loaded. Continuing from epoch {start_epoch + 1}.")
        except Exception as e:
            if rank == 0:
                print(f"Failed to load checkpoint ({e}). Starting from scratch.")
            start_epoch = 0

    train_model(
        rank, is_ddp, device, model, train_loader, val_loader, optimizer, scheduler,
        args.epochs, start_epoch, initial_best_val_loss_saving,
        initial_best_val_loss_early_stop, initial_epochs_no_improve,
        model_save_path, latest_checkpoint_path, args.lambda_grad
    )

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
