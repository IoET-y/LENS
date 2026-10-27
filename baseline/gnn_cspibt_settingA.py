# ====================================================================
# File: gnn_cspibt_pogema_benchmark.py
# Desc: GNN-Guided CS-PIBT for Pogema using ml-mapf-with-search definitions
# ====================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import time
import argparse
import yaml
import json
import pandas as pd
from collections import defaultdict, deque
from tqdm import tqdm
import logging
import heapq
from typing import Dict, List, Tuple, Optional, Set
import queue # For BFS

# --- PyTorch Geometric ---
try:
    import torch_geometric.nn as pyg_nn
    import torch_geometric.utils as pyg_utils # Changed import
    from torch_geometric.data import Data
    # from torch_geometric.utils import dense_to_sparse # Not needed for new data creation
except ImportError:
    logging.error("torch_geometric not found. Please install it.")
    exit(1)

# --- Pogema Imports ---
try:
    from pogema import GridConfig
    from pogema_toolbox.registry import ToolboxRegistry
    from pogema_toolbox.create_env import Environment
    from create_env import create_eval_env
except ImportError as e:
    logging.error(f"Error: Pogema or pogema_toolbox import failed: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Corrected format

# ====================================================================
# 1. Constants and Mappings (Same as before)
# ====================================================================
POGEMA_ACTIONS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
POGEMA_ACTION_NAMES = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
POGEMA_STAY = 0
CS_PIBT_DELTAS = np.array([[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]], dtype=int)
NUM_CS_PIBT_ACTIONS = 5
delta_to_pogema_action_idx = {v: k for k, v in POGEMA_ACTIONS.items()}
FREE_CELL = 0
OBSTACLE_CELL = 1
INVALID_BD_VALUE = 999999

# ====================================================================
# 2. Model Definition (Copied EXACTLY from ml-mapf-with-search/main_pys/model.py)
# ====================================================================

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, linear_dim, in_channels, out_channels,relu_type):
        super(CustomConv, self).__init__(aggr='add')
        self.lin = nn.Linear(linear_dim, out_channels)
        self.lin_self = nn.Linear(linear_dim, out_channels)
        # Assuming kernel_size=3, stride=1, padding=0 based on linear_dim calculation needs
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.conv_self = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.relu_type=relu_type

    def forward(self, x, bd_pred, edge_index):
        # x shape: (N, C, D, D) where C=in_channels, D=2k+1
        # bd_pred shape: (N, 5)
        # edge_index shape: (2, E)
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Apply 2D conv, expecting (N, C, H, W) input
        # Conv output size: (N, C, D-2, D-2)
        conv_out = self.conv_self(x)
        flattened_conv = torch.flatten(conv_out, start_dim=1) # Shape: (N, C*(D-2)*(D-2))

        # Dynamically check bd_pred shape before stacking
        # Make sure bd_pred is on the same device and dtype
        bd_pred_device = bd_pred.to(flattened_conv.device, dtype=flattened_conv.dtype)
        
        # Check if bd_pred has more than one dimension and more than 2 elements (basic sanity check)
        if bd_pred.ndim > 1 and bd_pred.shape[0]>0 and bd_pred.shape[1] > 0:
            combined_features = torch.hstack([flattened_conv, bd_pred_device]) # Shape (N, C*(D-2)*(D-2) + 5)
        else:
            # Handle cases where bd_pred might be empty or invalid (e.g., num_agents=0)
            combined_features = flattened_conv # Fallback if bd_pred is problematic


        if self.relu_type!="relu":
            activated_features = F.leaky_relu(combined_features)
        else:
            activated_features = F.relu(combined_features)

        self_x = self.lin_self(activated_features) # Output shape (N, out_channels)

        # Apply activation again for neighbor features before linear layer
        # Note: Using the same combined_features and activation as self_x calculation
        if self.relu_type!="relu":
             x_neighbors_pre_lin = F.leaky_relu(combined_features)
        else:
             x_neighbors_pre_lin = F.relu(combined_features)

        x_neighbors = self.lin(x_neighbors_pre_lin) # Output shape (N, out_channels)

        # Propagate neighbor features
        propagated_neighbors = self.propagate(edge_index, x=x_neighbors) # Shape (N, out_channels)

        self_and_propogated = self_x + propagated_neighbors # Shape (N, out_channels)
        return self_and_propogated

    def message(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out

class GNNStack(nn.Module):
    def __init__(self, linear_dim, in_channels, hidden_dim, output_dim, relu_type, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.relu_type = relu_type # Store relu_type for forward pass checks
        # Create conv layers: 1 CustomConv + 3 SAGEConv
        self.convs = nn.ModuleList([self.build_conv_model(linear_dim, in_channels, hidden_dim, True)]) # CustomConv
        for _ in range(3): # SAGEConv x 3
            self.convs.append(self.build_conv_model(linear_dim, hidden_dim, hidden_dim, False)) # SAGEConv expects hidden_dim input

        # Create LayerNorm layers (total 5 needed based on original code loop structure)
        self.lns = nn.ModuleList()
        for _ in range(5): # Create 5 LayerNorms
             self.lns.append(nn.LayerNorm(hidden_dim))

        # Post-message passing MLP
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.25), # Hardcoded dropout
            nn.Linear(hidden_dim, output_dim)
        )

        if task not in ['node', 'graph']:
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25 # Hardcoded dropout
        self.num_layers = 4 # Hardcoded number of conv layers

        # Define the relu function based on relu_type for use in forward
        if self.relu_type == "leaky_relu":
             self.activation = F.leaky_relu
        else: # Default to relu
             self.activation = F.relu

    def build_conv_model(self, linear_dim, in_channels, hidden_dim, is_first_layer):
        if is_first_layer:
            # First layer is CustomConv
            return CustomConv(linear_dim, in_channels, hidden_dim, self.relu_type)
        else:
            # Subsequent layers are SAGEConv
            # SAGEConv input dim = hidden_dim, output dim = hidden_dim
            return pyg_nn.SAGEConv(in_channels, hidden_dim)

    def forward(self, data):
        x, edge_index, batch, bd_pred = data.x, data.edge_index, data.batch, data.bd_pred

        # Handle case where x might be empty if num_agents is 0
        if x.shape[0] == 0:
             # Return empty tensors with correct final dimensions
             emb = torch.empty((0, args.gnn_hidden_dim), device=x.device) # Use hidden_dim from args
             output = torch.empty((0, NUM_CS_PIBT_ACTIONS), device=x.device) # Use output dim
             return emb, output # Return log_softmax later if needed or adjust post_mp

        # Apply first layer (CustomConv)
        x = self.convs[0](x, bd_pred, edge_index)
        # Note: LayerNorm seems misindexed in original code loop (starts lns[1]). Let's try matching that.
        # If LayerNorm is applied after first layer, it should be lns[0]
        # x = self.lns[0](x) # Optional: Apply LN after first layer if intended

        emb = x # Store intermediate embedding

        # Apply subsequent layers (SAGEConv)
        for i in range(1, self.num_layers): # Layers 1, 2, 3
            x = self.convs[i](x, edge_index) # SAGEConv only needs x, edge_index
            emb = x # Update embedding before activation/dropout/ln

            # Apply activation specified by relu_type
            x = self.activation(x)

            x = F.dropout(x, p=self.dropout, training=self.training)

            # Apply LayerNorm (original code applies lns[1], lns[2], lns[3])
            # Check index bounds: len(self.lns) should be at least self.num_layers + 1 if accessing lns[i] here
            # Assuming 5 lns were created: lns[0]...lns[4]
            if i < self.num_layers: # Apply LN for layers 1, 2, 3 (indices 1, 2, 3)
                 if i < len(self.lns): # Safety check
                     x = self.lns[i](x)
                 else:
                     logging.warning(f"LayerNorm index {i} out of bounds (max: {len(self.lns)-1}). Skipping LN.")


        # Global pooling if task is graph prediction (not used here)
        # if self.task == 'graph':
        #     x = pyg_nn.global_mean_pool(x, batch)

        # Apply post-message passing MLP
        x = self.post_mp(x)

        # Return final embedding (from last conv layer before activation) and log_softmax output
        return emb, F.log_softmax(x, dim=1)

    # loss function (not needed for inference)
    # def loss(self, pred, label, weights): ...


# ====================================================================
# 3. Input Feature Creation (Copied EXACTLY from ml-mapf-with-search/main_pys/model_inputs.py)
#    Minor tweaks for variable names (grid_map_padded, cur_locs_padded)
# ====================================================================

def normalize_graph_data(data, k, edge_normalize="k", bd_normalize="center"):
    """Modifies data in place"""
    ### Normalize edge attributes
    # data.edge_attr (num_edges,2) the deltas in each direction which can be negative
    assert(edge_normalize in ["k"])
    if edge_normalize == "k":
        data.edge_attr /= k # Normalize edge attributes
    else:
        raise KeyError("Invalid edge normalization method: {}".format(edge_normalize))

    ### Normalize bd
    assert(bd_normalize in ["center"])
    bd_grid = data.x # (N,3,D,D)
    center = bd_grid[:, 1, k, k].unsqueeze(1).unsqueeze(2) # (N,1,1)
    bd_grid[:, 1, :, :] -= center
    bd_grid[:, 1, :, :] *= (1 - bd_grid[:, 0, :, :]) # Zeros out obstacles
    bd_grid[:, 1, :, :] /= (2*k)
    bd_grid[:, 1, :, :] = torch.clamp(bd_grid[:, 1, :, :], min=-1.0, max=1.0)

    data.x = bd_grid
    return data


# Keep compute_bd_for_pogema as is
def compute_bd_for_pogema(grid_map_padded: np.ndarray, goals_padded: np.ndarray) -> np.ndarray:
    """
    Computes Breadth-First Search distance (perfect heuristic 'bd') for each agent.
    Args:
        grid_map_padded: (H_pad, W_pad) boolean numpy array, True for obstacles.
        goals_padded: (N, 2) numpy array of agent goal locations [(r, c)] in padded coordinates.
    Returns:
        bd: (N, H_pad, W_pad) numpy array of distances. Unreachable cells get INVALID_BD_VALUE.
    """
    num_agents = goals_padded.shape[0]
    h_pad, w_pad = grid_map_padded.shape
    bd = np.full((num_agents, h_pad, w_pad), INVALID_BD_VALUE, dtype=np.int32)
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up

    for i in range(num_agents):
        goal_r, goal_c = goals_padded[i]

        # Check if goal is valid
        if not (0 <= goal_r < h_pad and 0 <= goal_c < w_pad and not grid_map_padded[goal_r, goal_c]):
            logging.warning(f"Agent {i}'s padded goal {goals_padded[i]} is invalid (out of bounds or obstacle). BD will be all invalid.")
            continue # Leave bd[i] as all invalid

        q = queue.Queue()
        q.put((goal_r, goal_c))
        bd[i, goal_r, goal_c] = 0

        visited = np.zeros_like(grid_map_padded, dtype=bool)
        visited[goal_r, goal_c] = True

        while not q.empty():
            r, c = q.get()
            current_dist = bd[i, r, c]

            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h_pad and 0 <= nc < w_pad and \
                   not grid_map_padded[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    bd[i, nr, nc] = current_dist + 1
                    q.put((nr, nc))
    return bd

def create_data_object(cur_locs_padded, bd, grid_map_padded, k, m, goals_padded, labels=np.array([]), debug_checks=False):
    """
    Creates PyTorch Geometric Data object for ml-mapf-with-search GNNStack.
    Args:
        cur_locs_padded: (N,2) numpy array of agent positions in padded coords.
        bd: (N, H_pad, W_pad) numpy array of precomputed BFS distances.
        grid_map_padded: (H_pad, W_pad) boolean numpy array, True=Obstacle.
        k: Int, radius for local features (D = 2k+1).
        m: Int, number of closest neighbors.
        goals_padded: (N,2) numpy array of agent goals in padded coords.
        labels: Optional numpy array for training labels.
    Returns:
        torch_geometric.data.Data object
    """
    pos_list = cur_locs_padded # Rename for consistency
    bd_list = bd             # Rename
    grid = grid_map_padded   # Rename
    goal_locs = goals_padded # Rename

    num_layers = 3 # grid, bd_slices, agent_pos_slices initially
    num_agents = len(pos_list)

    # Handle empty agent list case
    if num_agents == 0:
        D = 2 * k + 1
        # Calculate expected linear_dim even if empty
        linear_dimensions = (D - 2)**2 * num_layers + 5 # Default bd_pred added
        return Data(x=torch.empty((0, num_layers, D, D), dtype=torch.float32),
                  edge_index=torch.empty((2, 0), dtype=torch.long),
                  edge_attr=torch.empty((0, 2), dtype=torch.float32), # Added empty edge_attr
                  bd_pred=torch.empty((0, 5), dtype=torch.float32),    # Use float for bd_pred consistency
                  lin_dim=linear_dimensions,
                  num_channels=num_layers,
                  y=torch.empty(0, dtype=torch.long), # Match label type if needed
                  batch=torch.empty(0, dtype=torch.long)) # Add batch

    range_num_agents = np.arange(num_agents)

    # --- Node Features (Image-like: Obstacles, BD, Agent Positions) ---
    rowLocs = pos_list[:,0][:, None] # (N,1)
    colLocs = pos_list[:,1][:, None] # (N,1)

    # Check if agents are on obstacles (using boolean grid)
    on_obstacle = grid[pos_list[:, 0], pos_list[:, 1]]
    if np.any(on_obstacle):
         logging.warning(f"Agents {np.where(on_obstacle)[0]} are starting on obstacles!")
         # Optional: Handle this? For now, proceed.
    D = 2 * k + 1 # Expected slice dimension
    expected_slice_shape = (num_agents, D, D)

    x_mesh, y_mesh = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1), indexing='ij') # Each is (D,D)
    x_mesh_abs = x_mesh[None, :, :] + rowLocs[:, None, None] # Broadcast for absolute coords (N, D, D)
    y_mesh_abs = y_mesh[None, :, :] + colLocs[:, None, None] # Broadcast for absolute coords (N, D, D)

    # Clip indices to be within grid bounds before indexing
    h_pad, w_pad = grid.shape
    x_mesh_abs = np.clip(x_mesh_abs, 0, h_pad - 1)
    y_mesh_abs = np.clip(y_mesh_abs, 0, w_pad - 1)
# Si x_mesh_abs tiene una forma (N, 1, D, D), corrígela a (N, D, D)
    if x_mesh_abs.ndim == 4 and x_mesh_abs.shape[1] == 1:
        x_mesh_abs = np.squeeze(x_mesh_abs, axis=1)
        y_mesh_abs = np.squeeze(y_mesh_abs, axis=1)
    # !!! --- FIN DE LA SOLUCIÓN --- !!!
    
# Extract slices using clipped absolute coordinates
    try:
        # --- FIX STARTS HERE ---
        # Extract grid slices (might have extra dim)
        extracted_grid_slices = grid[x_mesh_abs, y_mesh_abs] # Expect (N,D,D) or (N,1,D,D) boolean

        # Check shape and squeeze if necessary
        if extracted_grid_slices.ndim == 4 and extracted_grid_slices.shape[1] == 1:
            grid_slices = np.squeeze(extracted_grid_slices, axis=1)
        elif extracted_grid_slices.shape == expected_slice_shape:
            grid_slices = extracted_grid_slices
        else:
             raise ValueError(f"grid_slices unexpected shape after extraction: {extracted_grid_slices.shape}")

        if grid_slices.shape != expected_slice_shape:
            raise ValueError(f"grid_slices final shape incorrect! Expected {expected_slice_shape}, got {grid_slices.shape}")
        # --- FIX ENDS HERE ---

        # Use advanced indexing for bd_slices

        bd_slices = bd_list[range_num_agents[:,None,None], x_mesh_abs, y_mesh_abs] # Expect (N,D,D) float/int

        if bd_slices.shape != expected_slice_shape:
             # This is the most likely culprit if shapes differ
             raise ValueError(f"bd_slices shape mismatch! Expected {expected_slice_shape}, got {bd_slices.shape}")

        # --- REVISED Agent Position Slice Calculation (from previous fix) ---
        temp_agent_map = np.zeros_like(grid, dtype=bool)
        valid_mask = (pos_list[:, 0] >= 0) & (pos_list[:, 0] < h_pad) & \
                     (pos_list[:, 1] >= 0) & (pos_list[:, 1] < w_pad)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            temp_agent_map[pos_list[valid_indices, 0], pos_list[valid_indices, 1]] = True

        extracted_slices = temp_agent_map[x_mesh_abs, y_mesh_abs] # Extract slices (Maybe N, 1, D, D)

        # Squeeze and check shape
        if extracted_slices.ndim == 4 and extracted_slices.shape[1] == 1:
            agent_pos_slices = np.squeeze(extracted_slices, axis=1).astype(np.float32)
        elif extracted_slices.shape == expected_slice_shape:
            agent_pos_slices = extracted_slices.astype(np.float32)
        else:
            raise ValueError(f"agent_pos_slices unexpected shape after extraction: {extracted_slices.shape}")

        if agent_pos_slices.shape != expected_slice_shape:
             # Should not happen if above logic is correct, but as a safeguard
             raise ValueError(f"agent_pos_slices final shape incorrect! Expected {expected_slice_shape}, got {agent_pos_slices.shape}")

        # agent_pos_slices[:, k, k] = 0.0 # Set center to 0
        # --- END REVISED Section ---

    except IndexError as ie:
         # Catch potential index errors during slicing
         logging.error(f"Indexing error during slice creation: {ie}. Check coordinates and clipping.", exc_info=True)
         raise ValueError("Slice creation failed due to indexing.") from ie
    except ValueError as ve:
         # Catch shape mismatches or other value errors
         logging.error(f"Shape or value error during slice creation: {ve}", exc_info=True)
         raise ValueError("Slice creation failed.") from ve


    # --- Stack features ---
    # Convert types just before stacking
    grid_slices_float = grid_slices.astype(np.float32)
    bd_slices_float = bd_slices.astype(np.float32)
    # agent_pos_slices is already float32

    # Add Debug Prints
    # print(f"DEBUG: Stacking shapes: grid={grid_slices_float.shape}, bd={bd_slices_float.shape}, agents={agent_pos_slices.shape}")


    try:
        node_features = np.stack([grid_slices_float,
                                  bd_slices_float,
                                  agent_pos_slices], axis=1) # Target shape: (N, 3, D, D)
    except ValueError as e:
         # Log the shapes again if stacking fails
         logging.error(f"Stacking failed! Shapes were: grid={grid_slices_float.shape}, bd={bd_slices_float.shape}, agents={agent_pos_slices.shape}")
         raise e # Re-raise the original error

    node_features = node_features.astype(np.float32) # Ensure final type

    # --- Edge Index & Attributes ---
    agent_indices_src = np.arange(num_agents) # Source indices

    # Calculate pairwise deltas and squared distances
    deltas = pos_list[:, None, :] - pos_list[None, :, :] # Shape (N, N, 2)
    dists_sq = np.sum(deltas**2, axis=2).astype(np.float32) # Shape (N, N), use float

    # Mask distances for agents outside k-hop FOV (Manhattan distance > k)
    # Use Manhattan distance for FOV check as in original model_inputs.py logic approximation
    manhattan_dists = np.sum(np.abs(deltas), axis=2)
    fov_mask = np.any(np.abs(deltas) > k, axis=2)
    dists_sq[fov_mask] = np.inf
    np.fill_diagonal(dists_sq, np.inf) # Ignore self-distance

    # Get indices of M closest neighbors within FOV
    closest_neighbors_indices = np.argsort(dists_sq, axis=1)[:, :m] # Shape (N, m)

    # Create edge list efficiently
    source_nodes = np.repeat(agent_indices_src, m) # [0,0,..0, 1,1,..1, ...]
    target_nodes = closest_neighbors_indices.flatten() # [n1,n2..nm for agent 0, n1,n2..nm for agent 1, ...]

    # Filter out edges where distance was infinity (outside FOV or self)
    valid_edge_mask = dists_sq[source_nodes, target_nodes] != np.inf
    edge_indices = np.stack([source_nodes[valid_edge_mask], target_nodes[valid_edge_mask]]) # Shape (2, E)

    # Get edge features (deltas) for valid edges
    edge_features = deltas[edge_indices[0], edge_indices[1]].astype(np.float32) # Shape (E, 2)

    # --- BD Prediction Feature (bd_pred) ---
    # Extract 3x3 BD neighborhood around each agent
    bd_pred_arr = np.full((num_agents, 5), 0.0, dtype=np.float32) # Default to 0, use float
    if num_agents > 0: # Avoid errors if no agents
        r_coords = pos_list[:, 0]
        c_coords = pos_list[:, 1]
        # Check validity before indexing bd_list
        valid_bd_mask = (r_coords > 0) & (r_coords < h_pad - 1) & \
                        (c_coords > 0) & (c_coords < w_pad - 1)
        valid_bd_indices = np.where(valid_bd_mask)[0]

        if len(valid_bd_indices) > 0: # Proceed only if some agents have valid neighborhoods
             # Efficiently get 3x3 slices using broadcasting for valid agents
             r_valid = r_coords[valid_bd_indices]
             c_valid = c_coords[valid_bd_indices]
             idx_agent = valid_bd_indices[:, None, None]
             idx_r = np.array([-1, 0, 1])[None, :, None] + r_valid[:, None, None]
             idx_c = np.array([-1, 0, 1])[None, None, :] + c_valid[:, None, None]

             bd_3x3_slices = bd_list[idx_agent, idx_r, idx_c] # Shape (N_valid, 3, 3)

             # Extract Stop, Right, Down, Up, Left values relative to center (1,1)
             # Indices: Stop=(1,1), Right=(1,2), Down=(2,1), Up=(0,1), Left=(1,0)
             center_vals = bd_3x3_slices[:, 1, 1]
             neighbor_vals = bd_3x3_slices[:, [1, 1, 2, 0, 1], [1, 2, 1, 1, 0]] # Shape (N_valid, 5)

             # Find minimum value(s) in each row (for each agent)
             min_vals = np.min(neighbor_vals, axis=1, keepdims=True)
             is_min = (neighbor_vals == min_vals).astype(np.float32) # One-hot (or multi-hot) encoding

             # Assign to the corresponding rows in bd_pred_arr
             bd_pred_arr[valid_bd_indices] = is_min

             # Handle agents near border (where bd_3x3 couldn't be extracted fully) - leave as zeros


    # --- Calculate linear_dim ---
    # Output size of the conv layer: (D-2) x (D-2)
    # Number of channels: num_layers (3)
    # Size of bd_pred: 5
    D = 2 * k + 1
    linear_dimensions = (D - 2)**2 * num_layers + bd_pred_arr.shape[1] # C*(D-2)^2 + 5

    # Create Data object
    data = Data(x=torch.from_numpy(node_features), # (N, C, D, D)
              edge_index=torch.from_numpy(edge_indices).long(), # (2, E) Needs long type
              edge_attr=torch.from_numpy(edge_features), # (E, 2)
              bd_pred=torch.from_numpy(bd_pred_arr).float(), # (N, 5) Needs float type
              y=torch.from_numpy(labels).long() # Assuming labels are integers
             )
    # Add custom attributes needed by the model's forward pass
    data.lin_dim = linear_dimensions
    data.num_channels = num_layers
    data.num_nodes = num_agents # Add num_nodes explicitly

    # Add batch attribute if needed (for PyG batching, defaults to single graph)
    data.batch = torch.zeros(num_agents, dtype=torch.long)

    return data


# ====================================================================
# 4. Helper Functions (Path Conversion, Probability Conversion)
# ====================================================================
# Keep heuristic, path_coords_to_actions, convertProbsToPreferences

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def path_coords_to_actions(path_coords: List[Tuple[int, int]], start_pos: Tuple[int, int]) -> List[int]:
    """Converts a list of coordinate tuples to Pogema action indices."""
    actions = []
    current_path = [start_pos] + path_coords
    for i in range(len(current_path) - 1):
        dr = current_path[i+1][0] - current_path[i][0]
        dc = current_path[i+1][1] - current_path[i][1]
        action = delta_to_pogema_action_idx.get((dr, dc), POGEMA_STAY)
        actions.append(action)
    return actions if actions else [POGEMA_STAY]

def convertProbsToPreferences(probs, conversion_type="sampled"):
    """Converts probabilities to preferences (CS-PIBT action indices 0-4)"""
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    num_agents, num_actions = probs.shape
    if num_agents == 0: return np.empty((0, num_actions), dtype=int)

    if conversion_type == "sorted":
        preferences = np.argsort(-probs, axis=1)
    elif conversion_type == "sampled":
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        preferences_tensor = torch.zeros((num_agents, num_actions), dtype=torch.int64)
        all_zero_mask = torch.all(probs_tensor < 1e-7, dim=1)
        if torch.any(all_zero_mask):
             logging.debug(f"Handling {torch.sum(all_zero_mask)} agents with near-zero probabilities via default order.")
             default_prefs = torch.arange(num_actions, dtype=torch.int64).unsqueeze(0).repeat(num_agents, 1)
             preferences_tensor[all_zero_mask] = default_prefs[all_zero_mask]

        valid_mask = ~all_zero_mask
        if torch.any(valid_mask):
            valid_probs = probs_tensor[valid_mask]
            initial_sums = valid_probs.sum(dim=1, keepdims=True)
            safe_initial_sums = torch.where(initial_sums <= 1e-7, torch.ones_like(initial_sums), initial_sums)
            valid_probs = valid_probs / safe_initial_sums
            valid_probs = torch.clamp(valid_probs, min=0.0)

            temp_probs = valid_probs.clone()
            sampled_prefs = torch.zeros((temp_probs.shape[0], num_actions), dtype=torch.int64)
            epsilon = 1e-9

            for i in range(num_actions):
                probs_for_sampling = temp_probs + epsilon
                renorm_sums = probs_for_sampling.sum(dim=1, keepdims=True)
                # Handle division by zero during sampling renormalization
                safe_renorm_sums = torch.where(renorm_sums <= 1e-7, torch.ones_like(renorm_sums), renorm_sums)
                probs_for_sampling = probs_for_sampling / safe_renorm_sums
                probs_for_sampling[torch.isnan(probs_for_sampling)] = 1.0 / num_actions

                cur_sample = torch.multinomial(probs_for_sampling, num_samples=1)
                sampled_prefs[:, i] = cur_sample[:, 0]
                temp_probs.scatter_(1, cur_sample, 0)

                row_sums = temp_probs.sum(dim=1, keepdims=True)
                safe_row_sums = torch.where(row_sums <= 1e-7, torch.ones_like(row_sums), row_sums)
                temp_probs = temp_probs / safe_row_sums
                temp_probs = torch.clamp(temp_probs, min=0.0)

            preferences_tensor[valid_mask] = sampled_prefs
        preferences = preferences_tensor.numpy()
    else:
        raise ValueError('Invalid conversion type: {}'.format(conversion_type))
    return preferences

# --- Benchmark Helper Functions (load_maps, filter_maps, calculate_stats) ---
# Keep load_and_register_maps, filter_maps_by_pattern, calculate_overall_stats

def load_and_register_maps(maps_dir):
    """ Loads and registers map configurations. """
    maps_path = Path(maps_dir); map_count = 0; registered_maps = set()
    if not maps_path.is_dir(): logging.warning(f"Maps directory not found: {maps_dir}"); return False, []
    all_map_names = []
    for maps_file in maps_path.rglob('maps.yaml'): # Search recursively
        try:
            with open(maps_file, 'r') as f: maps_data = yaml.safe_load(f)
            if maps_data:
                 maps_data = {str(k): v for k, v in maps_data.items()}
                 new_maps = {k: v for k, v in maps_data.items() if k not in registered_maps}
                 if new_maps:
                     ToolboxRegistry.register_maps(new_maps)
                     logging.info(f"Registered {len(new_maps)} new maps from: {maps_file.name}")
                     registered_maps.update(new_maps.keys())
                     all_map_names.extend(list(new_maps.keys()))
                     map_count += len(new_maps)
                 else: logging.debug(f"No new maps to register in {maps_file.name}")
            else: logging.warning(f"No maps found or empty file: {maps_file}")
        except Exception as e: logging.error(f"Failed to load/register {maps_file}: {e}")
    if map_count == 0: logging.warning(f"No new maps registered from {maps_dir}."); return False, []
    logging.info(f"Total unique maps registered: {len(registered_maps)}")
    return True, sorted(list(registered_maps))

def filter_maps_by_pattern(all_map_names, pattern):
    """ Filters map names based on a simple wildcard pattern. """
    if pattern == "*" or not pattern: return all_map_names
    if pattern.startswith("*") and pattern.endswith("*"):
        substring = pattern[1:-1]; return [name for name in all_map_names if substring in name]
    elif pattern.startswith("*"):
        suffix = pattern[1:]; return [name for name in all_map_names if name.endswith(suffix)]
    elif pattern.endswith("*"):
        prefix = pattern[:-1]; return [name for name in all_map_names if name.startswith(prefix)]
    else: return [name for name in all_map_names if name == pattern] # Exact match

def calculate_overall_stats(detailed_results_list, args_config):
    """Calculates overall statistics from all trial results."""
    # This function seems okay, it aggregates based on dict keys like 'success',
    # 'num_agents_at_start', 'num_agents_reached_target', 'makespan', 'sum_of_costs'.
    # We will ensure run_cspibt_simulation returns these keys.
    # --- Omitting the full function body for brevity, assume it's the same ---
    # --- as in LPSS_Direct_mapf_benchmark.py but without cascade stats ---
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
        num_agents_set.add(r_dict.get('num_agents_at_start', r_dict.get('num_agents', 0)))
        maps_actually_tested_set.add(r_dict['map_name'])

        num_at_start = r_dict.get('num_agents_at_start', 0)
        num_reached = r_dict.get('num_agents_reached_target', 0)
        isr_trial = (num_reached / num_at_start) if num_at_start > 0 else (1.0 if r_dict.get('success', False) else 0.0)
        all_trial_isrs.append(isr_trial)

        if r_dict.get('success', False):
            successful_overall_runs_count += 1
            if 'sum_of_costs' in r_dict: successful_socs.append(r_dict['sum_of_costs'])
            if 'makespan' in r_dict: successful_makespans.append(r_dict['makespan'])

        if 'computation_time_sec' in r_dict: all_durations.append(r_dict['computation_time_sec'])

    total_runs_attempted = len(detailed_results_list)
    # Correct calculation for success rate based on attempted runs
    map_limit = args_config.map_limit if args_config.map_limit else None
    agent_counts_len = len(args_config.agent_counts) if args_config.agent_counts else 1
    num_trials = args_config.num_trials
    expected_total_runs = (map_limit if map_limit else len(maps_actually_tested_set)) * agent_counts_len * num_trials
    # Adjust expected if stop_on_first_success was used (complex to track exactly, use total_runs_attempted instead)

    stats['num_agents_configurations_tested'] = sorted(list(n for n in num_agents_set if n > 0))
    stats['avg_isr'] = np.mean(all_trial_isrs) if all_trial_isrs else 0.0
    stats['overall_success_rate'] = (successful_overall_runs_count / total_runs_attempted) if total_runs_attempted > 0 else 0.0
    stats['failure_rate'] = 1.0 - stats['overall_success_rate']
    stats['avg_soc_on_overall_success'] = np.mean(successful_socs) if successful_socs else float('nan')
    stats['std_soc_on_overall_success'] = np.std(successful_socs) if successful_socs else float('nan')
    stats['avg_makespan_on_overall_success'] = np.mean(successful_makespans) if successful_makespans else float('nan')
    stats['std_makespan_on_overall_success'] = np.std(successful_makespans) if successful_makespans else float('nan')
    stats['avg_duration_s_per_run'] = np.mean(all_durations) if all_durations else float('nan')
    stats['maps_tested_count'] = len(maps_actually_tested_set)
    stats['runs_attempted_total'] = total_runs_attempted

    return stats

# ====================================================================
# 5. CS-PIBT Core Logic (Keep pibt, pibtRecursive, updatePriorities)
# ====================================================================
# Keep updatePriorities, pibtRecursive, pibt functions exactly as before

def updatePriorities(prev_priorities: np.ndarray, at_goal: np.ndarray) -> np.ndarray:
    """ Updates agent priorities based on PIBT rules. """
    agent_priorities = prev_priorities.copy()
    agent_priorities[(prev_priorities <= 0) & at_goal] -= 1
    agent_priorities[(prev_priorities > 0) & at_goal] = 0
    not_at_goal_mask = ~at_goal
    agent_priorities[not_at_goal_mask] = np.maximum(prev_priorities[not_at_goal_mask], 0) + 1
    return agent_priorities

def pibtRecursive(
    grid_map: np.ndarray, agent_id: int, action_preferences: np.ndarray,
    planned_agents: np.ndarray, move_deltas: np.ndarray,
    occupied_nodes: np.ndarray, occupied_edges: Dict[Tuple[int, int, int, int], bool],
    current_locs: np.ndarray, current_locs_to_agent: np.ndarray,
    start_time: float, time_limit: float
) -> bool:
    """ Recursive part of PIBT. """
    if time_limit > 0 and (time.time() - start_time) > time_limit:
        return False # Timeout

    current_pos = tuple(current_locs[agent_id])
    map_h, map_w = grid_map.shape

    for action_idx in action_preferences[agent_id]:
        aMove = tuple(CS_PIBT_DELTAS[action_idx])
        next_loc = (current_pos[0] + aMove[0], current_pos[1] + aMove[1])

        # Collision Checks
        if not (0 <= next_loc[0] < map_h and 0 <= next_loc[1] < map_w): continue
        if grid_map[next_loc[0], next_loc[1]]: continue
        if occupied_nodes[next_loc[0], next_loc[1]]: continue
        reverse_edge_key = tuple([*next_loc, *current_pos])
        if occupied_edges.get(reverse_edge_key, False): continue

        # Tentatively Apply Move
        move_deltas[agent_id] = aMove
        planned_agents[agent_id] = True
        occupied_nodes[next_loc[0], next_loc[1]] = True
        edge_key = tuple([*current_pos, *next_loc])
        occupied_edges[edge_key] = True

        # Check Conflict with Lower Priority Agents
        conflicting_agent_id = current_locs_to_agent[next_loc[0], next_loc[1]]
        conflict_resolved = True
        if conflicting_agent_id != -1 and conflicting_agent_id != agent_id and not planned_agents[conflicting_agent_id]:
            conflict_resolved = pibtRecursive(
                grid_map, conflicting_agent_id, action_preferences, planned_agents,
                move_deltas, occupied_nodes, occupied_edges, current_locs,
                current_locs_to_agent, start_time, time_limit
            )

        # Finalize or Backtrack
        if conflict_resolved:
            return True
        else:
            planned_agents[agent_id] = False
            occupied_nodes[next_loc[0], next_loc[1]] = False
            occupied_edges[edge_key] = False
            move_deltas[agent_id] = [0, 0]
            if time_limit > 0 and (time.time() - start_time) > time_limit: return False

    move_deltas[agent_id] = [0,0] # Explicitly set stay if function returns False
    return False

def pibt(
    grid_map: np.ndarray, action_preferences: np.ndarray,
    current_locs: np.ndarray, agent_priorities: np.ndarray,
    time_limit_pibt: float = -1.0
) -> Tuple[np.ndarray, bool]:
    """ PIBT main function. """
    start_time_pibt = time.time()
    num_agents = len(agent_priorities)
    map_h, map_w = grid_map.shape
    agent_order = np.argsort(-agent_priorities) # Highest priority first

    move_deltas = np.zeros((num_agents, 2), dtype=int)
    planned_agents = np.zeros(num_agents, dtype=bool)
    occupied_nodes = np.zeros(grid_map.shape, dtype=bool)
    occupied_edges = defaultdict(bool)

    current_locs_to_agent = np.full(grid_map.shape, -1, dtype=int)
    valid_loc_indices = (current_locs[:, 0] >= 0) & (current_locs[:, 0] < map_h) & \
                        (current_locs[:, 1] >= 0) & (current_locs[:, 1] < map_w)
    valid_agents = np.arange(num_agents)[valid_loc_indices]
    if len(valid_agents) > 0: # Check if there are any valid agents to map
        valid_coords = current_locs[valid_loc_indices]
        current_locs_to_agent[valid_coords[:, 0], valid_coords[:, 1]] = valid_agents

    pibt_completed_fully = True
    for agent_idx in agent_order:
        if not planned_agents[agent_idx]:
             if not valid_loc_indices[agent_idx]:
                 logging.debug(f"Skipping PIBT for agent {agent_idx} (off-map). Setting action to stay.")
                 move_deltas[agent_idx] = [0, 0]
                 planned_agents[agent_idx] = True # Mark as planned (to stay)
                 continue

             success = pibtRecursive(
                 grid_map, agent_idx, action_preferences, planned_agents,
                 move_deltas, occupied_nodes, occupied_edges, current_locs,
                 current_locs_to_agent, start_time_pibt, time_limit_pibt
             )
             if not success:
                 pibt_completed_fully = False
                 move_deltas[agent_idx] = [0, 0] # Explicitly stay if failed
                 planned_agents[agent_idx] = True # Mark as planned (to stay)
                 if time_limit_pibt > 0 and (time.time() - start_time_pibt) > time_limit_pibt:
                      logging.warning(f"PIBT timed out during planning for agent {agent_idx}.")
                 else:
                      logging.warning(f"PIBT failed for agent {agent_idx} (could not find move). Agent forced to stay.")
                 break # Stop planning if one fails

    for i in range(num_agents):
        if not planned_agents[i]: move_deltas[i] = [0, 0]

    return move_deltas, pibt_completed_fully

# ====================================================================
# 6. GNN-Guided CS-PIBT Simulation Function (Updated for new model/data)
# ====================================================================

def run_gnn_cspibt_simulation(
    env, model, device, k_padding, m_neighbors,
    max_episode_steps=256, verbose=False, time_limit_pibt_step=-1.0,
):
    sim_start_time = time.time()
    model.eval()

    num_agents = env.grid_config.num_agents
    agents_global_positions_unpadded = np.array(env.get_agents_xy(), dtype=int)
    agents_global_goals_unpadded = np.array(env.get_targets_xy(), dtype=int)
    global_obstacle_map_unpadded = env.unwrapped.grid.get_obstacles().astype(bool)
    map_h, map_w = global_obstacle_map_unpadded.shape

    # Apply padding

    # Compute BD on PADDED map
    logging.info("Computing Perfect Heuristic (BD)...")
    bd_compute_start = time.time()
    bd = compute_bd_for_pogema(global_obstacle_map_unpadded, agents_global_goals_unpadded)
    logging.info(f"BD computation took {time.time() - bd_compute_start:.2f} seconds.")

    # Initialize state
    agents_active = np.array([True] * num_agents)
    initial_distances = np.array([heuristic(tuple(pos), tuple(goal)) for pos, goal in zip(agents_global_positions_unpadded, agents_global_goals_unpadded)], dtype=float)
    agent_priorities = initial_distances / (np.max(initial_distances) + 1e-6)

    executed_paths_global = {i: [tuple(agents_global_positions_unpadded[i])] for i in range(num_agents)}
    total_env_steps_taken = 0
    simulation_successful = True
    error_messages = []
    truncated = {i: False for i in range(num_agents)}

    # --- Main Simulation Loop ---
    while np.any(agents_active) and total_env_steps_taken < max_episode_steps:
        planning_step_start_time = time.time()
        active_agent_indices = np.where(agents_active)[0]
        if len(active_agent_indices) == 0: break

        current_locs_unpadded = agents_global_positions_unpadded

        at_goal = np.array([tuple(pos) == tuple(goal) for pos, goal in zip(current_locs_unpadded, agents_global_goals_unpadded)], dtype=bool)
        agent_priorities = updatePriorities(agent_priorities, at_goal)

        try:
            with torch.no_grad():
                # 1. Create Data object using the CORRECTED function
                data = create_data_object(
                    current_locs_unpadded, bd, global_obstacle_map_unpadded, 
                    k_padding, m_neighbors, agents_global_goals_unpadded,
                )
                # 2. Normalize graph data
                data = normalize_graph_data(data, k_padding)
                # 3. Move to device
                data = data.to(device)
                # 4. Model Inference (expects data.x, data.bd_pred, data.edge_index)
                _, predictions_logsoftmax = model(data) # Model returns log_softmax
                # 5. Convert log_softmax to probabilities
                probabilities = torch.exp(predictions_logsoftmax)
                probs_np = probabilities.cpu().numpy() # Shape (N, 5)

            # --- Mask Invalid Actions (using UNPADDED map/coords) ---
            action_mask = np.zeros_like(probs_np, dtype=bool)
            for agent_id in active_agent_indices:
                 # Check if agent is currently on map before masking
                 if agent_id >= len(current_locs_unpadded): continue # Should not happen if indices are correct
                 r, c = current_locs_unpadded[agent_id]
                 # Ensure r,c are valid indices themselves
                 if not (0 <= r < map_h and 0 <= c < map_w):
                      action_mask[agent_id, :] = True # Mask all if agent starts off-map
                      continue

                 for cs_pibt_idx in range(NUM_CS_PIBT_ACTIONS):
                     dr, dc = CS_PIBT_DELTAS[cs_pibt_idx]
                     nr, nc = r + dr, c + dc
                     if not (0 <= nr < map_h and 0 <= nc < map_w and not global_obstacle_map_unpadded[nr, nc]):
                         action_mask[agent_id, cs_pibt_idx] = True

            probs_np[action_mask] = 1e-9 # Apply mask
            row_sums = probs_np.sum(axis=1, keepdims=True)
            all_masked = (row_sums < 1e-8).flatten()
            if np.any(all_masked):
                 masked_agents = np.where(all_masked)[0]
                 logging.warning(f"Step {total_env_steps_taken+1}: Agents {masked_agents} have all actions masked (trapped?). Forcing Stay preference.")
                 probs_np[all_masked, :] = 1e-9
                 probs_np[all_masked, 0] = 1.0
                 row_sums[all_masked] = 1.0

            safe_row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
            probs_np = probs_np / safe_row_sums

            # --- Convert Probabilities to Preferences ---
            action_preferences = convertProbsToPreferences(probs_np, "sampled")

        except Exception as e:
            logging.error(f"Error during GNN inference/preference generation step {total_env_steps_taken + 1}: {e}", exc_info=True)
            logging.warning("Falling back to Stay action for all agents.")
            action_preferences = np.zeros((num_agents, NUM_CS_PIBT_ACTIONS), dtype=int)
            move_deltas = np.zeros((num_agents, 2), dtype=int)
            pibt_success = False
            simulation_successful = False
            error_messages.append(f"GNN_ERR@S{total_env_steps_taken+1}")
            # Proceed with Stay actions

        if simulation_successful: # Only run PIBT if GNN succeeded
            if verbose: logging.debug(f"Step {total_env_steps_taken + 1}: Running PIBT...")
            move_deltas, pibt_success = pibt(
                global_obstacle_map_unpadded, # Use unpadded map
                action_preferences,
                current_locs_unpadded, # Use unpadded locations
                agent_priorities,
                time_limit_pibt_step
            )
            if not pibt_success:
                 logging.warning(f"Step {total_env_steps_taken + 1}: PIBT failed or timed out.")

        # --- Convert Deltas to Pogema Actions ---
        actions_for_env = []
        expected_next_positions = {}
        for i in range(num_agents):
            if not agents_active[i]:
                actions_for_env.append(POGEMA_STAY)
                expected_next_positions[i] = tuple(agents_global_positions_unpadded[i])
            else:
                delta = tuple(move_deltas[i])
                pogema_action = delta_to_pogema_action_idx.get(delta, POGEMA_STAY)
                actions_for_env.append(pogema_action)
                expected_next_positions[i] = (agents_global_positions_unpadded[i][0] + delta[0],
                                              agents_global_positions_unpadded[i][1] + delta[1])

        planning_duration = time.time() - planning_step_start_time
        if verbose: logging.debug(f"  Planning took {planning_duration:.4f}s.")

        if not simulation_successful:
            logging.error("Stopping simulation due to error in planning phase.")
            break

        # --- Step Environment ---
        obs_list, rewards, terminated, truncated_dict, infos = env.step(actions_for_env)
        total_env_steps_taken += 1
        new_global_positions_unpadded = np.array(env.get_agents_xy(), dtype=int)

        # Update State
        agents_global_positions_unpadded = new_global_positions_unpadded

        step_had_truncation = False
        active_agents_now = 0
        for i in range(num_agents):
            if agents_active[i]:
                pos_tuple = tuple(new_global_positions_unpadded[i])
                if pos_tuple != expected_next_positions[i]:
                     logging.debug(f"  Pos mismatch Agent {i} Step {total_env_steps_taken}: Expected {expected_next_positions[i]}, Got {pos_tuple}.")
                executed_paths_global[i].append(pos_tuple)

                if terminated[i]:
                    agents_active[i] = False
                    if pos_tuple != tuple(agents_global_goals_unpadded[i]):
                        logging.warning(f"Agent {i} terminated at {pos_tuple}, goal was {tuple(agents_global_goals_unpadded[i])}")
                # CORRECTO
                elif truncated_dict[i]: # Check agent index in list
                    agents_active[i] = False
                    truncated[i] = True
                    simulation_successful = False
                    step_had_truncation = True
                    error_messages.append(f"A{i}_TRUNC@S{total_env_steps_taken}")
                else:
                    active_agents_now += 1

        if verbose: logging.debug(f"  Step {total_env_steps_taken} finished. Active agents: {active_agents_now}.")
        if not simulation_successful: break

    # --- Final Result Calculation --- (Same as before)
    sim_duration = time.time() - sim_start_time
    num_agents_finished_at_goal = 0
    all_agents_finished_correctly = True

    for i in range(num_agents):
        agent_final_pos = tuple(agents_global_positions_unpadded[i])
        agent_goal = tuple(agents_global_goals_unpadded[i])
        is_finished_without_trunc = not agents_active[i] and not truncated.get(i, False)

        if is_finished_without_trunc:
            if agent_final_pos == agent_goal:
                num_agents_finished_at_goal += 1
            else:
                 all_agents_finished_correctly = False
                 error_messages.append(f"A{i}_FIN_WRONG_LOC")
        elif agents_active[i]:
             all_agents_finished_correctly = False
             error_messages.append(f"A{i}_DID_NOT_FINISH")

    final_success = simulation_successful and not np.any(agents_active) and all_agents_finished_correctly
    valid_costs = [len(executed_paths_global[i]) - 1 for i in range(num_agents)
                   if not agents_active[i] and not truncated.get(i, False) and
                   tuple(agents_global_positions_unpadded[i]) == tuple(agents_global_goals_unpadded[i])]
    sum_of_costs = sum(valid_costs)

    final_result = {
        "success": final_success, "makespan": total_env_steps_taken, "sum_of_costs": sum_of_costs,
        "individual_costs": {i: len(p) - 1 for i, p in executed_paths_global.items()},
        "executed_paths_global": executed_paths_global,
        "error_summary": "; ".join(error_messages) if error_messages else "No errors.",
        "num_agents_at_start": num_agents, "num_agents_reached_target": num_agents_finished_at_goal,
        "computation_time_sec": sim_duration,
    }
    return final_result


# ====================================================================
# 7. Main Benchmark Execution Logic (Adapted for new GNN Model)
# ====================================================================

def run_benchmark_main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    animations_dir = output_dir / "animations"; animations_dir.mkdir(exist_ok=True)
    animations_fail_dir = output_dir / "animations_fail"; animations_fail_dir.mkdir(exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")

    maps_registered_ok, all_available_map_names = load_and_register_maps(args.map_config_dir)
    if not maps_registered_ok and not all_available_map_names: exit(1)
    maps_to_run_on = filter_maps_by_pattern(all_available_map_names, args.map_pattern)
    if not maps_to_run_on: logging.error(f"No maps matching '{args.map_pattern}'."); exit(1)
    if args.map_limit is not None and args.map_limit > 0 and args.map_limit < len(maps_to_run_on):
        maps_to_run_on = maps_to_run_on[:args.map_limit]
        logging.info(f"Limiting to first {args.map_limit} maps.")

    logging.info(f"Algorithm: GNN-Guided CS-PIBT (ml-mapf-with-search arch)")
    logging.info(f"Will run on {len(maps_to_run_on)} maps, agent counts: {args.agent_counts}.")

    # --- Load GNN Model (Robust Approach) ---
    model_path = Path(args.model_path)
    if not model_path.is_file(): logging.error(f"Model file not found: {model_path}"); exit(1)
    try:
        logging.info(f"Loading model checkpoint from {model_path}...")
        loaded_object = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = None

        if isinstance(loaded_object, torch.nn.Module):
            logging.info("Checkpoint contained a model instance. Extracting state_dict.")
            state_dict = loaded_object.state_dict()
        elif isinstance(loaded_object, dict):
            logging.info("Checkpoint contained a dictionary.")
            state_dict = loaded_object.get('model_state_dict', loaded_object)
        else:
            raise TypeError(f"Loaded checkpoint is of unexpected type: {type(loaded_object)}")
        if state_dict is None: raise ValueError("Failed to extract state_dict.")

        # --- Instantiate model with CORRECT ARCHITECTURE ---
        # Calculate linear_dim based on k and num_channels=3
        D = 2 * args.k + 1
        num_feat_channels = 3 # Obstacle, BD, Agents
        linear_dim_calc = (D - 2)**2 * num_feat_channels + 5 # 5 for bd_pred
        logging.info(f"Instantiating NEW GNNStack (ml-mapf arch) with linear_dim={linear_dim_calc}, in_channels={num_feat_channels}, hidden_dim={args.gnn_hidden_dim}, num_layers=4")

        model = GNNStack(
            linear_dim=linear_dim_calc,
            in_channels=num_feat_channels, # Input channels for Conv2D
            hidden_dim=args.gnn_hidden_dim,
            output_dim=NUM_CS_PIBT_ACTIONS,
            relu_type=args.gnn_relu_type
            # num_layers is hardcoded to 4 inside GNNStack now
        )

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys: logging.warning(f"State_dict loading: Missing keys: {missing_keys}")
        if unexpected_keys: logging.warning(f"State_dict loading: Unexpected keys: {unexpected_keys}")

        model.to(device).eval()
        logging.info(f"GNN model created from definition and loaded weights successfully on {device}.")

    except Exception as e:
        logging.error(f"Error loading/configuring GNN model from {model_path}: {e}", exc_info=True)
        exit(1) # Exit if model loading fails

    # --- Run Benchmark Loop ---
    all_run_results_detailed = []
    summary_results_per_scenario = []
    total_scenarios = len(maps_to_run_on) * len(args.agent_counts)
    scenario_pbar = tqdm(total=total_scenarios, desc="Scenarios")
    benchmark_overall_start_time = time.time()

    for map_name in maps_to_run_on:
        for num_agents in args.agent_counts:
            scenario_pbar.set_description(f"Map: {map_name}, A:{num_agents}")
            tqdm.write(f"\n--- Scenario: Map='{map_name}', Agents={num_agents} ---")

            scenario_trial_results = []
            scenario_succeeded_once = False

            for trial_idx in range(args.num_trials):
                if args.stop_scenario_on_first_success and scenario_succeeded_once:
                    tqdm.write(f"  Skipping Trial {trial_idx + 1} (already succeeded).")
                    break

                tqdm.write(f"  Trial {trial_idx + 1}/{args.num_trials}")
                current_seed = (args.seed + trial_idx) if args.seed is not None else int(time.time() * 1000) % (2**32)

                env = None
                try:
                    env_grid_config = Environment(
                        map_name=map_name, num_agents=num_agents, obs_radius=args.obs_radius,
                        observation_type="POMAPF", collision_system="soft",
                        max_episode_steps=args.max_steps, seed=current_seed, with_animation=True
                    )
                    env = create_eval_env(config=env_grid_config)
                except Exception as e:
                    logging.error(f"Failed to create env for {map_name}, A:{num_agents}, T:{trial_idx+1}: {e}", exc_info=True)
                    sim_result_dict = {"success": False, "error": f"Env creation failed: {e}",
                                       "map_name": map_name, "num_agents": num_agents, "trial": trial_idx + 1,
                                       "computation_time_sec": 0, "seed_used": current_seed,
                                       "num_agents_at_start": num_agents, "num_agents_reached_target": 0,
                                       "makespan": args.max_steps, "sum_of_costs": -1}
                    scenario_trial_results.append(sim_result_dict)
                    all_run_results_detailed.append(sim_result_dict)
                    continue

                trial_start_time = time.time()
                sim_result_dict = {}
                try:
                    sim_result_dict = run_gnn_cspibt_simulation(
                        env, model, device,
                        k_padding=args.k, # Use args.k for padding and k-neighbors
                        m_neighbors=args.m,
                        max_episode_steps=args.max_steps,
                        verbose=args.verbose_simulation,
                        time_limit_pibt_step=args.pibt_time_limit
                    )
                    trial_computation_time = time.time() - trial_start_time
                    sim_result_dict["computation_time_sec"] = trial_computation_time

                    # Save Animation
                    is_success = sim_result_dict.get('success', False)
                    anim_subdir = animations_dir if is_success else animations_fail_dir
                    sanitized_map_name = map_name.replace("/", "_").replace("\\", "_")
                    anim_filename = f"{sanitized_map_name}_A{num_agents}_T{trial_idx+1}{'_FAIL' if not is_success else ''}.svg"
                    anim_path = anim_subdir / anim_filename
                    try:
                        env.save_animation(str(anim_path))
                    except Exception as e_anim:
                        logging.error(f"Failed to save animation for {map_name}, T:{trial_idx+1}: {e_anim}")

                except Exception as e:
                    trial_computation_time = time.time() - trial_start_time
                    logging.error(f"Error during GNN CS-PIBT sim {map_name}, A:{num_agents}, T:{trial_idx+1}: {e}", exc_info=True)
                    sim_result_dict = {"success": False, "error": str(e),
                                       "computation_time_sec": trial_computation_time}
                finally:
                    if env is not None:
                        try: env.close()
                        except Exception as e_close: logging.warning(f"Error closing env: {e_close}")

                # Record Results
                sim_result_dict["map_name"] = map_name
                sim_result_dict["num_agents"] = num_agents
                sim_result_dict["trial"] = trial_idx + 1
                sim_result_dict["seed_used"] = current_seed
                sim_result_dict.setdefault("success", False)
                sim_result_dict.setdefault("makespan", args.max_steps if not sim_result_dict.get("success") else -1)
                sim_result_dict.setdefault("sum_of_costs", -1)
                sim_result_dict.setdefault("num_agents_at_start", num_agents)
                sim_result_dict.setdefault("num_agents_reached_target", 0)
                sim_result_dict.setdefault("computation_time_sec", trial_computation_time)

                scenario_trial_results.append(sim_result_dict)
                all_run_results_detailed.append(sim_result_dict)

                if sim_result_dict['success']: scenario_succeeded_once = True

                # Log trial result
                isr_val = sim_result_dict['num_agents_reached_target']; isr_tot = sim_result_dict['num_agents_at_start']
                tqdm.write(f"    Trial {trial_idx + 1} Result: Success={sim_result_dict['success']}, "
                           f"Makespan={sim_result_dict['makespan']}, SoC={sim_result_dict['sum_of_costs']}, "
                           f"ISR={isr_val}/{isr_tot}, Time={sim_result_dict['computation_time_sec']:.2f}s")
                if not sim_result_dict['success'] and ("error" in sim_result_dict or "error_summary" in sim_result_dict):
                    tqdm.write(f"      Error: {sim_result_dict.get('error_summary', sim_result_dict.get('error', 'Unknown'))}")


            # --- Aggregate Scenario Results ---
            if scenario_trial_results:
                # (Aggregation logic remains the same)
                actual_trials_run = len(scenario_trial_results)
                success_count = sum(1 for r in scenario_trial_results if r['success'])
                success_rate = (success_count / actual_trials_run) * 100.0 if actual_trials_run > 0 else 0.0
                valid_runs = [r for r in scenario_trial_results if r['success']]
                avg_makespan = np.mean([r['makespan'] for r in valid_runs]) if valid_runs else float('nan')
                avg_soc = np.mean([r['sum_of_costs'] for r in valid_runs]) if valid_runs else float('nan')
                avg_comp_time = np.mean([r['computation_time_sec'] for r in scenario_trial_results])
                scenario_isrs = []
                for r in scenario_trial_results:
                    num_start = r.get('num_agents_at_start', 0); num_reach = r.get('num_agents_reached_target', 0)
                    scenario_isrs.append((num_reach / num_start) if num_start > 0 else (1.0 if r.get('success') else 0.0))
                avg_isr_sc = np.mean(scenario_isrs) if scenario_isrs else 0.0

                summary_results_per_scenario.append({
                    'map': map_name, 'agents': num_agents, 'trials_run': actual_trials_run,
                    'seed_base': args.seed, 'success_rate_perc': success_rate, 'avg_isr': avg_isr_sc,
                    'avg_makespan_on_success': avg_makespan, 'avg_sum_of_costs_on_success': avg_soc,
                    'avg_trial_computation_time_sec': avg_comp_time,
                    'model_path': args.model_path, 'k': args.k, 'm': args.m,
                    'pibt_time_limit_step': args.pibt_time_limit,
                })
                tqdm.write(f"  Scenario Summary (Map: {map_name}, Agents: {num_agents}, Ran {actual_trials_run} trial(s)): "
                           f"SR: {success_rate:.1f}%, Avg ISR: {avg_isr_sc:.3f}, Avg Makespan(S): {avg_makespan:.1f}, "
                           f"Avg SoC(S): {avg_soc:.1f}, Avg Time: {avg_comp_time:.2f}s")
            scenario_pbar.update(1)
    scenario_pbar.close()

    # --- Overall Statistics and Saving --- (Same as before)
    benchmark_overall_duration = time.time() - benchmark_overall_start_time
    logging.info(f"\n--- Benchmark Completed in {benchmark_overall_duration:.2f} seconds ---")

    overall_stats = calculate_overall_stats(all_run_results_detailed, args)
    logging.info("\n--- Overall GNN CS-PIBT Benchmark Statistics ---")
    if "error" in overall_stats: logging.error(overall_stats["error"])
    else:
        print(f"  Agent Configurations Tested  : {overall_stats['num_agents_configurations_tested']}")
        print(f"  Maps Tested Count            : {overall_stats['maps_tested_count']}")
        print(f"  Total Runs Attempted         : {overall_stats['runs_attempted_total']}")
        print(f"  Avg. Individual Success Rate : {overall_stats['avg_isr']:.3f}")
        print(f"  Overall System Success Rate  : {overall_stats['overall_success_rate']:.3f}")
        print(f"  Avg. Sum of Costs (SoC)      : {overall_stats['avg_soc_on_overall_success']:.2f} (on overall success)")
        print(f"  Avg. Makespan                : {overall_stats['avg_makespan_on_overall_success']:.2f} (on overall success)")
        print(f"  Avg. Computation Time (s)    : {overall_stats['avg_duration_s_per_run']:.3f} (per run)")


    # --- Save Results ---
    if summary_results_per_scenario:
        df_summary = pd.DataFrame(summary_results_per_scenario)
        cols_summary = ['map', 'agents', 'trials_run', 'seed_base', 'success_rate_perc', 'avg_isr',
                        'avg_makespan_on_success', 'avg_sum_of_costs_on_success',
                        'avg_trial_computation_time_sec', 'model_path', 'k', 'm',
                        'pibt_time_limit_step']
        df_summary = df_summary.reindex(columns=[col for col in cols_summary if col in df_summary.columns])

        logging.info("\n--- Aggregated Per-Scenario Results (GNN CS-PIBT - ml-mapf Arch) ---")
        print(df_summary.to_string(index=False, float_format="%.3f"))

        time_str = time.strftime('%Y%m%d_%H%M%S')
        summary_csv_path = output_dir / f"gnn_cspibt_mlmapf_summary_{time_str}.csv"
        summary_json_path = output_dir / f"gnn_cspibt_mlmapf_full_data_{time_str}.json"
        try:
            df_summary.to_csv(summary_csv_path, index=False, float_format="%.3f")
            logging.info(f"Per-scenario summary saved to {summary_csv_path}")

            benchmark_output_data = {
                'algorithm': 'GNN-Guided CS-PIBT (ml-mapf Arch)',
                'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'overall_statistics': overall_stats,
                'summary_per_scenario': df_summary.to_dict(orient='records'),
                'detailed_trial_results': all_run_results_detailed
            }
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, Path): return str(obj)
                    return super(NumpyEncoder, self).default(obj)
            with open(summary_json_path, 'w') as f:
                json.dump(benchmark_output_data, f, indent=2, cls=NumpyEncoder)
            logging.info(f"Full benchmark data saved to {summary_json_path}")

        except Exception as e:
            logging.error(f"Error saving results: {e}", exc_info=True)
    else:
        logging.warning("No scenarios were successfully summarized.")


if __name__ == "__main__":
    # --- Default Arguments (Reflecting ml-mapf-with-search) ---
    DEFAULT_MAX_EPISODE_STEPS = 512
    DEFAULT_OBS_RADIUS = 5
    DEFAULT_NUM_TRIALS = 1
    DEFAULT_PIBT_TIME_LIMIT = -1.0
    DEFAULT_K_NEIGHBORS = 4 # Must match data generation
    DEFAULT_M_NEIGHBORS = 5 # Used for graph edges
    DEFAULT_GNN_HIDDEN_DIM = 128
    # DEFAULT_GNN_NUM_LAYERS = 4 # Hardcoded in model def
    DEFAULT_GNN_RELU_TYPE = 'relu'

    parser = argparse.ArgumentParser(description="GNN-Guided CS-PIBT (ml-mapf Arch) Benchmarking")

    # Paths & Scenario Selection
    parser.add_argument("--map_config_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="benchmark_output_gnn_cspibt_mlmapf")
    parser.add_argument("--map_pattern", type=str, default="*")
    parser.add_argument("--map_limit", type=int, default=None)
    parser.add_argument("--agent_counts", type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument("--num_trials", type=int, default=DEFAULT_NUM_TRIALS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop_scenario_on_first_success", action='store_true')

    # Model Parameters
    parser.add_argument("--model_path", type=str, default="./ML-MAPF-with-Search/model/ssil_model.pt", help="Path to the pre-trained GNN model checkpoint (.pt file).")
    parser.add_argument("--k", type=int, default=DEFAULT_K_NEIGHBORS, help="K radius for data object features (e.g., 4 => 9x9).")
    parser.add_argument("--m", type=int, default=DEFAULT_M_NEIGHBORS, help="M closest neighbors for graph.")
    parser.add_argument("--gnn_hidden_dim", type=int, default=DEFAULT_GNN_HIDDEN_DIM)
    # num_layers is fixed at 4 by the model definition
    parser.add_argument("--gnn_relu_type", type=str, default=DEFAULT_GNN_RELU_TYPE, choices=['relu', 'leaky_relu'])

    # Simulation Parameters
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--obs_radius", type=int, default=DEFAULT_OBS_RADIUS)
    parser.add_argument("--pibt_time_limit", type=float, default=DEFAULT_PIBT_TIME_LIMIT)

    # Execution & Logging
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--verbose_simulation", action="store_true")

    args = parser.parse_args()

    if args.verbose_simulation: logging.getLogger().setLevel(logging.DEBUG)
    else: logging.getLogger().setLevel(logging.INFO)

    if not Path(args.model_path).is_file():
         logging.error(f"FATAL: Model file not found at {args.model_path}")
         exit(1)

    run_benchmark_main(args)