# LENS_l_setA.py (Setting A: Known Static Obstacle Map, Unknown Other Agents)

import os
import torch
import numpy as np
import time
import random
import logging
from typing import Dict, List, Tuple, Optional, Set, Deque, Any, cast
import networkx as nx
from collections import defaultdict, deque
import heapq
import itertools
import numba

from unet_model_new import UNetPotentialField
from action_sequence_decoder_l import decode_action_sequence_refined
from local_cbs_solver_robust_l import solve_local_cbs_robust, detect_all_conflicts_spacetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.patches as patches

# Define custom colormaps for visualization
# p_map: 0=FREE (White), 1=OBSTACLE (Black), 2=UNKNOWN (Gray)
PMAP_COLORS = ['#FFFFFF', '#000000', '#808080']
PMAP_CMAP = mcolors.ListedColormap(PMAP_COLORS)
PMAP_NORM = mcolors.BoundaryNorm([0, 1, 2, 3], PMAP_CMAP.N)

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
ACTION_DELTAS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}; ACTION_STAY = 0
UNKNOWN_CELL, FREE_CELL, OBSTACLE_CELL = 2, 0, 1


def plot_p_map(p_map: np.ndarray, agents_pos: List[Tuple[int, int]], agents_goals: List[Tuple[int, int]], step: int, save_path: str):
    """Visualizes the persistent known map (p_map)."""
    plt.figure(figsize=(12, 12))
    plt.imshow(p_map, cmap=PMAP_CMAP, norm=PMAP_NORM, interpolation='nearest')
    
    # Plot agents (blue circles) and their goals (red stars)
    if agents_pos:
        pos_r, pos_c = zip(*agents_pos)
        plt.scatter(pos_c, pos_r, c='blue', marker='o', s=20, label='Agents')
    if agents_goals:
        goal_r, goal_c = zip(*agents_goals)
        plt.scatter(goal_c, goal_r, c='red', marker='*', s=30, label='Goals')

    plt.title(f'Global Known Map (p_map) at Step {step}')
    plt.legend()
    plt.savefig(f"{save_path}/p_map_step_{step:04d}.png")
    plt.close()

def plot_heuristic_map(h_map: np.ndarray, agent_pos: Tuple[int, int], agent_goal: Tuple[int, int], agent_id: int, step: int, save_path: str):
    """Visualizes an agent's true-distance heuristic map."""
    plt.figure(figsize=(12, 12))
    # Mask unreachable cells (-1) to show them as gray
    masked_h_map = np.ma.masked_where(h_map == -1, h_map)
    
    ax = sns.heatmap(masked_h_map, cmap='viridis_r', cbar=True, square=True)
    ax.set_facecolor('gray') # Color for unreachable cells
    
    # Overlay agent position and goal
    plt.scatter([agent_pos[1]+0.5], [agent_pos[0]+0.5], c='cyan', marker='o', s=50, label='Agent')
    plt.scatter([agent_goal[1]+0.5], [agent_goal[0]+0.5], c='magenta', marker='*', s=80, label='Goal')

    plt.title(f'Agent {agent_id} Heuristic Map at Step {step}')
    plt.legend()
    plt.savefig(f"{save_path}/agent_{agent_id}_h_map_step_{step:04d}.png")
    plt.close()

def plot_dezd_activation(p_map: np.ndarray, personal_obs_map: np.ndarray, trap_zone: Set[Tuple[int, int]], agent_pos: Tuple[int, int], agent_id: int, step: int, save_path: str,goals: Set[Tuple[int, int]]):
    """Visualizes the activation of the DEZD mechanism for an agent."""
    
    # Create a composite map: 0=Free, 1=Obstacle, 2=DEZD Trap
    viz_map = p_map.copy().astype(float)
    viz_map[personal_obs_map] = 2 # Mark all personal obstacles
    
    # Highlight the newly added trap zone in a different color
    for r, c in trap_zone:
        viz_map[r, c] = 3

    cmap = mcolors.ListedColormap(['#FFFFFF', '#000000', '#FF6347', '#FFD700']) # White, Black, Tomato (DEZD), Gold (New Trap)
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    plt.figure(figsize=(12, 12))
    plt.imshow(viz_map, cmap=cmap, norm=norm, interpolation='nearest')
    plt.scatter([agent_pos[1]+0.5], [agent_pos[0]+0.5], c='blue', marker='o', s=50, label=f'Agent {agent_id}')
    plt.scatter([goals[1] + 0.5], [goals[0] + 0.5], marker='*', s=120, c='magenta')

    # Create legend patches
    patches = [plt.Rectangle((0,0),1,1, color=c) for c in ['#000000', '#FF6347', '#FFD700']]
    labels = ['Known Obstacle', 'DEZD Personal Obstacle', 'Newly Added Trap Zone']
    plt.legend(handles=patches, labels=labels, loc='upper right')

    plt.title(f'Agent {agent_id} DEZD Activation at Step {step}')
    plt.savefig(f"{save_path}/agent_{agent_id}_dezd_step_{step:04d}.png")
    plt.close()

def plot_unet_debug(fov_obstacles: np.ndarray, potential_map: np.ndarray, decoded_path: List[Tuple[int, int]], agent_id: int, step: int, save_path: str):
    """Visualizes the U-Net input, potential field, and decoded path side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # 1. FOV Obstacle Map (Input to U-Net)
    axes[0].imshow(fov_obstacles, cmap='gray_r', interpolation='nearest')
    axes[0].set_title(f'Agent {agent_id} FOV Obstacle Input')
    axes[0].set_xlabel('Local Coords')
    axes[0].set_ylabel('Local Coords')
    
    # 2. U-Net Potential Field
    sns.heatmap(potential_map, cmap='viridis_r', ax=axes[1], cbar=True, square=True)
    axes[1].set_title('U-Net Potential Field')
    axes[1].set_xlabel('Local Coords')
    axes[1].set_ylabel('Local Coords')
    
    # 3. Decoded Path on FOV
    axes[2].imshow(fov_obstacles, cmap='gray_r', interpolation='nearest')
    if decoded_path:
        path_r, path_c = zip(*decoded_path)
        axes[2].plot(path_c, path_r, marker='o', color='cyan', linestyle='-', markersize=8, label='Decoded Path')
        # Mark start
        axes[2].plot(path_c[0], path_r[0], marker='o', color='lime', markersize=12, label='Start')
    axes[2].set_title('Decoded Path')
    axes[2].set_xlabel('Local Coords')
    axes[2].set_ylabel('Local Coords')
    axes[2].legend()

    fig.suptitle(f'U-Net Debug for Agent {agent_id} at Step {step}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_path}/agent_{agent_id}_unet_debug_step_{step:04d}.png")
    plt.close()

@numba.jit(nopython=True, cache=True)
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
def get_map_hash(grid_map: np.ndarray) -> int: return hash(grid_map.tobytes())


@numba.jit(nopython=True, cache=True)
def _compute_dist_map_numba(p_map, h, w):


    dist_map = np.full((h, w), 99999.0, dtype=np.float32)
    

    queue_r = np.zeros(h * w, dtype=np.int32)
    queue_c = np.zeros(h * w, dtype=np.int32)
    head = 0
    tail = 0


    for r in range(h):
        for c in range(w):
            if p_map[r, c] == 1: # OBSTACLE_CELL
                dist_map[r, c] = 0.0
                queue_r[tail] = r
                queue_c[tail] = c
                tail += 1
    
    # BFS loop
    dr_arr = np.array([-1, 1, 0, 0])
    dc_arr = np.array([0, 0, -1, 1])
    
    while head < tail:
        curr_r = queue_r[head]
        curr_c = queue_c[head]
        head += 1
        
        current_dist = dist_map[curr_r, curr_c]
        
        for i in range(4):
            nr = curr_r + dr_arr[i]
            nc = curr_c + dc_arr[i]
            
            if 0 <= nr < h and 0 <= nc < w:
                if dist_map[nr, nc] > current_dist + 1.0:
                    dist_map[nr, nc] = current_dist + 1.0
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1
                    
    return dist_map

@numba.jit(nopython=True, cache=True)
def _compute_components_numba(p_map, h, w):

    component_map = np.zeros((h, w), dtype=np.int32)
    component_id = 1
    

    stack_r = np.zeros(h * w, dtype=np.int32)
    stack_c = np.zeros(h * w, dtype=np.int32)
    
    dr_arr = np.array([-1, 1, 0, 0])
    dc_arr = np.array([0, 0, -1, 1])

    for r in range(h):
        for c in range(w):
    
            if p_map[r, c] == 1 and component_map[r, c] == 0:
         
                stack_top = 0
                stack_r[stack_top] = r
                stack_c[stack_top] = c
                stack_top += 1
                component_map[r, c] = component_id
                
                while stack_top > 0:
                    stack_top -= 1
                    curr_r = stack_r[stack_top]
                    curr_c = stack_c[stack_top]
                    
                    for i in range(4):
                        nr = curr_r + dr_arr[i]
                        nc = curr_c + dc_arr[i]
                        
                        if 0 <= nr < h and 0 <= nc < w:
                            if p_map[nr, nc] == 1 and component_map[nr, nc] == 0:
                                component_map[nr, nc] = component_id
                                stack_r[stack_top] = nr
                                stack_c[stack_top] = nc
                                stack_top += 1
                
                component_id += 1
                
    return component_map



class AgentMemory:

    def __init__(self, agent_id: int, map_shape: Tuple[int, int], sim_params: Dict, 
                 heuristic_manager: 'HeuristicManager', arg_planner: Optional['AdaptiveRegionGraph']):
        self.agent_id = agent_id
        self.map_shape = map_shape
        self.params = sim_params

        self.heuristic_manager = heuristic_manager
        self.arg_planner = arg_planner
        

        self.history_len = self.params.get('dezd_history_len', 50)
        self.goal_proximity_threshold = self.params.get('dezd_goal_proximity_threshold', 3)
        self.stagnation_check_len = self.params.get('dezd_stagnation_check_len', 6)
        self.state_cycle_len = self.params.get('dezd_state_cycle_len', 10)

        self.direct_cycle_check_len = self.params.get('dezd_direct_cycle_len', 30)

        self.incremental_block_ratio = self.params.get('dezd_incremental_block_ratio', 0.8)
 
        self.trajectory_windows = self.params.get('dezd_trajectory_windows', [5,8,10,15,20])
        self.trajectory_overlap_threshold = self.params.get('dezd_trajectory_overlap_threshold', 0.7)



        self.pattern_detection_windows = self.params.get('dezd_pattern_windows', [3,6,12,24]) 

        self.pattern_position_threshold = self.params.get('dezd_pattern_pos_threshold', 0.6) 
        self.pattern_state_threshold = self.params.get('dezd_pattern_state_threshold', 0.6)

        self.wall_pattering_window = self.params.get('dezd_wall_pattering_window', 3) 
        self.wall_hugging_threshold = self.params.get('dezd_wall_hugging_threshold', 2) 
        self.wall_pattering_pos_threshold = self.params.get('dezd_wall_pattering_pos_threshold', 0.6) #
        self.goal_near_wall_threshold = self.params.get('dezd_goal_near_wall_threshold', 3) 

 
        self.stagnation_window = self.params.get('dezd_stagnation_window', 12)
        self.stagnation_bbox_size = self.params.get('dezd_stagnation_bbox_size', 8) 

        self.position_history: Deque[Tuple[int, int]] = deque(maxlen=self.history_len)
        self.heuristic_history: Deque[int] = deque(maxlen=self.history_len)
        self.state_history: Deque[Tuple[Tuple[int, int], int]] = deque(maxlen=self.history_len)

        # State machine: NORMAL -> PATTERING_DETECTED -> ESCAPING -> NORMAL
        self.status = "NORMAL"
        self.pattering_cooldown_counter = 0


        self.personal_obstacle_map = np.zeros(map_shape, dtype=bool)
        self.last_trapped_pos: Optional[Tuple[int, int]] = None
        self.heuristic_at_trap: int = -1

        self.trap_region_id: Optional[int] = None
        self.last_trap_zone: Set[Tuple[int, int]] = set()

        self.open_space_radius = self.params.get('dezd_open_space_radius', 5) 
        self.open_space_threshold = self.params.get('dezd_open_space_threshold', 0.25) 


    def update_after_step(self, new_pos: Tuple[int, int], goal_pos: Tuple[int, int], p_map: np.ndarray, cached_h_map: Optional[np.ndarray] = None):

        current_h_val = heuristic(new_pos, goal_pos)
        self.position_history.append(new_pos)
        self.heuristic_history.append(current_h_val)
        self.state_history.append((new_pos, current_h_val))

        if self.pattering_cooldown_counter > 0:
            self.pattering_cooldown_counter -= 1

        if self.status == "ESCAPING" and self.heuristic_at_trap != -1 and self.arg_planner is not None:
 
            if cached_h_map is not None:
                current_true_h = cached_h_map[new_pos]
                progress_made = current_true_h != -1 and current_true_h < self.heuristic_at_trap - self.params.get('dezd_escape_h_diff', 1)
                
                current_region_id = self.arg_planner._get_region_id(new_pos)
                region_changed = self.trap_region_id is not None and current_region_id != self.trap_region_id
                
                if progress_made and region_changed:
                    logging.info(f"DEZD: Agent {self.agent_id} has truly escaped region {self.trap_region_id}. Sealing trap entrance and resetting.")
                    self._seal_trap_entrance()
                    self.status = "NORMAL"
                    self.last_trapped_pos = None
                    self.heuristic_at_trap = -1
                    self.trap_region_id = None
                    self.last_trap_zone = set()

        if self.status == "PATTERING_DETECTED":
             if len(self.heuristic_history) > 1 and current_h_val < self.heuristic_history[0] - 1:
                self.status = "NORMAL"


    def _find_nearest_obstacle(self, start_pos: Tuple[int, int], p_map: np.ndarray) -> Optional[Tuple[int, int]]:

        if p_map[start_pos] == OBSTACLE_CELL:
            return start_pos
        
        q = deque([start_pos])
        visited = {start_pos}
        h, w = p_map.shape

        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    if p_map[nr, nc] == OBSTACLE_CELL:
                        return (nr, nc)
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return None

    def _check_for_wall_pattering(self, current_pos: Tuple[int, int], goal_pos: Tuple[int, int], 
                                  obstacle_dist_map: np.ndarray, obstacle_component_map: np.ndarray) -> Tuple[bool, str]:

        history_len_now = len(self.position_history)
        if history_len_now < self.wall_pattering_window:
            return False, ""


        recent_positions = list(itertools.islice(self.position_history, history_len_now - self.wall_pattering_window, history_len_now))
        avg_dist_to_wall = sum(obstacle_dist_map[pos] for pos in recent_positions) / len(recent_positions)
        if avg_dist_to_wall > self.wall_hugging_threshold:
            return False, ""

        half_point = len(recent_positions) // 2
        pos_similarity = len(set(recent_positions[:half_point]).intersection(set(recent_positions[half_point:]))) / len(set(recent_positions))
        if pos_similarity < self.wall_pattering_pos_threshold:
            return False, ""

        goal_dist_to_wall = obstacle_dist_map[goal_pos]
        if goal_dist_to_wall <= self.goal_near_wall_threshold:

            agent_wall_pos = self._find_nearest_obstacle(current_pos, obstacle_dist_map > 0)
            goal_wall_pos = self._find_nearest_obstacle(goal_pos, obstacle_dist_map > 0)
            if agent_wall_pos and goal_wall_pos:
                agent_wall_id = obstacle_component_map[agent_wall_pos]
                goal_wall_id = obstacle_component_map[goal_wall_pos]
                recent_heuristics = list(itertools.islice(self.heuristic_history, history_len_now - self.wall_pattering_window, history_len_now))
                no_progress = recent_heuristics[-1] >= recent_heuristics[0] 

                if agent_wall_id > 0 and agent_wall_id == goal_wall_id and not no_progress:
                    logging.debug(f"Agent {self.agent_id} wall pattering exempted: Agent and Goal are near the same obstacle component (ID: {agent_wall_id}) and making progress.")
                    return False, ""


        return True, f"Wall Pattering detected (avg_dist: {avg_dist_to_wall:.2f}, overlap: {pos_similarity:.2f})."


    def _is_in_constrained_space(self, agent_pos: Tuple[int, int], p_map: np.ndarray) -> bool:

        r_min = max(0, agent_pos[0] - self.open_space_radius)
        r_max = min(self.map_shape[0], agent_pos[0] + self.open_space_radius + 1)
        c_min = max(0, agent_pos[1] - self.open_space_radius)
        c_max = min(self.map_shape[1], agent_pos[1] + self.open_space_radius + 1)
    
    
        total_cells = 0
        obstacle_count = 0
    

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                total_cells += 1
                pos = (r, c)

                if p_map[pos] == OBSTACLE_CELL or self.personal_obstacle_map[pos]:
                    obstacle_count += 1
        
        if total_cells == 0:
            return True 
    

        obstacle_ratio = obstacle_count / total_cells
        

        return obstacle_ratio >= self.open_space_threshold

    
    def check_and_handle_pattering(self, goal_pos: Tuple[int, int], p_map: np.ndarray, obstacle_dist_map: np.ndarray, obstacle_component_map: np.ndarray):

        if self.pattering_cooldown_counter > 0 or self.status != "NORMAL": return
        if not self.position_history or self.heuristic_history[-1] <= self.goal_proximity_threshold: return
        pattering_detected = False
        reason = ""
        current_pos = self.position_history[-1]
        history_len_now = len(self.position_history)

        if not self._is_in_constrained_space(current_pos, p_map):
            logging.debug(f"Agent {self.agent_id} pattering check skipped: in open space.")
            return

        if not pattering_detected:
            pattering_detected, reason = self._check_for_wall_pattering(current_pos, goal_pos, obstacle_dist_map, obstacle_component_map)

        if not pattering_detected:
            pattering_detected, reason = self._check_for_pattern_repetition()

        stagnation_window = self.stagnation_window
        if not pattering_detected:
            if history_len_now > stagnation_window:
                recent_positions = list(itertools.islice(self.position_history, history_len_now - stagnation_window, history_len_now))
                r_coords, c_coords = zip(*recent_positions)
                bbox_height = max(r_coords) - min(r_coords)
                bbox_width = max(c_coords) - min(c_coords)

                if bbox_height < self.stagnation_bbox_size and bbox_width < self.stagnation_bbox_size:
                    pattering_detected, reason = True, f"Spatial Stagnation detected in a {bbox_width}x{bbox_height} bbox for {stagnation_window} steps."
        


        
        if pattering_detected:
            logging.info(f"DEZD: Agent {self.agent_id} detected PATTERING. Reason: {reason}")
            self.status = "PATTERING_DETECTED"
            self.pattering_cooldown_counter = self.params.get('pattering_cooldown_steps', 0)
            self.last_trapped_pos = current_pos
            self.heuristic_at_trap = self.heuristic_history[-1]

    def _seal_trap_entrance(self):
        if not self.last_trap_zone: return
        entrances = set(); h, w = self.map_shape
        for r, c in self.last_trap_zone:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w): continue
                if (nr, nc) not in self.last_trap_zone and self.params['persistent_known_map'][nr, nc] == FREE_CELL: entrances.add((nr, nc))
        logging.debug(f"Agent {self.agent_id} sealing entrances of last trap: {entrances}")
        self._add_zone_to_personal_obstacles(entrances)
        

    
    def _check_for_pattern_repetition(self) -> Tuple[bool, str]:

        history_len_now = len(self.position_history)
        
        for window_size in self.pattern_detection_windows:

            if history_len_now < window_size:
                continue


            window_positions = list(itertools.islice(self.position_history, history_len_now - window_size, history_len_now))
            window_states = list(itertools.islice(self.state_history, history_len_now - window_size, history_len_now))


            half_point = window_size // 2
            first_half_pos = set(window_positions[:half_point])
            second_half_pos = set(window_positions[half_point:])
            
            first_half_states = set(window_states[:half_point])
            second_half_states = set(window_states[half_point:])


            pos_intersection = len(first_half_pos.intersection(second_half_pos))
            pos_union = len(first_half_pos.union(second_half_pos))
            if pos_union > 0:
                pos_similarity = pos_intersection / pos_union
                if pos_similarity >= self.pattern_position_threshold:
                    return True, f"High position overlap ({pos_similarity:.2f}) in last {window_size} steps."


            state_intersection = len(first_half_states.intersection(second_half_states))
            state_union = len(first_half_states.union(second_half_states))
            if state_union > 0:
                state_similarity = state_intersection / state_union
                if state_similarity >= self.pattern_state_threshold:
                    return True, f"High state (pos,h) overlap ({state_similarity:.2f}) in last {window_size} steps."

        return False, ""


    def _identify_trap_pocket(self, start_pos: Tuple[int, int], p_map: np.ndarray, h_map: np.ndarray, other_agents_pos: Set[Tuple[int,int]]) -> Tuple[Set[Tuple[int, int]], Optional[Tuple[int, int]]]:

        q: Deque[Tuple[int, int]] = deque([start_pos])
        pocket: Set[Tuple[int, int]] = {start_pos}
        

        min_h_in_pocket = h_map[start_pos]
        if min_h_in_pocket == -1: 
             return set(), None

        head = 0
        while head < len(q):
            r, c = q.popleft()
            head +=1 
            
            h_val = h_map[r, c]
            if h_val != -1 and h_val < min_h_in_pocket:
                min_h_in_pocket = h_val

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                if not (0 <= nr < self.map_shape[0] and 0 <= nc < self.map_shape[1]): continue
                if (nr, nc) in pocket: continue
                

                is_passable = p_map[nr, nc] == FREE_CELL and not self.personal_obstacle_map[nr, nc]
                h_neighbor = h_map[nr, nc]

                if is_passable and h_neighbor != -1 and h_neighbor >= min_h_in_pocket:
                    pocket.add((nr, nc))
                    q.append((nr, nc))

        best_exit_point = None
        min_exit_h = float('inf')

        potential_exits = []

        social_penalty_factor = 2.0 

        for r, c in pocket:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                if not (0 <= nr < self.map_shape[0] and 0 <= nc < self.map_shape[1]): continue
                if (nr, nc) in pocket: continue

                h_outside = h_map[nr, nc]
                if h_outside != -1 and h_outside < min_h_in_pocket:

                    min_dist_to_others = float('inf')
                    if other_agents_pos:
                        min_dist_to_others = min([heuristic((nr, nc), other_pos) for other_pos in other_agents_pos])
                    

                    social_penalty = social_penalty_factor / (min_dist_to_others + 1e-6)
                    

                    score = h_outside + social_penalty
                    potential_exits.append((score, (nr, nc)))


        if not potential_exits:
            return pocket, None


        best_exit_point = min(potential_exits, key=lambda x: x[0])[1]
        
        return pocket, best_exit_point


    def initiate_escape_sequence_3(self, p_map: np.ndarray, global_h_map: np.ndarray, obstacle_dist_map: np.ndarray, other_agents_pos: Set[Tuple[int,int]], sim_params: Dict, steps: int,goals: Set[Tuple[int,int]]) -> Optional[List[Tuple[int, int]]]:

        current_pos = self.position_history[-1]

        trap_zone, escape_point = self._identify_trap_pocket(current_pos, p_map, global_h_map, other_agents_pos)


        if not escape_point or not trap_zone:
            logging.error(f"DEZD failed for Agent {self.agent_id}: Cannot identify trap pocket or find an escape point.")
            self.status = "NORMAL" 
            return None


        self.last_trap_zone = trap_zone 
        

        q_depth = deque([(escape_point, 0)])
        visited_in_trap = {escape_point}
        depth_map = [] # (depth, pos)
        
        while q_depth:
            pos, depth = q_depth.popleft()
            if pos in trap_zone:
                depth_map.append((depth, pos))
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (pos[0] + dr, pos[1] + dc)
                if neighbor in trap_zone and neighbor not in visited_in_trap:
                    visited_in_trap.add(neighbor)
                    q_depth.append((neighbor, depth + 1))


        depth_map.sort(key=lambda x: x[0], reverse=True)
        num_to_block = int(len(depth_map) * self.incremental_block_ratio)
        if num_to_block == 0 and len(depth_map) > 1:
            num_to_block = 10 
            
        blocking_zone = {pos for _, pos in depth_map[:num_to_block]}
        logging.info(f"DEZD: Agent {self.agent_id} performing incremental escape. Blocking {len(blocking_zone)} cells (the deepest part).")
        self._add_zone_to_personal_obstacles(blocking_zone)
        

        if self.arg_planner:
            self.trap_region_id = self.arg_planner._get_region_id(current_pos)
        
 
        squeeze_out_path_coords = run_single_agent_astar(
            start_pos_global=current_pos,
            goal_pos_global=escape_point,
            grid_map=p_map,
            sim_params=sim_params,
            heuristic_map=global_h_map, 
            obstacle_dist_map=obstacle_dist_map, 
            agent_memory=self, 
            other_agent_positions=other_agents_pos,
            treat_other_agents_as_obstacles=True 
        )


        if not squeeze_out_path_coords:
            logging.error(f"DEZD failed for Agent {self.agent_id}: Cannot plan path to escape point {escape_point}.")
            self.status = "NORMAL"
            return None
        
 
        self.status = "ESCAPING"
        logging.info(f"DEZD: Agent {self.agent_id} initiating escape from trap of size {len(trap_zone)}. Escape point: {escape_point}")
        self._add_zone_to_personal_obstacles(trap_zone)


        vis_params = sim_params.get('visualization_params', {})
        if vis_params.get('enable', False) and vis_params.get('visualize_dezd', True):
            plot_dezd_activation(p_map, self.personal_obstacle_map, trap_zone, current_pos, self.agent_id, steps, vis_params['save_path'],goals)

        return [current_pos] + squeeze_out_path_coords
    
 

    def _add_zone_to_personal_obstacles(self, zone: Set[Tuple[int, int]]):

        for r, c in zone:
            self.personal_obstacle_map[r, c] = True
            
class Constraint:
    def __init__(self, agent_id, loc, timestep, is_edge_constraint=False, prev_loc=None):
        self.agent_id, self.location, self.timestep, self.is_edge_constraint, self.prev_location = agent_id, loc, timestep, is_edge_constraint, prev_loc
    def __eq__(self, other): return isinstance(other, Constraint) and self.__dict__ == other.__dict__
    def __hash__(self): return hash((self.agent_id, self.location, self.timestep, self.is_edge_constraint, self.prev_location))

class Conflict:
    VERTEX, EDGE = 1, 2
    def __init__(self, conflict_type, agent1_id, agent2_id, loc1, timestep, loc2=None):
        self.type, self.agent1_id, self.agent2_id, self.location1, self.timestep, self.location2 = conflict_type, agent1_id, agent2_id, loc1, timestep, loc2


class HeuristicManager:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.true_distance_cache: Dict[Tuple[Tuple[int, int], int], np.ndarray] = {}
        self.cache_max_size = self.config.get('heuristic_cache_max_size', 256)

    def _run_reverse_bfs(self, goal_pos: Tuple[int, int], p_map: np.ndarray) -> np.ndarray:
        h, w = p_map.shape
        heuristic_map = np.full((h, w), -1.0, dtype=np.float32) # Use -1 for unreachable

        if not (0 <= goal_pos[0] < h and 0 <= goal_pos[1] < w and p_map[goal_pos] == FREE_CELL):
            return heuristic_map 

        q: Deque[Tuple[Tuple[int, int], int]] = deque([(goal_pos, 0)])
        heuristic_map[goal_pos] = 0

        while q:
            (r, c), dist = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w and p_map[nr, nc] == FREE_CELL and heuristic_map[nr, nc] == -1.0):
                    heuristic_map[nr, nc] = dist + 1
                    q.append(((nr, nc), dist + 1))
        return heuristic_map

    def get_true_distance_heuristic(self, goal_pos: Tuple[int, int], p_map: np.ndarray) -> np.ndarray:
        map_h = get_map_hash(p_map)
        cache_key = (goal_pos, map_h)

        if cache_key in self.true_distance_cache:
            return self.true_distance_cache[cache_key]
        
        if len(self.true_distance_cache) >= self.cache_max_size:
            self.true_distance_cache.pop(next(iter(self.true_distance_cache)))

        if self.config.get('verbose', False):
            logging.debug(f"Heuristic cache miss for goal {goal_pos}. Calculating new heuristic map.")

        heuristic_map = self._run_reverse_bfs(goal_pos, p_map)
        self.true_distance_cache[cache_key] = heuristic_map
        return heuristic_map

class BaseRegionGraph:

    def __init__(self, map_shape: Tuple[int, int], config: Dict[str, Any], persistent_map: np.ndarray, region_size: int, heuristic_manager: HeuristicManager):
        self.map_h, self.map_w = map_shape
        self.config = config
        self.region_size = region_size
        if self.region_size <= 0:
            self.region_size = 16
            logging.debug(f"Region size was <= 0, reset to 16.")
        self.regions_h = (self.map_h + self.region_size - 1) // self.region_size
        self.regions_w = (self.map_w + self.region_size - 1) // self.region_size
        self.graph = nx.Graph()
        self.region_centers: Dict[int, Tuple[int, int]] = {}
        self.last_map_hash = None
        self.heuristic_manager = heuristic_manager 
        self._build_base_graph(persistent_map)

    def _get_region_id(self, pos: Tuple[int, int]) -> int:
        return (pos[0] // self.region_size) * self.regions_w + (pos[1] // self.region_size)

    def _build_base_graph(self, persistent_map: np.ndarray):
        for r_idx in range(self.regions_h):
            for c_idx in range(self.regions_w):
                region_id = r_idx * self.regions_w + c_idx
                center_r = r_idx * self.region_size + self.region_size // 2
                center_c = c_idx * self.region_size + self.region_size // 2
                self.region_centers[region_id] = (min(center_r, self.map_h - 1), min(center_c, self.map_w - 1))
                self.graph.add_node(region_id, penalty=0.0)
        self.update_graph_with_obstacles(persistent_map)

    def _check_boundary_passable(self, r_idx: int, c_idx: int, dr: int, dc: int, persistent_map: np.ndarray) -> bool:
        passable_cells = 0
        num_checked = 0
        if dc == 1:
            gateway_c = (c_idx + 1) * self.region_size
            if gateway_c >= self.map_w: return False
            start_r, end_r = r_idx * self.region_size, (r_idx + 1) * self.region_size
            for r in range(start_r, end_r, 2):
                if 0 <= r < self.map_h:
                    num_checked += 1
                    if persistent_map[r, gateway_c - 1] != OBSTACLE_CELL and persistent_map[r, gateway_c] != OBSTACLE_CELL:
                        passable_cells += 1
        elif dr == 1:
            gateway_r = (r_idx + 1) * self.region_size
            if gateway_r >= self.map_h: return False
            start_c, end_c = c_idx * self.region_size, (c_idx + 1) * self.region_size
            for c in range(start_c, end_c, 2):
                if 0 <= c < self.map_w:
                    num_checked += 1
                    if persistent_map[gateway_r - 1, c] != OBSTACLE_CELL and persistent_map[gateway_r, c] != OBSTACLE_CELL:
                        passable_cells += 1
        
        return passable_cells >= 2 or (num_checked > 0 and (passable_cells / num_checked) > 0.2)

    def update_graph_with_obstacles(self, persistent_map: np.ndarray):
        current_map_hash = get_map_hash(persistent_map)
        if current_map_hash == self.last_map_hash: return
        if self.config.get('verbose', False): logging.debug(f"({type(self).__name__}): Map changed, updating region graph.")
        self.last_map_hash = current_map_hash
        for r_idx in range(self.regions_h):
            for c_idx in range(self.regions_w):
                rid1 = self._get_region_id((r_idx * self.region_size, c_idx * self.region_size))
                for dr, dc in [(1, 0), (0, 1)]:
                    nr_idx, nc_idx = r_idx + dr, c_idx + dc
                    if not (0 <= nr_idx < self.regions_h and 0 <= nc_idx < self.regions_w): continue
                    rid2 = self._get_region_id((nr_idx * self.region_size, nc_idx * self.region_size))
                    is_passable = self._check_boundary_passable(r_idx, c_idx, dr, dc, persistent_map)
                    self._update_edge_status(rid1, rid2, is_passable, r_idx, c_idx, nr_idx, nc_idx, persistent_map)

    def _update_edge_status(self, u, v, is_passable, r1, c1, r2, c2, p_map):
        raise NotImplementedError

    def find_high_level_path(self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> Optional[List[int]]:
        start_region, goal_region = self._get_region_id(start_pos), self._get_region_id(goal_pos)
        if start_region == goal_region or not self.graph.has_node(start_region) or not self.graph.has_node(goal_region):
            return None
        try:
            path = nx.astar_path(self.graph, start_region, goal_region,
                                 heuristic=lambda u, v: heuristic(self.region_centers[u], self.region_centers[v]),
                                 weight='weight')
            return path
        except nx.NetworkXNoPath:
            if self.config.get('verbose'): logging.debug(f"({type(self).__name__}): No path found from region {start_region} to {goal_region}")
            return None
    
    def get_validated_subgoal_from_path(self, region_path: List[int], p_map: np.ndarray, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Retrieves and validates the next subgoal along the high-level region path."""
        if not region_path or len(region_path) < 2:
            return None
        

        for region_index_in_path in [1, 2]:
            if len(region_path) <= region_index_in_path:
                continue
            
            target_region_id = region_path[region_index_in_path]
            candidate_subgoal = self.region_centers.get(target_region_id)

            if not candidate_subgoal: continue
            

            if p_map[candidate_subgoal] == FREE_CELL:
                return candidate_subgoal

            if self.config.get('verbose'):
                logging.debug(f"Subgoal {candidate_subgoal} in region {target_region_id} is invalid. Searching for alternative...")

            q: Deque[Tuple[int, int]] = deque([candidate_subgoal])
            visited = {candidate_subgoal}
            
            start_r, start_c = (target_region_id // self.regions_w) * self.region_size, (target_region_id % self.regions_w) * self.region_size
            end_r, end_c = start_r + self.region_size, start_c + self.region_size

            while q:
                r, c = q.popleft()
                for dr, dc in random.sample([(-1, 0), (1, 0), (0, -1), (0, 1)], 4):
                    nr, nc = r + dr, c + dc
                    if (start_r <= nr < end_r and start_c <= nc < end_c and
                        0 <= nr < self.map_h and 0 <= nc < self.map_w and (nr, nc) not in visited):
                        
                        visited.add((nr, nc))
                        if p_map[nr, nc] == FREE_CELL:
                            logging.debug(f"Found valid alternative subgoal at {(nr, nc)}.")
                            return (nr, nc)
                        q.append((nr, nc))
        
        logging.debug(f"Could not find any valid subgoal in regions {region_path[1:]}.")
        return None

class AdaptiveRegionGraph(BaseRegionGraph):
    def __init__(self, map_shape: Tuple[int, int], config: Dict[str, Any], persistent_map: np.ndarray, heuristic_manager: HeuristicManager):
        region_size = config.get('region_size', 16)
        super().__init__(map_shape, config, persistent_map, region_size, heuristic_manager)
    
    def _get_region_freeness(self, r_idx: int, c_idx: int, persistent_map: np.ndarray) -> float:
        start_r, start_c = r_idx * self.region_size, c_idx * self.region_size
        end_r, end_c = min(start_r + self.region_size, self.map_h), min(start_c + self.region_size, self.map_w)
        region_slice = persistent_map[start_r:end_r, start_c:end_c]
        if region_slice.size == 0: return 0.0
        return np.sum(region_slice == FREE_CELL) / region_slice.size



    def _update_edge_status(self, u, v, is_passable, r1, c1, r2, c2, p_map):

        has_edge = self.graph.has_edge(u, v)
        
        if not is_passable:
            if has_edge: self.graph.remove_edge(u, v)
            return
            

        center_u, center_v = self.region_centers[u], self.region_centers[v]
        dist = abs(center_u[0] - center_v[0]) + abs(center_u[1] - center_v[1]) # Manhattan

        freeness1 = self._get_region_freeness(r1, c1, p_map)
        freeness2 = self._get_region_freeness(r2, c2, p_map)

        weight_multiplier = 1.0 / (freeness1 * freeness2 + 1e-6)
        
        final_weight = dist * weight_multiplier
        
        if has_edge:
            self.graph[u][v]['weight'] = final_weight
        else:
            self.graph.add_edge(u, v, weight=final_weight)


def get_spatial_features_from_obs(agent_obs_dict, local_window_size):
    num_spatial_channels = 4; h = w = local_window_size
    spatial_features = np.zeros((num_spatial_channels, h, w), dtype=np.float32)
    center_idx = h // 2
    spatial_features[0, :, :] = agent_obs_dict.get("obstacles", np.ones((h, w))).astype(np.float32)
    spatial_features[1, :, :] = agent_obs_dict.get("agents", np.zeros((h, w))).astype(np.float32)
    spatial_features[2, center_idx, center_idx] = 1.0
    spatial_features[3, :, :] = agent_obs_dict.get("target", np.zeros((h,w))).astype(np.float32)
    return torch.from_numpy(spatial_features)

def get_non_spatial_features(current_global_pos, global_goal_pos):
    dy = global_goal_pos[0] - current_global_pos[0]; dx = global_goal_pos[1] - current_global_pos[1]
    norm = np.sqrt(dx**2 + dy**2) if (dx**2 + dy**2) > 0 else 1.0
    norm_dy, norm_dx = dy / norm, dx / norm
    return torch.from_numpy(np.array([norm_dy, norm_dx], dtype=np.float32))

def global_to_local_coords(global_pos_target, agent_global_pos, obs_radius, window_size):
    gr_target, gc_target = global_pos_target; gr_agent, gc_agent = agent_global_pos
    lr = gr_target - (gr_agent - obs_radius); lc = gc_target - (gc_agent - obs_radius)
    if 0 <= lr < window_size and 0 <= lc < window_size: return (int(lr), int(lc))
    return None

def path_coords_to_actions(path_coords: List[Tuple[int, int]], start_pos: Tuple[int, int]) -> List[int]:
    actions = []
    current_path = [start_pos] + path_coords
    for i in range(len(current_path) - 1):
        dr = current_path[i+1][0] - current_path[i][0]
        dc = current_path[i+1][1] - current_path[i][1]
        action = next((act for act, (ddr, ddc) in ACTION_DELTAS.items() if ddr == dr and ddc == dc), ACTION_STAY)
        actions.append(action)
    return actions if actions else [ACTION_STAY]

def run_single_agent_astar_for_unet(
    start_pos_global,
    goal_pos_global,
    grid_map,
    sim_params,
    heuristic_map,
    agent_memory=None,
    other_agent_positions=None,
    treat_other_agents_as_obstacles=False
):

    h, w = grid_map.shape
    

    planning_obstacle_map = (grid_map == OBSTACLE_CELL)
    if agent_memory:
        planning_obstacle_map |= agent_memory.personal_obstacle_map


    if treat_other_agents_as_obstacles and other_agent_positions:
        for r, c in other_agent_positions:
            if 0 <= r < h and 0 <= c < w:
                planning_obstacle_map[r, c] = True

    if not (0 <= start_pos_global[0] < h and 0 <= start_pos_global[1] < w) or planning_obstacle_map[start_pos_global]:
        return None

    get_h_score = lambda pos: heuristic_map[pos]
    initial_h = get_h_score(start_pos_global)
    if initial_h == -1.0: 
        return None

    open_set = [(initial_h, 0, start_pos_global)]  # (f_score, g_score, pos)
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start_pos_global: 0}
    
    nodes_expanded = 0
    node_limit = sim_params.get('astar_node_limit', 500) #2500

    while open_set:
        f, g, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current == goal_pos_global:
            path_deque = deque()
            temp = current
            while temp in came_from:
                path_deque.appendleft(temp)
                temp = came_from[temp]
            return list(path_deque)

        if nodes_expanded > node_limit: break

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if not (0 <= neighbor[0] < h and 0 <= neighbor[1] < w and not planning_obstacle_map[neighbor]):
                continue

            move_cost = 1.0
            if grid_map[neighbor] == UNKNOWN_CELL:
                move_cost += sim_params.get('unknown_soft_cost', 2.0)

            new_g = g + move_cost
            if new_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor], g_score[neighbor] = current, new_g
                h_val = get_h_score(neighbor)
                if h_val == -1.0:
                    continue 
                
                new_f = new_g + h_val
                heapq.heappush(open_set, (new_f, new_g, neighbor))
    return None




def run_single_agent_astar(
    start_pos_global,
    goal_pos_global,
    grid_map,
    sim_params,
    heuristic_map,
    obstacle_dist_map,  # ### NEW: Pass in the distance map ###
    agent_memory=None,
    other_agent_positions=None,
    treat_other_agents_as_obstacles=False
):

    h, w = grid_map.shape

    # --- Tunable parameters for congestion avoidance ---
    WALL_REPULSION_FACTOR = sim_params.get('wall_repulsion_factor', 1.5)
    WALL_REPULSION_RANGE = sim_params.get('wall_repulsion_range', 5.0)
    GOAL_PROXIMITY_THRESHOLD = 4

    planning_obstacle_map = (grid_map == OBSTACLE_CELL)
    if agent_memory:
        planning_obstacle_map |= agent_memory.personal_obstacle_map


    if treat_other_agents_as_obstacles and other_agent_positions:
        for r, c in other_agent_positions:
            if 0 <= r < h and 0 <= c < w:
                planning_obstacle_map[r, c] = True

    if not (0 <= start_pos_global[0] < h and 0 <= start_pos_global[1] < w) or planning_obstacle_map[start_pos_global]:
        return None

    get_h_score = lambda pos: heuristic_map[pos]
    initial_h = get_h_score(start_pos_global)
    if initial_h == -1.0:
        return None

    open_set = [(initial_h, 0, start_pos_global)]  # (f_score, g_score, pos)
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start_pos_global: 0}
    
    nodes_expanded = 0
    node_limit = sim_params.get('astar_node_limit', 1500) #2500

    while open_set:
        f, g, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current == goal_pos_global:
            path_deque = deque()
            temp = current
            while temp in came_from:
                path_deque.appendleft(temp)
                temp = came_from[temp]
            return list(path_deque)

        if nodes_expanded > node_limit: break

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if not (0 <= neighbor[0] < h and 0 <= neighbor[1] < w and not planning_obstacle_map[neighbor]):
                continue
            
 
            move_cost = 1.0
            if grid_map[neighbor] == UNKNOWN_CELL:
                move_cost += sim_params.get('unknown_soft_cost', 2.0)


            apply_repulsion = True
            if heuristic(neighbor, goal_pos_global) < GOAL_PROXIMITY_THRESHOLD:
                apply_repulsion = False

            if apply_repulsion:
                dist_to_wall = obstacle_dist_map[neighbor]
                if dist_to_wall < WALL_REPULSION_RANGE:
                    penalty = WALL_REPULSION_FACTOR * (1.0 / (dist_to_wall + 1.0))
                    move_cost += penalty

            
            new_g = g + move_cost
            if new_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor], g_score[neighbor] = current, new_g
                h_val = get_h_score(neighbor)
                if h_val == -1.0:
                    continue 
                
                new_f = new_g + h_val
                heapq.heappush(open_set, (new_f, new_g, neighbor))
    return None



def _build_conflict_groups(
    initial_paths: Dict[int, List[Tuple[int, int]]], 
    active_agents: List[int], 
    max_timestep: int, 
    sim_params: Dict
) -> Tuple[List[Set[int]], Dict[Tuple[int, int], float]]:

    if not active_agents: 
        return [], {}
    
    paths_for_detection = {aid: p for aid, p in initial_paths.items() if aid in active_agents}
    conflicts = detect_all_conflicts_spacetime(paths_for_detection, max_timestep)

    temporary_conflict_costs: Dict[Tuple[int, int], float] = defaultdict(float)
    congestion_penalty = sim_params.get('congestion_penalty', 25.0)
    
    G = nx.Graph()
    G.add_nodes_from(active_agents)
    for conflict in conflicts:
        if G.has_node(conflict.agent1_id) and G.has_node(conflict.agent2_id):
            G.add_edge(conflict.agent1_id, conflict.agent2_id)

        if conflict.type == Conflict.VERTEX and hasattr(conflict, 'location1') and conflict.location1:
            pos = conflict.location1
            temporary_conflict_costs[pos] += congestion_penalty
    
    components = [set(c) for c in nx.connected_components(G) if len(c) > 0]
    

    return components, temporary_conflict_costs


def _find_retreat_goal(agent_id: int, current_pos: Tuple[int, int], persistent_known_map: np.ndarray, all_agent_positions: List[Tuple[int, int]], claimed_subgoals: Set[Tuple[int, int]], max_search_dist: int) -> Optional[Tuple[int, int]]:
    h, w = persistent_known_map.shape
    q: Deque[Tuple[Tuple[int,int], int]] = deque([(current_pos, 0)])
    visited = {current_pos}
    other_agent_locs = {p for i, p in enumerate(all_agent_positions) if i != agent_id}

    while q:
        pos, dist = q.popleft()
        if dist > 0:
            is_valid_retreat = (persistent_known_map[pos] == FREE_CELL and pos not in other_agent_locs and pos not in claimed_subgoals)
            if is_valid_retreat: return pos
        if dist >= max_search_dist: continue
        
        for dr, dc in random.sample([(-1,0), (1,0), (0,-1), (0,1)], 4):
            neighbor = (pos[0] + dr, pos[1] + dc)
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w and neighbor not in visited:
                visited.add(neighbor)
                if persistent_known_map[neighbor] != OBSTACLE_CELL:
                    q.append((neighbor, dist + 1))
    return None

                             
def get_spatial_features_v19_compatible(agent_obs_dict, window_size):

    # 0: obstacles, 1: other_agents, 2: self_pos, 3: target_hotspot
    num_spatial_channels = 4
    h = w = window_size
    spatial_features = np.zeros((num_spatial_channels, h, w), dtype=np.float32)
    center_idx = h // 2
    
    spatial_features[0, :, :] = agent_obs_dict.get("obstacles", np.ones((h, w))).astype(np.float32)
    spatial_features[1, :, :] = agent_obs_dict.get("agents", np.zeros((h, w))).astype(np.float32)
    spatial_features[2, center_idx, center_idx] = 1.0
    # target_hotspot channel is now intelligently populated before this function is called
    spatial_features[3, :, :] = agent_obs_dict.get("target", np.zeros((h,w))).astype(np.float32)
                
    return torch.from_numpy(spatial_features)

def solve_by_push_and_rotate(
    group_list: List[int], 
    cbs_map_for_solver: np.ndarray, 
    agents_global_positions: List[Tuple[int, int]], 
    sim_params: Dict[str, Any],
    heuristic_maps: Dict[int, np.ndarray], # Pass in pre-calculated heuristic maps
    agent_memories: Dict[int, AgentMemory],  obstacle_dist_map: np.ndarray ### NEW ###

) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    """ Push and Rotate, updated for the new A* signature. """
    verbose = sim_params.get('verbose', False)
    if verbose: 
        logging.debug(f"Attempting Layer 3 Defense: Push-and-Rotate for group {group_list}.")
    
    pos_map = {aid: tuple(agents_global_positions[aid]) for aid in group_list}
    h, w = cbs_map_for_solver.shape
    
    # Find a nearby empty cell to start the rotation
    q = deque([(p, [p]) for p in pos_map.values()])
    visited = set(pos_map.values())
    empty_pos = None
    
    while q:
        curr, path_to_curr = q.popleft()
        if curr not in pos_map.values():
            empty_pos = curr
            break
        if len(path_to_curr) > 20: continue
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (curr[0] + dr, curr[1] + dc)
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w and cbs_map_for_solver[neighbor] == 0 and neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, path_to_curr + [neighbor]))
                
    if not empty_pos:
        logging.error(f"Push-and-Rotate FAILED for group {group_list}: could not find a nearby empty cell.")
        return None

    solution_paths = {aid: [pos_map[aid]] for aid in group_list}
    try:
        current_empty = empty_pos
        moved_agents = set()
        for _ in range(len(group_list)):
            # Find the closest agent to the current empty spot to move
            closest_agent_id = -1
            min_dist = float('inf')
            for aid in group_list:
                if aid not in moved_agents:
                    dist = heuristic(solution_paths[aid][-1], current_empty)
                    if dist < min_dist:
                        min_dist = dist
                        closest_agent_id = aid
            
            if closest_agent_id == -1: break
            
            agent_start_pos = solution_paths[closest_agent_id][-1]
            
            # The other agents in the group are treated as obstacles for this tactical move
            other_agents_in_group_pos = {
                tuple(solution_paths[other_aid][-1]) 
                for other_aid in group_list 
                if other_aid != closest_agent_id and other_aid not in moved_agents
            }
            
            # Updated A* call
            path_to_empty = run_single_agent_astar(
                start_pos_global=agent_start_pos,
                goal_pos_global=current_empty,
                grid_map=cbs_map_for_solver,
                sim_params=sim_params,
                # Use the agent's pre-calculated heuristic map for its *final* goal
                heuristic_map=heuristic_maps[closest_agent_id],
                obstacle_dist_map=obstacle_dist_map,
                agent_memory=agent_memories[closest_agent_id],
                other_agent_positions=other_agents_in_group_pos,
                # In this tactical maneuver, we MUST avoid the other agents in the deadlock
                treat_other_agents_as_obstacles=True
            )
            
            if not path_to_empty:
                logging.error(f"Push-and-Rotate FAILED for group {group_list}: agent {closest_agent_id} could not path to empty cell.")
                return None

            solution_paths[closest_agent_id].extend(path_to_empty)
            current_empty = agent_start_pos
            moved_agents.add(closest_agent_id)

        # Pad paths to equal length
        max_len = max(len(p) for p in solution_paths.values()) if solution_paths else 0
        for aid in group_list:
            path = solution_paths[aid]
            path.extend([path[-1]] * (max_len - len(path)))
            
        logging.info(f"Push-and-Rotate succeeded for group {group_list}.")
        return solution_paths

    except Exception as e:
        logging.error(f"Push-and-Rotate encountered an exception for group {group_list}: {e}")
        return None

def solve_by_coordinated_retreat(
    group_list: List[int], 
    group_agents_data: List[Dict], 
    persistent_known_map: np.ndarray, 
    cbs_map_for_solver: np.ndarray, 
    agents_global_positions: List[Tuple[int, int]], 
    agents_global_goals: List[Tuple[int, int]], 
    sim_params: Dict[str, Any],
    heuristic_maps: Dict[int, np.ndarray],
    agent_memories: Dict[int, AgentMemory],obstacle_dist_map: np.ndarray
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    """ Coordinated Retreat, updated for the new A* signature. """
    if len(group_list) < 2: return None
    
    verbose = sim_params.get('verbose', False)
    # Decide which agents yield (further from goal) and which move
    agents_with_dist = sorted([{'id': aid, 'dist': heuristic(tuple(agents_global_positions[aid]), tuple(agents_global_goals[aid]))} for aid in group_list], key=lambda x: x['dist'], reverse=True)
    num_yielders = max(1, len(group_list) // 2)
    yielder_ids = {d['id'] for d in agents_with_dist[:num_yielders]}
    mover_ids = {d['id'] for d in agents_with_dist[num_yielders:]}
    
    if verbose: 
        logging.info(f"Coordinated Retreat: Yielders={yielder_ids}, Movers={mover_ids}")

    yielder_paths: Dict[int, List[Tuple[int, int]]] = {}
    dynamic_constraints: List[Constraint] = []
    claimed_retreat_goals: Set[Tuple[int, int]] = set()
    
    # Plan paths for the "yielder" agents first
    for aid in yielder_ids:
        pos = tuple(agents_global_positions[aid])
        
        # Find a safe nearby spot to retreat to
        search_depth = sim_params.get('retreat_search_depth', 8)
        retreat_goal = _find_retreat_goal(aid, pos, persistent_known_map, agents_global_positions, claimed_retreat_goals, search_depth)
        if not retreat_goal: 
            logging.debug(f"Retreat failed: Yielder {aid} no goal found.")
            return None
        claimed_retreat_goals.add(retreat_goal)
        
        # For this tactical move, other agents are obstacles
        other_agents_on_map_pos = {tuple(p) for i, p in enumerate(agents_global_positions) if i != aid}
        
        # Updated A* call
        path_coords = run_single_agent_astar(
            start_pos_global=pos,
            goal_pos_global=retreat_goal,
            grid_map=cbs_map_for_solver,
            sim_params=sim_params,
            # Retreating is a local move, but guide it with global intelligence
            heuristic_map=heuristic_maps[aid], 
            obstacle_dist_map=obstacle_dist_map,
            agent_memory=agent_memories[aid],
            other_agent_positions=other_agents_on_map_pos,
            treat_other_agents_as_obstacles=True
        )

        if not path_coords: 
            logging.debug(f"Retreat failed: Yielder {aid} no path.")
            return None
        
        # Store the yielder's path and create constraints for the movers
        full_path = [pos] + path_coords
        yielder_paths[aid] = full_path
        for t, p in enumerate(full_path):
            if t > 0:
                dynamic_constraints.append(Constraint(aid, p, t))
                dynamic_constraints.append(Constraint(aid, p, t, is_edge_constraint=True, prev_loc=full_path[t-1]))

    # Plan for movers using CBS, respecting the yielders' paths
    movers_data = [d for d in group_agents_data if d['id'] in mover_ids]
    if not movers_data: 
        return yielder_paths

    mover_paths = solve_local_cbs_robust(
        agents_data=movers_data, 
        obstacles_local_map=cbs_map_for_solver,
        max_plan_len=sim_params.get('local_plan_horizon_base', 80) - 1,
        max_cbs_iterations=int(sim_params.get('cbs_max_iterations', 120) * 0.5),
        initial_constraints_list=dynamic_constraints,
        agents_true_global_goals_abs={i: tuple(g) for i, g in enumerate(agents_global_goals)},
        persistent_map_bundle={'persistent_known_map': persistent_known_map},
        verbose_cbs_solver=verbose
    )

    if not mover_paths: 
        logging.debug("Retreat failed: CBS for movers failed.")
        return None
        
    logging.info(f"Coordinated Retreat strategy succeeded!")
    return {**yielder_paths, **mover_paths}
    
def solve_by_forced_shuffle(group_list, cbs_map_for_solver, agents_global_positions, verbose=False):
    if verbose: logging.critical(f"Attempting Final Defense: Forced Shuffle for group {group_list}.")
    solution_paths, shuffled_group = {}, random.sample(group_list, len(group_list))
    temp_obstacle_map = cbs_map_for_solver.copy()
    for agent_id in shuffled_group:
        current_pos = tuple(agents_global_positions[agent_id])
        possible_moves = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]; random.shuffle(possible_moves)
        move_found = False
        for dr, dc in possible_moves:
            next_pos = (current_pos[0] + dr, current_pos[1] + dc)
            if (0 <= next_pos[0] < temp_obstacle_map.shape[0] and 0 <= next_pos[1] < temp_obstacle_map.shape[1] and temp_obstacle_map[next_pos] == 0):
                solution_paths[agent_id], temp_obstacle_map[next_pos], move_found = [current_pos, next_pos], 1, True; break
        if not move_found: solution_paths[agent_id] = [current_pos, current_pos]
    if verbose: logging.debug("Forced Shuffle complete.")
    return solution_paths
 
def solve_by_group_evaporation(
    group_list: List[int], 
    cbs_map: np.ndarray, 
    agents_pos: List[Tuple[int, int]], 
    sim_params: Dict[str, Any],
    heuristic_maps: Dict[int, np.ndarray],
    agent_memories: Dict[int, AgentMemory],obstacle_dist_map: np.ndarray
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
 
    logging.debug(f"Attempting Group Evaporation for large stuck group of size {len(group_list)}.")
    h, w = cbs_map.shape
    group_pos = {aid: agents_pos[aid] for aid in group_list}
    

    centroid_r = int(np.mean([pos[0] for pos in group_pos.values()]))
    centroid_c = int(np.mean([pos[1] for pos in group_pos.values()]))

    q = deque(list(group_pos.values()))
    visited = set(q)
    escape_pods = []
    
    while q and len(escape_pods) < len(group_list) * 1.5:
        r, c = q.popleft()
        dist_from_centroid = heuristic((r, c), (centroid_r, centroid_c))
        
        if dist_from_centroid > sim_params.get('evaporation_min_dist', 8) and (r,c) not in agents_pos:
            escape_pods.append((r,c))
            
        if dist_from_centroid > sim_params.get('evaporation_max_dist', 30): continue

        for dr, dc in random.sample([(-1, 0), (1, 0), (0, -1), (0, 1)], 4):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not cbs_map[nr, nc] and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    
    if len(escape_pods) < len(group_list):
        logging.error(f"Evaporation failed: Not enough escape pods found ({len(escape_pods)}/{len(group_list)}).")
        return None

    # 2. Assign each agent to its nearest escape pod and plan a path
    solution_paths = {}
    claimed_pods = set()
    # Sort agents to plan for those in the center of the jam first
    for aid in sorted(group_list, key=lambda i: heuristic(group_pos[i], (centroid_r, centroid_c))):
        best_pod = None
        min_dist = float('inf')
        for pod in escape_pods:
            if pod not in claimed_pods:
                dist = heuristic(group_pos[aid], pod)
                if dist < min_dist:
                    min_dist = dist
                    best_pod = pod
        
        if best_pod:
            claimed_pods.add(best_pod)
            
            # Other agents are obstacles for this tactical escape
            other_agents_on_map_pos = {tuple(p) for i, p in enumerate(agents_pos) if i != aid}
            
            # Updated A* call
            path = run_single_agent_astar(
                start_pos_global=group_pos[aid],
                goal_pos_global=best_pod,
                grid_map=cbs_map,
                sim_params=sim_params,
                # Guide the escape towards the agent's final goal
                heuristic_map=heuristic_maps[aid],
                obstacle_dist_map = obstacle_dist_map,

                agent_memory=agent_memories[aid],
                other_agent_positions=other_agents_on_map_pos,
                treat_other_agents_as_obstacles=True
            )
            
            solution_paths[aid] = [group_pos[aid]] + path if path else [group_pos[aid]]

    logging.info(f"Group Evaporation successful, assigning {len(solution_paths)} agents to escape pods.")
    return solution_paths

 
def _solve_single_group_with_defense_cascade(
    group_list: List[int], 
    group_agents_data: List[Dict], 
    cbs_map_for_solver: np.ndarray,
    agents_global_positions: List[Tuple[int, int]], 
    agents_global_goals: List[Tuple[int, int]],
    consecutive_cbs_fails_count: Dict[frozenset, int], 
    sim_params: Dict[str, Any],
    heuristic_maps: Dict[int, np.ndarray], # Pass down all required new params
    agent_memories: Dict[int, AgentMemory],obstacle_dist_map: np.ndarray
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    """ The main defense cascade, now passing new parameters to its subroutines. """
    group_frozenset = frozenset(group_list)
    fails_count = consecutive_cbs_fails_count.get(group_frozenset, 0)
    dynamic_k = sim_params.get('dynamic_k', 10) 
    sim_params['local_plan_horizon_base'] = dynamic_k + 1
    # Strategy 1: Standard CBS
    solution = solve_local_cbs_robust(
        agents_data=group_agents_data, 
        obstacles_local_map=cbs_map_for_solver, 
        max_plan_len=dynamic_k, 
        max_cbs_iterations=sim_params['cbs_max_iterations'],
        agents_true_global_goals_abs={i: g for i, g in enumerate(agents_global_goals)},
        persistent_map_bundle={'persistent_known_map': sim_params['persistent_known_map']}, 
        verbose_cbs_solver=False
    )
    if solution:
        consecutive_cbs_fails_count[group_frozenset] = 0
        return solution
    
    consecutive_cbs_fails_count[group_frozenset] = fails_count + 1
    
    # Strategy 2: Group Evaporation (for large groups)
    if len(group_list) > sim_params.get('evaporation_group_size_threshold', 10):
        solution = solve_by_group_evaporation(group_list, cbs_map_for_solver, agents_global_positions, sim_params, heuristic_maps, agent_memories,obstacle_dist_map)
        if solution: return solution
        
    # Strategy 3: Coordinated Retreat
    solution = solve_by_coordinated_retreat(group_list, group_agents_data, sim_params['persistent_known_map'], cbs_map_for_solver, agents_global_positions, agents_global_goals, sim_params, heuristic_maps, agent_memories,obstacle_dist_map)
    if solution: return solution
    if len(group_list) <= sim_params.get('push_rotate_group_size_threshold', 6):
        solution = solve_by_push_and_rotate(group_list, cbs_map_for_solver, agents_global_positions, sim_params, heuristic_maps, agent_memories,obstacle_dist_map)
        if solution: return solution
    
    # Final Resort: Forced Shuffle
    return solve_by_forced_shuffle(group_list, cbs_map_for_solver, agents_global_positions, sim_params.get('verbose', False))


def _find_safe_exploration_target(
    start_pos: Tuple[int, int], 
    goal_pos: Tuple[int, int], 
    persistent_map: np.ndarray,
    max_radius: int = 15
) -> Optional[Tuple[int, int]]:

    h, w = persistent_map.shape
    q = deque([(start_pos, 0)])
    visited = {start_pos}
    candidates = []

    while q:
        pos, dist = q.popleft()
        if dist > max_radius:
            continue

        if persistent_map[pos] == FREE_CELL:
            h_val = heuristic(pos, goal_pos)
            candidates.append((h_val, pos)) 

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if persistent_map[nr, nc] != OBSTACLE_CELL:
                    visited.add((nr, nc))
                    q.append(((nr, nc), dist + 1))
    
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0])[1]  


def _compute_obstacle_distance_transform(p_map: np.ndarray) -> np.ndarray:

    h, w = p_map.shape
    dist_map = np.full((h, w), float('inf'), dtype=np.float32)
    q = deque()


    for r in range(h):
        for c in range(w):
            if p_map[r, c] == OBSTACLE_CELL:
                dist_map[r, c] = 0
                q.append((r, c))

    while q:
        r, c = q.popleft()
        current_dist = dist_map[r, c]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist_map[nr, nc] == float('inf'):
                dist_map[nr, nc] = current_dist + 1
                q.append((nr, nc))
    
    return dist_map

def _compute_obstacle_components(p_map: np.ndarray) -> np.ndarray:

    h, w = p_map.shape
    component_map = np.zeros((h, w), dtype=np.int32)
    component_id = 1

    for r in range(h):
        for c in range(w):

            if p_map[r, c] == OBSTACLE_CELL and component_map[r, c] == 0:
                q = deque([(r, c)])
                component_map[r, c] = component_id
                
                while q:
                    curr_r, curr_c = q.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (0 <= nr < h and 0 <= nc < w and
                                p_map[nr, nc] == OBSTACLE_CELL and
                                component_map[nr, nc] == 0):
                            component_map[nr, nc] = component_id
                            q.append((nr, nc))
                
                component_id += 1
    
    return component_map

def run_mapf_simulation(env, unet_model, device, max_episode_steps=512, config: Optional[Dict] = None, preset: str = "paper", **kwargs):
    # 1. Initialization
    unet_model.eval()
    obs_list, _ = env.reset()
    num_agents = env.grid_config.num_agents
    all_pos, all_goals = list(env.get_agents_xy()), list(env.get_targets_xy())
    map_h, map_w = env.unwrapped.grid.get_obstacles().shape

    # Setting A: static obstacle map is known a priori
    p_map = env.unwrapped.grid.get_obstacles().astype(np.int8)
    obs_radius, window_size = env.grid_config.obs_radius, env.grid_config.obs_radius * 2 + 1

    if preset == "original":
        default_config = {
            'congestion_penalty': 25.0,
            'unknown_soft_cost': 3,
            'n_exec_steps': 6,
            'evaporation_group_size_threshold': 8,
            'evaporation_min_dist': 6,
            'evaporation_max_dist': 30,
            'push_rotate_group_size_threshold': 6,
            'local_plan_horizon_base': 40,
            'region_size': 7,
            'use_arg_planner': True,
            'cbs_max_iterations': 120,
            'cbs_time_limit_s': 300,
            'endgame_threshold_ratio': 0.20,
            'dezd_stagnation_window': 15,
            'dezd_stagnation_bbox_size': 8,
            'verbose': False,
            'planning_priority': 'distance_to_goal',
            'wall_repulsion_factor': 2.5,
            'wall_repulsion_range': 2.0,
            'visualization_params': {'enable': False, 'visualize_agent_id': 0, 'visualize_p_map': True, 'p_map_interval': 5},
        }
    else:
        # Paper Section 4.2, 4.3, Appendix H.5, & Table 13 defaults
        default_config = {
            'congestion_penalty': 25.0,
            'unknown_soft_cost': 3,
            'n_exec_steps': 6,
            'evaporation_group_size_threshold': 12,
            'evaporation_min_dist': 6,
            'evaporation_max_dist': 30,
            'push_rotate_group_size_threshold': 6,
            'local_plan_horizon_base': 40,
            'region_size': 16,
            'use_arg_planner': True,
            'cbs_max_iterations': 100,
            'cbs_time_limit_s': 30.0,
            'endgame_threshold_ratio': 0.15,
            'dezd_stagnation_window': 10,
            'dezd_stagnation_bbox_size': 8,
            'verbose': False,
            'planning_priority': 'distance_to_goal',
            'wall_repulsion_factor': 2.5,
            'wall_repulsion_range': 2.0,
            'visualization_params': {'enable': False, 'visualize_agent_id': 0, 'visualize_p_map': True, 'p_map_interval': 5},
        }

    if config:
        default_config.update(config)
    sim_params = default_config
    sim_params.update(kwargs)
    if 'large_group_threshold' in kwargs and kwargs['large_group_threshold'] is not None:
        sim_params['evaporation_group_size_threshold'] = kwargs['large_group_threshold']
    if 'pattering_history_len' in kwargs and kwargs['pattering_history_len'] is not None:
        sim_params['dezd_stagnation_window'] = kwargs['pattering_history_len']
    sim_params.update({
        'persistent_known_map': p_map,
        'FREE_CELL': FREE_CELL,
        'OBSTACLE_CELL': OBSTACLE_CELL,
        'UNKNOWN_CELL': UNKNOWN_CELL,
        'map_h': map_h,
        'map_w': map_w,
    })
    endgame_threshold_ratio = sim_params.get('endgame_threshold_ratio', 0.15)
    endgame_triggered = False
    endgame_cooldown = 0

    stats = {
        'cbs_invocations': 0,
        'dezd_activations': 0,
        'conflicts_per_cycle': [],
    }
    
    vis_params = sim_params.get('visualization_params', {'enable': False})
    if vis_params.get('enable', False):
        save_path = vis_params.get('save_path', f'mapf_viz_v23')
        os.makedirs(save_path, exist_ok=True)
        vis_params['save_path'] = save_path # Ensure save_path is in the dict
    
    def _find_best_reachable_proxy_goal(
        agent_pos: Tuple[int, int],
        original_goal: Tuple[int, int],
        p_map: np.ndarray,
        heuristic_manager: 'HeuristicManager'
    ) -> Optional[Tuple[int, int]]:

        map_h, map_w = p_map.shape
    
 
        if p_map[original_goal] == FREE_CELL:

            temp_h_map = heuristic_manager.get_true_distance_heuristic(original_goal, p_map)
            if temp_h_map[agent_pos] != -1:
                return original_goal
    
        logging.info(f"Finding best reachable proxy goal for agent at {agent_pos} with original goal {original_goal}.")
    
 
        h_from_agent = heuristic_manager.get_true_distance_heuristic(agent_pos, p_map)
        if h_from_agent[agent_pos] == -1:
            logging.error(f"Agent at {agent_pos} is trapped in a completely isolated area. Cannot find any proxy goal.")
            return None 

        h_to_goal = heuristic_manager.get_true_distance_heuristic(original_goal, p_map)
    

        best_proxy_goal = None
        min_total_cost = float('inf')
    

        free_cells_indices = np.argwhere(p_map == FREE_CELL)
    
        for r, c in free_cells_indices:
            pos = (r, c)
            
            cost_from_agent = h_from_agent[pos]
            cost_to_goal = h_to_goal[pos]
    
   
            if cost_from_agent != -1:
                total_cost = cost_from_agent + cost_to_goal
                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    best_proxy_goal = pos
    
        if best_proxy_goal:
            logging.info(f"Found best proxy goal: {best_proxy_goal} with total estimated cost {min_total_cost}.")
        else:
            logging.warning(f"Could not find any cell reachable by both agent {agent_pos} and goal {original_goal}.")
            
        return best_proxy_goal
    

    @numba.jit(nopython=True, cache=True)
    def _update_p_map_core(p_map, agent_pos_arr, fov_obstacles, active_mask, obs_radius, map_h, map_w):
 
        changed = False
        num_agents = len(agent_pos_arr)
        window_size = 2 * obs_radius + 1

        for i in range(num_agents):
            if not active_mask[i]: 
                continue

            r, c = agent_pos_arr[i]
            tl_r = r - obs_radius
            tl_c = c - obs_radius


            for fr in range(window_size):
                for fc in range(window_size):
                    gr = tl_r + fr
                    gc = tl_c + fc

                    # Boundary check
                    if 0 <= gr < map_h and 0 <= gc < map_w:

                        if p_map[gr, gc] == 2: 
                            val = fov_obstacles[i, fr, fc]
                            new_val = 1 if val == 1 else 0
                            p_map[gr, gc] = new_val
                            changed = True
        return changed


    heuristic_manager = HeuristicManager(sim_params)
    arg_planner = AdaptiveRegionGraph((map_h, map_w), sim_params, p_map, heuristic_manager) if sim_params.get('use_arg_planner', False) else None

    p_map_hash = get_map_hash(p_map) 

    obstacle_dist_map = _compute_dist_map_numba(p_map.astype(np.int32), map_h, map_w)
    obstacle_component_map = _compute_components_numba(p_map.astype(np.int32), map_h, map_w)

    agent_memories = {
        i: AgentMemory(i, (map_h, map_w), sim_params, heuristic_manager, arg_planner) 
        for i in range(num_agents)
    }
    consecutive_cbs_fails: Dict[frozenset, int] = defaultdict(int)
    active = [True] * num_agents

    paths_history = {i: [tuple(all_pos[i])] for i in range(num_agents)}
    steps, success, errors, start_time = 0, True, [], time.time()
    
    timing_stats = defaultdict(float)

    heuristic_maps_for_all_agents: Dict[int, np.ndarray] = {}

    while any(active) and steps < max_episode_steps:
        loop_start_time = time.time()

        if time.time() - start_time > sim_params['cbs_time_limit_s']:
            success = False
            errors.append(f"TIMEOUT@{steps}")
            break
        
        active_ids = [i for i, a in enumerate(active) if a]
        if not active_ids: break
        time_sec_1_start = time.time()
        num_active = len(active_ids)

        if not endgame_triggered and num_active <= int(num_agents * endgame_threshold_ratio):
            logging.warning(f"ENDGAME MODE TRIGGERED: {num_active}/{num_agents} agents remaining. Switching to global CBS solver.")
            
            if endgame_cooldown > 0:
                endgame_triggered = False 
                endgame_cooldown -= 1
                logging.debug(f"Endgame cooling down... {endgame_cooldown} steps remaining.")
            else:
                logging.debug(f"Endgame started at step {steps}")
                endgame_triggered = True

                straggler_agents_data = []
                initial_paths = {}
                straggler_ids = active_ids
                
                for aid in straggler_ids:
                    pos = tuple(all_pos[aid])
                    goal = tuple(all_goals[aid])
                    straggler_agents_data.append({'id': aid, 'start_local': pos, 'goal_local': goal})

                    best_goal = _find_best_reachable_proxy_goal(pos, goal, p_map, heuristic_manager)
                    if best_goal:
                        goal = best_goal
                    else:
                        logging.error(f"Endgame agent {aid} at {pos} could not find ANY reachable proxy for goal {goal}. Skipping.")

                    path = run_single_agent_astar(
                        pos, goal, p_map, sim_params, 
                        heuristic_manager.get_true_distance_heuristic(goal, p_map), obstacle_dist_map=obstacle_dist_map
                    )
                    initial_paths[aid] = [pos] + (path if path else [])

                logging.info(f"Running final CBS for the {len(straggler_ids)} remaining agents...")
                final_cbs_paths = solve_local_cbs_robust(
                    agents_data=straggler_agents_data,
                    obstacles_local_map=(p_map != FREE_CELL),
                    max_plan_len=100,
                    max_cbs_iterations=sim_params['cbs_max_iterations'] * 2, 
                    agents_true_global_goals_abs={i: tuple(all_goals[i]) for i in straggler_ids},
                    persistent_map_bundle={'persistent_known_map': p_map},
                    verbose_cbs_solver=sim_params.get('verbose', False)
                )

                if not final_cbs_paths:
                    endgame_cooldown = 2
                    logging.error("ENDGAME FAILED: Final CBS could not find a solution for the remaining agents.")
                    endgame_triggered = False 
                else:
                    logging.info("Endgame CBS solution found. Executing final paths...")
                    endgame_action_sequences = {}
                    max_path_len = 0
                    for aid in straggler_ids:
                        path_coords = final_cbs_paths.get(aid, [])
                        start_pos = tuple(all_pos[aid])
                        actions = path_coords_to_actions(path_coords, start_pos)
                        endgame_action_sequences[aid] = actions
                        if len(actions) > max_path_len:
                            max_path_len = len(actions)

                    had_truncation = False
                    for k in range(max_path_len):
                        if not any(active): break 
                        
                        step_actions = [ACTION_STAY] * num_agents
                        for aid in straggler_ids:
                            if k < len(endgame_action_sequences[aid]):
                                step_actions[aid] = endgame_action_sequences[aid][k]

                        obs_list, _, term, trunc, _ = env.step(step_actions)
                        steps += 1
                        all_pos = list(env.get_agents_xy())
    
                        for i in straggler_ids:
                            if active[i]:
                                paths_history[i].append(tuple(all_pos[i]))
                                if term[i]: active[i] = False
                                if trunc[i]:
                                    active[i] = False
                                    success = False
                                    had_truncation = True
                                    errors.append(f"A{i}_TRUNC@S{steps}_IN_ENDGAME")
                        
                        if not any(active) or had_truncation: break
                    logging.debug("Endgame execution completed.")

        for aid in active_ids:
            agent_memories[aid].check_and_handle_pattering(
                tuple(all_goals[aid]), 
                p_map,
                obstacle_dist_map,
                obstacle_component_map
            )

        timing_stats['1_map_update_and_pattering'] += time.time() - time_sec_1_start

        time_sec_2_total_start = time.time()
        time_agg_heuristic_calcs = 0
        time_agg_dezd_planning = 0
        time_agg_arg_guidance = 0
        time_agg_unet_prep = 0

        if sim_params['planning_priority'] == 'random':
            active_ids_prioritized = random.sample(active_ids, len(active_ids))
        else:
            active_ids_prioritized = sorted(active_ids, key=lambda aid: heuristic(all_pos[aid], all_goals[aid]))

        proposed_paths: Dict[int, List[Tuple[int, int]]] = {}
        planned_spacetime_obstacles: Set[Tuple[int, int, int]] = set() 

        if steps in [0, 60, 120, 180]:
            for aid in active_ids_prioritized:
                pos, goal = tuple(all_pos[aid]), tuple(all_goals[aid])
                mem = agent_memories[aid]
                global_h_map = heuristic_manager.get_true_distance_heuristic(goal, p_map)
                heuristic_maps_for_all_agents[aid] = global_h_map

        
        agents_for_unet_batch: List[Dict] = []

        for aid in active_ids_prioritized:
            t_start = time.time()

            pos, goal = tuple(all_pos[aid]), tuple(all_goals[aid])
            mem = agent_memories[aid]



            if vis_params.get('enable', False) and aid == vis_params.get('visualize_agent_id', -1):
                plot_heuristic_map(global_h_map, pos, goal, aid, steps, vis_params['save_path'])
            current_path = None
            time_agg_heuristic_calcs += time.time() - t_start

         
            t_start = time.time()

            if mem.status == "PATTERING_DETECTED":
                other_agents_pos = {tuple(p) for i, p in enumerate(all_pos) if i != aid}
                if active.count(True) >= 0.4 * len(active): 
                    escape_path= None
                    
                else:
                    escape_path = mem.initiate_escape_sequence_3(
                    p_map=p_map, 
                    global_h_map=global_h_map,
                    obstacle_dist_map = obstacle_dist_map,
                    other_agents_pos=other_agents_pos, 
                    sim_params=sim_params,
                    steps=steps,goals= goal)
                    
                if escape_path:
                    current_path = escape_path
                    stats['dezd_activations'] += 1 
            time_agg_dezd_planning += time.time() - t_start
            if not current_path:
                t_start = time.time()

                target_for_unet = goal
                guidance_path_coords = None

                if arg_planner and heuristic(pos, goal) > obs_radius:
                    high_level_path = arg_planner.find_high_level_path(pos, goal)
                    
                    if high_level_path:
                        subgoal = arg_planner.get_validated_subgoal_from_path(high_level_path, p_map, pos)
                        
                        if subgoal:
                            target_for_unet = subgoal
                            guidance_path_coords = run_single_agent_astar_for_unet(
                                pos, subgoal, p_map, sim_params, global_h_map, mem, 
                                {p for t, r, c in planned_spacetime_obstacles for p in [(r,c)]},
                                treat_other_agents_as_obstacles=True
                            )
                time_agg_arg_guidance += time.time() - t_start

                t_start = time.time()
                temp_obs = obs_list[aid].copy()
                target_map = np.zeros((window_size, window_size), dtype=np.float32)

                if guidance_path_coords: # 
                    for i, p in enumerate(guidance_path_coords):
                        p_local = global_to_local_coords(p, pos, obs_radius, window_size)
                        if p_local:
                            value = 1.0 - (i / (len(guidance_path_coords) * 1.5 + 1))
                            target_map[p_local] = max(target_map[p_local], value)
                else: # 
                    target_local = global_to_local_coords(target_for_unet, pos, obs_radius, window_size)
                    if target_local: target_map[target_local] = 1.0

                temp_obs['target'] = target_map
                


                global_map_slice = np.zeros((window_size, window_size), dtype=np.float32)
                
                tl_r, tl_c = pos[0] - obs_radius, pos[1] - obs_radius
                

                for r_local in range(window_size):
                    for c_local in range(window_size):
                        g_r, g_c = tl_r + r_local, tl_c + c_local
                        

                        if 0 <= g_r < map_h and 0 <= g_c < map_w:
                            global_map_slice[r_local, c_local] = float(p_map[g_r, g_c])
                        else:

                            global_map_slice[r_local, c_local] = 1.0

                unet_obstacles = global_map_slice.copy()
                
                for r_local in range(window_size):
                    for c_local in range(window_size):
                        g_r, g_c = tl_r + r_local, tl_c + c_local
                        if 0 <= g_r < map_h and 0 <= g_c < map_w:
                            if mem.personal_obstacle_map[g_r, g_c]:
                                unet_obstacles[r_local, c_local] = 1.0
                            if (0, g_r, g_c) in planned_spacetime_obstacles:
                                unet_obstacles[r_local, c_local] = 1.0
                
                temp_obs['obstacles'] = unet_obstacles

                agents_for_unet_batch.append({
                    'id': aid,
                    'spatial': get_spatial_features_v19_compatible(temp_obs, window_size),
                    'non_spatial': get_non_spatial_features(pos, target_for_unet),
                    'potential_map_args': {
                        'obs_map': temp_obs.get("obstacles").astype(bool),
                        'start_node': (obs_radius, obs_radius),
                        'max_steps': sim_params['n_exec_steps'] * 2,
                        'goal_node': global_to_local_coords(goal, pos, obs_radius, window_size)
                    }
                })
                time_agg_unet_prep += time.time() - t_start

            if current_path:
                proposed_paths[aid] = current_path
                for t, p_step in enumerate(current_path[1:sim_params['n_exec_steps']+1]):
                    planned_spacetime_obstacles.add((t, p_step[0], p_step[1]))
                    

        timing_stats['2.0_highlevel_planning_total'] += time.time() - time_sec_2_total_start
        timing_stats['2.1_heuristic_calcs'] += time_agg_heuristic_calcs
        timing_stats['2.2_dezd_planning'] += time_agg_dezd_planning
        timing_stats['2.3_arg_and_guidance_path'] += time_agg_arg_guidance
        timing_stats['2.4_unet_input_prep'] += time_agg_unet_prep


    
        time_sec_3_start = time.time()
        if agents_for_unet_batch:
            s_batch = torch.stack([d['spatial'] for d in agents_for_unet_batch]).to(device)
            ns_batch = torch.stack([d['non_spatial'] for d in agents_for_unet_batch]).to(device)
            with torch.no_grad():
                potentials = unet_model(s_batch, ns_batch).squeeze(1).cpu().numpy()

            for i, agent_data in enumerate(agents_for_unet_batch):
                aid = agent_data['id']
                pos = tuple(all_pos[aid])
                args = agent_data['potential_map_args']
                _, path_local = decode_action_sequence_refined(potentials[i], args['obs_map'],args['start_node'],args['max_steps'],args['goal_node'],None)

                if vis_params.get('enable', False) and agent_data['id'] == vis_params.get('visualize_agent_id', -1):
                    plot_unet_debug(
                        args['obs_map'],
                        potentials[i],
                        path_local,
                        agent_data['id'],
                        steps,
                        vis_params['save_path']
                    )
                    
                current_path = [pos] + [(pos[0] - obs_radius + r, pos[1] - obs_radius + c) for r, c in path_local[1:]]
                proposed_paths[aid] = current_path
                for t, p_step in enumerate(current_path[1:sim_params['n_exec_steps']+1]):
                    planned_spacetime_obstacles.add((t, p_step[0], p_step[1]))
                    
        timing_stats['3_unet_planning'] += time.time() - time_sec_3_start

        time_sec_4_start = time.time()
        
        final_paths: Dict[int, List[Tuple[int, int]]] = {}
        conflict_components, _ = _build_conflict_groups(proposed_paths, active_ids, sim_params['n_exec_steps'], sim_params)
        
        handled_agents = set()

        for group_set in conflict_components:
            if len(group_set) < 2: continue
            group = sorted(list(group_set))
            g_data = [{'id': aid, 'start_local': tuple(all_pos[aid]), 'goal_local': proposed_paths.get(aid, [tuple(all_pos[aid])])[-1]} for aid in group]
            group_d_subgoal = max([heuristic(d['start_local'], d['goal_local']) for d in g_data])
            dynamic_k = max(min(group_d_subgoal, 10), sim_params['n_exec_steps'])
            sim_params['dynamic_k'] = dynamic_k 
            solution = _solve_single_group_with_defense_cascade(
                group, 
                g_data, 
                (p_map != FREE_CELL), 
                all_pos, 
                all_goals, 
                consecutive_cbs_fails, 
                sim_params,
                heuristic_maps_for_all_agents,  
                agent_memories,obstacle_dist_map                 
            )            
            if solution: final_paths.update(solution)
            else: final_paths.update({aid: [tuple(all_pos[aid])] for aid in group})
            handled_agents.update(group)

        for aid in active_ids:
            if aid not in handled_agents:
                final_paths[aid] = proposed_paths.get(aid, [tuple(all_pos[aid])])

        timing_stats['4_solve_conflit_planning'] += time.time() - time_sec_4_start

        time_sec_5_start = time.time()
        time_agg_env_step = 0
        time_agg_map_update_after_step = 0
        
        max_path_len = 0 if not final_paths else max(len(p) for p in final_paths.values()) if final_paths else 0
        n_exec = min(sim_params['n_exec_steps'], max_path_len - 1 if max_path_len > 0 else 0)
        if n_exec <= 0: n_exec = 1
        
        action_sequences = {}
        for aid in active_ids:
            path = final_paths.get(aid, [tuple(all_pos[aid])])
            actions = path_coords_to_actions(path[1:], path[0])
            actions.extend([ACTION_STAY] * (n_exec - len(actions)))
            action_sequences[aid] = actions

        map_changed_during_exec = False

        for k in range(n_exec):
            if not any(active): break
            step_actions = [action_sequences.get(i, [ACTION_STAY]*n_exec)[k] if active[i] else ACTION_STAY for i in range(num_agents)]
            
            t_env_step_start = time.time()
            obs_list, _, term, trunc, _ = env.step(step_actions)
            time_agg_env_step += time.time() - t_env_step_start

            steps += 1
            all_pos = list(env.get_agents_xy())
            had_truncation = False
            
 
            for i in range(num_agents):
                cached_h = heuristic_maps_for_all_agents.get(i)
                agent_memories[i].update_after_step(tuple(all_pos[i]), tuple(all_goals[i]), p_map, cached_h)

                if active[i]:
                    paths_history[i].append(tuple(all_pos[i]))
                    if term[i]: active[i] = False
                    if trunc[i]: active[i]=False; success=False; had_truncation=True; errors.append(f"A{i}_TRUNC@S{steps}")

            t_map_update_after_step_start = time.time()
            


            time_agg_map_update_after_step += time.time() - t_map_update_after_step_start

            if not any(active) or steps >= max_episode_steps or had_truncation: break
        

        if map_changed_during_exec:
            t_heavy_calc_start = time.time()
            

            p_map_int = p_map.astype(np.int32)
            obstacle_dist_map = _compute_dist_map_numba(p_map_int, map_h, map_w)
            obstacle_component_map = _compute_components_numba(p_map_int, map_h, map_w)
            
            p_map_hash = get_map_hash(p_map)
            
            if arg_planner: 
                arg_planner.update_graph_with_obstacles(p_map)
            
            time_agg_map_update_after_step += time.time() - t_heavy_calc_start
                
        timing_stats['5.0_path_execution_total'] += time.time() - time_sec_5_start
        timing_stats['5.1_env_step_calls'] += time_agg_env_step
        timing_stats['5.2_map_updates_in_exec_loop'] += time_agg_map_update_after_step

        if not success: break
        timing_stats['total_loop_time'] += time.time() - loop_start_time


    finished = sum(1 for i in range(num_agents) if tuple(all_pos[i]) == tuple(all_goals[i]))
    if sim_params.get('verbose', False):
        print("\n" + "=" * 50)
        print("Performance Analysis Results")
        print("=" * 50)
        total_time = timing_stats.get('total_loop_time', 0)
        num_steps_processed = steps / sim_params.get('n_exec_steps', 1)
        if num_steps_processed == 0:
            num_steps_processed = 1

        print(f"Total simulation time: {time.time() - start_time:.4f} seconds")
        print(f"Total steps executed: {steps}")
        print(f"Approximate main loops: {int(num_steps_processed)}")
        print("-" * 50)
        sorted_timings = sorted(timing_stats.items(), key=lambda item: item[1], reverse=True)

        for key, value in sorted_timings:
            if key != 'total_loop_time':
                percentage = (value / total_time) * 100 if total_time > 0 else 0
                avg_time_per_loop = value / num_steps_processed
                print(f"{key:<30s}: {value:>10.4f}s ({percentage:5.2f}%) | Avg per loop: {avg_time_per_loop:.6f}s")

    return {
        "success": success and not any(active),
        "makespan": steps,
        "sum_of_costs": sum(len(p) - 1 for p in paths_history.values()),
        "executed_paths_global": paths_history,
        "error_summary": "; ".join(errors) or "No errors.",
        "num_agents_at_start": num_agents,
        "num_agents_reached_target": finished,
    }