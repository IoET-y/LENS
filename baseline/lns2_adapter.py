# ====================================================================
# File: lns2_adapter.py
# Desc: High-fidelity adapter for LNS2-RL on POGEMA maps
#       Fixes: List Indexing Bug, Action Mapping, Ghost Obstacles, Heuristic Semantics
# ====================================================================

import numpy as np
import copy
import sys
from collections import deque

try:
    from LNS2_RL.alg_parameters import EnvParameters, NetParameters
    from LNS2_RL.world_property import State
    from LNS2_RL.dynamic_state import DyState
except ImportError:
    from alg_parameters import EnvParameters, NetParameters
    from world_property import State
    from dynamic_state import DyState

# ====================================================================
#  (Verified against world_property.py)
# LNS2 Native: 
#   1: (0, 1) -> Right (Col +1)
#   2: (1, 0) -> Down  (Row +1)
#   3: (0, -1) -> Left (Col -1)
#   4: (-1, 0) -> Up   (Row -1)
#
# POGEMA Native: 
#   0: Stay, 1: Up, 2: Down, 3: Left, 4: Right
# ====================================================================
LNS2_TO_POGEMA = {0: 0, 1: 4, 2: 2, 3: 3, 4: 1}

class PogemaLNS2Env:
    def __init__(self, pogema_grid, agents_xy, targets_xy, lns2_cpp_instance, safe_dim=None):
        self.lns2_model = lns2_cpp_instance
        self.pogema_shape = pogema_grid.shape 
        self.rows, self.cols = self.pogema_shape
        
        self.safe_dim = safe_dim if safe_dim else max(self.rows, self.cols)
        self.size = max(self.rows, self.cols)
        
        self.global_num_agent = len(agents_xy)
        self.map = np.zeros((self.rows, self.cols), dtype=int)
        self.map[pogema_grid == 1] = -1
        
        self.start_list = [(r, c) for r, c in agents_xy]
        self.goal_list = [(r, c) for r, c in targets_xy]
        self.fov_size = EnvParameters.FOV_SIZE
        
        self.fix_state = copy.copy(self.map)
        self.fix_state_dict = {}
        for i in range(self.rows):
            for j in range(self.cols):
                self.fix_state_dict[i, j] = []
        
        self.world = State(self.fix_state, self.fix_state_dict, self.global_num_agent, self.start_list, self.goal_list)
        
        self._recalc_heuristic_map()

        raw_paths = self.lns2_model.vector_path
        self.paths = [[tuple(pos) for pos in path] for path in raw_paths]
        
        self.dynamic_state = None 
        self.agent_util_map_action = []
        self.agent_util_map_vertex = []
        self.local_agents = []
        self.time_step = 0
        self.sipps_path = []
        self.all_next_poss = []


    def _recalc_heuristic_map(self):
 
        self.world.heuri_map = np.zeros((self.global_num_agent, 4, self.rows, self.cols), dtype=bool)
        SAFE_INF = 1000000000 # 

        for i in range(self.global_num_agent):
            goal = self.goal_list[i]
            dist_map = np.full((self.rows, self.cols), SAFE_INF, dtype=np.int32)
            q = deque([goal])
            dist_map[goal] = 0
            
            while q:
                curr_r, curr_c = q.popleft()
                curr_dist = dist_map[curr_r, curr_c]
                
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        # Do not traverse through obstacles (-1)
                        if self.map[nr, nc] != -1 and dist_map[nr, nc] > curr_dist + 1:
                            dist_map[nr, nc] = curr_dist + 1
                            q.append((nr, nc))
            
            # Match the Up/Down/Left/Right channel ordering in world_property.py
            # Up (Channel 0): dist(x-1) < dist(x)
            self.world.heuri_map[i, 0, 1:, :] = (dist_map[:-1, :] < dist_map[1:, :]) & (self.map[1:, :] != -1) & (self.map[:-1, :] != -1)
            # Down (Channel 1): dist(x+1) < dist(x)
            self.world.heuri_map[i, 1, :-1, :] = (dist_map[1:, :] < dist_map[:-1, :]) & (self.map[:-1, :] != -1) & (self.map[1:, :] != -1)
            # Left (Channel 2): dist(y-1) < dist(y)
            self.world.heuri_map[i, 2, :, 1:] = (dist_map[:, :-1] < dist_map[:, 1:]) & (self.map[:, 1:] != -1) & (self.map[:, :-1] != -1)
            # Right (Channel 3): dist(y+1) < dist(y)
            self.world.heuri_map[i, 3, :, :-1] = (dist_map[:, 1:] < dist_map[:, :-1]) & (self.map[:, :-1] != -1) & (self.map[:, 1:] != -1)


    def reset_for_planning(self, local_agents, first_time=False):
        self.local_agents = local_agents
        self.local_num_agents = len(local_agents)
        self.time_step = 0
        
        self.dynamic_state = DyState(self.paths, self.global_num_agent, (self.safe_dim, self.safe_dim))
        
        path_new_agent = {i: self.paths[i] for i in self.local_agents}
        self.sipp_coll_pair_num = self.lns2_model.calculate_sipps(self.local_agents)
        self.makespan = self.lns2_model.makespan
        
        raw_sipps = self.lns2_model.sipps_path
        self.sipps_path = [[tuple(pos) for pos in path] for path in raw_sipps]
        self.sipps_max_len = max([len(p) for p in self.sipps_path]) if self.sipps_path else 0
        
        self.dynamic_state.reset_local_tasks(self.local_agents, path_new_agent, None, None, self.makespan + 1)
        self.world.reset_local_tasks(self.fix_state, self.fix_state_dict, self.start_list, self.local_agents)
        
        self.agent_util_map_action = [np.zeros((5, self.rows, self.cols)) for _ in range(2)]
        self.agent_util_map_vertex = [np.zeros((self.rows, self.cols)) for _ in range(2)]
        
        for local_i in range(self.local_num_agents):
            pos = self.world.local_agents_poss[local_i]
            if 0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols:
                self.agent_util_map_vertex[-1][pos] += 1
            
        self.new_collision_pairs = set()
        self.update_ulti()
        self.predict_next()
        self.episode_len = max(EnvParameters.EPISODE_LEN[0], self.makespan + 32)

    def predict_next(self):
        self.all_next_poss = []
        if self.time_step != 0:
            for local_agent_index in range(self.local_num_agents):
                next_poss_list = []
                for k in range(EnvParameters.K_STEPS):
                    path_arr = np.array(self.sipps_path[local_agent_index])
                    curr_pos = self.world.local_agents_poss[local_agent_index]
                    if len(path_arr) == 0: 
                        next_poss = curr_pos
                    else:
                        if k == 0:
                            dis = np.sqrt(np.sum((curr_pos - path_arr)**2, axis=1))
                            idx = np.argmin(dis) + 1
                        else:
                            idx = min(self.time_step + k, len(path_arr)-1)
                        
                        if idx < len(path_arr):
                            next_poss = tuple(path_arr[idx])
                        else:
                            next_poss = tuple(path_arr[-1])
                    next_poss_list.append(next_poss)
                self.all_next_poss.append(next_poss_list)
        else:
            for local_agent_index in range(self.local_num_agents):
                next_poss_list = []
                for k in range(EnvParameters.K_STEPS):
                    if k + 1 < len(self.sipps_path[local_agent_index]):
                        next_poss = tuple(self.sipps_path[local_agent_index][k + 1])
                    else:
                        next_poss = tuple(self.sipps_path[local_agent_index][-1])
                    next_poss_list.append(next_poss)
                self.all_next_poss.append(next_poss_list)

    def update_ulti(self):
        self.space_ulti_action = np.zeros((5, self.rows, self.cols))
        self.space_ulti_vertex = np.zeros((self.rows, self.cols))
        
        for t in EnvParameters.UTI_WINDOWS:
            fut_t = self.time_step + t + 1
            if fut_t < 0: continue
            
            if fut_t < self.dynamic_state.max_lens:
                # DyState.state is numpy array (Time, H, W) -> Direct Slice OK
                state_slice = self.dynamic_state.state[fut_t, :self.rows, :self.cols]
                
                # DyState.util_map_action is List of numpy arrays -> List Index then Slice
                action_slice = self.dynamic_state.util_map_action[fut_t][:, :self.rows, :self.cols]
                
                self.space_ulti_vertex += state_slice
                self.space_ulti_action += action_slice

        self.space_ulti_vertex = 10 * self.space_ulti_vertex / self.global_num_agent
        self.space_ulti_action = 10 * self.space_ulti_action / self.global_num_agent

    def joint_step(self, actions_idx):
        self.time_step += 1
        self.agent_util_map_action.pop(0)
        self.agent_util_map_vertex.pop(0)
        self.agent_util_map_action.append(np.zeros((5, self.rows, self.cols)))
        self.agent_util_map_vertex.append(np.zeros((self.rows, self.cols)))
        
        for local_i, i in enumerate(self.local_agents):
            action = int(actions_idx[local_i])
            direction = self.world.get_dir(action)
            ax, ay = self.world.local_agents_poss[local_i]
            nx, ny = ax + direction[0], ay + direction[1]
            
            if 0 <= nx < self.rows and 0 <= ny < self.cols and self.map[nx, ny] >= 0:
                if i in self.world.state_dict[ax, ay]: self.world.state_dict[ax, ay].remove(i)
                if self.world.state[ax, ay] > 0: self.world.state[ax, ay] -= 1
                
                self.world.agents_poss[i] = (nx, ny)
                self.world.local_agents_poss[local_i] = (nx, ny)
                
                is_at_goal = ((nx, ny) == self.goal_list[i])
                if not is_at_goal:
                    self.world.state[nx, ny] += 1
                    self.world.state_dict[nx, ny].append(i)
                    self.agent_util_map_action[-1][action, nx, ny] += 1
                    self.agent_util_map_vertex[-1][nx, ny] += 1
            else:
                self.agent_util_map_action[-1][0, ax, ay] += 1
                self.agent_util_map_vertex[-1][ax, ay] += 1
                
        self.update_ulti()
        self.predict_next()
        
        all_done = True
        for i in self.local_agents:
            if self.world.agents_poss[i] != self.goal_list[i]:
                all_done = False
                break
        return (self.time_step >= self.episode_len) or all_done

    def observe(self, local_agent_index):
        agent_index = self.world.local_agents[local_agent_index]
        curr_pos = self.world.agents_poss[agent_index]
        
        top = max(curr_pos[0] - self.fov_size // 2, 0)
        bottom = min(curr_pos[0] + self.fov_size // 2 + 1, self.rows)
        left = max(curr_pos[1] - self.fov_size // 2, 0)
        right = min(curr_pos[1] + self.fov_size // 2 + 1, self.cols)
        
        top_left = (curr_pos[0] - self.fov_size // 2, curr_pos[1] - self.fov_size // 2)
        FOV_top = max(self.fov_size // 2 - curr_pos[0], 0)
        FOV_left = max(self.fov_size // 2 - curr_pos[1], 0)
        FOV_bottom = FOV_top + (bottom - top)
        FOV_right = FOV_left + (right - left)
        
        obs_shape = (self.fov_size, self.fov_size)
        
        goal_map = np.zeros(obs_shape)
        local_poss_map = np.zeros(obs_shape)
        local_goals_map = np.zeros(obs_shape)
        obs_map = np.ones(obs_shape)
        guide_map = np.zeros((4, self.fov_size, self.fov_size))
        dynamic_poss_maps = np.zeros((EnvParameters.NUM_TIME_SLICE, self.fov_size, self.fov_size))
        sipps_map = np.zeros(obs_shape)
        util_map = np.zeros(obs_shape)
        util_map_action = np.zeros((5, self.fov_size, self.fov_size))
        blank_map = np.zeros(obs_shape)
        occupy_map = np.zeros(obs_shape)
        next_step_map = np.zeros((EnvParameters.K_STEPS, self.fov_size, self.fov_size))

        # Fill Static
        goal_pos = self.goal_list[agent_index]
        if top <= goal_pos[0] < bottom and left <= goal_pos[1] < right:
            goal_map[goal_pos[0] - top_left[0], goal_pos[1] - top_left[1]] = 1
        
        local_poss_map[curr_pos[0] - top_left[0], curr_pos[1] - top_left[1]] = 1
        obs_map[FOV_top:FOV_bottom, FOV_left:FOV_right] = -self.map[top:bottom, left:right]
        guide_map[:, FOV_top:FOV_bottom, FOV_left:FOV_right] = self.world.heuri_map[agent_index][:, top:bottom, left:right]
        util_map[FOV_top:FOV_bottom, FOV_left:FOV_right] = self.space_ulti_vertex[top:bottom, left:right]
        util_map_action[:, FOV_top:FOV_bottom, FOV_left:FOV_right] = self.space_ulti_action[:, top:bottom, left:right]

        # SIPP
        if self.time_step - EnvParameters.WINDOWS < 0: min_time = 0
        elif self.time_step >= len(self.sipps_path[local_agent_index]): min_time = max(0, len(self.sipps_path[local_agent_index]) - EnvParameters.WINDOWS)
        else: min_time = self.time_step - EnvParameters.WINDOWS
        max_time = min(self.time_step + EnvParameters.WINDOWS, len(self.sipps_path[local_agent_index]))
        window_path = self.sipps_path[local_agent_index][min_time:max_time]
        
        visible_agents = set()
        
        for i in range(top_left[0], top_left[0] + self.fov_size):
            for j in range(top_left[1], top_left[1] + self.fov_size):
                rel_x, rel_y = i - top_left[0], j - top_left[1]
                
                if i >= self.rows or i < 0 or j >= self.cols or j < 0 or self.world.state[i, j] == -1:
                    occupy_map[rel_x, rel_y] = 1 - self.time_step / (self.episode_len + 1e-5)
                    continue
                
                if (i, j) in window_path: sipps_map[rel_x, rel_y] = 1
                
                for iter_a in range(self.local_num_agents):
                    if iter_a != local_agent_index:
                        for k in range(EnvParameters.K_STEPS):
                            if (i, j) == self.all_next_poss[iter_a][k]: 
                                next_step_map[k, rel_x, rel_y] += 1

                if self.world.state[i, j] > 0:
                    for item in self.world.state_dict[i, j]:
                        if item in self.world.local_agents and item != agent_index:
                            visible_agents.add(item)
                            local_poss_map[rel_x, rel_y] += 1
                
                for t in range(EnvParameters.NUM_TIME_SLICE):
                    fut_t = self.time_step + t
                    if fut_t < self.dynamic_state.max_lens:
                        dynamic_poss_maps[t, rel_x, rel_y] = self.dynamic_state.state[fut_t, i, j]

                occupy_t = 0
                blank_t = 0
                scan_end = int(self.episode_len + 1)
                
                for t_sim in range(self.time_step, scan_end):
                    if t_sim >= self.dynamic_state.max_lens: break
                    if self.dynamic_state.state[t_sim, i, j] > 0: occupy_t += 1
                    else: break
                
                for t_sim in range(self.time_step + 1, scan_end):
                    if t_sim >= self.dynamic_state.max_lens: 
                        blank_t += (self.episode_len - t_sim)
                        break
                    if self.dynamic_state.state[t_sim, i, j] == 0: blank_t += 1
                    else: break
                    
                occupy_map[rel_x, rel_y] = occupy_t / (self.episode_len + 1e-5)
                blank_map[rel_x, rel_y] = blank_t / (self.episode_len + 1e-5)

        zero_mask = local_poss_map == 0; local_poss_map = 0.5 + 0.5 * np.tanh((local_poss_map - 1) / 3); local_poss_map[zero_mask] = 0
        zero_mask = next_step_map == 0; next_step_map = 0.5 + 0.5 * np.tanh((next_step_map - 1) / 3); next_step_map[zero_mask] = 0
        zero_mask = dynamic_poss_maps == 0; dynamic_poss_maps = 0.5 + 0.5 * np.tanh((dynamic_poss_maps - 1) / 3); dynamic_poss_maps[zero_mask] = 0
        
        for vis in visible_agents:
            gx, gy = self.world.agents_goals[vis]
            px = max(top_left[0], min(top_left[0] + self.fov_size - 1, gx))
            py = max(top_left[1], min(top_left[1] + self.fov_size - 1, gy))
            local_goals_map[px - top_left[0], py - top_left[1]] = 1


        # [CRITICAL FIX]: Vector calculation logic
        dx = self.world.agents_goals[agent_index][0] - self.world.agents_poss[agent_index][0]
        dy = self.world.agents_goals[agent_index][1] - self.world.agents_poss[agent_index][1]
        mag = (dx ** 2 + dy ** 2) ** .5
        if mag != 0: dx, dy = dx/mag, dy/mag
        
        # Return base vector [dx, dy, mag].
        # Indices 3-7 of the 8-dim vector are assembled in the benchmark loop.
        vector = np.array([dx, dy, mag]) 
        
        return np.array([*dynamic_poss_maps, local_poss_map, goal_map, local_goals_map, obs_map, *guide_map, sipps_map, blank_map, occupy_map, util_map, *util_map_action, *next_step_map]), vector

    def list_next_valid_actions(self, local_agent_index):
        return self.world.list_next_valid_actions(local_agent_index)