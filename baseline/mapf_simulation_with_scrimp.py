# mapf_simulation_with_scrimp.py

import torch
import numpy as np
import logging

# --- SCRIMP Imports ---
# Assumes all SCRIMP .py files are in the same directory or accessible in PYTHONPATH
from scrimp_model.model import Model
from scrimp_model.episodic_buffer import EpisodicBuffer
from scrimp_model.alg_parameters import EnvParameters, NetParameters, IntrinsicParameters

# --- Constants for Action Mapping ---
# Translates SCRIMP's action space to Pogema's action space.
# Based on analysis of SCRIMP's `mapf_gym.py` dirDict and Pogema's action definitions.
# SCRIMP Action (Integer): Pogema Action (Integer)
SCRIMP_TO_POGEMA_ACTION_MAP = {
    0: 0,  # Stay   -> Stay
    4: 1,  # Up     -> Up
    2: 2,  # Down   -> Down
    3: 3,  # Left   -> Left
    1: 4,  # Right  -> Right
}

def calculate_heuristic_maps(grid_map, goals, num_agents):
    """
    Calculates the heuristic map for all agents using BFS. This is a required
    part of the observation for the SCRIMP model.
    """
    height, width = grid_map.shape
    dist_map = np.full((num_agents, height, width), fill_value=np.iinfo(np.int32).max, dtype=np.int32)
    
    for i in range(num_agents):
        goal_pos = tuple(goals[i])
        q = [goal_pos]
        dist_map[i, goal_pos[0], goal_pos[1]] = 0
        
        head = 0
        while head < len(q):
            (r, c) = q[head]
            head += 1
            dist = dist_map[i, r, c]

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and grid_map[nr, nc] == 0 and dist_map[i, nr, nc] > dist + 1:
                    dist_map[i, nr, nc] = dist + 1
                    q.append((nr, nc))

    heuri_map = np.zeros((num_agents, 4, height, width), dtype=bool)
    for i in range(num_agents):
        for r in range(height):
            for c in range(width):
                if grid_map[r, c] == 0:
                    dist = dist_map[i, r, c]
                    # Note: Channel order matches SCRIMP's internal format
                    # Up (towards smaller row index)
                    if r > 0 and dist_map[i, r - 1, c] < dist: heuri_map[i, 0, r, c] = 1
                    # Down
                    if r < height - 1 and dist_map[i, r + 1, c] < dist: heuri_map[i, 1, r, c] = 1
                    # Left
                    if c > 0 and dist_map[i, r, c - 1] < dist: heuri_map[i, 2, r, c] = 1
                    # Right
                    if c < width - 1 and dist_map[i, r, c + 1] < dist: heuri_map[i, 3, r, c] = 1
    return heuri_map

def pogema_to_scrimp_inputs(env, heuri_maps):
    """
    Translates the current Pogema state into the observation and vector
    formats expected by the SCRIMP model.
    """
    num_agents = env.grid_config.num_agents
    fov_size = EnvParameters.FOV_SIZE
    
    obs_batch = np.zeros((1, num_agents, NetParameters.NUM_CHANNEL, fov_size, fov_size), dtype=np.float32)
    vector_batch = np.zeros((1, num_agents, NetParameters.VECTOR_LEN), dtype=np.float32)
    
    agent_positions = env.get_agents_xy()
    agent_goals = env.get_targets_xy()
    grid_map = env.get_obstacles()
    map_h, map_w = grid_map.shape

    for i in range(num_agents):
        my_pos = agent_positions[i]
        
        # Part 1: Construct Image Observation (8 channels)
        top_left = (my_pos[0] - fov_size // 2, my_pos[1] - fov_size // 2)
        
        poss_map, goal_map, goals_map, obs_map = (np.zeros((fov_size, fov_size)) for _ in range(4))
        
        visible_agents_indices = []
        for r_fov in range(fov_size):
            for c_fov in range(fov_size):
                r_map, c_map = top_left[0] + r_fov, top_left[1] + c_fov

                if not (0 <= r_map < map_h and 0 <= c_map < map_w) or grid_map[r_map, c_map] == 1:
                    obs_map[r_fov, c_fov] = 1
                    continue
                
                if (r_map, c_map) == tuple(agent_goals[i]):
                    goal_map[r_fov, c_fov] = 1
                
                for other_idx, other_pos in enumerate(agent_positions):
                    if (r_map, c_map) == tuple(other_pos):
                        poss_map[r_fov, c_fov] = 1
                        if other_idx != i:
                           visible_agents_indices.append(other_idx)
        
        for other_idx in set(visible_agents_indices):
            other_goal = agent_goals[other_idx]
            proj_r = max(top_left[0], min(top_left[0] + fov_size - 1, other_goal[0]))
            proj_c = max(top_left[1], min(top_left[1] + fov_size - 1, other_goal[1]))
            proj_r_fov, proj_c_fov = proj_r - top_left[0], proj_c - top_left[1]
            goals_map[proj_r_fov, proj_c_fov] = 1

        guide_map = np.zeros((4, fov_size, fov_size))
        for r_fov in range(fov_size):
            for c_fov in range(fov_size):
                r_map, c_map = top_left[0] + r_fov, top_left[1] + c_fov
                if 0 <= r_map < map_h and 0 <= c_map < map_w:
                    # channel order: 0:up, 1:down, 2:left, 3:right
                    guide_map[:, r_fov, c_fov] = heuri_maps[i, :, r_map, c_map]

        obs_batch[0, i] = np.stack([poss_map, goal_map, goals_map, obs_map,
                                    guide_map[0], guide_map[1], guide_map[2], guide_map[3]])
        
        # Part 2: Construct Vector Input (initial dx, dy, mag part)
        my_goal = agent_goals[i]
        dr, dc = my_goal[0] - my_pos[0], my_goal[1] - my_pos[1]
        mag = (dr**2 + dc**2)**0.5
        dx, dy = (dr / mag, dc / mag) if mag != 0 else (0, 0)
        
        vector_batch[0, i, :3] = [dx, dy, mag]
    
    return obs_batch, vector_batch

def run_scrimp_simulation(env, model, device, max_episode_steps):
    """
    Runs a single simulation episode using the SCRIMP model in a Pogema environment.
    """
    num_agents = env.grid_config.num_agents
    env.reset()

    # --- Initializations ---
    hidden_state = (torch.zeros((num_agents, NetParameters.NET_SIZE // 2)).to(device),
                    torch.zeros((num_agents, NetParameters.NET_SIZE // 2)).to(device))
    message = torch.zeros((1, num_agents, NetParameters.NET_SIZE)).to(device)
    
    episodic_buffer = EpisodicBuffer(total_step=2e6, num_agent=num_agents)
    episodic_buffer.batch_add(env.get_agents_xy())

    grid_map = env.get_obstacles()
    goals = env.get_targets_xy()
    heuristic_maps = calculate_heuristic_maps(grid_map, goals, num_agents)

    agents_done = np.array([False] * num_agents)
    individual_costs = {i: 0 for i in range(num_agents)}
    
    scrimp_obs, scrimp_vector = pogema_to_scrimp_inputs(env, heuristic_maps)
    scrimp_vector[:, :, -1] = 0 # Previous action is 0 at start
    
    for step in range(max_episode_steps):
        if np.all(agents_done):
            break

        with torch.no_grad():
            scrimp_actions_np, hidden_state, _, _, message = model.final_evaluate(
                scrimp_obs, scrimp_vector, hidden_state, message, num_agents, greedy=False
            )
        scrimp_actions_np = scrimp_actions_np.astype(int)

        pogema_actions = [SCRIMP_TO_POGEMA_ACTION_MAP[a] for a in scrimp_actions_np]

        _, _, terminated, truncated, _ = env.step(pogema_actions)
        # FIX: Correctly update agent done status based on `terminated` flag.

        for i in range(num_agents):
            if not agents_done[i]:
                individual_costs[i] += 1
            if terminated[i] :
                agents_done[i] = True
            # if truncated[i]:
            #     agents_done[i] = False
        # --- Prepare inputs for the NEXT step ---
        new_xy = env.get_agents_xy()
        
        scrimp_rewards_ext = np.zeros((1, num_agents), dtype=np.float32)
        for i in range(num_agents):
            if terminated[i]:
                scrimp_rewards_ext[0, i] = EnvParameters.GOAL_REWARD
            elif pogema_actions[i] == 0:
                scrimp_rewards_ext[0, i] = EnvParameters.IDLE_COST
            else:
                scrimp_rewards_ext[0, i] = EnvParameters.ACTION_COST

        _, _, intrinsic_rewards, min_dists = episodic_buffer.if_reward(
            new_xy, scrimp_rewards_ext, np.all(agents_done), terminated
        )

        scrimp_obs, scrimp_vector = pogema_to_scrimp_inputs(env, heuristic_maps)
        scrimp_vector[:, :, 3] = scrimp_rewards_ext
        scrimp_vector[:, :, 4] = intrinsic_rewards
        scrimp_vector[:, :, 5] = min_dists
        scrimp_vector[:, :, -1] = scrimp_actions_np
        
    # FIX: Calculate metrics based on agents that successfully finished, even in a partial success.
    
    # Overall success is when ALL agents have finished.
    success = np.all(agents_done)
    # ISR is based on how many agents actually finished.
    num_reached = np.sum(agents_done)
    
    # Collect costs ONLY for agents that successfully reached their goal.
    finished_costs = [cost for i, cost in individual_costs.items() if agents_done[i]]
    
    # Makespan is the time of the last successful agent. If none, it's the timeout.
    makespan = max(finished_costs) if finished_costs else max_episode_steps
    # Sum of Costs is the sum of path lengths for successful agents. If none, it's -1.
    sum_of_costs = sum(finished_costs) if finished_costs else -1

    return {
        "success": bool(success),
        "makespan": int(makespan),
        "sum_of_costs": int(sum_of_costs),
        "individual_costs": {i: c for i, c in individual_costs.items() if agents_done[i]},
        "num_agents_reached_target": int(num_reached),
    }