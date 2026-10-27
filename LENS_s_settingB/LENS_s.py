# LENS_s.py (Setting B: Completely Unknown Environment)
from collections import defaultdict, deque
import heapq
import logging
from pathlib import Path
import random
import time
from typing import Deque, Dict, List, Optional, Set, Tuple

import networkx as nx
import numba
from numba.core import types
import numpy as np
import torch

from action_sequence_decoder_s import decode_action_sequence_refined
from local_cbs_solver_s import (
    CBSHighLevelNode,
    Conflict,
    Constraint,
    detect_all_conflicts_spacetime,
    solve_local_cbs_robust,
)
from unet_model_new import UNetPotentialField

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ACTION_DELTAS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
ACTION_NAMES = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
ACTION_STAY = 0
JIT_ACTION_DELTAS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)
POS_KEY_TYPE = types.UniTuple(types.int64, 3)

UNKNOWN_CELL = 2
FREE_CELL = 0
OBSTACLE_CELL = 1


def get_spatial_features_from_obs(agent_obs_dict, local_window_size):
    num_spatial_channels = 4
    h = w = local_window_size
    spatial_features = np.zeros((num_spatial_channels, h, w), dtype=np.float32)
    center_idx = h // 2
    spatial_features[0, :, :] = agent_obs_dict.get("obstacles", np.ones((h, w))).astype(np.float32)
    spatial_features[1, :, :] = agent_obs_dict.get("agents", np.zeros((h, w))).astype(np.float32)
    spatial_features[2, center_idx, center_idx] = 1.0
    spatial_features[3, :, :] = agent_obs_dict.get("target", np.zeros((h, w))).astype(np.float32)
    return torch.from_numpy(spatial_features).unsqueeze(0)


def get_non_spatial_features(current_global_pos, global_goal_pos):
    dy = global_goal_pos[0] - current_global_pos[0]
    dx = global_goal_pos[1] - current_global_pos[1]
    norm = np.sqrt(dx ** 2 + dy ** 2)
    norm_dy, norm_dx = (dy / norm, dx / norm) if norm > 0 else (0.0, 0.0)
    return torch.from_numpy(np.array([norm_dy, norm_dx], dtype=np.float32)).unsqueeze(0)


def global_to_local_coords(global_pos_target, agent_global_pos, obs_radius, window_size):
    gr_target, gc_target = global_pos_target
    gr_agent, gc_agent = agent_global_pos
    lr = gr_target - (gr_agent - obs_radius)
    lc = gc_target - (gc_agent - obs_radius)
    if 0 <= lr < window_size and 0 <= lc < window_size:
        return (int(lr), int(lc))
    return None


def path_coords_to_actions(path_coords: List[Tuple[int, int]], start_pos: Tuple[int, int]) -> List[int]:
    actions = []
    current_path = [start_pos] + path_coords
    for i in range(len(current_path) - 1):
        dr = current_path[i + 1][0] - current_path[i][0]
        dc = current_path[i + 1][1] - current_path[i][1]
        action = next(
            (act for act, (ddr, ddc) in ACTION_DELTAS.items() if ddr == dr and ddc == dc),
            ACTION_STAY,
        )
        actions.append(action)
    return actions if actions else [ACTION_STAY]


@numba.jit(nopython=True)
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def run_single_agent_astar(
    start_pos_global: Tuple[int, int],
    goal_pos_global: Tuple[int, int],
    grid_obstacle_map: np.ndarray,
    max_path_len: int,
) -> Optional[List[Tuple[int, int]]]:
    if (
        not (
            0 <= start_pos_global[0] < grid_obstacle_map.shape[0]
            and 0 <= start_pos_global[1] < grid_obstacle_map.shape[1]
        )
        or grid_obstacle_map[start_pos_global]
    ):
        return None
    open_set = [(heuristic(start_pos_global, goal_pos_global), 0, start_pos_global)]
    came_from = {}
    g_score = {start_pos_global: 0}
    h, w = grid_obstacle_map.shape
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal_pos_global:
            path = deque()
            while current in came_from:
                path.appendleft(current)
                current = came_from[current]
            return list(path)
        if g + 1 >= max_path_len:
            continue
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < h and 0 <= neighbor[1] < w and not grid_obstacle_map[neighbor[0], neighbor[1]]):
                continue
            new_g = g + 1
            if new_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = new_g
                heapq.heappush(open_set, (new_g + heuristic(neighbor, goal_pos_global), new_g, neighbor))
    return None


def _build_conflict_groups(
    initial_paths: Dict[int, List[Tuple[int, int]]],
    active_agents: List[int],
    max_timestep: int,
) -> List[Set[int]]:
    if not active_agents:
        return []

    paths_for_detection = {aid: p for aid, p in initial_paths.items() if aid in active_agents}
    conflicts = detect_all_conflicts_spacetime(paths_for_detection, max_timestep)

    G = nx.Graph()
    G.add_nodes_from(active_agents)
    for conflict in conflicts:
        if G.has_node(conflict.agent1_id) and G.has_node(conflict.agent2_id):
            G.add_edge(conflict.agent1_id, conflict.agent2_id)

    conflict_groups = [set(c) for c in nx.connected_components(G) if len(c) > 0]
    return conflict_groups


def _find_frontier_subgoal(
    persistent_map: np.ndarray,
    start_pos: Tuple[int, int],
    final_goal: Tuple[int, int],
    claimed_subgoals: Set[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    q = deque([start_pos])
    visited = {start_pos}
    frontier = []
    h, w = persistent_map.shape

    astar_grid_map = (persistent_map == OBSTACLE_CELL) | (persistent_map == UNKNOWN_CELL)
    if not (0 <= start_pos[0] < h and 0 <= start_pos[1] < w and not astar_grid_map[start_pos[0], start_pos[1]]):
        return None

    while q:
        r, c = q.popleft()
        is_frontier = False
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if persistent_map[nr, nc] == UNKNOWN_CELL:
                    is_frontier = True
                elif persistent_map[nr, nc] == FREE_CELL and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        if is_frontier:
            frontier.append((r, c))

    if not frontier:
        return None

    unclaimed_frontiers = [p for p in frontier if p not in claimed_subgoals]
    if not unclaimed_frontiers:
        logging.debug(f"  All discovered frontier points {frontier} have already been claimed.")
        return None

    best_frontier = min(unclaimed_frontiers, key=lambda p: heuristic(p, final_goal))
    return best_frontier


def _find_retreat_goal(
    agent_id: int,
    current_pos: Tuple[int, int],
    agent_pos_history: Deque[Tuple[int, int]],
    persistent_known_map: np.ndarray,
    all_agent_positions: List[Tuple[int, int]],
    claimed_subgoals: Set[Tuple[int, int]],
    max_search_dist: int = 10,
) -> Optional[Tuple[int, int]]:
    h, w = persistent_known_map.shape

    last_move_vec = (0, 0)
    if len(agent_pos_history) > 1:
        prev_pos = agent_pos_history[-2]
        last_move_vec = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])

    forward_dir = (last_move_vec[0], last_move_vec[1])
    backward_dir = (-last_move_vec[0], -last_move_vec[1])
    directions = [
        backward_dir,
        (forward_dir[1], forward_dir[0]),
        (-forward_dir[1], -forward_dir[0]),
        forward_dir,
    ]
    unique_dirs = []
    for d in directions:
        if d != (0, 0) and d not in unique_dirs:
            unique_dirs.append(d)
    for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if d not in unique_dirs:
            unique_dirs.append(d)

    q = deque([(current_pos, 0)])
    visited = {current_pos}
    other_agent_locs = set(p for i, p in enumerate(all_agent_positions) if i != agent_id)

    while q:
        pos, dist = q.popleft()

        if dist > 0:
            is_valid_retreat = (
                persistent_known_map[pos[0], pos[1]] == FREE_CELL
                and pos not in other_agent_locs
                and pos not in claimed_subgoals
            )
            if is_valid_retreat:
                logging.debug(f"  Deadlock intervention: Found retreat goal {pos} for agent {agent_id}.")
                return pos

        if dist >= max_search_dist:
            continue

        for dr, dc in unique_dirs:
            neighbor = (pos[0] + dr, pos[1] + dc)
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w and neighbor not in visited:
                visited.add(neighbor)
                if persistent_known_map[neighbor[0], neighbor[1]] != OBSTACLE_CELL:
                    q.append((neighbor, dist + 1))

    logging.warning(f"  Deadlock intervention: Could not find any retreat goal for agent {agent_id}.")
    return None


def solve_by_coordinated_retreat(
    group_list: List[int],
    group_agents_data: List[Dict],
    persistent_known_map: np.ndarray,
    cbs_map_for_solver: np.ndarray,
    agents_global_positions: List[Tuple[int, int]],
    agents_global_goals: List[Tuple[int, int]],
    agent_pos_history: Dict[int, Deque],
    max_plan_len: int,
    sub_cbs_max_iter: int,
    verbose: bool = False,
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    if len(group_list) < 2:
        return None

    agents_with_dist = []
    for agent_id in group_list:
        pos = tuple(agents_global_positions[agent_id])
        goal = tuple(agents_global_goals[agent_id])
        dist_to_goal = heuristic(pos, goal)
        agents_with_dist.append({"id": agent_id, "dist": dist_to_goal})

    agents_with_dist.sort(key=lambda x: x["dist"], reverse=True)

    num_yielders = max(1, len(group_list) // 2)
    yielder_ids = {d["id"] for d in agents_with_dist[:num_yielders]}
    mover_ids = {d["id"] for d in agents_with_dist[num_yielders:]}

    yielder_planning_map = cbs_map_for_solver.copy()
    for mover_id in mover_ids:
        mover_pos = tuple(agents_global_positions[mover_id])
        yielder_planning_map[mover_pos[0], mover_pos[1]] = 1

    yielder_paths: Dict[int, List[Tuple[int, int]]] = {}
    dynamic_constraints_for_movers: List[Constraint] = []
    claimed_retreat_goals = set()

    for agent_id in yielder_ids:
        current_pos = tuple(agents_global_positions[agent_id])
        retreat_goal = _find_retreat_goal(
            agent_id,
            current_pos,
            agent_pos_history[agent_id],
            persistent_known_map,
            agents_global_positions,
            claimed_retreat_goals,
        )
        if not retreat_goal:
            logging.warning(f"  Retreat failed: agent {agent_id} cannot find any retreat goal.")
            return None

        claimed_retreat_goals.add(retreat_goal)

        retreat_path_coords = run_single_agent_astar(
            current_pos, retreat_goal, yielder_planning_map, max_plan_len
        )

        if not retreat_path_coords:
            logging.warning(f"  Retreat failed: agent {agent_id} cannot reach {retreat_goal}.")
            return None

        full_path = [current_pos] + retreat_path_coords
        yielder_paths[agent_id] = full_path

        for t, pos in enumerate(full_path):
            if t > 0:
                dynamic_constraints_for_movers.append(Constraint(agent_id, pos, t))
                dynamic_constraints_for_movers.append(
                    Constraint(agent_id, pos, t, is_edge_constraint=True, prev_loc=full_path[t - 1])
                )

    movers_data = [d for d in group_agents_data if d["id"] in mover_ids]
    if not movers_data:
        return yielder_paths

    mover_paths = solve_local_cbs_robust(
        agents_data=movers_data,
        obstacles_local_map=cbs_map_for_solver,
        max_plan_len=max_plan_len,
        max_cbs_iterations=sub_cbs_max_iter,
        initial_constraints_list=dynamic_constraints_for_movers,
        agents_true_global_goals_abs={i: g for i, g in enumerate(agents_global_goals)},
        persistent_map_bundle={
            "persistent_known_map": persistent_known_map,
            "map_global_origin_r": 0,
            "map_global_origin_c": 0,
            "FREE_CELL": FREE_CELL,
        },
        verbose_cbs_solver=verbose,
    )

    if not mover_paths:
        return None

    final_solution = yielder_paths.copy()
    final_solution.update(mover_paths)
    return final_solution


def solve_by_forced_shuffle(
    group_list: List[int],
    cbs_map_for_solver: np.ndarray,
    agents_global_positions: List[Tuple[int, int]],
    verbose: bool = False,
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    solution_paths = {}
    shuffled_group = random.sample(group_list, len(group_list))

    temp_obstacle_map = cbs_map_for_solver.copy()
    unmoved_agents = set(group_list)

    for agent_id in shuffled_group:
        current_pos = tuple(agents_global_positions[agent_id])
        possible_moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(possible_moves)

        move_found = False
        for dr, dc in possible_moves:
            next_pos = (current_pos[0] + dr, current_pos[1] + dc)

            if (
                0 <= next_pos[0] < temp_obstacle_map.shape[0]
                and 0 <= next_pos[1] < temp_obstacle_map.shape[1]
                and temp_obstacle_map[next_pos[0], next_pos[1]] == 0
            ):
                solution_paths[agent_id] = [current_pos, next_pos]
                temp_obstacle_map[next_pos[0], next_pos[1]] = 1
                unmoved_agents.remove(agent_id)
                move_found = True
                break

        if not move_found:
            solution_paths[agent_id] = [current_pos, current_pos]

    for agent_id in unmoved_agents:
        pos = tuple(agents_global_positions[agent_id])
        solution_paths[agent_id] = [pos, pos]

    if verbose:
        logging.warning("  Applied forced shuffle fallback.")

    return solution_paths


def solve_by_local_search_shuffle(
    group_list: List[int],
    cbs_map_for_solver: np.ndarray,
    agents_global_positions: List[Tuple[int, int]],
    max_shuffle_steps: int = 20,
    verbose: bool = False,
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    num_agents_in_group = len(group_list)
    if num_agents_in_group == 0:
        return {}

    initial_positions = [tuple(agents_global_positions[aid]) for aid in group_list]
    target_positions = [initial_positions[(i + 1) % num_agents_in_group] for i in range(num_agents_in_group)]

    current_positions = list(initial_positions)
    paths = {aid: [pos] for aid, pos in zip(group_list, initial_positions)}

    map_h, map_w = cbs_map_for_solver.shape
    moves = np.array([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int32)

    for step in range(max_shuffle_steps):
        if all(cur == tgt for cur, tgt in zip(current_positions, target_positions)):
            logging.info(f"  Shuffle succeeded in {step} steps for group {group_list}")
            return paths

        next_positions_proposals = {}
        agent_indices_to_process = list(range(num_agents_in_group))
        random.shuffle(agent_indices_to_process)

        occupied_next_cells = set(current_positions)

        for agent_idx in agent_indices_to_process:
            if current_positions[agent_idx] == target_positions[agent_idx]:
                next_positions_proposals[agent_idx] = current_positions[agent_idx]
                continue

            best_move = current_positions[agent_idx]
            min_dist = heuristic(current_positions[agent_idx], target_positions[agent_idx])

            shuffled_move_indices = list(range(len(moves)))
            random.shuffle(shuffled_move_indices)

            for move_idx in shuffled_move_indices:
                dr, dc = moves[move_idx]
                next_pos = (current_positions[agent_idx][0] + dr, current_positions[agent_idx][1] + dc)

                if not (
                    0 <= next_pos[0] < map_h
                    and 0 <= next_pos[1] < map_w
                    and not cbs_map_for_solver[next_pos[0], next_pos[1]]
                ):
                    continue

                if next_pos in occupied_next_cells:
                    continue

                dist = heuristic(next_pos, target_positions[agent_idx])
                if dist < min_dist:
                    min_dist = dist
                    best_move = next_pos

            next_positions_proposals[agent_idx] = best_move
            if best_move != current_positions[agent_idx]:
                occupied_next_cells.remove(current_positions[agent_idx])
            occupied_next_cells.add(best_move)

        for i in range(num_agents_in_group):
            current_positions[i] = next_positions_proposals.get(i, current_positions[i])
            paths[group_list[i]].append(current_positions[i])

    return None


def _solve_single_group_with_defense_cascade(
    group_list: List[int],
    group_agents_data: List[Dict],
    initial_constraints: List[Constraint],
    cbs_map_for_solver: np.ndarray,
    agents_global_positions: List[Tuple[int, int]],
    agents_global_goals: List[Tuple[int, int]],
    agent_pos_history: Dict[int, Deque],
    consecutive_cbs_fails_count: Dict[frozenset, int],
    sim_params: Dict,
    cascade_stats: Dict[str, int],
) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    """
    Runs the hierarchical defense cascade for a single conflict group:
      Layer 1: Local CBS (L-CBS)
      Layer 2: Coordinated Retreat (Priority-Based Yielding)
      Layer 3: Local Search Shuffle (Rotational Shuffle)
      Layer 4: Forced Shuffle (Stochastic Perturbation)
    """
    group_frozenset = frozenset(group_list)
    verbose = sim_params.get("verbose", False)

    cascade_stats["total_groups_processed"] += 1
    dynamic_k = sim_params.get("dynamic_k", sim_params["local_plan_horizon"] - 1)

    # Layer 1: Standard Local CBS
    solution_paths = solve_local_cbs_robust(
        agents_data=group_agents_data,
        obstacles_local_map=cbs_map_for_solver,
        max_plan_len=dynamic_k,
        max_cbs_iterations=sim_params["cbs_max_iterations"],
        initial_constraints_list=initial_constraints,
        agents_true_global_goals_abs={i: g for i, g in enumerate(agents_global_goals)},
        persistent_map_bundle={
            "persistent_known_map": sim_params["persistent_known_map"],
            "map_global_origin_r": 0,
            "map_global_origin_c": 0,
            "FREE_CELL": sim_params["FREE_CELL"],
        },
        verbose_cbs_solver=verbose,
    )
    if solution_paths:
        cascade_stats["l1_cbs_success"] += 1
        consecutive_cbs_fails_count[group_frozenset] = 0
        return solution_paths

    consecutive_cbs_fails_count[group_frozenset] += 1
    if verbose:
        logging.debug(
            f"  Group {group_list} Layer 1 (CBS) failed. Consecutive fails: {consecutive_cbs_fails_count[group_frozenset]}"
        )
    if consecutive_cbs_fails_count[group_frozenset] < sim_params["max_consecutive_cbs_fails_for_intervention"]:
        return None

    # Layer 2: Coordinated Retreat
    sub_cbs_max_iter = int(sim_params["cbs_max_iterations"] * sim_params["sub_cbs_max_iterations_multiplier"])
    solution_paths = solve_by_coordinated_retreat(
        group_list,
        group_agents_data,
        sim_params["persistent_known_map"],
        cbs_map_for_solver,
        agents_global_positions,
        agents_global_goals,
        agent_pos_history,
        sim_params["local_plan_horizon"] - 1,
        sub_cbs_max_iter,
        verbose,
    )
    if solution_paths:
        cascade_stats["l2_retreat_success"] += 1
        return solution_paths

    # Layer 3: Local Search Shuffle
    solution_paths = solve_by_local_search_shuffle(
        group_list,
        cbs_map_for_solver,
        agents_global_positions,
        max_shuffle_steps=20,
        verbose=verbose,
    )
    if solution_paths:
        cascade_stats["l3_local_shuffle_success"] += 1
        return solution_paths

    # Layer 4: Forced Shuffle
    solution_paths = solve_by_forced_shuffle(
        group_list, cbs_map_for_solver, agents_global_positions, verbose
    )
    cascade_stats["l4_forced_shuffle_success"] += 1
    return solution_paths


def run_mapf_simulation(
    env,
    unet_model,
    device,
    max_episode_steps: int = 128,
    local_plan_horizon: int = 10,
    n_exec_steps: int = 3,
    cbs_max_iterations: int = 100,
    verbose: bool = False,
    visualization_output_dir: Optional[Path] = None,
    pattering_history_len: int = 10,
    pattering_unique_pos_threshold: int = 3,
    pattering_astar_bonus_horizon: int = 30,
    max_consecutive_cbs_fails_for_intervention: int = 1,
    sub_cbs_max_iterations_multiplier: float = 1.5,
    cbs_time_limit_s: float = 30.0,
    large_group_threshold: int = 12,
    preset: str = "paper",
):
    unet_model.eval()
    obs_list, _ = env.reset()
    num_agents = env.grid_config.num_agents
    agents_global_positions = list(env.get_agents_xy())
    agents_global_goals = list(env.get_targets_xy())

    # Setting B: static obstacle map is initially unknown and discovered online
    global_static_obstacle_map_ground_truth = env.unwrapped.grid.get_obstacles().astype(np.bool_)
    global_map_height, global_map_width = global_static_obstacle_map_ground_truth.shape
    persistent_known_map = np.full((global_map_height, global_map_width), UNKNOWN_CELL, dtype=np.int8)

    sim_obs_radius = env.grid_config.obs_radius
    sim_window_size = sim_obs_radius * 2 + 1

    agent_pos_history: Dict[int, Deque] = {
        i: deque(maxlen=pattering_history_len) for i in range(num_agents)
    }
    consecutive_cbs_fails_count: Dict[frozenset, int] = defaultdict(int)

    def update_persistent_map(current_obs_list, current_agent_positions, p_map, obs_rad, map_h, map_w):
        for agent_idx_updater in range(len(current_agent_positions)):
            if agent_idx_updater < len(current_obs_list) and current_obs_list[agent_idx_updater] is not None:
                agent_r, agent_c = current_agent_positions[agent_idx_updater]
                fov_obstacles = current_obs_list[agent_idx_updater].get("obstacles")
                if fov_obstacles is not None:
                    fov_h_actual, fov_w_actual = fov_obstacles.shape
                    fov_global_tl_r, fov_global_tl_c = agent_r - obs_rad, agent_c - obs_rad
                    for r_fov in range(fov_h_actual):
                        for c_fov in range(fov_w_actual):
                            glob_r, glob_c = fov_global_tl_r + r_fov, fov_global_tl_c + c_fov
                            if 0 <= glob_r < map_h and 0 <= glob_c < map_w:
                                if fov_obstacles[r_fov, c_fov]:
                                    p_map[glob_r, glob_c] = OBSTACLE_CELL
                                else:
                                    p_map[glob_r, glob_c] = FREE_CELL

    update_persistent_map(
        obs_list,
        agents_global_positions,
        persistent_known_map,
        sim_obs_radius,
        global_map_height,
        global_map_width,
    )
    agents_active = [True] * num_agents
    executed_paths_global = {i: [tuple(agents_global_positions[i])] for i in range(num_agents)}
    sim_center_pos_in_fov = (sim_obs_radius, sim_obs_radius)
    total_env_steps_taken = 0
    simulation_successful = True
    error_messages = []
    for i in range(num_agents):
        agent_pos_history[i].append(tuple(agents_global_positions[i]))
    start_time = time.time()

    cascade_stats = defaultdict(int)
    truncated = [False] * num_agents

    while any(agents_active) and total_env_steps_taken < max_episode_steps:
        if time.time() - start_time > cbs_time_limit_s:
            simulation_successful = False
            error_messages.append(f"TIMEOUT@{total_env_steps_taken}")
            break

        active_agent_ids_for_planning = [i for i, active in enumerate(agents_active) if active]
        if not active_agent_ids_for_planning:
            break

        claimed_subgoals_this_cycle = set()
        proposed_unet_actions_fov: Dict[int, List[int]] = {}

        for agent_id in active_agent_ids_for_planning:
            is_pattering_combined = (
                len(set(agent_pos_history[agent_id])) <= pattering_unique_pos_threshold
                if len(agent_pos_history[agent_id]) >= pattering_history_len
                else False
            )

            proposal_generated = False
            if is_pattering_combined:
                is_goal_area_known = persistent_known_map[agents_global_goals[agent_id]] != UNKNOWN_CELL
                if not is_goal_area_known:
                    frontier_subgoal = _find_frontier_subgoal(
                        persistent_known_map,
                        tuple(agents_global_positions[agent_id]),
                        agents_global_goals[agent_id],
                        claimed_subgoals_this_cycle,
                    )
                    if frontier_subgoal:
                        claimed_subgoals_this_cycle.add(frontier_subgoal)
                        path_to_frontier = run_single_agent_astar(
                            tuple(agents_global_positions[agent_id]),
                            frontier_subgoal,
                            (persistent_known_map != FREE_CELL),
                            local_plan_horizon,
                        )
                        if path_to_frontier:
                            proposed_unet_actions_fov[agent_id] = path_coords_to_actions(
                                path_to_frontier, tuple(agents_global_positions[agent_id])
                            )
                            proposal_generated = True

                if not proposal_generated:
                    astar_path = run_single_agent_astar(
                        tuple(agents_global_positions[agent_id]),
                        agents_global_goals[agent_id],
                        (persistent_known_map != FREE_CELL),
                        local_plan_horizon + pattering_astar_bonus_horizon,
                    )
                    if astar_path:
                        proposed_unet_actions_fov[agent_id] = path_coords_to_actions(
                            astar_path, tuple(agents_global_positions[agent_id])
                        )
                        proposal_generated = True

            if not proposal_generated:
                agent_obs_dict = obs_list[agent_id]
                spatial_features = get_spatial_features_from_obs(agent_obs_dict, sim_window_size).to(device)
                non_spatial_features = get_non_spatial_features(
                    tuple(agents_global_positions[agent_id]), tuple(agents_global_goals[agent_id])
                ).to(device)
                with torch.no_grad():
                    predicted_potential_field_fov = (
                        unet_model(spatial_features, non_spatial_features).squeeze().cpu().numpy()
                    )
                local_target = global_to_local_coords(
                    tuple(agents_global_goals[agent_id]),
                    tuple(agents_global_positions[agent_id]),
                    sim_obs_radius,
                    sim_window_size,
                )
                local_obstacles = agent_obs_dict.get(
                    "obstacles", np.ones((sim_window_size, sim_window_size))
                ).astype(np.bool_)
                actions, _ = decode_action_sequence_refined(
                    predicted_potential_field_fov,
                    local_obstacles,
                    sim_center_pos_in_fov,
                    local_plan_horizon,
                    local_target,
                )
                proposed_unet_actions_fov[agent_id] = actions

        cbs_map_for_solver = (persistent_known_map == OBSTACLE_CELL) | (persistent_known_map == UNKNOWN_CELL)
        initial_paths_in_cbs_map_coords: Dict[int, List[Tuple[int, int]]] = {}
        cbs_agents_data_transformed = []
        map_h_cbs, map_w_cbs = cbs_map_for_solver.shape
        for agent_id in active_agent_ids_for_planning:
            start_cbs = tuple(agents_global_positions[agent_id])
            path_cbs = [start_cbs]
            r, c = start_cbs
            for act in proposed_unet_actions_fov.get(agent_id, []):
                dr, dc = ACTION_DELTAS.get(act, (0, 0))
                nr, nc = r + dr, c + dc
                if not (0 <= nr < map_h_cbs and 0 <= nc < map_w_cbs and not cbs_map_for_solver[nr, nc]):
                    break
                path_cbs.append((nr, nc))
                r, c = nr, nc
            initial_paths_in_cbs_map_coords[agent_id] = path_cbs
            cbs_agents_data_transformed.append(
                {"id": agent_id, "start_local": start_cbs, "goal_local": path_cbs[-1]}
            )

        initial_conflicts = detect_all_conflicts_spacetime(
            initial_paths_in_cbs_map_coords, local_plan_horizon
        )
        dynamic_n_exec_steps = n_exec_steps
        if initial_conflicts and preset == "original":
            dynamic_n_exec_steps = 6

        final_paths_coords: Dict[int, List[Tuple[int, int]]] = {}
        final_k_step_action_sequences: Dict[int, List[int]] = {}

        sim_params = {
            "local_plan_horizon": local_plan_horizon,
            "cbs_max_iterations": cbs_max_iterations,
            "persistent_known_map": persistent_known_map,
            "FREE_CELL": FREE_CELL,
            "verbose": verbose,
            "max_consecutive_cbs_fails_for_intervention": max_consecutive_cbs_fails_for_intervention,
            "sub_cbs_max_iterations_multiplier": sub_cbs_max_iterations_multiplier,
        }

        all_cbs_agents_data_dict = {d["id"]: d for d in cbs_agents_data_transformed}
        all_components = _build_conflict_groups(
            initial_paths_in_cbs_map_coords, active_agent_ids_for_planning, local_plan_horizon
        )

        free_agent_ids = []
        conflict_groups = []
        for component in all_components:
            if len(component) > 1:
                conflict_groups.append(component)
            else:
                free_agent_ids.extend(list(component))

        if verbose:
            logging.info(
                f"Identified {len(free_agent_ids)} free agents and {len(conflict_groups)} conflict groups."
            )
        for aid in free_agent_ids:
            final_paths_coords[aid] = initial_paths_in_cbs_map_coords.get(
                aid, [tuple(agents_global_positions[aid])]
            )

        for group in conflict_groups:
            group_list = sorted(list(group))
            group_agents_data = [all_cbs_agents_data_dict[aid] for aid in group_list]

            group_d_subgoal = max([heuristic(d["start_local"], d["goal_local"]) for d in group_agents_data])
            dynamic_k = max(min(group_d_subgoal, local_plan_horizon - 1), dynamic_n_exec_steps)

            sim_params["dynamic_k"] = dynamic_k
            solution_paths = None
            if len(group_list) > large_group_threshold:
                solution_paths = solve_by_coordinated_retreat(
                    group_list,
                    group_agents_data,
                    persistent_known_map,
                    cbs_map_for_solver,
                    agents_global_positions,
                    agents_global_goals,
                    agent_pos_history,
                    local_plan_horizon - 1,
                    int(cbs_max_iterations * sub_cbs_max_iterations_multiplier),
                    verbose,
                )
                if not solution_paths:
                    solution_paths = solve_by_forced_shuffle(
                        group_list, cbs_map_for_solver, agents_global_positions, verbose
                    )
            else:
                if verbose:
                    logging.info(f"  Solving group {group_list} with defense cascade.")
                solution_paths = _solve_single_group_with_defense_cascade(
                    group_list,
                    group_agents_data,
                    [],
                    cbs_map_for_solver,
                    agents_global_positions,
                    agents_global_goals,
                    agent_pos_history,
                    consecutive_cbs_fails_count,
                    sim_params,
                    cascade_stats,
                )

            if solution_paths:
                final_paths_coords.update(solution_paths)
            else:
                cascade_stats["cascade_pre_intervention_fail"] += 1
                final_paths_coords.update(
                    {aid: [tuple(agents_global_positions[aid])] * 2 for aid in group_list}
                )

        for agent_id in active_agent_ids_for_planning:
            path_coords = final_paths_coords.get(agent_id, [tuple(agents_global_positions[agent_id])])
            actions = path_coords_to_actions(path_coords[1:], path_coords[0])
            plan = actions if actions else [ACTION_STAY]
            while len(plan) < local_plan_horizon:
                plan.append(ACTION_STAY)
            final_k_step_action_sequences[agent_id] = plan

        plan_is_valid = {i: True for i in range(num_agents)}
        for k in range(dynamic_n_exec_steps):
            if not any(agents_active):
                break

            actions_this_step = []
            expected_next_positions = {}
            for i in range(num_agents):
                if not agents_active[i] or not plan_is_valid[i]:
                    actions_this_step.append(ACTION_STAY)
                    expected_next_positions[i] = tuple(agents_global_positions[i])
                else:
                    plan = final_k_step_action_sequences.get(i)
                    action_to_take = plan[k] if plan and k < len(plan) else ACTION_STAY
                    actions_this_step.append(action_to_take)
                    dr, dc = ACTION_DELTAS.get(action_to_take, (0, 0))
                    expected_next_positions[i] = (
                        agents_global_positions[i][0] + dr,
                        agents_global_positions[i][1] + dc,
                    )

            obs_list, _, terminated, truncated, _ = env.step(actions_this_step)
            total_env_steps_taken += 1
            new_global_positions = list(env.get_agents_xy())

            for i in range(num_agents):
                if (
                    agents_active[i]
                    and plan_is_valid[i]
                    and tuple(new_global_positions[i]) != expected_next_positions[i]
                ):
                    plan_is_valid[i] = False

            agents_global_positions = new_global_positions

            step_had_truncation = False
            for i in range(num_agents):
                agent_pos_history[i].append(tuple(agents_global_positions[i]))
                if agents_active[i]:
                    executed_paths_global[i].append(tuple(agents_global_positions[i]))
                    if terminated[i]:
                        agents_active[i] = False
                    if truncated[i]:
                        agents_active[i] = False
                        simulation_successful = False
                        step_had_truncation = True
                        error_messages.append(f"A{i}_TRUNC@S{total_env_steps_taken}")

            if not any(agents_active) or total_env_steps_taken >= max_episode_steps or step_had_truncation:
                break

            update_persistent_map(
                obs_list,
                agents_global_positions,
                persistent_known_map,
                sim_obs_radius,
                global_map_height,
                global_map_width,
            )

        if not simulation_successful:
            break

    total_processed = cascade_stats.get("total_groups_processed", 0)
    if total_processed > 0 and verbose:
        logging.info("\n--- Defense Cascade Statistics ---")
        l1_s = cascade_stats.get("l1_cbs_success", 0)
        l2_s = cascade_stats.get("l2_retreat_success", 0)
        l3_s = cascade_stats.get("l3_local_shuffle_success", 0)
        l4_s = cascade_stats.get("l4_forced_shuffle_success", 0)
        pre_fail = cascade_stats.get("cascade_pre_intervention_fail", 0)

        logging.info(f"Total conflict groups processed: {total_processed}")
        logging.info(f"  L1: Standard CBS Success    : {l1_s:6d} ({l1_s / total_processed:8.2%})")
        logging.info(f"  L2: Coordinated Retreat     : {l2_s:6d} ({l2_s / total_processed:8.2%})")
        logging.info(f"  L3: Local Search Shuffle    : {l3_s:6d} ({l3_s / total_processed:8.2%})")
        logging.info(f"  L4: Forced Shuffle          : {l4_s:6d} ({l4_s / total_processed:8.2%})")
        logging.info(f"  Pre-Intervention Failures   : {pre_fail:6d} ({pre_fail / total_processed:8.2%})")

    num_agents_finished = sum(
        1
        for i in range(num_agents)
        if not agents_active[i] and not truncated[i] and agents_global_positions[i] == agents_global_goals[i]
    )
    final_result = {
        "success": simulation_successful and not any(agents_active),
        "makespan": total_env_steps_taken,
        "sum_of_costs": sum(len(p) - 1 for p in executed_paths_global.values() if p),
        "individual_costs": {i: len(p) - 1 for i, p in executed_paths_global.items() if p},
        "executed_paths_global": executed_paths_global,
        "error_summary": "; ".join(error_messages) if error_messages else "No errors.",
        "num_agents_at_start": num_agents,
        "num_agents_reached_target": num_agents_finished,
        "defense_cascade_stats": dict(cascade_stats),
    }
    return final_result
