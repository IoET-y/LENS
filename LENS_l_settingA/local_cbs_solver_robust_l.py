# local_cbs_solver_robust_l.py
from collections import defaultdict
import heapq
import itertools
import logging
import random
from typing import Dict, List, Optional, Tuple

import numba
from numba.core import types
from numba.typed import Dict as NumbaDict
import numpy as np

ACTION_DELTAS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)
POS_KEY_TYPE = types.UniTuple(types.int64, 3)


class Constraint:
    def __init__(self, agent_id, loc, timestep, is_edge_constraint=False, prev_loc=None):
        self.agent_id = agent_id
        self.location = loc
        self.timestep = timestep
        self.is_edge_constraint = is_edge_constraint
        self.prev_location = prev_loc

    def __eq__(self, other):
        if not isinstance(other, Constraint):
            return False
        return (
            self.agent_id == other.agent_id
            and self.location == other.location
            and self.timestep == other.timestep
            and self.is_edge_constraint == other.is_edge_constraint
            and self.prev_location == other.prev_location
        )

    def __hash__(self):
        return hash((self.agent_id, self.location, self.timestep, self.is_edge_constraint, self.prev_location))

    def __repr__(self):
        if self.is_edge_constraint:
            return f"EdgeConstraint(A{self.agent_id}: {self.prev_location}->{self.location} @t={self.timestep})"
        return f"VertexConstraint(A{self.agent_id}: {self.location} @t={self.timestep})"


class Conflict:
    VERTEX = 1
    EDGE = 2

    def __init__(self, conflict_type, agent1_id, agent2_id, loc1, timestep, loc2=None):
        self.type = conflict_type
        self.agent1_id = agent1_id
        self.agent2_id = agent2_id
        self.location1 = loc1
        self.timestep = timestep
        self.location2 = loc2

    def __repr__(self):
        if self.type == Conflict.VERTEX:
            return f"VertexConflict(A{self.agent1_id}, A{self.agent2_id} @ {self.location1}, t={self.timestep})"
        elif self.type == Conflict.EDGE:
            return (
                f"EdgeConflict(A{self.agent1_id} {self.location1[0]}->{self.location1[1]} "
                f"vs A{self.agent2_id} {self.location2[0]}->{self.location2[1]} @t={self.timestep})"
            )
        return "UnknownConflict"


class CBSHighLevelNode:
    _ids = itertools.count(0)

    def __init__(self, constraints: Optional[List[Constraint]] = None, paths=None, sum_of_costs=0):
        self.id = next(self._ids)
        self.constraints = constraints if constraints is not None else []
        self.paths = paths if paths is not None else {}
        self.sum_of_costs = sum_of_costs
        self.conflicts = []

    def __lt__(self, other):
        if len(self.conflicts) != len(other.conflicts):
            return len(self.conflicts) < len(other.conflicts)
        if self.sum_of_costs != other.sum_of_costs:
            return self.sum_of_costs < other.sum_of_costs
        return self.id < other.id


@numba.jit(nopython=True)
def _numba_low_level_astar(
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    obstacles_local: np.ndarray,
    constraints_array: np.ndarray,
    cost_map: np.ndarray,
    max_path_len: int,
):
    map_height, map_width = obstacles_local.shape

    def h_func(p1_r, p1_c, p2_r, p2_c):
        return abs(p1_r - p2_r) + abs(p1_c - p2_c)

    open_set = [(h_func(start_pos[0], start_pos[1], goal_pos[0], goal_pos[1]), 0, start_pos[0], start_pos[1], 0)]
    came_from = NumbaDict.empty(key_type=POS_KEY_TYPE, value_type=POS_KEY_TYPE)
    g_scores = NumbaDict.empty(key_type=POS_KEY_TYPE, value_type=types.int64)
    start_key = (start_pos[0], start_pos[1], 0)
    g_scores[start_key] = 0

    while len(open_set) > 0:
        f, g, r, c, t = heapq.heappop(open_set)
        current_key = (r, c, t)

        if g > g_scores.get(current_key, np.iinfo(np.int64).max):
            continue

        if (r, c) == goal_pos:
            path = []
            curr = current_key
            while curr in came_from:
                path.append((curr[0], curr[1]))
                curr = came_from[curr]
            path.append((start_pos[0], start_pos[1]))
            path.reverse()
            while len(path) < max_path_len + 1:
                path.append(goal_pos)
            return path

        if t >= max_path_len:
            continue

        for dr_dc_idx in [1, 2, 3, 4, 0]:
            dr, dc = ACTION_DELTAS[dr_dc_idx]
            nr, nc = r + dr, c + dc
            nt = t + 1

            if not (0 <= nr < map_height and 0 <= nc < map_width and not obstacles_local[nr, nc]):
                continue

            is_move_valid = True
            for i in range(constraints_array.shape[0]):
                constr = constraints_array[i]
                if constr[3] == 0:
                    if constr[2] == nt and constr[0] == nr and constr[1] == nc:
                        is_move_valid = False
                        break
                else:
                    if constr[2] == nt and constr[4] == r and constr[5] == c and constr[0] == nr and constr[1] == nc:
                        is_move_valid = False
                        break
            if not is_move_valid:
                continue

            move_cost = 1 + int(cost_map[nr, nc])
            new_g = g + move_cost

            neighbor_key = (nr, nc, nt)
            if new_g < g_scores.get(neighbor_key, np.iinfo(np.int64).max):
                g_scores[neighbor_key] = new_g
                new_f = new_g + h_func(nr, nc, goal_pos[0], goal_pos[1])
                heapq.heappush(open_set, (new_f, new_g, nr, nc, nt))
                came_from[neighbor_key] = current_key

    return None


def low_level_astar_for_cbs(
    agent_id,
    start_pos,
    original_goal_pos,
    obstacles_local,
    constraints,
    max_plan_len: int,
    dynamic_cost_map: Optional[np.ndarray] = None,
    true_agent_global_goal_abs: Optional[Tuple[int, int]] = None,
    persistent_map_info: Optional[Dict] = None,
    verbose_cbs: bool = False,
):
    final_target_goal_for_astar = original_goal_pos

    agent_constraints = []
    for c in constraints:
        if c.agent_id == agent_id:
            if c.is_edge_constraint:
                agent_constraints.append(
                    [c.location[0], c.location[1], c.timestep, 1, c.prev_location[0], c.prev_location[1]]
                )
            else:
                agent_constraints.append([c.location[0], c.location[1], c.timestep, 0, -1, -1])

    if not agent_constraints:
        constraints_array = np.empty((0, 6), dtype=np.int32)
    else:
        constraints_array = np.array(agent_constraints, dtype=np.int32)

    if dynamic_cost_map is None:
        dynamic_cost_map = np.zeros_like(obstacles_local, dtype=np.int32)

    path_tuples = _numba_low_level_astar(
        start_pos,
        final_target_goal_for_astar,
        obstacles_local,
        constraints_array,
        dynamic_cost_map,
        max_plan_len,
    )
    if path_tuples is not None:
        return [tuple(p) for p in path_tuples]
    return None


def detect_all_conflicts_spacetime(paths_dict: Dict[int, List[Tuple[int, int]]], max_timestep: int):
    all_conflicts = []
    arrival_times = {}
    for aid, path in paths_dict.items():
        if path is None:
            return []
        goal_pos = path[-1]
        try:
            arrival_time = path.index(goal_pos)
        except ValueError:
            arrival_time = len(path)
        arrival_times[aid] = arrival_time

    vertex_reservation = defaultdict(list)
    edge_reservation = defaultdict(list)
    for agent_id, path in paths_dict.items():
        agent_arrival_time = arrival_times[agent_id]
        for t, pos in enumerate(path):
            if t > agent_arrival_time:
                break
            vertex_reservation[(t, pos)].append(agent_id)
        for t in range(len(path) - 1):
            if t + 1 > agent_arrival_time:
                break
            pos1, pos2 = path[t], path[t + 1]
            if pos1 != pos2:
                edge_reservation[(t + 1, pos1, pos2)].append(agent_id)

    for (timestep, pos), occupants in vertex_reservation.items():
        if len(occupants) > 1:
            for agent1_id, agent2_id in itertools.combinations(occupants, 2):
                all_conflicts.append(Conflict(Conflict.VERTEX, agent1_id, agent2_id, pos, timestep))

    for (timestep, pos1, pos2), occupants1 in edge_reservation.items():
        if (timestep, pos2, pos1) in edge_reservation:
            occupants2 = edge_reservation[(timestep, pos2, pos1)]
            for agent1_id in occupants1:
                for agent2_id in occupants2:
                    if agent1_id < agent2_id:
                        all_conflicts.append(
                            Conflict(Conflict.EDGE, agent1_id, agent2_id, (pos1, pos2), timestep, (pos2, pos1))
                        )
    return all_conflicts


def solve_local_cbs_robust(
    agents_data: List[Dict],
    obstacles_local_map: np.ndarray,
    max_plan_len: int,
    max_cbs_iterations: int = 100,
    agent_memories: Optional[Dict[int, "AgentMemory"]] = None,
    agents_true_global_goals_abs: Optional[Dict[int, Tuple[int, int]]] = None,
    persistent_map_bundle: Optional[Dict] = None,
    verbose_cbs_solver: bool = False,
    initial_constraints_list: Optional[List[Constraint]] = None,
):
    def is_cardinal(conflict: Conflict, agent_id: int, current_node: CBSHighLevelNode) -> bool:
        original_path = current_node.paths.get(agent_id)
        if not original_path:
            return False
        original_goal = original_path[-1]
        try:
            original_cost = original_path.index(original_goal)
        except ValueError:
            original_cost = len(original_path) - 1

        if conflict.type == Conflict.VERTEX:
            new_constr = Constraint(agent_id, conflict.location1, conflict.timestep)
        elif conflict.type == Conflict.EDGE:
            edge = conflict.location1 if agent_id == conflict.agent1_id else conflict.location2
            new_constr = Constraint(agent_id, edge[1], conflict.timestep, True, edge[0])
        else:
            return False

        temp_constraints = current_node.constraints + [new_constr]
        agent_info = next((a for a in agents_data if a["id"] == agent_id), None)
        if not agent_info:
            return False

        dynamic_cost_map_agent = (
            agent_memories[agent_id].dynamic_cost_map
            if agent_memories and agent_id in agent_memories
            else None
        )
        new_path = low_level_astar_for_cbs(
            agent_id,
            agent_info["start_local"],
            agent_info["goal_local"],
            obstacles_local_map,
            temp_constraints,
            max_plan_len,
            dynamic_cost_map=dynamic_cost_map_agent,
            true_agent_global_goal_abs=agents_true_global_goals_abs.get(agent_id)
            if agents_true_global_goals_abs
            else None,
            persistent_map_info=persistent_map_bundle,
            verbose_cbs=False,
        )
        if new_path is None:
            return True
        new_goal = new_path[-1]
        try:
            new_cost = new_path.index(new_goal)
        except ValueError:
            new_cost = len(new_path) - 1
        return new_cost > original_cost

    open_list = []
    iteration_count = 0
    root_node = CBSHighLevelNode(constraints=initial_constraints_list if initial_constraints_list is not None else [])

    for agent_info in agents_data:
        agent_id = agent_info["id"]
        dynamic_cost_map_agent = (
            agent_memories[agent_id].dynamic_cost_map
            if agent_memories and agent_id in agent_memories
            else None
        )

        path_coords = low_level_astar_for_cbs(
            agent_id,
            agent_info["start_local"],
            agent_info["goal_local"],
            obstacles_local_map,
            root_node.constraints,
            max_plan_len,
            dynamic_cost_map=dynamic_cost_map_agent,
            true_agent_global_goal_abs=agents_true_global_goals_abs.get(agent_id)
            if agents_true_global_goals_abs
            else None,
            persistent_map_info=persistent_map_bundle,
            verbose_cbs=verbose_cbs_solver,
        )
        if path_coords is None:
            if verbose_cbs_solver:
                logging.warning(f"  [CBS RootFail] Agt {agent_id} initial path fail.")
            return None
        root_node.paths[agent_id] = path_coords
        goal = path_coords[-1]
        try:
            cost = path_coords.index(goal)
        except ValueError:
            cost = len(path_coords) - 1
        root_node.sum_of_costs += cost

    root_node.conflicts = detect_all_conflicts_spacetime(root_node.paths, max_plan_len)
    heapq.heappush(open_list, root_node)

    while open_list and iteration_count < max_cbs_iterations:
        current_node = heapq.heappop(open_list)
        iteration_count += 1

        if not current_node.conflicts:
            if verbose_cbs_solver:
                logging.info(f"[CBS] Solution found! Iterations={iteration_count}")
            return current_node.paths

        classified_conflicts = []
        for c in current_node.conflicts:
            is_cardinal_for_a1 = is_cardinal(c, c.agent1_id, current_node)
            is_cardinal_for_a2 = is_cardinal(c, c.agent2_id, current_node)
            priority = 0 if (is_cardinal_for_a1 or is_cardinal_for_a2) else 1
            classified_conflicts.append((priority, c))
        classified_conflicts.sort(key=lambda x: (x[0], x[1].timestep, x[1].type, x[1].agent1_id, x[1].agent2_id))
        conflict = classified_conflicts[0][1]
        agents_to_constrain = [conflict.agent1_id, conflict.agent2_id]
        if classified_conflicts[0][0] == 1:
            random.shuffle(agents_to_constrain)

        for agent_to_constrain in agents_to_constrain:
            new_constraints_for_child = list(current_node.constraints)
            if conflict.type == Conflict.VERTEX:
                new_constr = Constraint(agent_to_constrain, conflict.location1, conflict.timestep)
            elif conflict.type == Conflict.EDGE:
                edge = conflict.location1 if agent_to_constrain == conflict.agent1_id else conflict.location2
                new_constr = Constraint(agent_to_constrain, edge[1], conflict.timestep, True, edge[0])
            else:
                continue
            if new_constr in new_constraints_for_child:
                continue
            new_constraints_for_child.append(new_constr)
            child_node = CBSHighLevelNode(constraints=new_constraints_for_child, paths={}, sum_of_costs=0)
            path_valid_for_child = True
            temp_child_paths = {}
            constrained_agent_info = next((a for a in agents_data if a["id"] == agent_to_constrain), None)

            if not constrained_agent_info:
                path_valid_for_child = False
            else:
                dynamic_cost_map_constrained = (
                    agent_memories[agent_to_constrain].dynamic_cost_map
                    if agent_memories and agent_to_constrain in agent_memories
                    else None
                )
                new_path_constrained_agent = low_level_astar_for_cbs(
                    agent_to_constrain,
                    constrained_agent_info["start_local"],
                    constrained_agent_info["goal_local"],
                    obstacles_local_map,
                    child_node.constraints,
                    max_plan_len,
                    dynamic_cost_map=dynamic_cost_map_constrained,
                    true_agent_global_goal_abs=agents_true_global_goals_abs.get(agent_to_constrain)
                    if agents_true_global_goals_abs
                    else None,
                    persistent_map_info=persistent_map_bundle,
                    verbose_cbs=verbose_cbs_solver,
                )
                if new_path_constrained_agent is None:
                    path_valid_for_child = False
                else:
                    temp_child_paths[agent_to_constrain] = new_path_constrained_agent
                    for other_agent_info in agents_data:
                        other_id = other_agent_info["id"]
                        if other_id in current_node.paths:
                            path_to_copy = temp_child_paths.get(other_id, current_node.paths[other_id])
                            temp_child_paths[other_id] = path_to_copy
                            goal = path_to_copy[-1]
                            try:
                                cost = path_to_copy.index(goal)
                            except ValueError:
                                cost = len(path_to_copy) - 1
                            child_node.sum_of_costs += cost
                        else:
                            path_valid_for_child = False
                            break
            if path_valid_for_child:
                child_node.paths = temp_child_paths
                child_node.conflicts = detect_all_conflicts_spacetime(child_node.paths, max_plan_len)
                heapq.heappush(open_list, child_node)

    if verbose_cbs_solver:
        logging.warning(f"[CBS] Failed. Iterations: {iteration_count}, OpenListEmpty: {not open_list}")
    return None
