# action_sequence_decoder_s.py
from typing import List, Optional, Tuple
import numpy as np

ACTION_DELTAS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
ACTION_NAMES = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
ACTION_STAY = 0
CENTER_POS = (5, 5)
WINDOW_SIZE = 11

# Paper Eq. (1) action extraction coefficients:
# \lambda_visit = 0.5, \lambda_stay = 0.6, \lambda_mom = 0.2
VISITED_CELL_PENALTY = 0.5
MOMENTUM_BONUS = 0.2
STAY_PENALTY_WHEN_NOT_AT_TARGET = 0.6


def get_best_move_with_anti_pattering(
    current_pos_local: Tuple[int, int],
    potential_field_local: np.ndarray,
    obstacles_local: np.ndarray,
    is_at_local_target: bool,
    path_coords_in_current_decode: List[Tuple[int, int]],
    last_action_in_current_decode: Optional[int],
) -> Optional[Tuple[int, Tuple[int, int]]]:
    r, c = current_pos_local
    move_options = []

    visited_set_for_penalty = (
        set(path_coords_in_current_decode[:-1])
        if path_coords_in_current_decode and len(path_coords_in_current_decode) > 1
        else set()
    )

    for action_idx, (dr, dc) in ACTION_DELTAS.items():
        nr, nc = r + dr, c + dc

        if not (0 <= nr < WINDOW_SIZE and 0 <= nc < WINDOW_SIZE and not obstacles_local[nr, nc]):
            continue

        base_potential = potential_field_local[nr, nc]
        adjusted_potential = base_potential

        if action_idx == ACTION_STAY and not is_at_local_target:
            adjusted_potential += STAY_PENALTY_WHEN_NOT_AT_TARGET

        if (nr, nc) in visited_set_for_penalty:
            adjusted_potential += VISITED_CELL_PENALTY

        if (
            last_action_in_current_decode is not None
            and action_idx == last_action_in_current_decode
            and action_idx != ACTION_STAY
        ):
            adjusted_potential -= MOMENTUM_BONUS

        move_options.append((adjusted_potential, action_idx, (nr, nc)))

    if not move_options:
        return None

    move_options.sort(key=lambda x: x[0])

    _, best_action, next_pos = move_options[0]
    return best_action, next_pos


def decode_action_sequence_refined(
    initial_potential_field: np.ndarray,
    initial_obstacles_local: np.ndarray,
    start_pos_local: Tuple[int, int] = CENTER_POS,
    max_seq_len: int = 10,
    local_target_pos: Optional[Tuple[int, int]] = None,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    if not (initial_potential_field.shape == initial_obstacles_local.shape):
        raise ValueError("Potential field and obstacle maps must have the same shape")

    action_sequence: List[int] = []
    current_pos: Tuple[int, int] = start_pos_local
    path_taken_coords: List[Tuple[int, int]] = [current_pos]
    last_decoded_action: Optional[int] = None

    for _ in range(max_seq_len):
        is_at_target = local_target_pos is not None and current_pos == local_target_pos

        if is_at_target:
            best_action, next_pos = ACTION_STAY, current_pos
        else:
            move_result = get_best_move_with_anti_pattering(
                current_pos,
                initial_potential_field,
                initial_obstacles_local,
                is_at_target,
                path_taken_coords,
                last_decoded_action,
            )

            if not move_result:
                break

            best_action, next_pos = move_result

        action_sequence.append(best_action)
        current_pos = next_pos
        path_taken_coords.append(current_pos)
        last_decoded_action = best_action

    return action_sequence, path_taken_coords
