# region_graph.py
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

FREE_CELL = 0
OBSTACLE_CELL = 1
UNKNOWN_CELL = 2


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build_adaptive_region_graph(
    persistent_known_map: np.ndarray,
    min_region_size: int = 100,
    max_region_size: int = 500,
) -> Tuple[nx.Graph, np.ndarray, Dict[int, Tuple[int, int]]]:
    """
    Builds an Adaptive Region Graph (ARG) from the known map.

    Returns:
        - nx.Graph: High-level graph where nodes are region IDs.
        - np.ndarray: 2D array mapping each cell to its region ID.
        - Dict[int, Tuple[int, int]]: Map from region_id to its centroid coordinate.
    """
    h, w = persistent_known_map.shape
    region_map = np.zeros_like(persistent_known_map, dtype=np.int32)
    visited = (persistent_known_map != FREE_CELL)
    region_id_counter = 1

    # 1. Flood-fill to find connected free-space clusters
    for r in range(h):
        for c in range(w):
            if not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                current_cluster_pixels = []
                while q:
                    curr_r, curr_c = q.popleft()
                    current_cluster_pixels.append((curr_r, curr_c))
                    region_map[curr_r, curr_c] = region_id_counter
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))

                # Split oversized clusters into bounded regions
                if len(current_cluster_pixels) > max_region_size:
                    cluster_set = set(current_cluster_pixels)
                    sub_q = deque([current_cluster_pixels[0]])
                    sub_visited = {current_cluster_pixels[0]}
                    count = 0
                    while sub_q:
                        sr, sc = sub_q.popleft()
                        count += 1
                        if count > max_region_size:
                            region_id_counter += 1
                            count = 0

                        region_map[sr, sc] = region_id_counter
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = sr + dr, sc + dc
                            if (nr, nc) in cluster_set and (nr, nc) not in sub_visited:
                                sub_visited.add((nr, nc))
                                sub_q.append((nr, nc))

                region_id_counter += 1

    # 2. Extract region properties and build graph
    regions = {}
    centroids = {}
    unique_ids = np.unique(region_map)
    for rid in unique_ids:
        if rid == 0:
            continue
        mask = (region_map == rid)
        if np.sum(mask) < min_region_size:
            region_map[mask] = 0
            continue
        regions[int(rid)] = mask

        dist_map = distance_transform_edt(mask)
        centroid_r, centroid_c = np.unravel_index(np.argmax(dist_map), dist_map.shape)
        centroids[int(rid)] = (int(centroid_r), int(centroid_c))

    G = nx.Graph()
    for rid in regions:
        G.add_node(rid, centroid=centroids[rid])

    # 3. Find gateways and add edges to the graph
    for rid1 in regions:
        dilated_mask = binary_dilation(regions[rid1])
        boundary_mask = dilated_mask & (~regions[rid1])

        neighbor_ids = np.unique(region_map[boundary_mask])
        for rid2 in neighbor_ids:
            rid2 = int(rid2)
            if rid2 == 0 or rid2 <= rid1:
                continue
            if not G.has_node(rid2):
                continue

            dist = heuristic(centroids[rid1], centroids[rid2])
            G.add_edge(rid1, rid2, weight=dist)

    return G, region_map, centroids


def find_region_path(
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    region_map: np.ndarray,
    region_graph: nx.Graph,
) -> Optional[List[int]]:
    """Finds a path of region IDs from start to goal."""
    start_region = int(region_map[start_pos])
    goal_region = int(region_map[goal_pos])

    if (
        start_region == 0
        or goal_region == 0
        or not region_graph.has_node(start_region)
        or not region_graph.has_node(goal_region)
    ):
        return None

    if start_region == goal_region:
        return [start_region]

    try:
        path = nx.astar_path(
            region_graph,
            start_region,
            goal_region,
            heuristic=lambda u, v: heuristic(
                region_graph.nodes[u]["centroid"], region_graph.nodes[v]["centroid"]
            ),
            weight="weight",
        )
        return path
    except nx.NetworkXNoPath:
        return None
