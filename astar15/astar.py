import numpy as np
from astar15.slide15 import BLANK_SPACE, SIDE_LENGTH, Directions, Slide15, tile_distance
from numpy.typing import NDArray


def node_hash(node: NDArray[np.int_]) -> str:
    """Hash function for numpy array."""
    return ",".join([str(node) for node in node.flatten()])


def node_unhash(node_hash: str, side_length: int = SIDE_LENGTH) -> NDArray[np.int_]:
    """Convert hash back to array."""
    num_list_str = node_hash.split(",")
    return np.array([int(num) for num in num_list_str]).reshape(
        side_length, side_length
    )


def reconstruct_path(came_from: dict[str, str], current: str) -> list[str]:
    """Recursive function call to keep kgetting parent node of current node until back at root node starting at end to construct path.

    Parameters
    ----------
    came_from: dict[str, str] Dictionary of current node to its parent
    current: str Hash of current node of path

    Return
    ------
    list[str] Path of nodes
    """
    if current in came_from:
        path = reconstruct_path(came_from, came_from[current])
        return path + [current]
    else:
        return [current]


def reconstruct_moves(path: list[str]) -> list[Directions]:
    """Based on path, determine moves to make path by comparing consecutive previous and next states.

    Turn hash to actual node, then figure out movement of blank space to see what direction move. If row changes then UP/DOWN, if col then LEFT/RIGHT.

    Parameters
    ----------
    list[str] Path of nodes

    Returns
    -------
    list[Directions] List of moves that form the path.
    """
    directions = []
    for state_prev, state_next in zip(path[:-1], path[1:]):
        prev_np = node_unhash(state_prev)
        now_np = node_unhash(state_next)
        prev_rows, prev_cols = np.where(prev_np == BLANK_SPACE)
        prev_row, prev_col = prev_rows[0], prev_cols[0]
        now_rows, now_cols = np.where(now_np == BLANK_SPACE)
        now_row, now_col = now_rows[0], now_cols[0]
        if prev_row - now_row == 1:
            directions.append(Directions.DOWN)
        if prev_row - now_row == -1:
            directions.append(Directions.UP)
        if prev_col - now_col == 1:
            directions.append(Directions.RIGHT)
        if prev_col - now_col == -1:
            directions.append(Directions.LEFT)

    return directions


def potential(node: NDArray[np.int_]) -> int:
    """Potential function is total distance for each number to its proper place.

    Do not include blank tile. Just get distance for each tile of expected location to current one and sum.

    Parameters
    ----------
    (NDArray[np.int_]) The game board with size side_length x side_length, values 0..side_length**2 -1

    Returns
    -------
    int Heiuristic potential for current state from goal state.
    """
    potential = 0
    for row, vals in enumerate(node):
        for col, tile_number in enumerate(vals):
            if tile_number > 0:
                potential += tile_distance(
                    tile_number,
                    row,
                    col,
                )
    return potential


def astar(
    start: NDArray[np.int_], end: NDArray[np.int_], know_first: bool = True
) -> list[Directions]:
    """A* Algorithm.

    Calculate hashes for np array use in dictionaries and hash map to map those.
    G score is path length weight, f score is gscore + potential heuristic of each node.
    Add start node to open set, calculate neighbors, get f scores of each neighbor.
    If new neighbor not in open set and lower fscore add to open set and fscore map.
    On next pass find node with lowest fscore, move to closed set, find its neighbors, and repeat.

    Parameters
    ----------
    start (NDArray[np.int_]) The game board with size side_length x side_length, values 0..side_length**2 -1, random start state
    end (NDArray[np.int_]) The game board with size side_length x side_length, values 0..side_length**2 -1, goal state
    know_first: bool Determines if True by default, do not use g score or path length just potential.

    Returns
    -------
    list[Directions] List of moves that form the path.
    """
    slide_game = Slide15()
    start_hash = node_hash(start)
    end_hash = node_hash(end)
    open_nodes = {start_hash}
    closed_nodes = set()
    hash_map = {start_hash: start}
    came_from: dict[str, str] = {}
    g_score = {start_hash: 0}
    f_score = {start_hash: g_score[start_hash] + potential(start)}
    edge_len = 1

    while len(open_nodes) > 0:
        cur_node_hash = min(open_nodes, key=lambda open_hash: f_score.get(open_hash, 0))
        cur_node = hash_map[cur_node_hash]

        open_nodes.remove(cur_node_hash)
        closed_nodes.add(cur_node_hash)
        if cur_node_hash == end_hash:
            path = reconstruct_path(came_from, end_hash)
            directions = reconstruct_moves(path)
            return [direct.name for direct in directions]

        slide_game.board = cur_node
        for neighbor in slide_game.neighbors():
            neighbor_hash = node_hash(neighbor)
            if neighbor_hash not in hash_map:
                hash_map[neighbor_hash] = neighbor
            tent_gscore = g_score.get(cur_node_hash, 0) + edge_len
            tent_fscore = (
                potential(neighbor) if know_first else potential(neighbor) + tent_gscore
            )
            if (
                neighbor_hash in closed_nodes
                and neighbor_hash in f_score
                and tent_fscore >= f_score.get(neighbor_hash, 0)
            ):
                continue
            if neighbor_hash not in open_nodes or (
                neighbor_hash in f_score and tent_fscore < f_score.get(neighbor_hash, 0)
            ):
                came_from[neighbor_hash] = cur_node_hash
                g_score[neighbor_hash] = tent_gscore
                f_score[neighbor_hash] = tent_fscore
                if neighbor_hash not in open_nodes:
                    open_nodes.add(neighbor_hash)

    return []
