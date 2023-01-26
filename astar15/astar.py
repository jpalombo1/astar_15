from astar15.slide15 import Slide15, tile_distance
import numpy as np
from numpy.typing import NDArray


np.random.seed(1234)


def potential(node: NDArray[np.int_]) -> int:
    """Potential function is total distance for each number to its proper place."""
    potential = 0
    for row, vals in enumerate(node):
        for col, tile_number in enumerate(vals):
            potential += tile_distance(tile_number, row, col)
    return potential


def astar(start: NDArray[np.int_], end: NDArray[np.int_]) -> None:
    start_hash = hash(start.tostring())
    end_hash = hash(end.tostring())
    open_nodes = [start_hash]
    closed_nodes = []
    hash_map = {start_hash: start}
    distances = {start_hash: 0}
    edge_len = 1

    while len(open_nodes) > 0:
        nodevals = [
            distances.get(node_hash, 0) + potential(hash_map[node_hash])
            for node_hash in open_nodes
        ]
        cur_node_hash = open_nodes[np.argmin(nodevals)]
        open_nodes.remove(cur_node_hash)
        closed_nodes.append(cur_node_hash)
        if cur_node_hash == end_hash:
            return distances[end_hash]

        cur_node = hash_map[cur_node_hash]
        slide_game.board = cur_node

        for neighbor in slide_game.neighbors():
            neighbor_hash = hash(neighbor.tostring())
            hash_map[neighbor_hash] = neighbor
            distances[neighbor_hash] = min(
                distances.get(neighbor_hash, 0),
                distances.get(cur_node_hash, 0) + edge_len,
            )
            if neighbor_hash not in closed_nodes and neighbor_hash not in open_nodes:
                open_nodes.append(neighbor_hash)
        print(len(open_nodes))
        print([hash_map[nhash] for nhash in open_nodes])
        print(distances)
    return


slide_game = Slide15(random=True)
start = np.copy(slide_game.board)
end = np.copy(Slide15(random=False).generate_board())
astar(start, end)
potential(start)
print(f"Board Start:\n{start}\n")
print(f"Board Goal:\n{end}\n")
