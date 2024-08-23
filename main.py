import numpy as np
from astar15.astar import astar
from astar15.slide15 import Slide15

np.random.seed(1234)
TEST_CASES = [
    [[5, 1, 7, 3], [9, 2, 11, 4], [13, 6, 15, 8], [0, 10, 14, 12]],
    [[2, 5, 13, 12], [1, 0, 3, 15], [9, 7, 14, 6], [10, 11, 8, 4]],
    [[5, 2, 4, 8], [10, 0, 3, 14], [13, 6, 11, 12], [1, 15, 9, 7]],
    [[11, 4, 12, 2], [5, 10, 3, 15], [14, 1, 6, 7], [0, 9, 8, 13]],
    [[5, 8, 7, 11], [1, 6, 12, 2], [9, 0, 13, 10], [14, 3, 4, 15]],
]


def main():
    """Create game with random start, determined end"""
    for test_case in TEST_CASES:
        start = np.array(test_case)
        end = np.copy(Slide15(random=False).generate_board())
        directions = astar(start, end, know_first=True)
        print(f"Board Start:\n{start}\n")
        print(f"Board Goal:\n{end}\n")
        print(directions, "\n", len(directions))


main()
