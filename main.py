import numpy as np

from astar15.astar import astar
from astar15.slide15 import Slide15

np.random.seed(1234)
TEST_CASE = [[5, 1, 7, 3], [9, 2, 11, 4], [13, 6, 15, 8], [0, 10, 14, 12]]


def main():
    """Create game with random start, determined end"""
    start = np.array(TEST_CASE)
    end = np.copy(Slide15(random=False).generate_board())
    directions = astar(start, end)
    print(f"Board Start:\n{start}\n")
    print(f"Board Goal:\n{end}\n")
    print(directions)


main()
