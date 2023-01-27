from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

MAX_SLOTS = 16
SIDE_LENGTH = 4
BLANK_SPACE = 0


class Directions(Enum):
    """Enumerate possible slide directions. Values are row and column offsets set by direction."""

    UP = [-1, 0]
    DOWN = [1, 0]
    LEFT = [0, -1]
    RIGHT = [0, 1]


def tile_distance(tile_number: int, tile_row: int, tile_col: int) -> int:
    """Calculate L1 norm distance tile to expected spot.
    0 is last number so tile number off by 1 since row 0 col 0 expected tile num is 1.
    """
    expected_row = (tile_number - 1) // SIDE_LENGTH
    expected_col = (tile_number - 1) % SIDE_LENGTH
    return np.abs(expected_row - tile_row) + np.abs(expected_col - tile_col)


@dataclass
class Slide15:
    """Slide 15 Game."""

    random: bool = True

    def __post_init__(self) -> None:
        """Post init."""
        if self.random:
            self.board = self.generate_random_board()
        else:
            self.board = self.generate_board()

    def generate_board(self) -> NDArray[np.int_]:
        """Generate standard ordered board, 4x4 15 tiles, 0 empty tile in top left."""
        return np.roll(np.arange(MAX_SLOTS), -1).reshape((SIDE_LENGTH, SIDE_LENGTH))

    def generate_random_board(self) -> NDArray[np.int_]:
        """Generate 4x4 board of 15 tiles, 1 empty 0 tile, randomly shuffled"""
        board = np.arange(MAX_SLOTS)
        np.random.shuffle(board)
        return board.reshape((SIDE_LENGTH, SIDE_LENGTH))

    def neighbors(self) -> list[NDArray[np.int_]]:
        """Generate neighbors of board by trying to slide all directions into blank space.

        To slide in direction of empty space, must get tile in opposite direction in relation to tile so multiply by -1.
        """
        empty_rows, empty_cols = np.where(self.board == BLANK_SPACE)
        empty_row, empty_col = empty_rows[0], empty_cols[0]
        neighbors = []
        for direction in Directions:
            try:
                tile_row = empty_row + direction.value[0] * -1
                tile_col = empty_col + direction.value[1] * -1
                neighbors.append(self.slide(tile_row, tile_col, direction))
            except ValueError:
                continue
        return neighbors

    def slide(
        self,
        tile_row: int,
        tile_col: int,
        direction: Directions,
        set_board: bool = False,
    ) -> NDArray[np.int_]:
        """Performs sliding given tile to slide row/column, direction to slide. Returns the bew board.

        Also if set_board is set, the new board state is updated to the new board.
        """
        if not self._check_boundaries(tile_row, tile_col, direction):
            raise ValueError("Not possible move, slide off board, try again!")
        if not self._can_slide(tile_row, tile_col, direction):
            raise ValueError("Not possible move, sliding into non-empty space!")
        new_row = tile_row + direction.value[0]
        new_col = tile_col + direction.value[1]
        new_board = np.copy(self.board)
        new_board[new_row, new_col] = new_board[tile_row, tile_col]
        new_board[tile_row, tile_col] = BLANK_SPACE
        if set_board:
            self.board = new_board
        return new_board

    def _can_slide(self, row: int, col: int, direction: Directions) -> bool:
        """Check if tile can slide/ would move into an empty space."""
        row += direction.value[0]
        col += direction.value[1]
        return self.board[row, col] == BLANK_SPACE

    def _check_boundaries(self, row: int, col: int, direction: Directions) -> bool:
        """Check if sliding tile at position in given direction both stays within the board and even a valid position to start.

        e.g. if col is 0 and slide left, col now -1 so invalid.
        """
        if row >= SIDE_LENGTH or row < 0:
            return False
        if col >= SIDE_LENGTH or col < 0:
            return False

        row += direction.value[0]
        col += direction.value[1]

        if row >= SIDE_LENGTH or row < 0:
            return False
        if col >= SIDE_LENGTH or col < 0:
            return False
        return True
