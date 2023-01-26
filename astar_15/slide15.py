from dataclasses import dataclass
import numpy as np
from enum import Enum, auto
from numpy.typing import NDArray

MAX_SLOTS = 16
SIDE_LENGTH = 4
BLANK_SPACE = 0


class Directions(Enum):
    """Enumerate possible slide directions."""

    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


def tile_distance(self, tile_number: int, tile_row: int, tile_col: int) -> int:
    """Calculate L1 norm distance tile to expected spot."""
    expected_row = tile_number // SIDE_LENGTH
    expected_col = tile_number % SIDE_LENGTH
    return np.abs(expected_row - tile_row) + np.abs(expected_col - tile_col)


@dataclass
class Slide15:
    """Slide 15 Game."""

    def __post_init__(self) -> None:
        """Post init."""
        self.board = self.generate_random_board()
        print(self.board)
        return

    def generate_board(self) -> NDArray[np.int_]:
        """Generate standard ordered board, 4x4 15 tiles, 0 empty tile in top left."""
        return np.arange(MAX_SLOTS).reshape((SIDE_LENGTH, SIDE_LENGTH))

    def generate_random_board(self) -> NDArray[np.int_]:
        """Generate 4x4 board of 15 tiles, 1 empty 0 tile, randomly shuffled"""
        board = np.arange(MAX_SLOTS)
        np.random.shuffle(board)
        return board.reshape((SIDE_LENGTH, SIDE_LENGTH))

    def neighbors(self) -> list[NDArray[np.int_]]:
        """Generate neighbors of board by trying to slide all directions into blank space."""
        empty_rows, empty_cols = np.where(self.board == BLANK_SPACE)
        empty_row, empty_col = empty_rows[0], empty_cols[0]
        neighbors = []
        print(empty_row, empty_col)
        for direction in Directions:
            try:
                neighbors.append(self.slide(empty_row, empty_col, direction))
                print(f"{direction.name}: Valid!")
            except ValueError as e:
                print(f"{direction.name}: {e}")
                continue
        return neighbors

    def slide(
        self, tile_row: int, tile_col: int, direction: Directions
    ) -> NDArray[np.int_]:
        if not self._check_boundaries(tile_row, tile_col, direction):
            raise ValueError("Not possible move, slide off board, try again!")
        if not self._can_slide(tile_row, tile_col, direction):
            raise ValueError("Not possible move, sliding into non-empty space!")
        new_row = (
            tile_row - 1
            if direction == Directions.UP
            else tile_row + 1
            if direction == Directions.DOWN
            else tile_row
        )
        new_col = (
            tile_col - 1
            if direction == Directions.LEFT
            else tile_col + 1
            if direction == Directions.RIGHT
            else tile_col
        )
        new_board = np.copy(self.board)
        new_board[new_row, new_col] = new_board[tile_row, tile_col]
        new_board[tile_row, tile_col] = BLANK_SPACE
        return new_board

    def _can_slide(self, row: int, col: int, direction: Directions) -> bool:
        """Check if tile can slide into empty space."""
        if direction == Directions.UP:
            return self.board[row - 1, col] == BLANK_SPACE
        elif direction == Directions.DOWN:
            return self.board[row + 1, col] == BLANK_SPACE
        if direction == Directions.LEFT:
            return self.board[row, col - 1] == BLANK_SPACE
        elif direction == Directions.RIGHT:
            return self.board[row, col - 1] == BLANK_SPACE
        return False

    def _check_boundaries(self, row: int, col: int, direction: Directions) -> bool:
        """Check if sliding tile at xy position in given direction stays within the board board.

        e.g. if xpos is 0 and slide left, xpos now -1 so invalid.
        """
        if row + 1 > SIDE_LENGTH - 1 and direction == Directions.DOWN:
            return False
        if row - 1 < 0 and direction == Directions.UP:
            return False
        if col + 1 > SIDE_LENGTH - 1 and direction == Directions.RIGHT:
            return False
        if col - 1 < 0 and direction == Directions.LEFT:
            return False
        return True


slide15 = Slide15()
print(slide15.neighbors())
