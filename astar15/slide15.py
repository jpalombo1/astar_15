from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

BLANK_SPACE: int = 0
SIDE_LENGTH: int = 4


class Directions(Enum):
    """Enumerate possible slide directions. Values are row and column offsets set by direction."""

    UP = [-1, 0]
    DOWN = [1, 0]
    LEFT = [0, -1]
    RIGHT = [0, 1]


def tile_distance(
    tile_number: int, tile_row: int, tile_col: int, side_length: int = SIDE_LENGTH
) -> int:
    """Calculate L1 norm distance tile to expected spot.
    0 is last number so tile number off by 1 since row 0 col 0 expected tile num is 1.

    Parameters
    ----------
    tile_number: int Number of tile for expected spot.
    tile_row: int Row of specified tile to slide.
    tile_col: int Col of specified tile to slide
    direction: Directions Direction of slide
    side_length: int Default side length

    Returns
    -------
    int Heuristic of tile number epxected loation to current location in grid moves.
    """
    expected_row = (tile_number - 1) // side_length
    expected_col = (tile_number - 1) % side_length
    return np.abs(expected_row - tile_row) + np.abs(expected_col - tile_col)


@dataclass
class Slide15:
    """Slide 15 Game. Game where you slide tiles in grid with 1 empty and order numbers.

    Attributes
    ----------
    random: (bool) Flag to determine if use random board or goal one.
    side_length: (int) Length of side for square board.
    board: (NDArray[np.int_]) The game board with size side_length x side_length, values 0..side_length**2 -1
    """

    random: bool = True
    side_length: int = SIDE_LENGTH

    def __post_init__(self) -> None:
        """Post init to make the board."""
        if self.random:
            self.board = self.generate_random_board()
        else:
            self.board = self.generate_board()

    def generate_board(self) -> NDArray[np.int_]:
        """Generate standard ordered board, side_lenxside_len side_len**2 tiles, 0 empty tile in top left.

        Returns
        -------
        (NDArray[np.int_]) The game board with size side_length x side_length, values 0..side_length**2 -1
        """
        return np.roll(np.arange(self.side_length**2), -1).reshape(
            (self.side_length, self.side_length)
        )

    def generate_random_board(self) -> NDArray[np.int_]:
        """Generate side_lenxside_len side_len**2 tiles, 1 empty 0 tile, randomly shuffled.

        Returns
        -------
        (NDArray[np.int_]) The game board with size side_length x side_length, values 0..side_length**2 -1
        """
        board = np.arange(self.side_length**2)
        np.random.shuffle(board)
        return board.reshape((self.side_length, self.side_length))

    def neighbors(self) -> list[NDArray[np.int_]]:
        """Generate neighbors of board by trying to slide all directions into blank space. Only include valid slides.

        To slide in direction of empty space, must get tile in opposite direction in relation to tile so multiply by -1.

        Returns
        -------
        list[NDArray[np.int_]] : List of neighbors where neighbor is board state.
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
        """Performs sliding given tile to slide row/column, direction to slide. Returns the new board.

        Makes sure slide is valid so piece not off board or hit into non-blank piece, then perform slide.
        Also if set_board is set, the new board state is updated to the new board.

        Parameters:
        tile_row: int Row of specified tile to slide.
        tile_col: int Col of specified tile to slide
        direction: Directions Direction of slide
        set_board: bool Decide to set board state to slide result, default False

        Returns:
        NDArray[np.int_] Board state after slide.
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
        """Check if tile can slide/ would move into an empty space.

        Parameters
        ----------
        row: (int) Row of tile to slide
        col: (int) Col of tile to slide
        direction: (Directions) Direction to slide tile

        Returns
        -------
        bool: Check if tile would slide into blank space.
        """
        row += direction.value[0]
        col += direction.value[1]
        return self.board[row, col] == BLANK_SPACE

    def _check_boundaries(self, row: int, col: int, direction: Directions) -> bool:
        """Check if sliding tile at position in given direction both stays within the board and even a valid position to start.

        e.g. if col is 0 and slide left, col now -1 so invalid.

        Parameters
        ----------
        row: (int) Row of tile to slide
        col: (int) Col of tile to slide
        direction: (Directions) Direction to slide tile

        Returns
        -------
        bool: Check if tile is both not off the board and won't slide off board.
        """
        if row >= self.side_length or row < 0:
            return False
        if col >= self.side_length or col < 0:
            return False

        row += direction.value[0]
        col += direction.value[1]

        if row >= self.side_length or row < 0:
            return False
        if col >= self.side_length or col < 0:
            return False
        return True
