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


@dataclass
class Slide15:
    def __post_init__(self) -> None:
        """Post init."""
        self.board = self.generate_random_board()
        print(self.board)
        return

    def generate_random_board(self) -> NDArray[np.int_]:
        """Generate 4x4 board of 15 tiles, 1 empty 0 tile, randomly shuffled"""
        board = np.arange(MAX_SLOTS).reshape((SIDE_LENGTH, SIDE_LENGTH))
        return board

    def can_slide(self, direction: Directions, tile_x: int, tile_y: int):
        """Check if tile can slide to direction."""
