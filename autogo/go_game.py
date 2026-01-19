import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class GoGameState:
    board: np.ndarray # 0 is empty, 1 is black, 2 is white
    current_player: int # 1 is black, 2 is white
    last_move: Optional[Tuple[int, int]]
    ko_point: Optional[Tuple[int, int]]
    move_count: int
    captured_black: int
    captured_white: int
    consecutive_passes: int # track consecutive passes. determine end of game
    last_was_pass: bool # track if the last move was a pass
    board_history: List[np.ndarray] # track board history for superko rule

class GoGame:
    def __init__(self, board_size: int=9):
        self.board_size = board_size
        self.reset()

    def reset(self) -> GoGameState:
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        return GoGameState(
            board=board,
            current_player=1,  # Black starts
            last_move=None,
            ko_point=None,
            move_count=0,
            captured_black=0,
            captured_white=0,
            consecutive_passes=0,
            last_was_pass=False,
            board_history=[]  # Start with empty history
        )

    def is_valid_move(self, state: GoGameState, move: Tuple[int, int]) -> bool:
        if not self.is_on_board(move):
            return False

        x, y = move
        if state.board[x, y] != 0:
            return False

        # check ko
        if state.ko_point == move:
            return False

        # check for suicide rule
        temp_board = state.board.copy()
        temp_board[x, y] = state.current_player

        # check if this move would capture opponent stones
        opponent_color = 3 - state.current_player
        would_capture = False

        for neighbor in self.get_neighbors(move):
            nx, ny = neighbor
            if temp_board[nx, ny] == opponent_color:
                group = self.get_group_with_board(temp_board, neighbor)
                if self.get_liberties_with_board(temp_board, group) == 0:
                    would_capture = True
                    break

        # apply captures to temp board for accurate superko checking
        if would_capture:
            for neighbor in self.get_neighbors(move):
                nx, ny = neighbor
                if temp_board[nx, ny] == opponent_color:
                    group = self.get_group_with_board(temp_board, neighbor)
                    if self.get_liberties_with_board(temp_board, group) == 0:
                        for gx, gy in group:
                            temp_board[gx, gy] = 0

        # check for superko rule (positional)
        if self.would_repeat_position(state.board_history, temp_board):
            return False

        # check if placed stone has liberties
        placed_group = self.get_group_with_board(temp_board, move)
        if self.get_liberties_with_board(temp_board, placed_group) == 0:
            return False

        return True



    def is_on_board(self, move: Tuple[int, int]) -> bool:
        x, y = move
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                neighbors.append((nx, ny))
        return neighbors

    @staticmethod
    def would_repeat_position(board_history: List[np.ndarray], prospective_board: np.ndarray) -> bool:
        """Check if a move would repeat a previous board position (superko rule)."""
        for historical_board in board_history:
            if np.array_equal(historical_board, prospective_board):
                return True
        return False

    def get_group(self, state: GoGameState, pos: Tuple[int, int]) -> set:
        """Get all stones in the same group as position."""
        return self.get_group_with_board(state.board, pos)

    def get_group_with_board(self, board: np.ndarray, pos: Tuple[int, int]) -> set:
        """Get all stones in the connected group at position pos on the given board."""
        x, y = pos
        color = board[x, y]
        if color == 0:
            return set()

        group = set()
        stack = [pos]

        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)

            for neighbor in self.get_neighbors(current):
                nx, ny = neighbor
                if board[nx, ny] == color and neighbor not in group:
                    stack.append(neighbor)

        return group

    def get_liberties(self, state: GoGameState, group: set) -> int:
        """Count liberties of a group."""
        return self.get_liberties_with_board(state.board, group)

    def get_liberties_with_board(self, board: np.ndarray, group: set) -> int:
        """Count liberties of a group using custom board."""
        liberties = set()
        for x, y in group:
            for neighbor in self.get_neighbors((x, y)):
                nx, ny = neighbor
                if board[nx, ny] == 0:
                    liberties.add((nx, ny))
        return len(liberties)
