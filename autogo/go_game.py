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

    def remove_captured_stones(self, state: GoGameState, last_move: Tuple[int, int]) -> Tuple[int, Optional[Tuple[int, int]]]:
        x, y = last_move
        opponent_color = 3 - state.board[x, y]  # switch between 1 and 2
        captured = 0
        ko_point = None

        # check each neighboring opponent group
        for neighbor in self.get_neighbors(last_move):
            nx, ny = neighbor
            if state.board[nx, ny] == opponent_color:
                group = self.get_group(state, (nx, ny))
                if self.get_liberties(state, group) == 0:
                    captured += len(group)

                    # check for ko: single stone capture that creates a ko situation
                    if len(group) == 1:
                        # check if the captured stone was only adjacent to the placed stone
                        captured_neighbors = self.get_neighbors((nx, ny))
                        adjacent_to_placed = {last_move}
                        other_neighbors = [n for n in captured_neighbors if n not in adjacent_to_placed]

                        # check if all other neighbors are occupied by current player's stones
                        all_own_color = True
                        for ox, oy in other_neighbors:
                            if state.board[ox, oy] != state.current_player:
                                all_own_color = False
                                break

                        # check if placing this stone created a group with only one liberty (the captured position)
                        if all_own_color and len(other_neighbors) == 3:  # Edge or corner
                            placed_group = self.get_group(state, last_move)
                            if self.get_liberties(state, placed_group) == 1:
                                ko_point = (nx, ny)

                    # remove the captured stones
                    for gx, gy in group:
                        state.board[gx, gy] = 0

        return captured, ko_point

    def make_move(self, state: GoGameState, move: Tuple[int, int]) -> Optional[GoGameState]:
        """Make a move and return new state if valid."""
        if not self.is_valid_move(state, move):
            return None

        # create new state
        new_state = GoGameState(
            board=state.board.copy(),
            current_player=3 - state.current_player,
            last_move=move,
            ko_point=None,
            move_count=state.move_count + 1,
            captured_black=state.captured_black,
            captured_white=state.captured_white,
            consecutive_passes=0,  # reset pass tracking for regular moves
            last_was_pass=False,
            board_history=state.board_history + [state.board.copy()]  # add current board to history
        )

        # place stone
        x, y = move
        new_state.board[x, y] = state.current_player

        # remove captured stones and detect ko
        captured, ko_point = self.remove_captured_stones(new_state, move)
        if state.current_player == 1:  # black captured white
            new_state.captured_white += captured
        else:
            new_state.captured_black += captured

        # set ko point if detected
        new_state.ko_point = ko_point

        return new_state

    def make_pass(self, state: GoGameState) -> GoGameState:
        new_state = GoGameState(
            board=state.board.copy(),
            current_player=3 - state.current_player,
            last_move=None,
            ko_point=None, # ko is cleared after a pass
            move_count=state.move_count + 1,
            captured_black=state.captured_black,
            captured_white=state.captured_white,
            consecutive_passes=state.consecutive_passes + 1 if state.last_was_pass else 1,
            last_was_pass=True,
            board_history=state.board_history # pass does not change board history
        )

        return new_state

    def get_valid_moves(self, state: GoGameState) -> List[Tuple[int, int]]:
        # will probably be useful for analysis and for the model to know valid moves
        valid_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.is_valid_move(state, (x, y)):
                    valid_moves.append((x, y))
        return valid_moves

    @staticmethod
    def is_game_over(state: GoGameState) -> bool:
        return state.consecutive_passes >= 2

    def get_winner(self, state: GoGameState) -> Optional[int]:
        if not self.is_game_over(state):
            return None

        # get territory
        black_territory, white_territory = self.calculate_territory(state.board)

        # score: stones + territory + captures + komi
        black_score = (np.sum(state.board == 1) +
                      black_territory +
                      state.captured_white)

        white_score = (np.sum(state.board == 2) +
                      white_territory +
                      state.captured_black +
                      6.5)  # Komi

        return 1 if black_score > white_score else 2

    def calculate_territory(self, board: np.ndarray) -> Tuple[int, int]:
        visited = np.zeros_like(board.shape, dtype=bool)
        black_territory = 0
        white_territory = 0

        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x, y] == 0 and not visited[x, y]:
                    # found an empty point, start flood fill
                    territory, borders = self.flood_fill_territory(board, (x, y), visited)

                    # determine ownership
                    borders_set = set(borders)
                    black_border = 1 in borders_set
                    white_border = 2 in borders_set

                    if black_border and not white_border:
                        black_territory += territory
                    elif white_border and not black_border:
                        white_territory += territory
                    # if both or none, neutral territory

        return black_territory, white_territory

    def flood_fill_territory(self, board: np.ndarray, start: Tuple[int, int], visited: np.ndarray) -> Tuple[int, List[int]]:
        if board[start[0], start[1]] != 0:
            return 0, []

        stack = [start]
        territory = 0
        borders = []

        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue

            visited[x, y] = True
            territory += 1

            for neighbor in self.get_neighbors((x, y)):
                nx, ny = neighbor
                if board[nx, ny] == 0 and not visited[nx, ny]:
                    stack.append((nx, ny))
                elif board[nx, ny] != 0:
                    borders.append(board[nx, ny])

        return territory, borders