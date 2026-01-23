import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GoGameState:
    board: np.ndarray  # 0 is empty, 1 is black, 2 is white
    current_player: int  # 1 is black, 2 is white
    last_move: Optional[Tuple[int, int]]
    ko_point: Optional[Tuple[int, int]]
    move_count: int
    captured_black: int
    captured_white: int
    consecutive_passes: int  # track consecutive passes. determine end of game
    last_was_pass: bool  # track if the last move was a pass
    board_history: List[np.ndarray]  # track board history for superko rule
    board_hashes: set[int]  # track board hashes for fast superko checking


class GoGame:
    # O(n^2) where n = board_size
    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        self._neighbors = {}
        for x in range(board_size):
            for y in range(board_size):
                neighbors = []
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < board_size and 0 <= ny < board_size:
                        neighbors.append((nx, ny))
                self._neighbors[(x, y)] = tuple(neighbors)
        self.reset()

    # O(n^2) - board creation and hash computation
    def reset(self) -> GoGameState:
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        board_hash = hash(board.tobytes())
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
            board_history=[],
            board_hashes={board_hash},
        )

    # O(h * n^2) where h = history length (worst case), O(n^2) average with hash optimization
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

        # check if this move would capture opponent stones and apply in single pass
        opponent_color = 3 - state.current_player

        for neighbor in self.get_neighbors(move):
            nx, ny = neighbor
            if temp_board[nx, ny] == opponent_color:
                group = self.get_group_with_board(temp_board, neighbor)
                if self.get_liberties_with_board(temp_board, group) == 0:
                    # apply capture immediately
                    for gx, gy in group:
                        temp_board[gx, gy] = 0

        # check for superko rule (using hash for fast lookup)
        temp_hash = hash(temp_board.tobytes())
        if temp_hash in state.board_hashes:
            # verify with actual comparison to avoid hash collisions
            if self.would_repeat_position(state.board_history, temp_board):
                return False

        # check if placed stone has liberties
        placed_group = self.get_group_with_board(temp_board, move)
        if self.get_liberties_with_board(temp_board, placed_group) == 0:
            return False

        return True

    # O(1) - simple bounds check
    def is_on_board(self, move: Tuple[int, int]) -> bool:
        x, y = move
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    # O(1) - cached neighbor lookup
    def get_neighbors(self, pos: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        return self._neighbors.get(pos, ())

    # O(h * n^2) where h = history length, each array_equal is O(n^2)
    @staticmethod
    def would_repeat_position(
        board_history: List[np.ndarray], prospective_board: np.ndarray
    ) -> bool:
        """Check if a move would repeat a previous board position (superko rule)."""
        for historical_board in board_history:
            if np.array_equal(historical_board, prospective_board):
                return True
        return False

    # O(1) - wrapper, delegates to get_group_with_board
    def get_group(self, state: GoGameState, pos: Tuple[int, int]) -> set:
        """Get all stones in the same group as position."""
        return self.get_group_with_board(state.board, pos)

    # O(n^2) - worst case visits entire board (single connected group)
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

    # O(1) - wrapper, delegates to get_liberties_with_board
    def get_liberties(self, state: GoGameState, group: set) -> int:
        """Count liberties of a group."""
        return self.get_liberties_with_board(state.board, group)

    # O(g) where g = group size (bounded by O(n^2))
    def get_liberties_with_board(self, board: np.ndarray, group: set) -> int:
        """Count liberties of a group using custom board."""
        liberties = set()
        for x, y in group:
            for neighbor in self.get_neighbors((x, y)):
                nx, ny = neighbor
                if board[nx, ny] == 0:
                    liberties.add((nx, ny))
        return len(liberties)

    # O(n^2) - checks up to 4 neighbor groups, each group operation O(n^2) worst case
    def remove_captured_stones(
        self, state: GoGameState, last_move: Tuple[int, int]
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
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
                        other_neighbors = [
                            n for n in captured_neighbors if n not in adjacent_to_placed
                        ]

                        # check if all other neighbors are occupied by current player's stones
                        all_own_color = True
                        for ox, oy in other_neighbors:
                            if state.board[ox, oy] != state.current_player:
                                all_own_color = False
                                break

                        # check if placing this stone created a group with only one liberty (the captured position)
                        if (
                            all_own_color and len(other_neighbors) == 3
                        ):  # Edge or corner
                            placed_group = self.get_group(state, last_move)
                            if self.get_liberties(state, placed_group) == 1:
                                ko_point = (nx, ny)

                    # remove the captured stones
                    for gx, gy in group:
                        state.board[gx, gy] = 0

        return captured, ko_point

    # O(h * n^2) - dominated by is_valid_move call and board copy/hashing
    def make_move(
        self, state: GoGameState, move: Tuple[int, int]
    ) -> Optional[GoGameState]:
        """Make a move and return new state if valid."""
        if not self.is_valid_move(state, move):
            return None

        # create new state
        new_board = state.board.copy()
        x, y = move
        new_board[x, y] = state.current_player

        # remove captured stones and detect ko
        # temporarily use new_board for capture detection
        temp_state = GoGameState(
            board=new_board,
            current_player=state.current_player,
            last_move=move,
            ko_point=None,
            move_count=state.move_count + 1,
            captured_black=state.captured_black,
            captured_white=state.captured_white,
            consecutive_passes=0,
            last_was_pass=False,
            board_history=state.board_history,
            board_hashes=state.board_hashes,
        )
        captured, ko_point = self.remove_captured_stones(temp_state, move)

        new_state = GoGameState(
            board=new_board,
            current_player=3 - state.current_player,
            last_move=move,
            ko_point=ko_point,
            move_count=state.move_count + 1,
            captured_black=state.captured_black
            + (captured if state.current_player == 2 else 0),
            captured_white=state.captured_white
            + (captured if state.current_player == 1 else 0),
            consecutive_passes=0,
            last_was_pass=False,
            board_history=state.board_history + [state.board.copy()],
            board_hashes=state.board_hashes | {hash(new_board.tobytes())},
        )

        return new_state

    # O(1) - creates new state without board operations
    def make_pass(self, state: GoGameState) -> GoGameState:
        return GoGameState(
            board=state.board,
            current_player=3 - state.current_player,
            last_move=None,
            ko_point=None,  # ko is cleared after a pass
            move_count=state.move_count + 1,
            captured_black=state.captured_black,
            captured_white=state.captured_white,
            consecutive_passes=state.consecutive_passes + 1
            if state.last_was_pass
            else 1,
            last_was_pass=True,
            board_history=state.board_history,  # pass does not change board history
            board_hashes=state.board_hashes,
        )

    # O(h * n^2) - is_valid_move called for each empty position with adjacent stones
    def get_valid_moves(self, state: GoGameState) -> List[Tuple[int, int]]:
        # will probably be useful for analysis and for the model to know valid moves
        valid_moves = []
        empty_spots = np.argwhere(state.board == 0)

        for x, y in empty_spots:
            # only check positions with at least one adjacent stone
            has_adjacent = False
            for nx, ny in self.get_neighbors((x, y)):
                if state.board[nx, ny] != 0:
                    has_adjacent = True
                    break

            if has_adjacent or state.move_count == 0:  # first move can be anywhere
                if self.is_valid_move(state, (x, y)):
                    valid_moves.append((x, y))

        return valid_moves

    # O(1) - simple field access
    @staticmethod
    def is_game_over(state: GoGameState) -> bool:
        return state.consecutive_passes >= 2

    # O(n^2) - largely affected by calculate_territory
    def get_winner(self, state: GoGameState) -> Optional[int]:
        if not self.is_game_over(state):
            return None

        # get territory
        black_territory, white_territory = self.calculate_territory(state.board)

        # score: stones + territory + captures + komi
        black_score = np.sum(state.board == 1) + black_territory + state.captured_white

        white_score = (
            np.sum(state.board == 2) + white_territory + state.captured_black + 6.5
        )  # Komi

        return 1 if black_score > white_score else 2

    # O(n^2) - visits each position once, flood_fill is O(n^2) total across all calls
    def calculate_territory(self, board: np.ndarray) -> Tuple[int, int]:
        visited = np.zeros((self.board_size, self.board_size), dtype=bool)
        black_territory = 0
        white_territory = 0

        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[x, y] == 0 and not visited[x, y]:
                    # found an empty point, start flood fill
                    territory, borders = self.flood_fill_territory(
                        board, (x, y), visited
                    )

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

    # O(n^2) - worst case visits entire board
    def flood_fill_territory(
        self, board: np.ndarray, start: Tuple[int, int], visited: np.ndarray
    ) -> Tuple[int, List[int]]:
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

    # O(n^2 * h) - dominated by get_valid_moves and get_liberty_channels calls
    def to_tensor(self, state: GoGameState) -> np.ndarray:
        """Convert game state to tensor representation for model input."""
        # channels: 0-1 (stones), 2 (empty), 3 (turn), 4 (ko), 5 (last move), 6-10 (liberties), 11 (legal), 12-18 (history)
        # VERY MUCH inspired by alphago's implementation
        channels = 26
        tensor = np.zeros(
            (channels, self.board_size, self.board_size), dtype=np.float32
        )

        # channel 0: black stones
        tensor[0] = (state.board == 1).astype(np.float32)

        # channel 1: white stones
        tensor[1] = (state.board == 2).astype(np.float32)

        # channel 2: empty points
        tensor[2] = (state.board == 0).astype(np.float32)

        # channel 3: current player (1 for black, 0 for white)
        tensor[3] = np.full(
            (self.board_size, self.board_size),
            1.0 if state.current_player == 1 else 0.0,
            dtype=np.float32,
        )

        # channel 4: ko point
        if state.ko_point:
            kx, ky = state.ko_point
            tensor[4, kx, ky] = 1.0

        # channel 5: last move
        if state.last_move:
            lx, ly = state.last_move
            tensor[5, lx, ly] = 1.0

        # channels 6-10: liberties (1, 2, 3, 4, 5+)
        liberty_channels = self.get_liberty_channels(state)
        tensor[6:11] = liberty_channels

        # channel 11: legal moves
        valid_moves = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        for move in self.get_valid_moves(state):
            valid_moves[move[0], move[1]] = 1.0
        tensor[11] = valid_moves

        # channels 12-18: board history (last 7 positions, using 2 planes per position)
        history_len = min(7, len(state.board_history))
        for i in range(history_len):
            past_board = state.board_history[-(i + 1)]
            base_idx = 12 + 2 * i
            tensor[base_idx] = (past_board == 1).astype(np.float32)  # black stones
            tensor[base_idx + 1] = (past_board == 2).astype(np.float32)  # white stones

        return tensor

    # O(n^4) worst case: O(n^2) positions, each potentially scanning O(n^2) group
    def get_liberty_channels(self, state: GoGameState) -> np.ndarray:
        """Create liberty channels for each position."""
        # liberties are free intersections adjacent to a group
        liberty_channels = np.zeros(
            (5, self.board_size, self.board_size), dtype=np.float32
        )
        liberties_grid = np.full((self.board_size, self.board_size), -1, dtype=np.int8)

        for x in range(self.board_size):
            for y in range(self.board_size):
                if state.board[x, y] != 0:
                    group = self.get_group(state, (x, y))
                    if group:
                        liberties = self.get_liberties(state, group)
                        # only set for the first stone of each group to avoid duplicate work
                        if (x, y) == min(group):
                            for gx, gy in group:
                                liberties_grid[gx, gy] = liberties

        # encode liberties into binary channels
        liberty_channels[0] = (liberties_grid == 1).astype(np.float32)
        liberty_channels[1] = (liberties_grid == 2).astype(np.float32)
        liberty_channels[2] = (liberties_grid == 3).astype(np.float32)
        liberty_channels[3] = (liberties_grid == 4).astype(np.float32)
        liberty_channels[4] = (liberties_grid >= 5).astype(np.float32)

        return liberty_channels
