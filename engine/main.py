from typing import Optional

import torch
import numpy
import uvicorn
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)

from engine.go_game import GoGame, GoGameState
from engine.redis_client import GoRedisClient


class GoModel(torch.nn.Module):
    def __init__(self, board_size: int =9):
        super().__init__()
        self.board_size = board_size
        self.action_size = self.board_size * self.board_size + 1  # +1 for pass move

        # The initial convolutional layers to process the board state
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(26, 64, kernel_size=3, padding=1), # matches 19 channels in @go_game.py
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )

        # The policy head to output move probabilities
        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 2, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * self.board_size * self.board_size, self.action_size),
            torch.nn.Softmax(dim=-1)
        )

        # The value head to output the game outcome prediction
        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 1, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(self.board_size * self.board_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

class GameLoop:
    """Public game loop that runs the main Go game for display"""

    def __init__(self, redis_client: GoRedisClient):
        self.redis = redis_client
        self.game = GoGame()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = self.game.reset()
        self.pass_count = 0
        self.game_over = False

    def load_latest_model(self) -> bool:
        """Load the latest model from Redis."""
        version = self.redis.get_current_model_version()
        if version == 0:
            logger.warning("No trained model available, using random moves")
            return False

        model_data = self.redis.load_model(version)
        if not model_data:
            logger.warning("Failed to load model data")
            return False

        self.model = GoModel()
        self.model.load_state_dict(pickle.loads(model_data))
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model version {version}")
        return True

    def select_move(self, state: GoGameState) -> Optional[tuple]:
        if self.model is None:
            # random move
            valid_moves = self.game.get_valid_moves(state)
            if valid_moves:
                return valid_moves[np.random.randint(len(valid_moves))]
            return None

            # model-based move
        state_tensor = torch.from_numpy(self.game.to_tensor(state)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, _ = self.model(state_tensor)
            policy = policy.squeeze().cpu().numpy()

        # filter to valid moves
        valid_moves = self.game.get_valid_moves(state)
        if not valid_moves:
            return None

        valid_mask = np.zeros(self.model.action_size, dtype=bool)
        for x, y in valid_moves:
            valid_mask[x * self.game.board_size + y] = True

        valid_policy = policy.copy()
        valid_policy[~valid_mask] = 0
        valid_policy = valid_policy / valid_policy.sum()

        # sample move
        move_idx = np.random.choice(self.model.action_size, p=valid_policy)

        if move_idx == self.model.action_size - 1:  # Pass
            return None
        else:
            x = move_idx // self.game.board_size
            y = move_idx % self.game.board_size
            return x, y

    def step_game(self):
        """Execute one move in the game."""
        if self.game_over:
            return

        move = self.select_move(self.state)
        if move is None:
            # pass move
            pass_count = getattr(self, 'pass_count', 0) + 1
            if pass_count >= 2:
                self.game_over = True
                winner = self.game.get_winner(self.state)
                self.redis.update_win_stats(winner)
                logger.info(f"Game over. Winner: {'Black' if winner == 1 else 'White'}")
        else:
            # make move
            new_state = self.game.make_move(self.state, move)
            if new_state is None:
                logger.error(f"Invalid move: {move}")
                return
            self.state = new_state
            self.pass_count = 0

        # save state to Redis
        self.redis.save_public_game_state(self.state)

    def reset_game(self):
        """Start a new game."""
        self.state = self.game.reset()
        self.game_over = False
        self.pass_count = 0
        self.redis.save_public_game_state(self.state)
        logger.info("New game started")



