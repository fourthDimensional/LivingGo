import asyncio
import pickle
import logging
import random
import time
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import torch
import numpy as np

from engine.go_game import GoGame, GoGameState
from engine.redis_client import GoRedisClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoModel(torch.nn.Module):
    def __init__(self, board_size: int = 9):
        super().__init__()
        self.board_size = board_size
        self.action_size = self.board_size * self.board_size + 1  # +1 for pass move

        # The initial convolutional layers to process the board state
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                26, 64, kernel_size=3, padding=1
            ),  # matches 19 channels in @go_game.py
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
            torch.nn.Softmax(dim=-1),
        )

        # The value head to output the game outcome prediction
        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 1, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(self.board_size * self.board_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Tanh(),
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
        state_tensor = (
            torch.from_numpy(self.game.to_tensor(state)).unsqueeze(0).to(self.device)
        )

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
            pass_count = getattr(self, "pass_count", 0) + 1
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting Go Game API")

    # load initial model
    game_loop.load_latest_model()

    # start background game loop
    asyncio.create_task(game_loop_task())

    # start initial game
    game_loop.reset_game()

    yield

    logger.info("Shutting down Go Game API")


app = FastAPI(title="Go Game API", lifespan=lifespan)
redis_client = GoRedisClient()
game_loop = GameLoop(redis_client)
connected_websockets: List[WebSocket] = []


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Go Game API", "status": "running"}


@app.get("/status")
async def get_status():
    """Get current game status."""
    current_version = redis_client.get_current_model_version()
    trajectory_count = redis_client.get_trajectory_count()
    training_steps = redis_client.get_training_steps()
    win_stats = redis_client.get_win_stats()

    return {
        "model_version": current_version,
        "trajectory_count": trajectory_count,
        "training_steps": training_steps,
        "win_stats": win_stats,
        "game_active": not game_loop.game_over,
    }


@app.get("/game")
async def get_game_state():
    """Get current game state."""
    state = redis_client.load_public_game_state()
    if state is None:
        return {"error": "No game state available"}

    # Convert numpy types to native Python types
    last_move = None
    if state.last_move:
        last_move = (int(state.last_move[0]), int(state.last_move[1]))

    return {
        "board": state.board.tolist(),
        "current_player": int(state.current_player),
        "move_count": int(state.move_count),
        "last_move": last_move,
        "captured_black": int(state.captured_black),
        "captured_white": int(state.captured_white),
        "game_over": game_loop.game_over,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time game updates."""
    await websocket.accept()
    connected_websockets.append(websocket)

    try:
        # send current state immediately
        state = redis_client.load_public_game_state()
        if state:
            # Convert numpy types to native Python types
            last_move = None
            if state.last_move:
                last_move = (int(state.last_move[0]), int(state.last_move[1]))

            await websocket.send_json(
                {
                    "type": "game_update",
                    "board": state.board.tolist(),
                    "current_player": int(state.current_player),
                    "move_count": int(state.move_count),
                    "last_move": last_move,
                    "captured_black": int(state.captured_black),
                    "captured_white": int(state.captured_white),
                    "game_over": game_loop.game_over,
                }
            )

        # keep connection alive
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # handle any client messages if needed
            except asyncio.TimeoutError:
                # send ping to keep connection alive
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
        logger.info("WebSocket disconnected")


async def broadcast_game_state():
    """Broadcast game state to all connected websockets."""
    if not connected_websockets:
        return

    state = redis_client.load_public_game_state()
    if state is None:
        return

    # Convert numpy types to native Python types
    last_move = None
    if state.last_move:
        last_move = (int(state.last_move[0]), int(state.last_move[1]))

    game_message = {
        "type": "game_update",
        "board": state.board.tolist(),
        "current_player": int(state.current_player),
        "move_count": int(state.move_count),
        "last_move": last_move,
        "captured_black": int(state.captured_black),
        "captured_white": int(state.captured_white),
        "game_over": game_loop.game_over,
    }

    disconnected = []
    for websocket in connected_websockets:
        try:
            await websocket.send_json(game_message)
        except Exception as e:
            logger.error(f"Failed to send game state to websocket: {e}")
            disconnected.append(websocket)

    # remove disconnected websockets
    for ws in disconnected:
        if ws in connected_websockets:
            connected_websockets.remove(ws)


async def broadcast_performance_update():
    """Broadcast performance metrics to all connected websockets."""
    if not connected_websockets:
        return

    # get current system metrics
    current_version = redis_client.get_current_model_version()
    trajectory_count = redis_client.get_trajectory_count()
    training_steps = redis_client.get_training_steps()
    win_stats = redis_client.get_win_stats()

    # get recent training metrics
    recent_metrics = {}
    metrics = ["policy_loss", "value_loss", "entropy", "total_loss"]
    for metric in metrics:
        data = redis_client.get_training_metrics_history(metric, 10)  # last 10 entries
        if data:
            recent_metrics[metric] = data[-1]["value"] if data else 0

    performance_message = {
        "type": "performance_update",
        "timestamp": int(time.time() * 1000),
        "current_version": current_version,
        "trajectory_count": trajectory_count,
        "training_steps": training_steps,
        "win_stats": win_stats,
        "recent_metrics": recent_metrics,
    }

    disconnected = []
    for websocket in connected_websockets:
        try:
            await websocket.send_json(performance_message)
        except Exception as e:
            logger.error(f"Failed to send performance update to websocket: {e}")
            disconnected.append(websocket)

    # remove disconnected websockets
    for ws in disconnected:
        if ws in connected_websockets:
            connected_websockets.remove(ws)


async def game_loop_task():
    """Background task for running the game loop."""
    await asyncio.sleep(5)  # Initial delay
    performance_counter = 0

    while True:
        try:
            # check if we need to load a new model
            current_version = redis_client.get_current_model_version()
            if current_version > getattr(game_loop, "loaded_version", 0):
                game_loop.load_latest_model()
                game_loop.loaded_version = current_version

            # check if we need to reset the game
            if game_loop.game_over:
                game_loop.reset_game()

            # make a move
            game_loop.step_game()

            # broadcast state
            await broadcast_game_state()

            performance_counter += 1
            if performance_counter >= 30:
                await broadcast_performance_update()
                performance_counter = 0

            # wait 1 second before next move
            await asyncio.sleep(random.randint(10, 2000) / 1000.0)

        except Exception as e:
            logger.error(f"Game loop error: {e}")
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    uvicorn.run("livinggo.main:app", host="0.0.0.0", port=8000, reload=True)
