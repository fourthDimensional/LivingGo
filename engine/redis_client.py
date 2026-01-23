import redis
import pickle
import os
from typing import Optional, List, Dict, Any
import logging

from engine.go_game import GoGameState

logger = logging.getLogger(__name__)


class GoRedisClient:
    def __init__(self, host: str = None, port: int = None, db: int = 0):
        host = host or os.getenv('REDIS_HOST', 'redis') # default to 'redis' for Docker setup
        port = port or int(os.getenv('REDIS_PORT', 6379))
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.client.ping() # will raise an error if connection fails

    # model management stuff
    def get_current_model_version(self) -> int:
        version = self.client.get('current_model_version')
        return int(version) if version else 0

    def set_current_model_version(self, version: int):
        self.client.set('current_model_version', version)

    def save_model(self, version: int, model_data: bytes):
        self.client.set(f'model:{version}', model_data)

    def load_model(self, version: int):
        return self.client.get(f'model:{version}')

    # game state management stuff
    def save_game_state(self, state: GoGameState):
        # this is the public, singular game state
        board_key = 'public:board_state'
        self.client.set(board_key, pickle.dumps(state.board))

        # save metadata
        self.client.set("public:move_count", state.move_count)
        self.client.set("public:current_player", state.current_player)

        if state.last_move:
            self.client.set("public:last_move", pickle.dumps(state.last_move))

        metadata = {
            "captured_black": state.captured_black,
            "captured_white": state.captured_white,
            "ko_point": state.ko_point
        }
        self.client.set("public:metadata", pickle.dumps(metadata))

    def load_game_state(self) -> Optional[GoGameState]:
        board_data = self.client.get("public:board_state")
        if not board_data:
            return None

        board = pickle.loads(board_data)
        move_count = int(self.client.get("public:move_count") or 0)
        current_player = int(self.client.get("public:current_player") or 1)

        last_move_data = self.client.get("public:last_move")
        last_move = pickle.loads(last_move_data) if last_move_data else None

        metadata_data = self.client.get("public:metadata")
        metadata = pickle.loads(metadata_data) if metadata_data else {}

        return GoGameState(
            board=board,
            current_player=current_player,
            last_move=last_move,
            ko_point=None,  # not stored for public display
            move_count=move_count,
            captured_black=metadata.get("captured_black", 0),
            captured_white=metadata.get("captured_white", 0),
            consecutive_passes=0,  # not tracked for public display
            last_was_pass=False,  # not tracked for public display
            board_history=[],  # not tracked for public display
            board_hashes={hash(board.tobytes())}
        )

    # training stuff
    def add_trajectory(self, trajectory: Dict[str, Any]):
        trajectory_data = pickle.dumps(trajectory)
        result = self.client.lpush("trajectories", trajectory_data)
        logger.info(
            f"Added trajectory to Redis, result: {result}, total trajectories: {self.client.llen('trajectories')}")

    def add_trajectories_batch(self, trajectories: List[Dict[str, Any]]):
        """Add multiple training trajectories in a single Redis pipeline operation."""
        if not trajectories:
            logger.info("No trajectories to add")
            return 0

        # use Redis pipeline for bulk insert
        pipeline = self.client.pipeline()
        trajectory_data_list = []

        # prepare trajectory data with validation
        for i, traj in enumerate(trajectories):
            try:
                # validate trajectory structure
                if not isinstance(traj, dict) or "states" not in traj:
                    logger.warning(f"Skipping invalid trajectory at index {i}: missing required fields")
                    continue

                trajectory_data = pickle.dumps(traj)
                trajectory_data_list.append(trajectory_data)
            except (pickle.PickleError, TypeError) as e:
                logger.warning(f"Failed to serialize trajectory at index {i}: {e}")
                continue

        if not trajectory_data_list:
            logger.error("No valid trajectories to add after validation")
            return 0

        # add all trajectories to the pipeline
        for trajectory_data in trajectory_data_list:
            pipeline.lpush("trajectories", trajectory_data)

        # execute all operations in a single roundtrip
        results = pipeline.execute()

        total_added = len([r for r in results if r])  # count successful additions
        total_trajectories = self.client.llen("trajectories")

        logger.info(
            f"Added {total_added}/{len(trajectory_data_list)} trajectories to Redis in batch operation, total trajectories: {total_trajectories}")

        # log any failures
        failed_count = len(trajectory_data_list) - total_added
        if failed_count > 0:
            logger.warning(f"{failed_count} trajectories failed to add to Redis")

        return total_added

    def get_trajectories(self, count: int = 10) -> List[Dict[str, Any]]:
        trajectory_data_list = self.client.lrange("trajectories", 0, count - 1)
        trajectories = [pickle.loads(data) for data in trajectory_data_list]

        logger.info(f"Retrieved {len(trajectories)} trajectories from Redis (requested {count})")

        # remove retrieved trajectories
        if trajectories:
            self.client.ltrim("trajectories", count, -1)
            remaining = self.client.llen("trajectories")
            logger.info(f"Removed {len(trajectories)} trajectories, {remaining} remaining")
        else:
            logger.info("No trajectories available to retrieve")

        return trajectories

    def get_trajectory_count(self) -> int:
        return self.client.llen("trajectories")

    # training statistics
    def increment_training_steps(self):
        self.client.incr("training:steps")

    def get_training_steps(self) -> int:
        steps = self.client.get("training:steps")
        return int(steps) if steps else 0

    def update_win_stats(self, winner: int):
        key = f"public:wins:{'black' if winner == 1 else 'white'}"
        self.client.incr(key)

    def update_win_stats_batch(self, winners: List[int]):
        """Update win statistics for multiple games."""
        if not winners:
            return

        black_wins = sum(1 for w in winners if w == 1)
        white_wins = sum(1 for w in winners if w == 2)

        if black_wins:
            self.client.incrby("public:wins:black", black_wins)
        if white_wins:
            self.client.incrby("public:wins:white", white_wins)

        logger.info(f"Updated win stats: {black_wins} black, {white_wins} white")

    def get_win_stats(self) -> Dict[str, int]:
        """Get win statistics."""
        black_wins = int(self.client.get("public:wins:black") or 0)
        white_wins = int(self.client.get("public:wins:white") or 0)
        return {"black": black_wins, "white": white_wins}

    # cleanup and maintenance
    def clear_trajectories(self):
        """Clear all training trajectories."""
        self.client.delete("trajectories")

    def clear_public_state(self):
        """Clear public game state."""
        keys = [
            "public:board_state",
            "public:move_count",
            "public:current_player",
            "public:last_move",
            "public:metadata"
        ]
        self.client.delete(*keys)

    def clear_all_data(self):
        """Clear all Go-related data from Redis."""
        logger.info("Clearing all data from Redis...")

        # clear all Go-related keys
        patterns = [
            "model:*",           # All saved models
            "current_model_version",
            "trajectories",      # Training trajectories
            "training:*",        # Training metadata
            "public:*",          # Public game state
            "public:wins:*",     # Win statistics
            "rq:queue:*",        # RQ job queues
            "rq:job:*",          # RQ job data
            "rq:worker:*",       # RQ worker data
            "rq:failed:*",       # RQ failed jobs
        ]

        # collect all keys to delete
        all_keys = []
        for pattern in patterns:
            keys = self.client.keys(pattern)
            if keys:
                all_keys.extend(keys)
                logger.info(f"Found {len(keys)} keys matching pattern '{pattern}'")

        # also try individual keys
        individual_keys = [
            "current_model_version",
            "trajectories",
            "training:steps",
            "public:wins:black",
            "public:wins:white",
        ]

        for key in individual_keys:
            if self.client.exists(key):
                all_keys.append(key)
                logger.info(f"Found individual key '{key}'")

        # remove duplicates and delete
        if all_keys:
            unique_keys = list(set(all_keys))
            self.client.delete(*unique_keys)
            logger.info(f"Deleted {len(unique_keys)} keys from Redis")
        else:
            logger.info("No keys found to delete")

        logger.info("Redis data clearing completed")

    def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server info."""
        return self.client.info()

    # backup and restore
    def backup_state(self) -> Dict[str, bytes]:
        """Backup all relevant keys."""
        keys = self.client.keys("*")
        backup = {}
        for key in keys:
            backup[key] = self.client.get(key)
        return backup

    def restore_state(self, backup: Dict[str, bytes]):
        """Restore state from backup."""
        self.client.flushdb()
        for key, value in backup.items():
            self.client.set(key, value)