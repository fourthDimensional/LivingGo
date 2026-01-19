import torch
import numpy as np
from engine.go_game import GoGame
from engine.main import GoModel


def select_move_from_policy(policy, game, state):
    """Select a move from the policy output."""
    # get valid moves
    valid_moves = game.get_valid_moves(state)

    if not valid_moves:
        return None  # pass

    # convert moves to indices
    move_probs = []
    for x, y in valid_moves:
        idx = x * game.board_size + y
        move_probs.append((idx, policy[0, idx].item()))

    # sort by probability and select top move
    move_probs.sort(key=lambda x: x[1], reverse=True)
    best_idx, best_prob = move_probs[0]
    move = (best_idx // game.board_size, best_idx % game.board_size)

    return move


def main():
    # initialize game
    game = GoGame(board_size=9)
    state = game.reset()

    print(f"Game initialized with board size: {game.board_size}")
    print(f"Initial state - Current player: {'Black' if state.current_player == 1 else 'White'}")

    # initialize model
    model = GoModel(board_size=9)
    model.eval()

    print("\nModel initialized")

    # move 1: black
    print("MOVE 1 - BLACK")

    state_tensor = game.to_tensor(state)
    with torch.no_grad():
        policy, value = model(torch.from_numpy(state_tensor).unsqueeze(0))

    print(f"Value: {value.item():.4f} (win prob for black)")

    move = select_move_from_policy(policy, game, state)
    if move:
        state = game.make_move(state, move)
        print(f"Move played: {move}")
    else:
        state = game.make_pass(state)
        print("Move: PASS")

    # move 2: white
    print("MOVE 2 - WHITE")

    state_tensor = game.to_tensor(state)
    with torch.no_grad():
        policy, value = model(torch.from_numpy(state_tensor).unsqueeze(0))

    print(f"Value: {value.item():.4f} (win prob for white)")

    move = select_move_from_policy(policy, game, state)
    if move:
        state = game.make_move(state, move)
        print(f"Move played: {move}")
    else:
        state = game.make_pass(state)
        print("Move: PASS")

    print("Initial board state (after 2 moves):")
    print(state.board)
    print(f"Current player: {'Black' if state.current_player == 1 else 'White'}")


if __name__ == "__main__":
    main()
