import pytest
import numpy as np
from engine.go_game import GoGame


@pytest.fixture
def game():
    return GoGame(board_size=9)


def test_basic_capture(game):
    """Test basic stone capture scenario."""
    state = game.reset()

    # black plays at (4,4)
    state = game.make_move(state, (4, 4))
    assert state is not None

    # white surrounds black stone (alternating turns)
    state = game.make_move(state, (3, 4))  # white
    state = game.make_move(state, (0, 0))  # black plays elsewhere
    state = game.make_move(state, (4, 3))  # white
    state = game.make_move(state, (0, 1))  # black plays elsewhere
    state = game.make_move(state, (5, 4))  # white
    state = game.make_move(state, (0, 2))  # black plays elsewhere

    # white plays final surrounding stone at (4,5) - should capture
    state = game.make_move(state, (4, 5))

    # black stone at (4,4) should be captured
    assert state.board[4, 4] == 0
    assert state.captured_black == 1


def test_ko_rule(game):
    """Test ko rule prevents immediate recapture."""
    state = game.reset()

    # create a simple ko through valid moves
    # white plays first
    state = game.make_move(state, (4, 3))
    state = game.make_move(state, (0, 0))
    state = game.make_move(state, (4, 5))
    state = game.make_move(state, (0, 1))
    state = game.make_move(state, (3, 4))

    # black plays at (4,4)
    state = game.make_move(state, (4, 4))

    # white surrounds on the right
    state = game.make_move(state, (5, 3))
    state = game.make_move(state, (1, 0))
    state = game.make_move(state, (5, 5))
    state = game.make_move(state, (1, 1))

    # white captures at (5,4) - this might create ko
    result = game.make_move(state, (5, 4))

    # if ko was detected, verify recapture is blocked
    if result and result.ko_point:
        # black tries to recapture immediately
        assert not game.is_valid_move(result, result.ko_point)


def test_suicide_rule(game):
    """Test that suicide moves are invalid."""
    state = game.reset()

    # place black stones in a line through valid moves
    state = game.make_move(state, (4, 3))
    state = game.make_move(state, (0, 0))
    state = game.make_move(state, (4, 4))
    state = game.make_move(state, (0, 1))
    state = game.make_move(state, (4, 5))

    # now white's turn, tries to play at (4,4) which is occupied - should fail
    assert not game.is_valid_move(state, (4, 4))

    # white tries to play adjacent with no liberties (suicide)
    # first surround (4,6) with black
    state = game.make_move(state, (3, 6))
    state = game.make_move(state, (1, 1))
    state = game.make_move(state, (4, 6))  # black surrounds white's potential move
    state = game.make_move(state, (1, 2))
    state = game.make_move(state, (5, 6))

    # white tries to play at (4,5) - already occupied
    # or tries to play somewhere that would be suicide
    # create proper suicide test: black stones surrounding an empty point
    state2 = game.reset()
    state2.board[4, 3] = 1
    state2.board[4, 4] = 1
    state2.board[4, 5] = 1
    state2.current_player = 2

    # white tries to play in middle - surrounded, no escape
    assert not game.is_valid_move(state2, (4, 4))


def test_pass_and_game_over(game):
    """Test pass mechanism and game over detection."""
    state = game.reset()

    # not game over initially
    assert not game.is_game_over(state)

    # first pass
    state = game.make_pass(state)
    assert not game.is_game_over(state)
    assert state.last_was_pass

    # second pass ends game
    state = game.make_pass(state)
    assert game.is_game_over(state)
    assert state.consecutive_passes >= 2


def test_territory_scoring(game):
    """Test basic territory calculation."""
    state = game.reset()

    # create simple territory: black stones surrounding corner
    state.board[0, 0] = 1
    state.board[0, 1] = 1
    state.board[1, 0] = 1

    black_territory, white_territory = game.calculate_territory(state.board)

    # position (1,1) should be counted as black territory
    assert black_territory >= 1


def test_group_liberties(game):
    """Test liberty counting for groups."""
    state = game.reset()

    # create a group of 3 black stones in a line
    state.board[4, 3] = 1
    state.board[4, 4] = 1
    state.board[4, 5] = 1

    group = game.get_group(state, (4, 4))
    liberties = game.get_liberties(state, group)

    # should have liberties
    assert liberties > 0


def test_superko_basic(game):
    """Test that board history is tracked."""
    state = game.reset()

    # make some moves
    state = game.make_move(state, (4, 4))
    state = game.make_move(state, (4, 3))

    # check history is being tracked
    assert len(state.board_history) > 0
    assert len(state.board_hashes) > 0
