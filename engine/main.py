import torch
import numpy
import uvicorn

class GoModel(torch.nn.Module):
    def __init__(self, board_size: int =9):
        super().__init__()
        self.board_size = board_size
        self.action_size = self.board_size * self.board_size + 1  # +1 for pass move

        # The initial convolutional layers to process the board state
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(19, 64, kernel_size=3, padding=1), # matches 19 channels in @go_game.py
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


