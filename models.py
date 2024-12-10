from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        # First conv: 65x65 -> 22x22
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=7, stride=3, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second conv: 22x22 -> 8x8
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        \
        # Final conv: maintain 8x8 but process features
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.repr_dim = 32
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # -> 22x22
        x = F.relu(self.bn2(self.conv2(x)))  # -> 8x8
        x = F.relu(self.bn3(self.conv3(x)))  # -> 8x8
        return x  # Output shape: [B, 32, 8, 8]

class TransitionModel(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Conv2d(2, 16, 1),  # First go to intermediate dimension
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, hidden_dim, 1),  # Then to full hidden_dim
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Transition model
        self.transition = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim)
        )
        
    def forward(self, state, action):
        """
        Args:
            state: [B, hidden_dim, 8, 8] - Current state representation
            action: [B, 2] - (dx, dy) action
        """
        B, _, H, W = state.shape
        
        # Expand action to spatial dimensions and embed
        action = action.view(B, 2, 1, 1).expand(-1, -1, H, W)
        action_embedding = self.action_embed(action)
        
        # Combine state and action
        combined = torch.cat([state, action_embedding], dim=1)
        
        # Predict state change
        delta = self.transition(combined)
        
        # Residual connection
        next_state = state + delta
        
        return next_state

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(input_channels=2)
        self.predictor = TransitionModel(hidden_dim=32)
        
    def forward(self, states, actions):
        """
        Args:
            states: [B, 1, 2, 65, 65] - Initial state only
            actions: [B, T-1, 2] - Sequence of T-1 actions
        Returns:
            predictions: [B, T, 32, 8, 8] - Predicted representations
        """
        B, _, _, H, W = states.shape
        T = actions.shape[1] + 1
        
        # Get initial state encoding
        curr_state = self.encoder(states.squeeze(1))
        predictions = [curr_state]
        
        # Predict future states
        for t in range(T-1):
            curr_state = self.predictor(curr_state, actions[:, t])
            predictions.append(curr_state)
            
        predictions = torch.stack(predictions, dim=1)
        return predictions

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output