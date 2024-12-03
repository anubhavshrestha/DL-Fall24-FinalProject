from typing import List, Tuple, Optional
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import torchvision.transforms as T


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, channels: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)       # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)      # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)     # 8x8 -> 4x4
        
        # Input features to fc layer: 256 channels * 4 * 4 = 4096
        self.fc = nn.Linear(4096, 256)
        self.repr_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        x = F.relu(self.conv1(x))  # [B, 32, 32, 32]
        x = F.relu(self.conv2(x))  # [B, 64, 16, 16]
        x = F.relu(self.conv3(x))  # [B, 128, 8, 8]
        x = F.relu(self.conv4(x))  # [B, 256, 4, 4]
        
        x = x.reshape(x.size(0), -1)  # [B, 256*4*4]
        x = self.fc(x)               # [B, 256]
        return x


class Predictor(nn.Module):
    def __init__(self, state_dim: int = 256, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class JEPAVICReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()
        self.repr_dim = self.encoder.repr_dim

        # VICReg hyperparameters
        self.sim_coeff = 25.0  # lambda
        self.std_coeff = 25.0  # mu
        self.cov_coeff = 1.0   # nu
        
        # Data augmentation
        self.augment = T.Compose([
            T.RandomRotation(10),
            T.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
        ])

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass for evaluation"""
        batch_size, seq_len = states.shape[:2]
        predictions = []
        
        # Initial state
        current_state = self.encoder(states[:, 0])
        predictions.append(current_state)
        
        # Predict future states
        for t in range(seq_len - 1):
            current_state = self.predictor(current_state, actions[:, t])
            predictions.append(current_state)
            
        return torch.stack(predictions, dim=1)

    @staticmethod
    def off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """Return a flattened view of the off-diagonal elements of a square matrix"""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def compute_vicreg_loss(
        self, 
        z_a: torch.Tensor, 
        z_b: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VICReg losses"""
        # Invariance loss
        sim_loss = F.mse_loss(z_a, z_b)

        # Variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))

        # Covariance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (batch_size - 1)
        cov_z_b = (z_b.T @ z_b) / (batch_size - 1)
        cov_loss = (
            self.off_diagonal(cov_z_a).pow_(2).sum() / self.repr_dim +
            self.off_diagonal(cov_z_b).pow_(2).sum() / self.repr_dim
        )
        
        return sim_loss, std_loss, cov_loss

    def training_step(
        self, 
        states: torch.Tensor,
        actions: torch.Tensor,
        target_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Training step incorporating both JEPA prediction and VICReg regularization"""
        batch_size, seq_len = states.shape[:2]
        
        # Create augmented views
        states_aug = torch.stack([self.augment(states[:, t]) for t in range(seq_len)], dim=1)
        
        # Initialize losses
        pred_loss = 0
        sim_loss = 0
        std_loss = 0
        cov_loss = 0
        
        # Initial encoded states
        z_t = self.encoder(states[:, 0])
        z_t_aug = self.encoder(states_aug[:, 0])
        
        # Apply VICReg to initial encodings
        sim, std, cov = self.compute_vicreg_loss(z_t, z_t_aug, batch_size)
        sim_loss += sim
        std_loss += std
        cov_loss += cov
        
        for t in range(seq_len - 1):
            # Predict next state representation
            z_pred = self.predictor(z_t, actions[:, t])
            
            # Get target representation
            if target_states is not None:
                # Teacher forcing
                z_target = self.encoder(target_states[:, t + 1])
                z_t = z_target
            else:
                # Use predicted state
                z_target = self.encoder(states[:, t + 1])
                z_t = z_pred
            
            # Prediction loss
            pred_loss += F.mse_loss(z_pred, z_target)
            
            # VICReg between predicted and target
            sim, std, cov = self.compute_vicreg_loss(z_pred, z_target, batch_size)
            sim_loss += sim
            std_loss += std
            cov_loss += cov
        
        # Combine losses
        total_loss = (
            pred_loss + 
            self.sim_coeff * sim_loss + 
            self.std_coeff * std_loss + 
            self.cov_coeff * cov_loss
        )
        
        losses = {
            'pred_loss': pred_loss.item(),
            'sim_loss': sim_loss.item(),
            'std_loss': std_loss.item(),
            'cov_loss': cov_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, losses


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