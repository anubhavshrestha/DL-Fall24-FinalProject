import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

##### DO NOT TOUCH ######################################################################################################
##### DO NOT TOUCH ######################################################################################################
##### DO NOT TOUCH ######################################################################################################
##### DO NOT TOUCH ######################################################################################################
##### DO NOT TOUCH ######################################################################################################
##### DO NOT TOUCH ######################################################################################################


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


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


##### DO NOT TOUCH ######################################################################################################
##### DO NOT TOUCH ######################################################################################################
##### DO NOT TOUCH ######################################################################################################
##### DO NOT TOUCH ######################################################################################################
##### DO NOT TOUCH ######################################################################################################
##### DO NOT TOUCH ######################################################################################################



class VICRegLoss(nn.Module):
    def __init__(self, lambda_param=25.0, mu_param=25.0, nu_param=1.0):
        super().__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
    
    def off_diagonal(self, x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
    
    def forward(self, z_a, z_b):
        N = z_a.shape[0]
        D = z_a.shape[1]
        
        # Invariance loss
        sim_loss = F.mse_loss(z_a, z_b)
        
        # Variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
        
        # Covariance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        
        cov_z_a = (z_a.T @ z_a) / (N - 1)
        cov_z_b = (z_b.T @ z_b) / (N - 1)
        
        cov_loss = (self.off_diagonal(cov_z_a).pow_(2).sum() / D +
                   self.off_diagonal(cov_z_b).pow_(2).sum() / D)
        
        total_loss = (self.lambda_param * sim_loss + 
                     self.mu_param * std_loss + 
                     self.nu_param * cov_loss)
        
        losses = {
            'sim_loss': sim_loss.item(),
            'std_loss': std_loss.item(),
            'cov_loss': cov_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, losses

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param=0.005):
        super().__init__()
        self.lambda_param = lambda_param
        
    def forward(self, z_a, z_b):
        N = z_a.shape[0]
        D = z_a.shape[1]
        
        # Normalize representations
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-6)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-6)
        
        # Cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N
        
        # Loss computation
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = torch.triu(c.pow(2), diagonal=1).sum() + torch.tril(c.pow(2), diagonal=-1).sum()
        
        total_loss = on_diag + self.lambda_param * off_diag
        
        losses = {
            'on_diag_loss': on_diag.item(),
            'off_diag_loss': off_diag.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, losses

class NormalizedCombinedLoss(nn.Module):
    def __init__(self, 
                 vicreg_sim_coef=25.0,
                 vicreg_std_coef=25.0,
                 vicreg_cov_coef=1.0,
                 barlow_lambda=0.005,
                 loss_weight=0.5):
        super().__init__()
        self.vicreg_sim_coef = vicreg_sim_coef
        self.vicreg_std_coef = vicreg_std_coef
        self.vicreg_cov_coef = vicreg_cov_coef
        self.barlow_lambda = barlow_lambda
        self.loss_weight = loss_weight
        
        # Running statistics for normalization
        self.register_buffer('vicreg_mean', torch.tensor(0.0))
        self.register_buffer('barlow_mean', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        self.momentum = 0.9
    
    def off_diagonal(self, x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
    
    def update_means(self, vicreg_loss, barlow_loss):
        with torch.no_grad():
            if self.count == 0:
                self.vicreg_mean = vicreg_loss.detach()
                self.barlow_mean = barlow_loss.detach()
            else:
                self.vicreg_mean = self.momentum * self.vicreg_mean + (1 - self.momentum) * vicreg_loss.detach()
                self.barlow_mean = self.momentum * self.barlow_mean + (1 - self.momentum) * barlow_loss.detach()
            self.count += 1
    
    def forward(self, z_a, z_b):
        N = z_a.shape[0]
        D = z_a.shape[1]
        
        # VICReg components
        sim_loss = F.mse_loss(z_a, z_b)
        
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
        
        z_a_c = z_a - z_a.mean(dim=0)
        z_b_c = z_b - z_b.mean(dim=0)
        
        cov_z_a = (z_a_c.T @ z_a_c) / (N - 1)
        cov_z_b = (z_b_c.T @ z_b_c) / (N - 1)
        
        vicreg_cov_loss = (self.off_diagonal(cov_z_a).pow_(2).sum() / D +
                          self.off_diagonal(cov_z_b).pow_(2).sum() / D)
        
        # Compute full VICReg loss
        vicreg_loss = (self.vicreg_sim_coef * sim_loss +
                      self.vicreg_std_coef * std_loss +
                      self.vicreg_cov_coef * vicreg_cov_loss)
        
        # Barlow Twins components
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-6)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-6)
        
        c = torch.mm(z_a_norm.T, z_b_norm) / N
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = torch.triu(c.pow(2), diagonal=1).sum() + torch.tril(c.pow(2), diagonal=-1).sum()
        barlow_loss = on_diag + self.barlow_lambda * off_diag
        
        # Update running means
        self.update_means(vicreg_loss, barlow_loss)
        
        # Normalize losses using running means
        eps = 1e-6
        if self.count > 0:
            vicreg_loss_norm = vicreg_loss / (self.vicreg_mean + eps)
            barlow_loss_norm = barlow_loss / (self.barlow_mean + eps)
        else:
            vicreg_loss_norm = vicreg_loss
            barlow_loss_norm = barlow_loss
        
        # Combine normalized losses
        total_loss = self.loss_weight * vicreg_loss_norm + (1 - self.loss_weight) * barlow_loss_norm
        
        component_losses = {
            'total_loss': total_loss.item(),
            'vicreg_loss': vicreg_loss.item(),
            'vicreg_norm': vicreg_loss_norm.item(),
            'barlow_loss': barlow_loss.item(),
            'barlow_norm': barlow_loss_norm.item(),
            'vicreg_mean': self.vicreg_mean.item(),
            'barlow_mean': self.barlow_mean.item()
        }
        
        return total_loss, component_losses

class Encoder(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        # First conv: 65x65 -> 22x22
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=5, stride=3, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second conv: 22x22 -> 8x8
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.repr_dim = 32
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # -> 22x22
        x = F.relu(self.bn2(self.conv2(x)))  # -> 8x8
        return x  # Output shape: [B, 32, 8, 8]

class TransitionModel(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Conv2d(2, hidden_dim // 2, 1),  # First go to intermediate dimension
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 1),  # Then to full hidden_dim
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
    def __init__(self, loss_type='vicreg', lambda_param=25.0, mu_param=25.0, nu_param=1.0, encoder='small', predictor='small'):
        super().__init__()
        if encoder == 'small':
            self.encoder = Encoder()
        else:
            pass 

        if predictor == 'small':
            self.predictor = TransitionModel()
        elif predictor == 'medium':
            self.predictor = TransitionModelMedium()
        elif predictor == 'large':
            self.predictor = TransitionModelLarge()
        elif predictor == 'xl':
            self.predictor = TransitionModelXL()

        self.loss_type = loss_type

        
        # Initialize appropriate loss function
        if loss_type == 'vicreg':
            self.criterion = VICRegLoss(lambda_param, mu_param, nu_param)
        elif loss_type == 'barlow':
            self.criterion = BarlowTwinsLoss(lambda_param)
        elif loss_type == 'both':
            self.criterion = NormalizedCombinedLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        self.repr_dim = 32 * 8 * 8
    
    def forward(self, states, actions):
        """
        Forward pass for evaluation
        Args:
            states: [B, T, 2, H, W] - Full sequence of states
            actions: [B, T-1, 2] - Sequence of T-1 actions
        Returns:
            predictions: [B, T, D] - Predicted representations
        """
        init_states = states[:, 0:1]
        predictions = self.forward_prediction(init_states, actions)
        B, T, C, H, W = predictions.shape
        predictions = predictions.view(B, T, -1)
        return predictions
    
    def forward_prediction(self, states, actions):
        """
        Forward pass for prediction of future states
        Args:
            states: [B, 1, 2, H, W] - Initial state only
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
    
    def training_step(self, batch):
        states = batch.states
        actions = batch.actions
        
        # Get initial state
        init_states = states[:, 0:1]
        
        # Get predictions for all steps
        predictions = self.forward_prediction(init_states, actions)
        
        # Initialize losses
        total_loss = 0.0
        accumulated_losses = None
        
        # Compute loss for each timestep
        for t in range(actions.shape[1]):
            pred_state = predictions[:, t+1]
            target_obs = states[:, t+1]
            
            # Get target encoding and compute loss
            target_state = self.encoder(target_obs)
            pred_flat = pred_state.flatten(start_dim=1)
            target_flat = target_state.flatten(start_dim=1)
            
            loss, component_losses = self.criterion(pred_flat, target_flat)
            total_loss += loss
            
            # Initialize accumulated_losses on first iteration using actual component names
            if accumulated_losses is None:
                accumulated_losses = {k: 0.0 for k in component_losses}
            
            # Accumulate component losses
            for k in component_losses:
                accumulated_losses[k] += component_losses[k]
        
        # Average losses over timesteps
        for k in accumulated_losses:
            accumulated_losses[k] /= actions.shape[1]
        
        return total_loss / actions.shape[1], predictions, accumulated_losses

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

class TransitionModelMedium(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Action embedding with more capacity
        self.action_embed = nn.Sequential(
            nn.Conv2d(2, hidden_dim // 2, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),  # Additional conv layer
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Transition model with residual blocks
        self.transition = nn.Sequential(
            # First residual block
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            
            # Second residual block
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim)
        )
        
    def forward(self, state, action):
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

class TransitionModelLarge(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Action embedding with increased capacity
        self.action_embed = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU()
        )
        
        # Initial projection
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim * 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU()
        )
        
        # Transition model with deeper residual blocks
        self.transition_blocks = nn.ModuleList([
            # Three residual blocks with bottleneck
            nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim * 2, 1),
                nn.BatchNorm2d(hidden_dim * 2)
            ) for _ in range(3)
        ])
        
        # Final projection
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim)
        )
        
    def forward(self, state, action):
        B, _, H, W = state.shape
        
        # Expand action to spatial dimensions and embed
        action = action.view(B, 2, 1, 1).expand(-1, -1, H, W)
        action_embedding = self.action_embed(action)
        
        # Combine state and action
        combined = torch.cat([state, action_embedding], dim=1)
        x = self.proj(combined)
        
        # Apply residual blocks
        for block in self.transition_blocks:
            x = F.relu(x + block(x))
        
        # Final projection to state space
        delta = self.final(x)
        
        # Residual connection
        next_state = state + delta
        
        return next_state

class TransitionModelXL(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Sophisticated action embedding with attention
        self.action_embed = nn.Sequential(
            nn.Conv2d(2, hidden_dim * 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU()
        )
        
        # Initial projection with increased channels
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim * 5, hidden_dim * 4, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU()
        )
        
        # Multi-scale processing branches
        self.branches = nn.ModuleList([
            # Standard resolution branch
            nn.Sequential(
                nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, padding=1),
                nn.BatchNorm2d(hidden_dim * 2),
                nn.ReLU(),
                nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
                nn.BatchNorm2d(hidden_dim * 2)
            ),
            # Reduced resolution branch
            nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, padding=1),
                nn.BatchNorm2d(hidden_dim * 2),
                nn.ReLU(),
                nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
                nn.BatchNorm2d(hidden_dim * 2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        ])
        
        # Transition blocks with attention
        self.transition_blocks = nn.ModuleList([
            nn.Sequential(
                # First conv + attention
                nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 1),
                nn.BatchNorm2d(hidden_dim * 2),
                nn.ReLU(),
                # Spatial attention
                nn.Conv2d(hidden_dim * 2, 1, 1),
                nn.Sigmoid(),
                # Second conv
                nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 1),
                nn.BatchNorm2d(hidden_dim * 4)
            ) for _ in range(4)
        ])
        
        # Final projection layers
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim)
        )
        
    def forward(self, state, action):
        B, _, H, W = state.shape
        
        # Expand action to spatial dimensions and embed
        action = action.view(B, 2, 1, 1).expand(-1, -1, H, W)
        action_embedding = self.action_embed(action)
        
        # Combine state and action
        combined = torch.cat([state, action_embedding], dim=1)
        x = self.proj(combined)
        
        # Multi-scale processing
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        
        # Apply transition blocks with attention
        for block in self.transition_blocks:
            attention = block[:4](x)  # Extract attention weights
            x_transformed = block[4:](x)  # Transform features
            x = F.relu(x + attention * x_transformed)
        
        # Final projection to state space
        delta = self.final(x)
        
        # Residual connection
        next_state = state + delta
        
        return next_state