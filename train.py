import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from tqdm import tqdm
from dataset import WallSample, create_wall_dataloader
from models import Encoder, TransitionModel, WorldModel, Prober
import wandb
from evaluator import ProbingEvaluator

def load_data(device):
    data_path = "/drive_reader/as16386/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds

def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)


    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")
    
    return avg_losses
    

class VICRegLoss(nn.Module):
    def __init__(self, lambda_param=25.0, mu_param=25.0, nu_param=1.0):
        super().__init__()
        self.lambda_param = lambda_param  # invariance loss coefficient
        self.mu_param = mu_param         # variance loss coefficient
        self.nu_param = nu_param         # covariance loss coefficient
    
    def off_diagonal(self, x):
        """Return off-diagonal elements of a square matrix"""
        n = x.shape[0]
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
    
    def forward(self, z_a, z_b):
        """
        Args:
            z_a, z_b: Batch of representations [N, D]
        Returns:
            total_loss: Combined VICReg loss
            losses: Dictionary containing individual loss components
        """
        N = z_a.shape[0]  # batch size
        D = z_a.shape[1]  # dimension
        
        # Invariance loss (MSE)
        sim_loss = F.mse_loss(z_a, z_b)
        
        # Variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = (torch.mean(F.relu(1 - std_z_a)) + 
                   torch.mean(F.relu(1 - std_z_b)))
        
        # Covariance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        
        cov_z_a = (z_a.T @ z_a) / (N - 1)
        cov_z_b = (z_b.T @ z_b) / (N - 1)
        
        cov_loss = (self.off_diagonal(cov_z_a).pow_(2).sum() / D +
                   self.off_diagonal(cov_z_b).pow_(2).sum() / D)
        
        # Combine losses
        total_loss = (self.lambda_param * sim_loss + 
                     self.mu_param * std_loss + 
                     self.nu_param * cov_loss)
        
        # Return individual losses for logging
        losses = {
            'sim_loss': sim_loss.item(),
            'std_loss': std_loss.item(),
            'cov_loss': cov_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, losses

# class WorldModelVICReg(nn.Module):
#     def __init__(self, lambda_param=25.0, mu_param=25.0, nu_param=1.0):
#         super().__init__()
#         self.encoder = Encoder(input_channels=2)
#         self.predictor = TransitionModel(hidden_dim=32)
#         self.criterion = VICRegLoss(lambda_param, mu_param, nu_param)
#         self.repr_dim = 32 * 8 * 8
    
#     def compute_vicreg_loss(self, pred_state, target_obs):
#         """Compute VICReg loss between predicted and encoded target states"""
#         # Get target encoding
#         target_state = self.encoder(target_obs)
        
#         # Flatten spatial dimensions: [B, 32, 8, 8] -> [B, 2048]
#         pred_flat = pred_state.flatten(start_dim=1)
#         target_flat = target_state.flatten(start_dim=1)
        
#         # Compute VICReg losses
#         total_loss, component_losses = self.criterion(pred_flat, target_flat)
#         return total_loss, component_losses
    
#     def training_step(self, batch):
#         states = batch.states
#         actions = batch.actions
        
#         # Get initial state
#         init_states = states[:, 0:1]
        
#         # Get predictions for all steps
#         predictions = self.forward_prediction(init_states, actions)
        
#         # Initialize losses
#         total_loss = 0.0
#         accumulated_losses = {
#             'sim_loss': 0.0,
#             'std_loss': 0.0,
#             'cov_loss': 0.0,
#             'total_loss': 0.0
#         }
        
#         # Compute VICReg loss for each timestep
#         for t in range(actions.shape[1]):
#             pred_state = predictions[:, t+1]
#             target_obs = states[:, t+1]
            
#             loss, component_losses = self.compute_vicreg_loss(pred_state, target_obs)
#             total_loss += loss
            
#             # Accumulate component losses
#             for k in accumulated_losses:
#                 accumulated_losses[k] += component_losses[k]
        
#         # Average losses over timesteps
#         for k in accumulated_losses:
#             accumulated_losses[k] /= actions.shape[1]
        
#         return total_loss / actions.shape[1], predictions, accumulated_losses


#     def forward_prediction(self, states, actions):
#         """
#         Forward pass for prediction of future states
#         Args:
#             states: [B, 1, 2, 65, 65] - Initial state only
#             actions: [B, T-1, 2] - Sequence of T-1 actions
#         Returns:
#             predictions: [B, T, 32, 8, 8] - Predicted representations
#         """
#         B, _, _, H, W = states.shape
#         T = actions.shape[1] + 1
        
#         # Get initial state encoding
#         curr_state = self.encoder(states.squeeze(1))
#         predictions = [curr_state]
        
#         # Predict future states
#         for t in range(T-1):
#             curr_state = self.predictor(curr_state, actions[:, t])
#             predictions.append(curr_state)
            
#         predictions = torch.stack(predictions, dim=1)
#         return predictions

class WorldModelVICReg(nn.Module):
    def __init__(self, lambda_param=25.0, mu_param=25.0, nu_param=1.0):
        super().__init__()
        self.encoder = Encoder(input_channels=2)
        self.predictor = TransitionModel(hidden_dim=32)
        self.criterion = VICRegLoss(lambda_param, mu_param, nu_param)
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
        # Get initial state only
        init_states = states[:, 0:1]  # [B, 1, 2, H, W]
        
        # Use forward_prediction for consistency
        predictions = self.forward_prediction(init_states, actions)
        
        # Reshape predictions to [B, T, D]
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
    
    def compute_vicreg_loss(self, pred_state, target_obs):
        """Compute VICReg loss between predicted and encoded target states"""
        # Get target encoding
        target_state = self.encoder(target_obs)
        
        # Flatten spatial dimensions: [B, 32, 8, 8] -> [B, 2048]
        pred_flat = pred_state.flatten(start_dim=1)
        target_flat = target_state.flatten(start_dim=1)
        
        # Compute VICReg losses
        total_loss, component_losses = self.criterion(pred_flat, target_flat)
        return total_loss, component_losses
    
    def training_step(self, batch):
        states = batch.states
        actions = batch.actions
        
        # Get initial state
        init_states = states[:, 0:1]
        
        # Get predictions for all steps
        predictions = self.forward_prediction(init_states, actions)
        
        # Initialize losses
        total_loss = 0.0
        accumulated_losses = {
            'sim_loss': 0.0,
            'std_loss': 0.0,
            'cov_loss': 0.0,
            'total_loss': 0.0
        }
        
        # Compute VICReg loss for each timestep
        for t in range(actions.shape[1]):
            pred_state = predictions[:, t+1]
            target_obs = states[:, t+1]
            
            loss, component_losses = self.compute_vicreg_loss(pred_state, target_obs)
            total_loss += loss
            
            # Accumulate component losses
            for k in accumulated_losses:
                accumulated_losses[k] += component_losses[k]
        
        # Average losses over timesteps
        for k in accumulated_losses:
            accumulated_losses[k] /= actions.shape[1]
        
        return total_loss / actions.shape[1], predictions, accumulated_losses

import os 


import wandb

class WorldModelTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=3e-5, 
                 device='cuda', log_dir='runs', probe_train_data=None, probe_val_data=None):
        self.model = model.to(device)
        print(f"Using device: {device}")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.probe_train_data = probe_train_data
        self.probe_val_data = probe_val_data
        
        # Initialize TensorBoard writer
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(f'{log_dir}/{current_time}')
        
        # Save hyperparameters
        self.writer.add_text('hyperparameters', f'''
        learning_rate: {learning_rate}
        batch_size: {train_loader.batch_size}
        model_channels: {model.encoder.repr_dim}
        ''')

        # Initialize WandB
        wandb.init(
            project="world-model-vicreg",  # Replace with your project name
            config={
                "learning_rate": learning_rate,
                "batch_size": train_loader.batch_size,
                "num_epochs": 100,  # Default value; update as necessary
                "device": device,
            }
        )
        wandb.watch(model, log="all", log_freq=10)

    def validate(self, epoch):
        self.model.eval()
        total_val_loss = 0.0
        val_losses = {
            'total_loss': 0.0,
            'sim_loss': 0.0,
            'std_loss': 0.0,
            'cov_loss': 0.0
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                states = batch.states.to(self.device)
                actions = batch.actions.to(self.device)
                
                loss, _, component_losses = self.model.training_step(
                    WallSample(states=states, actions=actions, locations=batch.locations)
                )
                
                total_val_loss += loss.item()
                for k in val_losses:
                    val_losses[k] += component_losses[k]
        
        # Average the losses
        num_batches = len(self.val_loader)
        total_val_loss /= num_batches
        for k in val_losses:
            val_losses[k] /= num_batches
            
        return total_val_loss, val_losses

    def train_epoch(self, epoch):
        self.model.train()
        total_train_loss = 0.0
        train_losses = {
            'total_loss': 0.0,
            'sim_loss': 0.0,
            'std_loss': 0.0,
            'cov_loss': 0.0
        }
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            self.optimizer.zero_grad()
            
            states = batch.states.to(self.device)
            actions = batch.actions.to(self.device)
            
            loss, _, component_losses = self.model.training_step(
                WallSample(states=states, actions=actions, locations=batch.locations)
            )
            
            # Add backpropagation steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            # Update running losses
            total_train_loss += loss.item()
            for k in train_losses:
                train_losses[k] += component_losses[k]
        
        num_batches = len(self.train_loader)
        total_train_loss /= num_batches
        for k in train_losses:
            train_losses[k] /= num_batches
            
        return total_train_loss, train_losses  

    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        os.makedirs('cnn_based_checkpoints', exist_ok=True)
        path = f'cnn_based_checkpoints/checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss, train_losses = self.train_epoch(epoch)
            val_loss, val_losses = self.validate(epoch)
            
            # Log metrics to WandB
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{k}": v for k, v in train_losses.items()},
                **{f"val_{k}": v for k, v in val_losses.items()},
            })

            print("\nEpoch Summary:")
            print("Training Losses:")
            for k, v in train_losses.items():
                print(f"{k}: {v:.4f}")
            print("\nValidation Losses:")
            for k, v in val_losses.items():
                print(f"{k}: {v:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                print("New best model saved!")
            
            losses = evaluate_model(self.device, self.model, self.probe_train_data, self.probe_val_data)
            normal_loss = losses['normal']
            wall_loss = losses['wall']
            wandb.log({
                "Normal_loss:": normal_loss,
                "wall_loss": wall_loss
            })
            

        # Finish WandB session
        wandb.finish()


def create_train_val_loaders(data_path, train_samples=10000, val_samples=2000):

    train_loader = create_wall_dataloader(
        data_path=data_path,
        probing=False,
        device="cuda",
        batch_size=128,
        train=True,
        num_samples=train_samples
    )
    
    val_loader = create_wall_dataloader(
        data_path=data_path,
        probing=False,
        device="cuda",
        batch_size=128,
        train=False,
        num_samples=val_samples
    )
    
    return train_loader, val_loader

model = WorldModelVICReg(
    lambda_param=25.0,
    mu_param=25.0,
    nu_param=1.0
)

train_loader, val_loader = create_train_val_loaders(
    data_path="/drive_reader/as16386/DL24FA/train",
    train_samples=None,
    val_samples=None
)
probe_train_ds, probe_val_ds = load_data("cuda")

trainer = WorldModelTrainer(model = model, train_loader = train_loader, val_loader = val_loader, probe_train_data=probe_train_ds, probe_val_data=probe_val_ds)
trainer.train(num_epochs=100)

