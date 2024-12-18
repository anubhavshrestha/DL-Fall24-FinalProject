import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from dataset import create_wall_dataloader, WallDataset
from models import WorldModel
from evaluator import ProbingEvaluator
from normalizer import Normalizer

class PathVisualizer(ProbingEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_predictions = []
        self.stored_targets = []
        
    def evaluate_pred_prober(self, prober, val_ds, prefix=""):
        self.stored_predictions = []
        self.stored_targets = []
        
        model = self.model
        probing_losses = []
        prober.eval()

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_ds, desc="Evaluating predictions")):
                # Forward pass through model
                init_states = batch.states[:, 0:1]
                pred_encs = model(states=init_states, actions=batch.actions)
                pred_encs = pred_encs.transpose(0, 1)

                # Get target locations and normalize
                target = getattr(batch, "locations").cuda()
                target = self.normalizer.normalize_location(target)

                # Get predictions
                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
                
                # Store normalized predictions and targets
                self.stored_predictions.append(pred_locs.cpu())
                self.stored_targets.append(target.cpu())
                
                # Calculate losses
                losses = self.location_losses(pred_locs, target)
                probing_losses.append(losses.cpu())

        losses_t = torch.stack(probing_losses, dim=0).mean(dim=0)
        losses_t = self.normalizer.unnormalize_mse(losses_t)
        losses_t = losses_t.mean(dim=-1)
        average_eval_loss = losses_t.mean().item()

        return average_eval_loss
    
    def get_stored_trajectories(self, num_samples: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get random samples of predicted and target trajectories."""
        all_preds = torch.cat(self.stored_predictions, dim=0)
        all_targets = torch.cat(self.stored_targets, dim=0)
        
        # Unnormalize the predictions and targets
        all_preds = self.normalizer.unnormalize_location(all_preds)
        all_targets = self.normalizer.unnormalize_location(all_targets)
        
        # Randomly sample trajectories
        num_trajectories = all_preds.shape[0]
        indices = torch.randperm(num_trajectories)[:num_samples]
        
        return all_preds[indices], all_targets[indices]

def visualize_trajectories(predictions: torch.Tensor, 
                         targets: torch.Tensor,
                         title: str = "Predicted vs Actual Trajectories"):
    """
    Visualize multiple trajectories.
    
    Args:
        predictions: Tensor of shape (N, T, 2) where N is number of samples and T is timesteps
        targets: Tensor of shape (N, T, 2)
        title: Plot title
    """
    num_samples = predictions.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(min(num_samples, 4)):
        ax = axes[i]
        pred_traj = predictions[i].numpy()
        target_traj = targets[i].numpy()
        
        # Plot trajectories
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-', label='Predicted', alpha=0.7)
        ax.plot(target_traj[:, 0], target_traj[:, 1], 'b-', label='Actual', alpha=0.7)
        
        # Plot start and end points
        ax.scatter(pred_traj[0, 0], pred_traj[0, 1], c='red', marker='o', s=100, label='Start')
        ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], c='red', marker='x', s=100, label='End')
        ax.scatter(target_traj[0, 0], target_traj[0, 1], c='blue', marker='o', s=100)
        ax.scatter(target_traj[-1, 0], target_traj[-1, 1], c='blue', marker='x', s=100)
        
        # Add arrows to show direction
        arrow_indices = np.linspace(0, len(pred_traj)-1, 5).astype(int)[:-1]
        for idx in arrow_indices:
            # Predicted trajectory arrows
            ax.arrow(pred_traj[idx, 0], pred_traj[idx, 1],
                    (pred_traj[idx+1, 0] - pred_traj[idx, 0])*0.2,
                    (pred_traj[idx+1, 1] - pred_traj[idx, 1])*0.2,
                    head_width=0.5, head_length=0.8, fc='r', ec='r', alpha=0.5)
            # Actual trajectory arrows
            ax.arrow(target_traj[idx, 0], target_traj[idx, 1],
                    (target_traj[idx+1, 0] - target_traj[idx, 0])*0.2,
                    (target_traj[idx+1, 1] - target_traj[idx, 1])*0.2,
                    head_width=0.5, head_length=0.8, fc='b', ec='b', alpha=0.5)
        
        ax.set_title(f'Trajectory {i+1}')
        ax.grid(True)
        ax.legend()
        ax.set_aspect('equal')
    
    plt.suptitle(title, y=1.02, size=16)
    plt.tight_layout()
    return fig

def main(checkpoint_path: str):
    # Load data
    probe_train_ds = create_wall_dataloader(
        data_path="/drive_reader/as16386/DL24FA/probe_normal/train",
        probing=True,
        device="cuda",
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path="/drive_reader/as16386/DL24FA/probe_normal/val",
        probing=True,
        device="cuda",
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path="/drive_reader/as16386/DL24FA/probe_wall/val",
        probing=True,
        device="cuda",
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}
    
    # Initialize model
    model = WorldModel(loss_type="vicreg")  # Adjust parameters as needed
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    
    # Initialize visualizer and train prober
    visualizer = PathVisualizer(
        device="cuda",
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
    )
    
    prober = visualizer.train_pred_prober()
    
    # Evaluate and get trajectories for both normal and wall scenarios
    print("Evaluating normal scenarios...")
    visualizer.evaluate_pred_prober(prober, probe_val_ds["normal"], "normal")
    normal_preds, normal_targets = visualizer.get_stored_trajectories(4)
    
    print("Evaluating wall scenarios...")
    visualizer.evaluate_pred_prober(prober, probe_val_ds["wall"], "wall")
    wall_preds, wall_targets = visualizer.get_stored_trajectories(4)
    
    # Visualize trajectories
    normal_fig = visualize_trajectories(normal_preds, normal_targets, 
                                      "Normal Scenarios: Predicted vs Actual Trajectories")
    wall_fig = visualize_trajectories(wall_preds, wall_targets,
                                    "Wall Scenarios: Predicted vs Actual Trajectories")
    
    return normal_fig, wall_fig

# Usage example:
if __name__ == "__main__":
    checkpoint_path = "path/to/your/checkpoint.pt"  # Replace with actual checkpoint path
    normal_fig, wall_fig = main(checkpoint_path)
    plt.show()