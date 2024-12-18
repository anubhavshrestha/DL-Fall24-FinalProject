import argparse
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import wandb
import os
import random
from models import WorldModel
from dataset import WallSample, create_wall_dataloader
from evaluator import ProbingEvaluator
import numpy as np 

def set_all_seeds(seed):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train World Model')
    
    # Existing arguments
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')

    parser.add_argument('--loss_type', type=str, default='vicreg', choices=['vicreg', 'barlow', 'both'],
                      help='Loss function to use (default: vicreg)')
    parser.add_argument('--use_validation', action='store_true',
                      help='Whether to perform validation (default: False)')
    parser.add_argument('--wandb_project', type=str, default='world-model',
                      help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='WandB entity/username')
    parser.add_argument('--save_frequency', type=int, default=1,
                      help='Save checkpoints every N epochs (default: 1)')
    parser.add_argument('--wandb_name', type=str, required=True,
                      help='Name for this run (used for checkpoint directory)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3e-,
                      help='Learning rate (default: 3e-5)')
    parser.add_argument('--weight_decay', type=float, default=0,
                      help='Weight decay (default: 1e-5)')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=200,
                      help='Number of epochs to train (default: 200)')
    parser.add_argument('--prober_lr', type=float, default=1e-3,
                      help='Learning rate for prober (default: 1e-3)')
    parser.add_argument('--scheduler', type=str, default='none', 
                      choices=['cosine', 'onecycle', 'none'],
                      help='Learning rate scheduler (default: none)')
    parser.add_argument('--encoder', type=str, default='small', 
                      choices=['small'],
                      help='Encoder Type (if multiple options available)')  
    parser.add_argument('--predictor', type=str, default='small', 
                      choices=['small', 'medium', 'large', 'xl'],
                      help='Predictor Type (if multiple options available)') 

    # Loss function hyperparameters
    parser.add_argument('--lambda_param', type=float, default=None,
                      help='Lambda parameter (default: 25.0 for VICReg, 0.005 for Barlow)')
    parser.add_argument('--mu_param', type=float, default=None,
                      help='Mu parameter for VICReg (default: 25.0)')
    parser.add_argument('--nu_param', type=float, default=None,
                      help='Nu parameter for VICReg (default: 1.0)')

    return parser.parse_args()


def load_data(device):
    """
    Load probe datasets for evaluation.
    Returns train dataset and validation datasets (normal and wall scenarios)
    """
    data_path = "/drive_reader/as16386/DL24FA"

    # Load probe training data
    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    # Load probe validation data for both scenarios
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
    """
    Evaluate model using probe datasets.
    Returns average losses for normal and wall scenarios.
    """
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
        learning_rate=args.prober_lr 
    )

    # Train and evaluate prober
    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")
    
    return avg_losses

class WorldModelTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=3e-4, 
             device='cuda', probe_train_data=None, probe_val_data=None,
             use_validation=False, wandb_project=None, wandb_name=None,
             checkpoint_dir='checkpoints', save_frequency=1, scheduler_type='cosine', seed=42):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.probe_train_data = probe_train_data
        self.probe_val_data = probe_val_data
        self.use_validation = use_validation
        self.checkpoint_dir = os.path.join(checkpoint_dir, wandb_name)
        self.save_frequency = save_frequency
        self.seed = seed 
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5
        )
        print('parameter count', sum(p.numel() for p in model.parameters()))

        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.num_epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=learning_rate * 10,
                epochs=args.num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.2  # 40% of training for warmup
            )
        else:
            self.scheduler = None

            
        # Initialize WandB
        if wandb_project:
            wandb.init(
                project=wandb_project,
                config={
                    "learning_rate": learning_rate,
                    "batch_size": train_loader.batch_size,
                    "loss_type": model.loss_type,
                    "use_validation": use_validation,
                    "seed": seed
                }
            )
            # Log model architecture
            wandb.watch(model, log="all", log_freq=10)

    def validate(self, epoch):
        """Run validation loop and compute losses."""
        self.model.eval()
        total_val_loss = 0.0
        val_losses = None
        
        with torch.no_grad():
            for batch in self.val_loader:
                states = batch.states.to(self.device)
                actions = batch.actions.to(self.device)
                
                loss, _, component_losses = self.model.training_step(
                    WallSample(states=states, actions=actions, locations=batch.locations)
                )
                
                # Initialize val_losses on first batch
                if val_losses is None:
                    val_losses = {k: 0.0 for k in component_losses}
                
                total_val_loss += loss.item()
                for k in component_losses:
                    val_losses[k] += component_losses[k]
        
        # Average the losses
        num_batches = len(self.val_loader)
        total_val_loss /= num_batches
        for k in val_losses:
            val_losses[k] /= num_batches
            
        return total_val_loss, val_losses

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_train_loss = 0.0
        
        # Initialize train_losses based on first batch components
        train_losses = None
            
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            self.optimizer.zero_grad()
            
            states = batch.states.to(self.device)
            actions = batch.actions.to(self.device)
            
            # Forward pass and compute loss
            loss, _, component_losses = self.model.training_step(
                WallSample(states=states, actions=actions, locations=batch.locations)
            )
            
            # Initialize train_losses on first batch
            if train_losses is None:
                train_losses = {k: 0.0 for k in component_losses}
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Step OneCycleLR scheduler if used
            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # Update running losses
            total_train_loss += loss.item()
            for k in component_losses:
                train_losses[k] += component_losses[k]
            
            # Log batch metrics to WandB
            if wandb.run is not None and batch_idx % 10 == 0:
                wandb.log({
                    "batch": epoch * len(self.train_loader) + batch_idx,
                    "batch_loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    **{f"batch_{k}": v for k, v in component_losses.items()}
                })
        
        # Average losses over epoch
        num_batches = len(self.train_loader)
        total_train_loss /= num_batches
        for k in train_losses:
            train_losses[k] /= num_batches
            
        # Step other schedulers that need epoch-level stepping
        if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.scheduler.step()
                
        return total_train_loss, train_losses

    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint."""
        # Only save if it's time to do so
        if (epoch + 1) % self.save_frequency != 0:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, path)
        
        if wandb.run is not None:
            wandb.save(path)  # Save checkpoint to WandB
    
    def train(self, num_epochs):
        """Main training loop."""
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss, train_losses = self.train_epoch(epoch)
            
            # Validation if enabled
            if self.use_validation:
                val_loss, val_losses = self.validate(epoch)
                print("\nValidation Losses:")
                for k, v in val_losses.items():
                    print(f"{k}: {v:.4f}")
            
            # Print training summary
            print("\nEpoch Summary:")
            print("Training Losses:")
            for k, v in train_losses.items():
                print(f"{k}: {v:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss)
            print("Model saved!")
            
            # Evaluate probing performance
            losses = evaluate_model(self.device, self.model, 
                                  self.probe_train_data, self.probe_val_data)
            normal_loss = losses['normal']
            wall_loss = losses['wall']
            
            # Log epoch metrics to WandB
            if wandb.run is not None:
                wandb_logs = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "normal_probe_loss": normal_loss,
                    "wall_probe_loss": wall_loss,
                    **{f"train_{k}": v for k, v in train_losses.items()}
                }
                
                if self.use_validation:
                    wandb_logs.update({
                        "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in val_losses.items()}
                    })
                    
                wandb.log(wandb_logs)

def create_train_val_loaders(data_path, train_samples=None, val_samples=None):
    """Create training and validation dataloaders."""
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


if __name__ == "__main__":
    # Parse arguments
    args = get_args()

    print(f"Setting random seed: {args.seed}")
    set_all_seeds(args.seed)

    # Create model with specified loss
    model = WorldModel(
        loss_type=args.loss_type,
        lambda_param=25.0 if args.loss_type == 'vicreg' else 0.005,
        mu_param=25.0 if args.loss_type == 'vicreg' else None,
        nu_param=1.0 if args.loss_type == 'vicreg' else None, 
        encoder=args.encoder, 
        predictor=args.predictor
    )

    # Create dataloaders
    train_loader, val_loader = create_train_val_loaders(
        data_path="/drive_reader/as16386/DL24FA/train",
        train_samples=None,
        val_samples=None
    )
    
    # Load probe datasets
    probe_train_ds, probe_val_ds = load_data("cuda")

    # Initialize trainer
    trainer = WorldModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        probe_train_data=probe_train_ds,
        probe_val_data=probe_val_ds,
        use_validation=args.use_validation,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        checkpoint_dir=args.checkpoint_dir,
        save_frequency=args.save_frequency, 
        scheduler_type=args.scheduler, 
        seed=args.seed, 
        learning_rate=args.learning_rate
    )
    
    # Start training
    trainer.train(num_epochs=300)
