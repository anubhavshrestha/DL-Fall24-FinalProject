import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import JEPAVICReg
from dataset import create_wall_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_jepa():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JEPAVICReg().to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)
    
    # Use correct data path
    train_loader = create_wall_dataloader(
        data_path="/scratch/an3854/DL24FA/train",  # Fixed path
        probing=False,
        device=device,
        batch_size=64,
        train=True,
    )
    
    # Validation loader with correct path
    val_loader = create_wall_dataloader(
        data_path="/scratch/an3854/DL24FA/probe_normal/val",  # Fixed path
        probing=False,
        device=device,
        batch_size=64,
        train=False,
    )
    
    # Rest of the code remains the same...
    n_epochs = 100
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_val_loss = float('inf')
    
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    epochs = []
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        total_loss = 0
        n_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in progress_bar:
            optimizer.zero_grad()
            
            loss, metrics = model.training_step(
                states=batch.states,
                actions=batch.actions
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            progress_bar.set_postfix({
                'loss': total_loss / n_batches,
                'pred_loss': metrics['pred_loss'],
                'sim_loss': metrics['sim_loss'],
                'std_loss': metrics['std_loss'],
                'cov_loss': metrics['cov_loss']
            })
        
        # Calculate average training loss
        avg_train_loss = total_loss / n_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss, _ = model.training_step(
                    states=batch.states,
                    actions=batch.actions
                )
                val_loss += loss.item()
                n_val_batches += 1
        
        avg_val_loss = val_loss / n_val_batches
        print(f"Epoch {epoch+1} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")
        
        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs.append(epoch + 1)
        
        scheduler.step()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_weights.pth')
        
        # Periodic checkpoint and plot
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
            
            # Create and save loss plot
            plot_losses(epochs, train_losses, val_losses, epoch + 1)

    # Save final model and plot
    torch.save(model.state_dict(), 'final_model_weights.pth')
    plot_losses(epochs, train_losses, val_losses, n_epochs)

def plot_losses(epochs, train_losses, val_losses, current_epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_plot_epoch_{current_epoch}.png')
    plt.close()

if __name__ == "__main__":
    train_jepa()