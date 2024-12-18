import optuna
import torch
import json
import os
from datetime import datetime
import numpy as np
from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator, ProbingConfig
from models import WorldModel

class StudyTracker:
    def __init__(self, save_path: str = "optuna_results/lr_trials.json"):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.load_trials()
        self.best_trial = {'value': float('inf')}

    def load_trials(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                self.trials = json.load(f)
        else:
            self.trials = {}

    def save_trial(self, trial_number: int, learning_rate: float, losses: dict, avg_loss: float):
        self.trials[str(trial_number)] = {
            'learning_rate': learning_rate,
            'losses': losses,
            'avg_loss': avg_loss,
            'timestamp': datetime.now().isoformat()
        }

        if avg_loss < self.best_trial['value']:
            self.best_trial = {
                'number': trial_number,
                'value': avg_loss,
                'learning_rate': learning_rate,
                'losses': losses
            }
            print("\nNew best trial found!")
            print(f"  Learning rate: {learning_rate:.6e}")
            print(f"  Average loss: {avg_loss:.6f}")

        with open(self.save_path, 'w') as f:
            json.dump({
                'trials': self.trials,
                'best_trial': self.best_trial
            }, f, indent=2)

tracker = StudyTracker()

def evaluate_learning_rate(trial_number: int, learning_rate: float, device: str = "cuda") -> float:
    # Load probe datasets
    data_path = "/drive_reader/as16386/DL24FA"
    
    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_ds = {
        "normal": create_wall_dataloader(
            data_path=f"{data_path}/probe_normal/val",
            probing=True,
            device=device,
            train=False,
        ),
        "wall": create_wall_dataloader(
            data_path=f"{data_path}/probe_wall/val",
            probing=True,
            device=device,
            train=False,
        ),
        "wall_other": create_wall_dataloader(
            data_path=f"{data_path}/probe_wall_other/val",
            probing=True,
            device=device,
            train=False,
        ),
        "expert": create_wall_dataloader(
            data_path=f"{data_path}/probe_expert/val",
            probing=True,
            device=device,
            train=False,
        )
    }

    # Load model from checkpoint
    model = WorldModel(loss_type='both')
    checkpoint = torch.load('/drive_reader/as16386/DL-final-proj/Chill-Pill/checkpoints/baseline_combined_lr_reduced/checkpoint_epoch_87.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()

    config = ProbingConfig(lr=learning_rate)
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
        config=config
    )

    prober = evaluator.train_pred_prober()
    losses = evaluator.evaluate_all(prober=prober)
    avg_loss = np.mean(list(losses.values()))
    
    tracker.save_trial(trial_number, learning_rate, losses, avg_loss)
    
    print(f"\nTrial {trial_number}:")
    print(f"  Learning rate: {learning_rate:.6e}")
    print(f"  Average loss: {avg_loss:.6f}")
    print("  Individual losses:")
    for scenario, loss in losses.items():
        print(f"    {scenario}: {loss:.6f}")
    
    return avg_loss

def objective(trial):
    # Restricted search range: 0.001 to 0.01
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01, log=True)
    return evaluate_learning_rate(trial.number, learning_rate)

def optimize_learning_rate(n_trials: int = 10000, n_jobs: int = 4):
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        print("\nOptimization interrupted!")
        print("\nBest trial found:")
        print(f"  Trial number: {tracker.best_trial['number']}")
        print(f"  Average loss: {tracker.best_trial['value']:.6f}")
        print(f"  Learning rate: {tracker.best_trial['learning_rate']:.6e}")
        print("\nIndividual losses for best trial:")
        for scenario, loss in tracker.best_trial['losses'].items():
            print(f"  {scenario}: {loss:.6f}")

if __name__ == "__main__":
    optimize_learning_rate(n_trials=10000, n_jobs=4)