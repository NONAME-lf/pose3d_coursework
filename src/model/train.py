"""
Training script for exercise quality assessment model.
Trains on KIMORE dataset and evaluates performance.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import build_model


# ============================================================
# Dataset
# ============================================================
class KimoreDataset(Dataset):
    def __init__(self, body_parts, labels):
        """
        body_parts: dict of numpy arrays {part_name: (n_samples, seq_len, features)}
        labels: numpy array (n_samples,)
        """
        self.body_parts = {k: torch.FloatTensor(v) for k, v in body_parts.items()}
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        parts = {k: v[idx] for k, v in self.body_parts.items()}
        return parts, self.labels[idx]


def collate_fn(batch):
    """Custom collate for dict-based inputs."""
    parts_batch = {}
    labels = []
    
    for parts, label in batch:
        for k, v in parts.items():
            if k not in parts_batch:
                parts_batch[k] = []
            parts_batch[k].append(v)
        labels.append(label)
    
    parts_batch = {k: torch.stack(v) for k, v in parts_batch.items()}
    labels = torch.stack(labels)
    return parts_batch, labels


# ============================================================
# Training function
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for parts, labels in loader:
        parts = {k: v.to(device) for k, v in parts.items()}
        labels = labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(parts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
    
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for parts, labels in loader:
            parts = {k: v.to(device) for k, v in parts.items()}
            labels = labels.to(device)
            
            predictions = model(parts)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)


# ============================================================
# Main training loop with K-Fold cross-validation
# ============================================================
def train_exercise(exercise_name, data, n_epochs=100, n_folds=5, lr=0.001):
    """Train and evaluate model for one exercise using K-Fold CV."""
    
    print(f"\n{'='*60}")
    print(f"Training on {exercise_name}")
    print(f"{'='*60}")
    
    body_parts = data['body_parts']
    labels = data['labels']
    n_samples = len(labels)
    
    print(f"Samples: {n_samples}, Label range: {labels.min():.3f}-{labels.max():.3f}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # K-Fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_fold_preds = np.zeros(n_samples)
    all_fold_labels = np.zeros(n_samples)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(labels)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        # Split data
        train_parts = {k: v[train_idx] for k, v in body_parts.items()}
        val_parts = {k: v[val_idx] for k, v in body_parts.items()}
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        # Create datasets and loaders
        train_dataset = KimoreDataset(train_parts, train_labels)
        val_dataset = KimoreDataset(val_parts, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        
        # Build model
        model = build_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        
        # Training
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_preds, val_true = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= 30:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model and get final predictions
        model.load_state_dict(best_model_state)
        _, val_preds, val_true = evaluate(model, val_loader, criterion, device)
        
        all_fold_preds[val_idx] = val_preds
        all_fold_labels[val_idx] = val_true
        
        fold_mse = mean_squared_error(val_true, val_preds)
        fold_mae = mean_absolute_error(val_true, val_preds)
        fold_corr, _ = spearmanr(val_true, val_preds)
        
        print(f"  Fold {fold+1} - MSE: {fold_mse:.4f}, MAE: {fold_mae:.4f}, Spearman: {fold_corr:.4f}")
        fold_results.append({'mse': fold_mse, 'mae': fold_mae, 'spearman': fold_corr})
    
    # Overall results
    overall_mse = mean_squared_error(all_fold_labels, all_fold_preds)
    overall_mae = mean_absolute_error(all_fold_labels, all_fold_preds)
    overall_corr, overall_p = spearmanr(all_fold_labels, all_fold_preds)
    
    print(f"\n{'='*40}")
    print(f"Overall results for {exercise_name}:")
    print(f"  MSE:  {overall_mse:.4f}")
    print(f"  MAE:  {overall_mae:.4f}")
    print(f"  Spearman correlation: {overall_corr:.4f} (p={overall_p:.4f})")
    print(f"{'='*40}")
    
    return {
        'exercise': exercise_name,
        'predictions': all_fold_preds,
        'labels': all_fold_labels,
        'mse': overall_mse,
        'mae': overall_mae,
        'spearman': overall_corr,
        'fold_results': fold_results
    }


# ============================================================
# Visualization
# ============================================================
def plot_results(results, save_dir='results/training'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot predicted vs actual for each exercise
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results):
        ax.scatter(res['labels'], res['predictions'], alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.plot([0.4, 1.0], [0.4, 1.0], 'r--', label='Perfect prediction')
        ax.set_xlabel('True Score (Clinical)')
        ax.set_ylabel('Predicted Score')
        ax.set_title(f"{res['exercise']}\nSpearman={res['spearman']:.3f}")
        ax.legend()
        ax.set_xlim(0.4, 1.05)
        ax.set_ylim(0.4, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions_vs_actual.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to {save_dir}/predictions_vs_actual.png")
    
    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Exercise':<12} {'MSE':<10} {'MAE':<10} {'Spearman':<10}")
    print(f"{'-'*60}")
    for res in results:
        print(f"{res['exercise']:<12} {res['mse']:<10.4f} {res['mae']:<10.4f} {res['spearman']:<10.4f}")
    print(f"{'='*60}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Loading processed KIMORE data...")
    with open('data/processed/kimore_processed.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    # Train on exercises 1 and 2 (arm lifting and trunk tilt - most relevant for rehab)
    # You can add more exercises by extending this list
    exercises_to_train = ['ex1', 'ex2']
    
    all_results = []
    for ex_name in exercises_to_train:
        result = train_exercise(ex_name, processed_data[ex_name], n_epochs=100, n_folds=5)
        all_results.append(result)
    
    plot_results(all_results)
    
    # Save results
    os.makedirs('results/evaluation', exist_ok=True)
    with open('results/evaluation/training_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("\nTraining complete! Results saved.")