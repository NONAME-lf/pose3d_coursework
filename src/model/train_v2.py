"""
Training v2: Simpler model + better hyperparameters for small dataset.
Uses a lighter architecture more suitable for ~75 samples.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os


# ============================================================
# Simpler model for small dataset
# ============================================================
class SimpleExerciseModel(nn.Module):
    """
    Lighter model: 1D Conv per body part -> concat -> FC layers.
    Better suited for small datasets (~75 samples).
    """
    def __init__(self, body_part_dims):
        super().__init__()
        
        self.part_encoders = nn.ModuleDict()
        total_features = 0
        
        for name, dim in body_part_dims.items():
            encoder = nn.Sequential(
                nn.Conv1d(dim, 16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 8, kernel_size=5, padding=2),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.part_encoders[name] = encoder
            total_features += 8
        
        self.regressor = nn.Sequential(
            nn.Linear(total_features, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, body_parts):
        features = []
        for name, encoder in self.part_encoders.items():
            x = body_parts[name].permute(0, 2, 1)  # (batch, features, time)
            x = encoder(x).squeeze(-1)  # (batch, 8)
            features.append(x)
        
        combined = torch.cat(features, dim=1)
        return self.regressor(combined).squeeze(-1)


# ============================================================
# Also try a baseline: flatten + simple MLP on joint angles
# ============================================================
class StatisticalFeatureModel(nn.Module):
    """
    Extract statistical features (mean, std, min, max, range) 
    from each joint, then use a small MLP.
    Most robust approach for very small datasets.
    """
    def __init__(self, n_joints=25, n_coords=3):
        super().__init__()
        # 5 stats per coordinate per joint
        input_dim = n_joints * n_coords * 5
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, full_skeleton):
        # full_skeleton: (batch, time, 75)
        # Compute statistical features
        mean_feat = full_skeleton.mean(dim=1)
        std_feat = full_skeleton.std(dim=1)
        min_feat = full_skeleton.min(dim=1).values
        max_feat = full_skeleton.max(dim=1).values
        range_feat = max_feat - min_feat
        
        features = torch.cat([mean_feat, std_feat, min_feat, max_feat, range_feat], dim=1)
        return self.mlp(features).squeeze(-1)


# ============================================================
# Dataset
# ============================================================
class KimoreDataset(Dataset):
    def __init__(self, body_parts, full_skeleton, labels):
        self.body_parts = {k: torch.FloatTensor(v) for k, v in body_parts.items()}
        self.full_skeleton = torch.FloatTensor(full_skeleton)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        parts = {k: v[idx] for k, v in self.body_parts.items()}
        return parts, self.full_skeleton[idx], self.labels[idx]


def collate_fn(batch):
    parts_batch = {}
    skeletons = []
    labels = []
    for parts, skeleton, label in batch:
        for k, v in parts.items():
            if k not in parts_batch:
                parts_batch[k] = []
            parts_batch[k].append(v)
        skeletons.append(skeleton)
        labels.append(label)
    parts_batch = {k: torch.stack(v) for k, v in parts_batch.items()}
    return parts_batch, torch.stack(skeletons), torch.stack(labels)


# ============================================================
# Training
# ============================================================
def train_model(model_class, model_kwargs, exercise_data, exercise_name,
                n_epochs=200, n_folds=5, lr=0.001, use_body_parts=True):
    
    print(f"\n{'='*60}")
    print(f"Training {model_class.__name__} on {exercise_name}")
    print(f"{'='*60}")
    
    body_parts = exercise_data['body_parts']
    full_skeleton = exercise_data['full_skeleton']
    labels = exercise_data['labels']
    n_samples = len(labels)
    
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_preds = np.zeros(n_samples)
    all_labels = np.zeros(n_samples)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(labels)):
        # Split
        train_parts = {k: v[train_idx] for k, v in body_parts.items()}
        val_parts = {k: v[val_idx] for k, v in body_parts.items()}
        
        train_ds = KimoreDataset(train_parts, full_skeleton[train_idx], labels[train_idx])
        val_ds = KimoreDataset(val_parts, full_skeleton[val_idx], labels[val_idx])
        
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=len(val_idx), shuffle=False, collate_fn=collate_fn)
        
        # Model
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
        best_val_loss = float('inf')
        best_state = None
        
        for epoch in range(n_epochs):
            # Train
            model.train()
            for parts, skeleton, lbl in train_loader:
                parts = {k: v.to(device) for k, v in parts.items()}
                skeleton = skeleton.to(device)
                lbl = lbl.to(device)
                
                optimizer.zero_grad()
                if use_body_parts:
                    pred = model(parts)
                else:
                    pred = model(skeleton)
                loss = criterion(pred, lbl)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Validate
            model.eval()
            with torch.no_grad():
                for parts, skeleton, lbl in val_loader:
                    parts = {k: v.to(device) for k, v in parts.items()}
                    skeleton = skeleton.to(device)
                    lbl = lbl.to(device)
                    if use_body_parts:
                        pred = model(parts)
                    else:
                        pred = model(skeleton)
                    val_loss = criterion(pred, lbl).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Best model predictions
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for parts, skeleton, lbl in val_loader:
                parts = {k: v.to(device) for k, v in parts.items()}
                skeleton = skeleton.to(device)
                if use_body_parts:
                    pred = model(parts)
                else:
                    pred = model(skeleton)
                all_preds[val_idx] = pred.cpu().numpy()
                all_labels[val_idx] = lbl.numpy()
        
        fold_corr, _ = spearmanr(all_labels[val_idx], all_preds[val_idx])
        print(f"  Fold {fold+1}: val_loss={best_val_loss:.4f}, Spearman={fold_corr:.3f}")
    
    # Overall
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    corr, p = spearmanr(all_labels, all_preds)
    
    print(f"\n  Overall: MSE={mse:.4f}, MAE={mae:.4f}, Spearman={corr:.4f} (p={p:.4f})")
    
    return {
        'exercise': exercise_name,
        'model': model_class.__name__,
        'predictions': all_preds,
        'labels': all_labels,
        'mse': mse, 'mae': mae, 'spearman': corr
    }


# ============================================================
# Visualization
# ============================================================
def plot_all_results(results_list, save_dir='results/training'):
    os.makedirs(save_dir, exist_ok=True)
    
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, res in zip(axes, results_list):
        ax.scatter(res['labels'], res['predictions'], alpha=0.6, 
                   edgecolors='black', linewidth=0.5, s=40)
        ax.plot([0.4, 1.0], [0.4, 1.0], 'r--', linewidth=1, label='Ideal')
        ax.set_xlabel('Clinical Score (True)', fontsize=11)
        ax.set_ylabel('Predicted Score', fontsize=11)
        ax.set_title(f"{res['exercise']} — {res['model']}\n"
                     f"Spearman={res['spearman']:.3f}, MAE={res['mae']:.3f}", fontsize=10)
        ax.legend()
        ax.set_xlim(0.4, 1.05)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results_v2.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to {save_dir}/results_v2.png")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Loading processed KIMORE data...")
    with open('data/processed/kimore_processed.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    all_results = []
    
    body_part_dims = {
        'trunk': 15, 'left_arm': 18, 'right_arm': 18,
        'left_leg': 12, 'right_leg': 12
    }
    
    for ex_name in ['ex1', 'ex2', 'ex3', 'ex4', 'ex5']:
        # Model 1: Conv-based with body part decomposition
        res1 = train_model(
            SimpleExerciseModel, {'body_part_dims': body_part_dims},
            processed_data[ex_name], ex_name,
            n_epochs=200, use_body_parts=True, lr=0.0005
        )
        all_results.append(res1)
        
        # Model 2: Statistical features + MLP
        res2 = train_model(
            StatisticalFeatureModel, {},
            processed_data[ex_name], ex_name,
            n_epochs=200, use_body_parts=False, lr=0.001
        )
        all_results.append(res2)
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Exercise':<10} {'Model':<25} {'MSE':<10} {'MAE':<10} {'Spearman':<10}")
    print(f"{'-'*70}")
    for res in all_results:
        print(f"{res['exercise']:<10} {res['model']:<25} {res['mse']:<10.4f} {res['mae']:<10.4f} {res['spearman']:<10.4f}")
    print(f"{'='*70}")
    
    plot_all_results(all_results)
    
    with open('results/evaluation/training_results_v2.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("\nDone!")