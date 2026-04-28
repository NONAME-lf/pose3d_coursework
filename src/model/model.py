"""
Deep Spatio-Temporal model for exercise quality assessment.
Based on: "A Deep Learning Framework for Assessing Physical Rehabilitation Exercises" (Liao et al., 2019)
Adapted for KIMORE dataset with 25 Kinect joints.
"""

import torch
import torch.nn as nn


class TemporalPyramidBlock(nn.Module):
    """1D convolution block with multi-scale temporal processing."""
    
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
            for k in kernel_sizes
        ])
    
    def forward(self, x):
        # x: (batch, channels, time)
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)  # concat along channel dim


class BodyPartSubNetwork(nn.Module):
    """Sub-network for one body part with temporal pyramid."""
    
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        
        # Temporal pyramid: process at 4 time scales
        self.conv_block = TemporalPyramidBlock(input_dim, hidden_dim)
        # After TemporalPyramidBlock: 3 branches * hidden_dim channels
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = hidden_dim * 3  # 3 kernel sizes
        
    def forward(self, x):
        # x: (batch, time, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, time)
        
        # Process at multiple temporal scales
        scales = []
        for rate in [1, 2, 4, 8]:
            downsampled = x[:, :, ::rate]  # downsample by rate
            out = self.conv_block(downsampled)  # (batch, hidden*3, time/rate)
            pooled = self.pool(out).squeeze(-1)  # (batch, hidden*3)
            scales.append(pooled)
        
        # Combine all scales
        combined = torch.stack(scales, dim=-1).mean(dim=-1)  # (batch, hidden*3)
        return combined


class ExerciseQualityModel(nn.Module):
    """
    Full model: 5 body part sub-networks -> LSTM -> regression score.
    """
    
    def __init__(self, body_part_dims, hidden_dim=32, lstm_hidden=64, lstm_layers=2):
        """
        body_part_dims: dict with dimensions for each body part
            e.g., {'trunk': 15, 'left_arm': 18, 'right_arm': 18, 'left_leg': 12, 'right_leg': 12}
        """
        super().__init__()
        
        # Create sub-network for each body part
        self.sub_networks = nn.ModuleDict({
            name: BodyPartSubNetwork(dim, hidden_dim)
            for name, dim in body_part_dims.items()
        })
        
        # Total features from all sub-networks
        total_features = sum(
            net.output_dim for net in self.sub_networks.values()
        )
        
        # LSTM for temporal analysis of combined features
        self.lstm = nn.LSTM(
            input_size=total_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()  # output between 0 and 1
        )
        
    def forward(self, body_parts):
        """
        body_parts: dict of tensors {part_name: (batch, time, features)}
        """
        # Process each body part through its sub-network
        part_features = []
        for name, subnet in self.sub_networks.items():
            feat = subnet(body_parts[name])  # (batch, hidden*3)
            part_features.append(feat)
        
        # Concatenate all body part features
        combined = torch.cat(part_features, dim=1)  # (batch, total_features)
        
        # Add time dimension for LSTM (treat as single timestep sequence)
        combined = combined.unsqueeze(1)  # (batch, 1, total_features)
        
        # LSTM
        lstm_out, _ = self.lstm(combined)  # (batch, 1, lstm_hidden)
        lstm_out = lstm_out[:, -1, :]  # take last output
        
        # Predict score
        score = self.fc(lstm_out).squeeze(-1)  # (batch,)
        return score


def build_model():
    """Build model with KIMORE body part dimensions."""
    body_part_dims = {
        'trunk': 15,      # 5 joints * 3 coords
        'left_arm': 18,   # 6 joints * 3 coords
        'right_arm': 18,  # 6 joints * 3 coords
        'left_leg': 12,   # 4 joints * 3 coords
        'right_leg': 12,  # 4 joints * 3 coords
    }
    model = ExerciseQualityModel(body_part_dims)
    return model


if __name__ == '__main__':
    # Quick test
    model = build_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy data
    batch_size = 4
    seq_len = 300
    dummy_input = {
        'trunk': torch.randn(batch_size, seq_len, 15),
        'left_arm': torch.randn(batch_size, seq_len, 18),
        'right_arm': torch.randn(batch_size, seq_len, 18),
        'left_leg': torch.randn(batch_size, seq_len, 12),
        'right_leg': torch.randn(batch_size, seq_len, 12),
    }
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.detach().numpy()}")