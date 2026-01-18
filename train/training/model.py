"""Transformer-based policy network for BeeWalker."""

import torch
import torch.nn as nn
import numpy as np


class TransformerPolicy(nn.Module):
    """
    Small transformer policy network optimized for Raspberry Pi Pico deployment.
    
    Architecture:
        - Input embedding: obs_dim -> d_model
        - Positional embedding for sequence context
        - Transformer encoder layers
        - Action head: Gaussian policy (mean + log_std)
        - Value head: Critic for PPO
    
    Target size: ~50-100k parameters to fit in Pico's 264KB RAM.
    """
    
    def __init__(self, obs_dim, act_dim, d_model=32, n_head=2, n_layers=2, 
                 context_len=8, dim_feedforward=64):
        super().__init__()
        
        self.d_model = d_model
        self.context_len = context_len
        
        # Input embedding
        self.obs_embedding = nn.Linear(obs_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, context_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=0.0,  # No dropout for small model
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer normalization
        self.ln = nn.LayerNorm(d_model)
        
        # Action head (Gaussian policy)
        self.action_mean = nn.Linear(d_model, act_dim)
        self.action_log_std = nn.Parameter(torch.zeros(1, act_dim))
        
        # Value head (Critic)
        self.value_head = nn.Linear(d_model, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Observation tensor of shape (batch, obs_dim) or (batch, seq, obs_dim)
            
        Returns:
            action_mean: (batch, act_dim)
            action_std: (batch, act_dim)
            value: (batch, 1)
        """
        # Add sequence dimension if missing
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        B, T, _ = x.shape
        
        # Embed observations
        x = self.obs_embedding(x)
        
        # Add positional embedding (truncate if sequence is shorter)
        pos_embed = self.pos_embedding[:, :T, :]
        x = x + pos_embed
        
        # Transform
        x = self.transformer(x)
        x = self.ln(x)
        
        # Use last state for prediction
        last_state = x[:, -1, :]
        
        # Action distribution
        action_mean = self.action_mean(last_state)
        action_std = torch.exp(self.action_log_std).expand_as(action_mean)
        
        # Value estimate
        value = self.value_head(last_state)
        
        return action_mean, action_std, value
    
    def get_param_count(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def export_for_pico(self, filepath):
        """
        Export weights in a format suitable for Pico deployment.
        Saves as numpy arrays that can be converted to C arrays.
        """
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        np.savez(filepath, **weights)
        print(f"Exported {len(weights)} weight tensors to {filepath}")


if __name__ == "__main__":
    # Test model parameters for BeeWalker
    obs_dim = 18  # 6 joints + 6 velocities + 6 IMU
    act_dim = 6   # 6 servo commands
    
    model = TransformerPolicy(obs_dim, act_dim, d_model=32, n_head=2, n_layers=2)
    param_count = model.get_param_count()
    print(f"Model parameters: {param_count:,}")
    print(f"Estimated size: {param_count * 4 / 1024:.1f} KB (float32)")
    
    # Test forward pass
    batch = torch.randn(4, obs_dim)
    mean, std, value = model(batch)
    print(f"Action mean shape: {mean.shape}")
    print(f"Action std shape: {std.shape}")
    print(f"Value shape: {value.shape}")
