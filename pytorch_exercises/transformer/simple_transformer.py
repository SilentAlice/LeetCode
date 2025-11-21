"""
Simple Transformer Implementation

This exercise demonstrates:
- Building a basic transformer encoder
- Self-attention mechanism
- Positional encoding
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :]
        return x


class SimpleTransformerEncoder(nn.Module):
    """Simple Transformer Encoder"""
    
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super(SimpleTransformerEncoder, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
    
    def forward(self, src):
        # src shape: (batch, seq_len, d_model)
        # Transpose for positional encoding: (seq_len, batch, d_model)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        # Transpose back: (batch, seq_len, d_model)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src)
        return output


if __name__ == "__main__":
    print("Simple Transformer Exercise\n")
    
    # Generate synthetic data
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    src = torch.randn(batch_size, seq_len, d_model)
    
    # Create transformer encoder
    model = SimpleTransformerEncoder(d_model=d_model, nhead=8, num_layers=2)
    
    # Forward pass
    output = model(src)
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nExercise completed!")
