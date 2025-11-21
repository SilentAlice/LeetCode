"""
Simple RNN/LSTM Implementation

This exercise demonstrates:
- Building RNN and LSTM models
- Sequence processing
- Handling variable-length sequences
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleRNN(nn.Module):
    """Simple RNN for sequence classification"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        # Take the last output
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class SimpleLSTM(nn.Module):
    """Simple LSTM for sequence classification"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the last output
        out = out[:, -1, :]
        out = self.fc(out)
        return out


if __name__ == "__main__":
    print("Simple RNN/LSTM Exercise\n")
    
    # Generate synthetic sequence data
    batch_size = 32
    seq_len = 10
    input_size = 5
    num_classes = 2
    
    X = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, num_classes, (batch_size,))
    
    # Test RNN
    print("Testing RNN...")
    rnn_model = SimpleRNN(input_size, hidden_size=64, num_layers=2, num_classes=num_classes)
    output = rnn_model(X)
    print(f"RNN output shape: {output.shape}")
    
    # Test LSTM
    print("\nTesting LSTM...")
    lstm_model = SimpleLSTM(input_size, hidden_size=64, num_layers=2, num_classes=num_classes)
    output = lstm_model(X)
    print(f"LSTM output shape: {output.shape}")
    
    print("\nExercise completed!")
