"""
Chapter 4: Multilayer Perceptrons - MLP Implementation

D2L Exercise: Building and training a Multi-Layer Perceptron from scratch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add plotting_tools to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../plotting_tools'))
try:
    from plotting_tools import plot_training_history
except ImportError:
    print("Warning: plotting_tools not found. Install it or add to PYTHONPATH.")
    plot_training_history = None


class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def generate_data(n_samples=1000, input_size=10):
    """Generate synthetic data for training"""
    X = torch.randn(n_samples, input_size)
    # Simple target: sum of first 3 features
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).long()
    return X, y


def train_model(model, train_loader, val_loader=None, num_epochs=10, learning_rate=0.01):
    """Training function"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs


if __name__ == "__main__":
    print("D2L Chapter 4: Multilayer Perceptrons\n")
    
    # Hyperparameters
    input_size = 10
    hidden_size = 64
    output_size = 2
    batch_size = 32
    num_epochs = 20
    dropout = 0.2
    
    # Generate data
    print("Generating data...")
    X_train, y_train = generate_data(1000, input_size)
    X_val, y_val = generate_data(200, input_size)
    X_test, y_test = generate_data(200, input_size)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = MLP(input_size, hidden_size, output_size, dropout=dropout)
    print(f"\nModel architecture:\n{model}\n")
    
    # Train model
    print("Training model...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs
    )
    
    # Plot training history
    if plot_training_history:
        plot_training_history(train_losses, val_losses, train_accs, val_accs)
    else:
        print("\nInstall plotting_tools to visualize training history.")
    
    print("\nExercise completed!")
