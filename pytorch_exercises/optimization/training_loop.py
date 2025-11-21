"""
Advanced Training Loop with Various Optimizers

This exercise demonstrates:
- Different optimizers (SGD, Adam, AdamW, RMSprop)
- Learning rate scheduling
- Gradient clipping
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os


class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=2):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_with_optimizer(model, train_loader, optimizer_name='adam', 
                         num_epochs=10, learning_rate=0.001, 
                         use_scheduler=False, clip_gradients=False):
    """
    Train model with different optimizers
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        optimizer_name: 'sgd', 'adam', 'adamw', 'rmsprop'
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        use_scheduler: Whether to use learning rate scheduler
        clip_gradients: Whether to clip gradients
    """
    criterion = nn.CrossEntropyLoss()
    
    # Select optimizer
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
    
    return losses


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath}")
    return epoch, loss


if __name__ == "__main__":
    print("Advanced Training Loop Exercise\n")
    
    # Generate synthetic data
    X_train = torch.randn(1000, 10)
    y_train = (X_train[:, 0] > 0).long()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Test different optimizers
    optimizers = ['adam', 'sgd', 'adamw', 'rmsprop']
    
    for opt_name in optimizers:
        print(f"\n{'='*50}")
        print(f"Training with {opt_name.upper()} optimizer")
        print(f"{'='*50}")
        
        model = SimpleModel()
        losses = train_with_optimizer(
            model, 
            train_loader, 
            optimizer_name=opt_name,
            num_epochs=10,
            learning_rate=0.001,
            use_scheduler=True,
            clip_gradients=True
        )
    
    print("\nExercise completed!")
