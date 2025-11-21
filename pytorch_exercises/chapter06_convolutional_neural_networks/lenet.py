"""
Chapter 6: Convolutional Neural Networks - LeNet Implementation

D2L Exercise: Building and training a CNN (LeNet-style) for image classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import sys
import os

# Add plotting_tools to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../plotting_tools'))
try:
    from plotting_tools import visualize_tensor, plot_training_history
except ImportError:
    print("Warning: plotting_tools not found. Install it or add to PYTHONPATH.")
    visualize_tensor = None
    plot_training_history = None


class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional block 1
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Convolutional block 2
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Convolutional block 3
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def generate_synthetic_images(n_samples=1000, img_size=28):
    """Generate synthetic grayscale images"""
    images = torch.randn(n_samples, 1, img_size, img_size)
    # Simple classification: based on average pixel value
    labels = (images.mean(dim=(1, 2, 3)) > 0).long()
    return images, labels


if __name__ == "__main__":
    print("Simple CNN Exercise\n")
    
    # Hyperparameters
    num_classes = 2
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # Generate synthetic data
    print("Generating synthetic image data...")
    X_train, y_train = generate_synthetic_images(1000, 28)
    X_test, y_test = generate_synthetic_images(200, 28)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SimpleCNN(num_classes=num_classes)
    print(f"\nModel architecture:\n{model}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Training model...")
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
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluation
    print("\nEvaluating model...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    
    print("\nExercise completed!")
