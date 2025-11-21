"""
Simple Multi-Layer Perceptron (MLP) Implementation

This exercise demonstrates:
- Building a simple neural network using nn.Module
- Forward pass
- Training loop
- Loss calculation and backpropagation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def generate_data(n_samples=1000, input_size=10):
    """Generate synthetic data for training"""
    X = torch.randn(n_samples, input_size)
    # Simple target: sum of first 3 features
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).long()
    return X, y


def train_model(model, train_loader, num_epochs=10, learning_rate=0.01):
    """Training function"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(num_epochs):
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
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return losses


def evaluate_model(model, test_loader):
    """Evaluate model accuracy"""
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
    return accuracy


if __name__ == "__main__":
    print("Simple MLP Exercise\n")
    
    # Hyperparameters
    input_size = 10
    hidden_size = 64
    output_size = 2
    batch_size = 32
    num_epochs = 20
    
    # Generate data
    print("Generating data...")
    X_train, y_train = generate_data(1000, input_size)
    X_test, y_test = generate_data(200, input_size)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SimpleMLP(input_size, hidden_size, output_size)
    print(f"\nModel architecture:\n{model}\n")
    
    # Train model
    print("Training model...")
    losses = train_model(model, train_loader, num_epochs)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy = evaluate_model(model, test_loader)
    
    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()
    
    print("\nExercise completed!")
