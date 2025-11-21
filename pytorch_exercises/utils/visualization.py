"""
Visualization utilities for PyTorch exercises.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(train_losses, val_losses=None, train_accs=None, val_accs=None):
    """
    Plot training history including loss and accuracy curves.
    
    Args:
        train_losses: List of training losses
        val_losses: Optional list of validation losses
        train_accs: Optional list of training accuracies
        val_accs: Optional list of validation accuracies
    """
    fig, axes = plt.subplots(1, 2 if train_accs is not None else 1, figsize=(12, 4))
    
    if train_accs is None:
        axes = [axes]
    
    # Plot loss
    ax = axes[0]
    ax.plot(train_losses, label='Train Loss')
    if val_losses:
        ax.plot(val_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot accuracy if provided
    if train_accs is not None:
        ax = axes[1]
        ax.plot(train_accs, label='Train Acc')
        if val_accs:
            ax.plot(val_accs, label='Val Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def visualize_tensor(tensor, title="Tensor Visualization"):
    """
    Visualize a tensor (useful for images).
    
    Args:
        tensor: PyTorch tensor to visualize
        title: Title for the plot
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    if len(tensor.shape) == 4:  # Batch of images
        tensor = tensor[0]
    if len(tensor.shape) == 3:  # Single image
        if tensor.shape[0] == 3 or tensor.shape[0] == 1:  # CHW format
            tensor = tensor.transpose(1, 2, 0)
        if tensor.shape[2] == 1:  # Grayscale
            tensor = tensor.squeeze(2)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(tensor, cmap='gray' if len(tensor.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()
