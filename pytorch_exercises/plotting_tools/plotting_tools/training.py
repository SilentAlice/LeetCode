"""
Training-related visualization functions.

Functions for plotting training history, loss curves, accuracy curves, etc.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union


def plot_training_history(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None
):
    """
    Plot training history including loss and accuracy curves.
    
    Args:
        train_losses: List of training losses
        val_losses: Optional list of validation losses
        train_accs: Optional list of training accuracies
        val_accs: Optional list of validation accuracies
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the figure
    """
    num_plots = 2 if train_accs is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    if num_plots == 1:
        axes = [axes]
    
    # Plot loss
    ax = axes[0]
    ax.plot(train_losses, label='Train Loss', linewidth=2)
    if val_losses:
        ax.plot(val_losses, label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot accuracy if provided
    if train_accs is not None:
        ax = axes[1]
        ax.plot(train_accs, label='Train Acc', linewidth=2)
        if val_accs:
            ax.plot(val_accs, label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_loss_curve(
    losses: Union[List[float], np.ndarray],
    label: str = "Loss",
    title: str = "Loss Curve",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    figsize: tuple = (8, 5),
    save_path: Optional[str] = None
):
    """
    Plot a single loss curve.
    
    Args:
        losses: List or array of loss values
        label: Label for the curve
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    plt.plot(losses, label=label, linewidth=2)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_accuracy_curve(
    accuracies: Union[List[float], np.ndarray],
    label: str = "Accuracy",
    title: str = "Accuracy Curve",
    xlabel: str = "Epoch",
    ylabel: str = "Accuracy",
    figsize: tuple = (8, 5),
    save_path: Optional[str] = None
):
    """
    Plot a single accuracy curve.
    
    Args:
        accuracies: List or array of accuracy values
        label: Label for the curve
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    plt.plot(accuracies, label=label, linewidth=2)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_curves(
    train_metrics: dict,
    val_metrics: Optional[dict] = None,
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None
):
    """
    Plot multiple learning curves from a metrics dictionary.
    
    Args:
        train_metrics: Dictionary of training metrics {metric_name: [values]}
        val_metrics: Optional dictionary of validation metrics
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    num_metrics = len(train_metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, train_values) in enumerate(train_metrics.items()):
        ax = axes[idx]
        ax.plot(train_values, label=f'Train {metric_name}', linewidth=2)
        
        if val_metrics and metric_name in val_metrics:
            ax.plot(val_metrics[metric_name], label=f'Val {metric_name}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
