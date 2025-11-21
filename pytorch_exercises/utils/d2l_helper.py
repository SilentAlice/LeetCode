"""
D2L-specific helper functions and utilities.

This module contains helper functions commonly used in D2L exercises.
"""

import torch
import torch.nn as nn
from typing import Tuple


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute the number of correct predictions.
    
    Args:
        y_hat: Predicted labels (logits or probabilities)
        y: True labels
    
    Returns:
        Accuracy as a float
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net: nn.Module, data_iter: torch.utils.data.DataLoader, device: torch.device = None) -> float:
    """
    Evaluate the accuracy of a model on a dataset.
    
    Args:
        net: Neural network model
        data_iter: DataLoader for the dataset
        device: Device to run on (CPU or CUDA)
    
    Returns:
        Accuracy as a float
    """
    if isinstance(net, nn.Module):
        net.eval()
    if not device:
        device = next(iter(net.parameters())).device
    
    metric = 0.0
    total = 0
    
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            metric += accuracy(y_hat, y)
            total += y.numel()
    
    return metric / total


def train_epoch_ch3(net: nn.Module, train_iter: torch.utils.data.DataLoader, 
                    loss: nn.Module, updater: torch.optim.Optimizer,
                    device: torch.device = None) -> Tuple[float, float]:
    """
    Train a model for one epoch (D2L Chapter 3 style).
    
    Args:
        net: Neural network model
        train_iter: DataLoader for training data
        loss: Loss function
        updater: Optimizer
        device: Device to run on
    
    Returns:
        Tuple of (average loss, accuracy)
    """
    if isinstance(net, nn.Module):
        net.train()
    if not device:
        device = next(iter(net.parameters())).device
    
    metric = 0.0
    total_loss = 0.0
    total = 0
    
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric += accuracy(y_hat, y)
        total_loss += float(l.sum())
        total += y.numel()
    
    return total_loss / total, metric / total
