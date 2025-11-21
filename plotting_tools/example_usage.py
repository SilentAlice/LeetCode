"""
Example usage of plotting_tools package.

This demonstrates how to use the plotting tools in your projects.
"""

import numpy as np
import torch
import sys
import os

# Import plotting tools
from plotting_tools import (
    plot_training_history,
    plot_loss_curve,
    plot_accuracy_curve,
    visualize_tensor,
    visualize_image_batch,
    plot_distribution,
    set_style
)

# Example 1: Plot training history
print("Example 1: Training History")
train_losses = [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2]
val_losses = [2.6, 1.9, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3]
train_accs = [45, 60, 72, 80, 85, 88, 90, 92, 93, 94]
val_accs = [44, 59, 71, 79, 84, 87, 89, 91, 92, 93]

# Uncomment to see plot:
# plot_training_history(train_losses, val_losses, train_accs, val_accs)

# Example 2: Visualize tensor/image
print("\nExample 2: Tensor Visualization")
image_tensor = torch.randn(1, 3, 32, 32)  # Batch, Channels, Height, Width
# Uncomment to see plot:
# visualize_tensor(image_tensor, title="Random Image")

# Example 3: Plot distribution
print("\nExample 3: Data Distribution")
data = np.random.normal(0, 1, 1000)
# Uncomment to see plot:
# plot_distribution(data, title="Normal Distribution", bins=30)

# Example 4: Single loss curve
print("\nExample 4: Loss Curve")
losses = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15]
# Uncomment to see plot:
# plot_loss_curve(losses, label="Training Loss", title="Loss Over Time")

print("\nAll examples completed!")
print("Uncomment the plot calls to see visualizations.")
