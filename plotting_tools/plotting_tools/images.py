"""
Image and tensor visualization functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
import torch


def visualize_tensor(
    tensor: Union[torch.Tensor, np.ndarray],
    title: str = "Tensor Visualization",
    figsize: tuple = (6, 6),
    cmap: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize a tensor (useful for images).
    
    Args:
        tensor: PyTorch tensor or numpy array to visualize
        title: Title for the plot
        figsize: Figure size tuple
        cmap: Colormap (None for RGB, 'gray' for grayscale)
        save_path: Optional path to save the figure
    """
    # Convert to numpy if tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    # Handle different tensor shapes
    if len(tensor.shape) == 4:  # Batch of images: (B, C, H, W) or (B, H, W, C)
        tensor = tensor[0]
    
    if len(tensor.shape) == 3:  # Single image
        # Check if CHW format
        if tensor.shape[0] in [1, 3] and tensor.shape[0] < tensor.shape[1]:
            tensor = tensor.transpose(1, 2, 0)
        # Handle grayscale
        if tensor.shape[2] == 1:
            tensor = tensor.squeeze(2)
            if cmap is None:
                cmap = 'gray'
    
    # Normalize to [0, 1] if needed
    if tensor.max() > 1.0:
        tensor = tensor.astype(np.float32)
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
    plt.figure(figsize=figsize)
    plt.imshow(tensor, cmap=cmap)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_image_batch(
    images: Union[torch.Tensor, np.ndarray],
    num_images: int = 16,
    ncols: int = 4,
    titles: Optional[list] = None,
    figsize: tuple = (12, 12),
    save_path: Optional[str] = None
):
    """
    Visualize a batch of images in a grid.
    
    Args:
        images: Batch of images (B, C, H, W) or (B, H, W, C)
        num_images: Number of images to display
        ncols: Number of columns in the grid
        titles: Optional list of titles for each image
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    # Convert to numpy if tensor
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    num_images = min(num_images, len(images))
    nrows = (num_images + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for i in range(num_images):
        img = images[i]
        
        # Handle CHW format
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:
            img = img.transpose(1, 2, 0)
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze(2)
        
        # Normalize
        if img.max() > 1.0:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        cmap = 'gray' if len(img.shape) == 2 else None
        axes[i].imshow(img, cmap=cmap)
        axes[i].axis('off')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=10)
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def show_image_grid(
    images: list,
    ncols: int = 4,
    titles: Optional[list] = None,
    figsize: tuple = (12, 12),
    save_path: Optional[str] = None
):
    """
    Show a grid of images from a list.
    
    Args:
        images: List of image arrays/tensors
        ncols: Number of columns in the grid
        titles: Optional list of titles
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    num_images = len(images)
    nrows = (num_images + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for i, img in enumerate(images):
        # Convert to numpy if tensor
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        # Handle different formats
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:
            img = img.transpose(1, 2, 0)
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze(2)
        
        # Normalize
        if img.max() > 1.0:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        cmap = 'gray' if len(img.shape) == 2 else None
        axes[i].imshow(img, cmap=cmap)
        axes[i].axis('off')
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=10)
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
