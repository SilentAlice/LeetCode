"""
Data visualization functions for distributions, confusion matrices, etc.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, List

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_distribution(
    data: Union[np.ndarray, List[float]],
    bins: int = 30,
    title: str = "Data Distribution",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    figsize: tuple = (8, 5),
    save_path: Optional[str] = None
):
    """
    Plot a histogram of data distribution.
    
    Args:
        data: Array or list of values
        bins: Number of bins for histogram
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: tuple = (8, 6),
    normalize: bool = False,
    save_path: Optional[str] = None
):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix (2D array)
        class_names: Optional list of class names
        figsize: Figure size tuple
        normalize: Whether to normalize the confusion matrix
        save_path: Optional path to save the figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    
    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
    else:
        # Fallback to matplotlib if seaborn not available
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        if class_names:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(
    features: List[str],
    importances: Union[np.ndarray, List[float]],
    top_k: Optional[int] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot feature importance (e.g., from tree-based models).
    
    Args:
        features: List of feature names
        importances: Array or list of importance values
        top_k: Optional number of top features to show
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    importances = np.array(importances)
    features = np.array(features)
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    if top_k:
        indices = indices[:top_k]
    
    sorted_features = features[indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(sorted_features)), sorted_importances)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Importance', fontsize=12)
    plt.title('Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
