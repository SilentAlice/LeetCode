"""
General-purpose plotting and visualization tools.

This package provides utilities for:
- Training history visualization
- Image and tensor visualization
- Data distribution plots
- General plotting utilities
"""

from .training import (
    plot_training_history,
    plot_loss_curve,
    plot_accuracy_curve,
    plot_learning_curves
)

from .images import (
    visualize_tensor,
    visualize_image_batch,
    show_image_grid
)

from .data import (
    plot_distribution,
    plot_confusion_matrix,
    plot_feature_importance
)

from .utils import (
    set_style,
    save_figure,
    create_subplots
)

__version__ = "0.1.0"
__all__ = [
    # Training plots
    'plot_training_history',
    'plot_loss_curve',
    'plot_accuracy_curve',
    'plot_learning_curves',
    # Image plots
    'visualize_tensor',
    'visualize_image_batch',
    'show_image_grid',
    # Data plots
    'plot_distribution',
    'plot_confusion_matrix',
    'plot_feature_importance',
    # Utilities
    'set_style',
    'save_figure',
    'create_subplots',
]
