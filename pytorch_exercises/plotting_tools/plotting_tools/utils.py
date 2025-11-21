"""
General plotting utilities and helper functions.
"""

import matplotlib.pyplot as plt
import matplotlib.style as style
from typing import Optional, Tuple


def set_style(style_name: str = 'default'):
    """
    Set matplotlib style.
    
    Args:
        style_name: Style name ('default', 'seaborn', 'ggplot', etc.)
    """
    try:
        plt.style.use(style_name)
    except OSError:
        print(f"Style '{style_name}' not found, using default")


def save_figure(fig, path: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """
    Save a figure to file.
    
    Args:
        fig: Matplotlib figure object
        path: Path to save the figure
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box in inches
    """
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figure saved to {path}")


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create subplots with consistent styling.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size tuple
        **kwargs: Additional arguments for plt.subplots
    
    Returns:
        Tuple of (figure, axes)
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes
