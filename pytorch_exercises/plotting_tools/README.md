# Plotting Tools

A general-purpose Python plotting and visualization package built on matplotlib.

## Features

- Training history visualization (loss, accuracy curves)
- Tensor/image visualization
- Data distribution plots
- Model architecture visualization
- General-purpose plotting utilities

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from plotting_tools import plot_training_history, visualize_tensor

# Plot training curves
plot_training_history(train_losses, val_losses, train_accs, val_accs)

# Visualize tensors/images
visualize_tensor(image_tensor, title="My Image")
```

## Structure

- `plotting_tools/` - Main package directory
  - `training.py` - Training-related plots (loss, accuracy)
  - `images.py` - Image and tensor visualization
  - `data.py` - Data distribution and analysis plots
  - `utils.py` - General plotting utilities

## Requirements

- matplotlib >= 3.7.0
- numpy >= 1.24.0
