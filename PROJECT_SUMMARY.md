# Project Summary

This workspace now contains three main projects:

## 1. LeetCode Python (`leetcode_python/`)

Python implementations of LeetCode problems with:
- Problem solutions in `problems/`
- Utility classes (ListNode, TreeNode) in `utils/`
- Test framework setup
- Example solutions for problems 1 and 2

## 2. PyTorch Exercises - D2L (`pytorch_exercises/`)

Exercises from the "Dive into Deep Learning" book, organized by chapters:

- **Chapter 2**: Preliminaries (tensor operations)
- **Chapter 3**: Linear Neural Networks
- **Chapter 4**: Multilayer Perceptrons (MLP implementation)
- **Chapter 5**: Deep Learning Computation (custom datasets)
- **Chapter 6**: Convolutional Neural Networks (LeNet)
- **Chapter 7**: Modern CNN (placeholder for AlexNet, VGG, ResNet, etc.)
- **Chapter 8**: Recurrent Neural Networks (RNN/LSTM)
- **Chapter 9**: Modern RNN (placeholder)
- **Chapter 10**: Attention Mechanisms (Transformer encoder)
- **Chapter 11**: Optimization Algorithms (optimizer comparison)
- **Chapter 12**: Computational Performance (placeholder)
- **Chapter 13**: Computer Vision (placeholder)
- **Chapter 14**: Natural Language Processing (placeholder)

**Note**: The project uses the separate `plotting_tools` package for visualization.

## 3. Plotting Tools (`plotting_tools/`)

A general-purpose plotting and visualization package with:

- **Training plots**: `plot_training_history()`, `plot_loss_curve()`, `plot_accuracy_curve()`
- **Image visualization**: `visualize_tensor()`, `visualize_image_batch()`, `show_image_grid()`
- **Data plots**: `plot_distribution()`, `plot_confusion_matrix()`, `plot_feature_importance()`
- **Utilities**: `set_style()`, `save_figure()`, `create_subplots()`

### Installation

```bash
# Install plotting_tools
cd plotting_tools
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/workspace/plotting_tools"
```

## Usage Example

In your D2L exercises, you can use plotting_tools like this:

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../plotting_tools'))
from plotting_tools import plot_training_history

# After training...
plot_training_history(train_losses, val_losses, train_accs, val_accs)
```

## Project Structure

```
/workspace/
├── leetcode_python/          # LeetCode Python solutions
├── pytorch_exercises/         # D2L book exercises (chapter-based)
│   ├── chapter02_preliminaries/
│   ├── chapter04_multilayer_perceptrons/
│   ├── chapter06_convolutional_neural_networks/
│   ├── chapter08_recurrent_neural_networks/
│   ├── chapter10_attention_mechanisms/
│   ├── chapter11_optimization_algorithms/
│   └── utils/                 # D2L-specific helpers
└── plotting_tools/            # General-purpose plotting package
    └── plotting_tools/
        ├── training.py        # Training-related plots
        ├── images.py          # Image visualization
        ├── data.py            # Data distribution plots
        └── utils.py           # General utilities
```

## Next Steps

1. Install dependencies for each project
2. Start working through D2L chapters
3. Use plotting_tools for all visualization needs
4. Add more exercises as you progress through the book
