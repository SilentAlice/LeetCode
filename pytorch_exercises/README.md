# PyTorch Exercises - Dive into Deep Learning (D2L)

Exercises and implementations from the "Dive into Deep Learning" book using PyTorch.

## Structure

This project follows the D2L book chapter organization:

- `chapter02_preliminaries/` - Data manipulation, linear algebra, calculus basics
- `chapter03_linear_neural_networks/` - Linear regression, softmax regression
- `chapter04_multilayer_perceptrons/` - MLPs, activation functions, dropout
- `chapter05_deep_learning_computation/` - Model construction, parameter management
- `chapter06_convolutional_neural_networks/` - Convolutional layers, pooling, LeNet
- `chapter07_modern_cnn/` - Modern CNN architectures (AlexNet, VGG, ResNet, etc.)
- `chapter08_recurrent_neural_networks/` - RNNs, GRUs, LSTMs
- `chapter09_modern_rnn/` - Modern RNN architectures and techniques
- `chapter10_attention_mechanisms/` - Attention mechanisms, Transformers
- `chapter11_optimization_algorithms/` - Optimization techniques, learning rate scheduling
- `chapter12_computational_performance/` - GPU computation, parallelization
- `chapter13_computer_vision/` - Image augmentation, object detection, semantic segmentation
- `chapter14_natural_language_processing/` - Word embeddings, text classification
- `utils/` - D2L-specific utilities and helper functions

## Usage

Each chapter contains exercises that can be run independently:

```bash
# Example: Run chapter 4 exercise
python chapter04_multilayer_perceptrons/mlp_from_scratch.py
```

## Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- plotting_tools (separate package for visualization)

## Installation

1. Install PyTorch dependencies:
```bash
pip install -r requirements.txt
```

2. Install plotting_tools package (from parent directory):
```bash
cd ../plotting_tools
pip install -e .
```

Or add plotting_tools to your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/plotting_tools"
```

## Reference

- Book: [Dive into Deep Learning](https://d2l.ai/)
- PyTorch Version: [D2L PyTorch](https://github.com/d2l-ai/d2l-pytorch)
