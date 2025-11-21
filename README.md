# LeetCode & PyTorch Exercises

This repository contains solutions to LeetCode problems and PyTorch exercises from "Dive into Deep Learning" (D2L).

## Structure

### LeetCode Solutions (`leetcode/`)

Solutions organized by language and algorithm type:

- **C Solutions** (`leetcode/c/`)
  - Organized by algorithm/data structure type:
    - `binary-tree/` - Binary tree problems
    - `bit-manipulation/` - Bitwise operations
    - `graph/` - Graph algorithms
    - `linked-list/` - Linked list problems
    - `sliding-window/` - Sliding window technique
    - `sort/` - Sorting algorithms
    - `string-processing/` - String manipulation
    - `tree/` - General tree problems
    - `union-find/` - Union-find/disjoint-set problems
  - Files follow naming pattern: `lc{number}-{descriptive-title}.c`

- **Python Solutions** (`leetcode/python/`)
  - Problem solutions in `problems/`
  - Utility classes (ListNode, TreeNode) in `utils/`
  - Test framework in `tests/`

### PyTorch Exercises (`pytorch_exercises/`)

Exercises from "Dive into Deep Learning" book, organized by chapters:

- `chapter02_preliminaries/` - Data manipulation, linear algebra basics
- `chapter03_linear_neural_networks/` - Linear regression, softmax regression
- `chapter04_multilayer_perceptrons/` - MLPs, activation functions, dropout
- `chapter05_deep_learning_computation/` - Model construction, parameter management
- `chapter06_convolutional_neural_networks/` - Convolutional layers, pooling, LeNet
- `chapter07_modern_cnn/` - Modern CNN architectures
- `chapter08_recurrent_neural_networks/` - RNNs, GRUs, LSTMs
- `chapter09_modern_rnn/` - Modern RNN architectures
- `chapter10_attention_mechanisms/` - Attention mechanisms, Transformers
- `chapter11_optimization_algorithms/` - Optimization techniques
- `chapter12_computational_performance/` - GPU computation, parallelization
- `chapter13_computer_vision/` - Image augmentation, object detection
- `chapter14_natural_language_processing/` - Word embeddings, text classification
- `plotting_tools/` - Visualization package for training plots and data analysis
- `utils/` - D2L-specific utilities and helper functions

### Manual C (`manualc/`)

C programming exercises and reference implementations.

### Drafts (`drafts/`)

Work-in-progress solutions and experimental code.

## Usage

### LeetCode C Solutions

```bash
cd leetcode/c
gcc sort/lc1-two-sum.c -o lc1
./lc1
```

### LeetCode Python Solutions

```bash
cd leetcode/python
pip install -r requirements.txt
python problems/lc1.py
pytest tests/
```

### PyTorch Exercises

```bash
cd pytorch_exercises
pip install -r requirements.txt
python chapter04_multilayer_perceptrons/mlp_from_scratch.py
```

## Contact

- Blog: https://silentming.net
- Email: yumingwu233@gmail.com

## References

- LeetCode: https://leetcode.com
- D2L Book: https://d2l.ai/
- D2L PyTorch: https://github.com/d2l-ai/d2l-pytorch
- Manual C: ["手写代码必备手册(C)"](https://github.com/kuke/acm-cheatsheet)
