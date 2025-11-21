"""
Chapter 2: Preliminaries - Tensor Operations

This exercise covers fundamental tensor operations as introduced in D2L Chapter 2.
"""

import torch
import numpy as np
import sys
import os

# Add plotting_tools to path (adjust path as needed)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../plotting_tools'))
try:
    from plotting_tools import plot_distribution
except ImportError:
    print("Warning: plotting_tools not found. Install it or add to PYTHONPATH.")
    plot_distribution = None


def exercise_tensor_creation():
    """Exercise: Creating tensors"""
    print("=" * 50)
    print("Tensor Creation")
    print("=" * 50)
    
    # Create tensor from list
    x = torch.tensor([1, 2, 3, 4, 5])
    print(f"From list: {x}")
    
    # Create tensor with specific shape
    zeros = torch.zeros(3, 4)
    ones = torch.ones(2, 3)
    print(f"\nZeros tensor:\n{zeros}")
    print(f"\nOnes tensor:\n{ones}")
    
    # Create random tensor
    random_tensor = torch.randn(2, 3)
    print(f"\nRandom tensor:\n{random_tensor}")


def exercise_tensor_operations():
    """Exercise: Tensor operations"""
    print("\n" + "=" * 50)
    print("Tensor Operations")
    print("=" * 50)
    
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([2.0, 3.0, 4.0, 5.0])
    
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {x + y}")
    print(f"x * y = {x * y}")
    print(f"x ** 2 = {x ** 2}")
    
    # Matrix multiplication
    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    C = torch.matmul(A, B)
    print(f"\nMatrix A shape: {A.shape}")
    print(f"Matrix B shape: {B.shape}")
    print(f"Matrix multiplication result shape: {C.shape}")


def exercise_indexing_slicing():
    """Exercise: Indexing and slicing"""
    print("\n" + "=" * 50)
    print("Indexing and Slicing")
    print("=" * 50)
    
    X = torch.arange(24).reshape(4, 6)
    print(f"Original tensor:\n{X}")
    print(f"\nFirst row: {X[0]}")
    print(f"First column: {X[:, 0]}")
    print(f"Element at [1, 2]: {X[1, 2]}")
    print(f"Slice [1:3, 2:5]:\n{X[1:3, 2:5]}")


def exercise_broadcasting():
    """Exercise: Broadcasting"""
    print("\n" + "=" * 50)
    print("Broadcasting")
    print("=" * 50)
    
    a = torch.arange(3).reshape(3, 1)
    b = torch.arange(2).reshape(1, 2)
    print(f"a shape: {a.shape}, a:\n{a}")
    print(f"b shape: {b.shape}, b:\n{b}")
    print(f"a + b (broadcasted):\n{a + b}")


def exercise_memory_operations():
    """Exercise: Memory operations"""
    print("\n" + "=" * 50)
    print("Memory Operations")
    print("=" * 50)
    
    X = torch.arange(12).reshape(3, 4)
    Y = X
    Y = Y + X
    print(f"After Y = Y + X, id(Y) == id(X): {id(Y) == id(X)}")
    
    # In-place operations
    Z = torch.zeros_like(X)
    Z[:] = X + Y
    print(f"Z after assignment:\n{Z}")


if __name__ == "__main__":
    print("D2L Chapter 2: Preliminaries - Tensor Operations\n")
    
    exercise_tensor_creation()
    exercise_tensor_operations()
    exercise_indexing_slicing()
    exercise_broadcasting()
    exercise_memory_operations()
    
    print("\n" + "=" * 50)
    print("All exercises completed!")
    print("=" * 50)
