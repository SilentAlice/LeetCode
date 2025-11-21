"""
Basic Tensor Operations in PyTorch

This exercise covers fundamental tensor operations including:
- Creating tensors
- Basic arithmetic operations
- Indexing and slicing
- Reshaping and transformations
"""

import torch
import numpy as np


def exercise_1_create_tensors():
    """Exercise 1: Create tensors from different sources"""
    print("=" * 50)
    print("Exercise 1: Creating Tensors")
    print("=" * 50)
    
    # Create tensor from list
    tensor1 = torch.tensor([1, 2, 3, 4, 5])
    print(f"From list: {tensor1}")
    
    # Create tensor from numpy array
    np_array = np.array([1, 2, 3])
    tensor2 = torch.from_numpy(np_array)
    print(f"From numpy: {tensor2}")
    
    # Create zeros tensor
    zeros = torch.zeros(3, 4)
    print(f"Zeros tensor:\n{zeros}")
    
    # Create ones tensor
    ones = torch.ones(2, 3)
    print(f"Ones tensor:\n{ones}")
    
    # Create random tensor
    random_tensor = torch.randn(2, 3)
    print(f"Random tensor:\n{random_tensor}")
    
    # Create tensor with specific range
    range_tensor = torch.arange(0, 10, 2)
    print(f"Range tensor: {range_tensor}")


def exercise_2_arithmetic():
    """Exercise 2: Basic arithmetic operations"""
    print("\n" + "=" * 50)
    print("Exercise 2: Arithmetic Operations")
    print("=" * 50)
    
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    print(f"a ** 2 = {a ** 2}")
    
    # Matrix multiplication
    matrix_a = torch.randn(2, 3)
    matrix_b = torch.randn(3, 4)
    result = torch.matmul(matrix_a, matrix_b)
    print(f"\nMatrix A shape: {matrix_a.shape}")
    print(f"Matrix B shape: {matrix_b.shape}")
    print(f"Matrix multiplication result shape: {result.shape}")


def exercise_3_indexing_slicing():
    """Exercise 3: Indexing and slicing"""
    print("\n" + "=" * 50)
    print("Exercise 3: Indexing and Slicing")
    print("=" * 50)
    
    tensor = torch.arange(24).reshape(4, 6)
    print(f"Original tensor:\n{tensor}")
    print(f"\nFirst row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Element at [1, 2]: {tensor[1, 2]}")
    print(f"Slice [1:3, 2:5]:\n{tensor[1:3, 2:5]}")


def exercise_4_reshaping():
    """Exercise 4: Reshaping and transformations"""
    print("\n" + "=" * 50)
    print("Exercise 4: Reshaping")
    print("=" * 50)
    
    tensor = torch.arange(24)
    print(f"Original shape: {tensor.shape}")
    
    reshaped = tensor.reshape(4, 6)
    print(f"Reshaped to (4, 6):\n{reshaped}")
    
    flattened = reshaped.flatten()
    print(f"Flattened: {flattened}")
    
    transposed = reshaped.T
    print(f"Transposed shape: {transposed.shape}")
    
    # View (similar to reshape but shares memory)
    viewed = tensor.view(2, 12)
    print(f"Viewed shape: {viewed.shape}")


def exercise_5_device_operations():
    """Exercise 5: GPU/CPU operations"""
    print("\n" + "=" * 50)
    print("Exercise 5: Device Operations")
    print("=" * 50)
    
    tensor = torch.randn(3, 3)
    print(f"Default device: {tensor.device}")
    
    if torch.cuda.is_available():
        tensor_gpu = tensor.cuda()
        print(f"GPU tensor device: {tensor_gpu.device}")
        tensor_cpu = tensor_gpu.cpu()
        print(f"Back to CPU: {tensor_cpu.device}")
    else:
        print("CUDA not available, using CPU")


if __name__ == "__main__":
    print("PyTorch Basic Tensor Operations Exercises\n")
    
    exercise_1_create_tensors()
    exercise_2_arithmetic()
    exercise_3_indexing_slicing()
    exercise_4_reshaping()
    exercise_5_device_operations()
    
    print("\n" + "=" * 50)
    print("All exercises completed!")
    print("=" * 50)
