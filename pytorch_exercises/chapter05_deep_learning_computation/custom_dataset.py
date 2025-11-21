"""
Custom Dataset Implementation

This exercise demonstrates:
- Creating custom Dataset classes
- Data preprocessing and augmentation
- Using DataLoader for batching
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomDataset(Dataset):
    """Custom dataset example"""
    
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: Input data (numpy array or tensor)
            labels: Corresponding labels
            transform: Optional transform to be applied on a sample
        """
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


class RandomTransform:
    """Example transform for data augmentation"""
    
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
    
    def __call__(self, sample):
        # Add random noise
        noise = torch.randn_like(sample) * self.noise_level
        return sample + noise


if __name__ == "__main__":
    print("Custom Dataset Exercise\n")
    
    # Generate synthetic data
    n_samples = 100
    feature_dim = 10
    
    data = torch.randn(n_samples, feature_dim)
    labels = torch.randint(0, 2, (n_samples,))
    
    # Create dataset without transform
    dataset_no_transform = CustomDataset(data, labels)
    print(f"Dataset size: {len(dataset_no_transform)}")
    
    # Create dataset with transform
    transform = RandomTransform(noise_level=0.1)
    dataset_with_transform = CustomDataset(data, labels, transform=transform)
    
    # Create data loader
    batch_size = 32
    dataloader = DataLoader(dataset_with_transform, batch_size=batch_size, shuffle=True)
    
    # Iterate through batches
    print(f"\nIterating through batches (batch_size={batch_size}):")
    for i, (batch_data, batch_labels) in enumerate(dataloader):
        print(f"Batch {i+1}: data shape={batch_data.shape}, labels shape={batch_labels.shape}")
        if i >= 2:  # Show first 3 batches
            break
    
    print("\nExercise completed!")
