#!/usr/bin/env python3
"""Quick test to verify training setup works"""
import os
import sys
import numpy as np
from PIL import Image
import torch
from train import ClothingDataset

def create_dummy_dataset():
    """Create a dummy dataset for testing"""
    os.makedirs("test_dataset/images", exist_ok=True)
    os.makedirs("test_dataset/masks", exist_ok=True)
    
    for i in range(5):
        # Create dummy RGB image
        img = Image.fromarray(np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8))
        img.save(f"test_dataset/images/{i:03d}.png")
        
        # Create dummy mask (random white blobs)
        mask = Image.fromarray(np.random.randint(0, 255, (320, 320), dtype=np.uint8))
        mask.save(f"test_dataset/masks/{i:03d}.png")
    
    print("Created dummy dataset in test_dataset/")

if __name__ == "__main__":
    create_dummy_dataset()
    
    # Test dataset
    dataset = ClothingDataset("test_dataset/images", "test_dataset/masks")
    print(f"Dataset size: {len(dataset)}")
    
    # Test one batch
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print("Dataset setup looks good!")
    print("\nTo train on your own data, prepare:")
    print("  dataset/images/  - your clothing photos")
    print("  dataset/masks/   - corresponding masks (white=clothing, black=background)")
    print("\nThen run: python train.py")
