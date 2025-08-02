#!/usr/bin/env python3
"""
split_dataset.py

Dataset splitting script for waste classification project.

This script splits the waste image dataset into training, validation, and test sets
following standard machine learning practices. Additionally, it makes use of stratified sampling to ensure
balanced class distribution across splits.

References:
- Chollet, François. Deep Learning with Python. 2nd ed., Manning, 2021.
- "Getting started with Classification - GeeksforGeeks."
- Scikit-learn documentation on train_test_split.

Author: Zinhle Maurice-Mopp (210125870)
Date: 2025-08-02
"""

# Run with: python scripts/split_dataset.py --input dataset/images --output dataset/split

"""
The command python scripts/split_dataset.py --input dataset/images --output dataset/split is used to organise the raw images 
from dataset/images into distinct training, validation, and test sets within dataset/split. 
This separation is for ensuring the model is trained on one set of data and evaluated on unseen data, 
thus preventing overfitting and providing an accurate measure of its real-world performance. 

This aligns with standard practices for preparing datasets for supervised learning, 
as detailed in Chollet's Deep Learning with Python.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse

def count_images_per_class(dataset_path):
    """
    Count the number of images in each class directory.
    
    Args:
        dataset_path (str): Path to the dataset directory containing class folders
        
    Returns:
        dict: Dictionary mapping class names to image counts
        
    References:
        Standard file system traversal patterns for dataset organisation.
    """
    class_counts = defaultdict(int)
    
    # Walk through all subdirectories to find images
    for root, dirs, files in os.walk(dataset_path):
        # Skip if this is the root dataset directory
        if root == dataset_path:
            continue
            
        # Count image files (common extensions)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
        
        if image_files:
            # Extract class name from directory structure
            # Uses file structure: dataset/images/category/subcategory/[real_world|default]/
            path_parts = Path(root).parts
            if 'images' in path_parts:
                images_idx = path_parts.index('images')
                if len(path_parts) > images_idx + 2:
                    # Format: category_subcategory (e.g., "glass_glass_beverage_bottles")
                    category = path_parts[images_idx + 1]
                    subcategory = path_parts[images_idx + 2]
                    class_name = f"{category}_{subcategory}"
                    class_counts[class_name] += len(image_files)
    
    return dict(class_counts)

def create_split_directories(output_path):
    """
    Create train/val/test directory structure.
    
    Args:
        output_path (str): Base path where split directories will be created
        
    References:
        Standard ML dataset organisation following Chollet's recommendations.
    """
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_path = Path(output_path) / split
        split_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {split_path}")

def split_class_images(class_images, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split a list of image paths into train/val/test sets using stratified sampling.
    
    Args:
        class_images (list): List of image file paths for a single class
        train_ratio (float): Proportion for training set (default 0.7)
        val_ratio (float): Proportion for validation set (default 0.15)  
        test_ratio (float): Proportion for test set (default 0.15)
        
    Returns:
        tuple: (train_images, val_images, test_images) lists
        
    References:
        Chollet, François. Deep Learning with Python. Chapter 3: "Getting started with neural networks."
        Standard 70/15/15 split recommended for medium-sized datasets.
    """
    # Validate split ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Shuffle images randomly for unbiased sampling
    random.shuffle(class_images)
    
    n_total = len(class_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split into three sets
    train_images = class_images[:n_train]
    val_images = class_images[n_train:n_train + n_val]
    test_images = class_images[n_train + n_val:]
    
    return train_images, val_images, test_images

def copy_images_to_split(image_paths, class_name, split_name, output_path):
    """
    Copy images to the appropriate split directory maintaining class structure.
    
    Args:
        image_paths (list): List of source image file paths
        class_name (str): Name of the class (e.g., "glass_glass_beverage_bottles")
        split_name (str): Split name ("train", "val", or "test")
        output_path (str): Base output directory path
        
    References:
        Standard file operations for dataset preparation in ML pipelines.
    """
    # Create class directory in split folder
    class_dir = Path(output_path) / split_name / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy each image to the split directory
    for img_path in image_paths:
        src_path = Path(img_path)
        dst_path = class_dir / src_path.name
        
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Warning: Failed to copy {src_path} to {dst_path}: {e}")
    
    print(f"✓ Copied {len(image_paths)} images to {split_name}/{class_name}")

def main():
    """
    Main function to execute dataset splitting workflow.
    
    Implements stratified sampling across all waste classes to ensure balanced
    representation in train/validation/test splits, following best practices
    for supervised learning dataset preparation.
    
    References:
        "Getting started with Classification - GeeksforGeeks."
        Scikit-learn documentation on stratified sampling principles.
    """
    parser = argparse.ArgumentParser(description='Split waste classification dataset')
    parser.add_argument('--input', default='dataset/images', 
                       help='Input dataset path (default: dataset/images)')
    parser.add_argument('--output', default='dataset/split', 
                       help='Output split dataset path (default: dataset/split)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible splits
    random.seed(args.seed)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Validate input directory exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_path}")
    
    print(f"Splitting dataset from: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Split ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
    
    # Count images per class for reporting
    class_counts = count_images_per_class(input_path)
    print(f"\nFound {len(class_counts)} classes:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} images")
    
    # Create output directory structure
    create_split_directories(output_path)
    
    # Process each class separately for stratified sampling
    total_processed = 0
    
    for root, dirs, files in os.walk(input_path):
        # Skip if this is the root dataset directory
        if root == input_path:
            continue
            
        # Find image files in current directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [os.path.join(root, f) for f in files 
                      if Path(f).suffix.lower() in image_extensions]
        
        if not image_files:
            continue
            
        # Extract class name from directory structure
        path_parts = Path(root).parts
        if 'images' in path_parts:
            images_idx = path_parts.index('images')
            if len(path_parts) > images_idx + 2:
                category = path_parts[images_idx + 1]
                subcategory = path_parts[images_idx + 2]
                class_name = f"{category}_{subcategory}"
                
                # Split images for this class
                train_imgs, val_imgs, test_imgs = split_class_images(
                    image_files, args.train_ratio, args.val_ratio, args.test_ratio
                )
                
                # Copy images to respective split directories
                copy_images_to_split(train_imgs, class_name, 'train', output_path)
                copy_images_to_split(val_imgs, class_name, 'val', output_path)
                copy_images_to_split(test_imgs, class_name, 'test', output_path)
                
                total_processed += len(image_files)
                
                print(f"  Split {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    print(f"\n✓ Dataset splitting complete!")
    print(f"✓ Total images processed: {total_processed}")
    print(f"✓ Split dataset saved to: {output_path}")
    
    # Verify split directories
    for split in ['train', 'val', 'test']:
        split_path = output_path / split
        if split_path.exists():
            n_classes = len([d for d in split_path.iterdir() if d.is_dir()])
            print(f"✓ {split} set: {n_classes} classes")

if __name__ == "__main__":
    main()