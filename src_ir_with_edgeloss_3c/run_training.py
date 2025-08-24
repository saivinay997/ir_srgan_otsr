#!/usr/bin/env python3
"""
Simple training script for SRGAN with edge loss
This script provides a quick way to start training with default parameters
"""

import os
import sys
from train import main

def run_training():
    """Run training with default parameters"""
    
    # Default paths - modify these according to your setup
    HR_train = "imgsResizedHR"  # Path to high-resolution training images
    HR_val = "imgsResizedHR"    # Path to high-resolution validation images
    
    # Configuration and output paths
    ymlpath = "opt.yml"
    val_results_path = "val_results"
    trained_model_path = "trained_models"
    note = "SRGAN training with edge loss"
    
    # Check if training data exists
    if not os.path.exists(HR_train):
        print(f"Error: Training data not found at {HR_train}")
        print("Please ensure your high-resolution images are in the correct directory.")
        print("You can modify the HR_train path in this script or organize your data accordingly.")
        return
    
    # Create output directories
    os.makedirs(val_results_path, exist_ok=True)
    os.makedirs(trained_model_path, exist_ok=True)
    
    print("Starting SRGAN training with edge loss...")
    print(f"Training data: {HR_train}")
    print(f"Validation data: {HR_val}")
    print(f"Configuration: {ymlpath}")
    print(f"Results will be saved to: {val_results_path}")
    print(f"Models will be saved to: {trained_model_path}")
    print(f"Note: {note}")
    print("-" * 50)
    
    try:
        # Start training
        main(HR_train, HR_val, ymlpath, val_results_path, trained_model_path, note)
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Partial results may be available in the output directories.")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("Please check your data paths and configuration.")

if __name__ == "__main__":
    run_training()
