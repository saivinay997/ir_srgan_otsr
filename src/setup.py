#!/usr/bin/env python3
"""
Setup script for IR-SRGAN-OTSR project
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.3.0",
        "PyYAML>=5.4.0",
        "tqdm>=4.62.0",
        "wandb>=0.12.0",
        "PuLP>=2.6.0",
        "scikit-image>=0.18.0",
        "matplotlib>=3.4.0",
        "tensorboard>=2.7.0"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        "trained_models",
        "val_results", 
        "datasets/train/HR",
        "datasets/train/LR",
        "datasets/val/HR",
        "datasets/val/LR",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available. Found {torch.cuda.device_count()} GPU(s)")
            print(f"  Current device: {torch.cuda.get_device_name()}")
        else:
            print("⚠ CUDA is not available. Training will use CPU (slower)")
    except ImportError:
        print("✗ PyTorch not installed")

def validate_config():
    """Validate configuration file"""
    print("Validating configuration...")
    
    try:
        import yaml
        with open("opt.yml", "r") as f:
            config = yaml.safe_load(f)
        
        required_keys = [
            "datasets.train.batch_size",
            "train.lr_G",
            "train.lr_D", 
            "train.niter",
            "scale"
        ]
        
        for key in required_keys:
            keys = key.split(".")
            value = config
            for k in keys:
                if k not in value:
                    print(f"✗ Missing config key: {key}")
                    return False
                value = value[k]
        
        print("✓ Configuration file is valid")
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Setting up IR-SRGAN-OTSR project...")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements")
        return False
    
    # Create directories
    create_directories()
    
    # Check CUDA
    check_cuda()
    
    # Validate config
    if not validate_config():
        print("Configuration validation failed")
        return False
    
    print("=" * 50)
    print("✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Place your training images in datasets/train/HR/")
    print("2. Place your validation images in datasets/val/HR/")
    print("3. Run: python train.py")
    print("4. For inference: python inference.py --model_path path/to/model --input_path image.jpg --output_path output.jpg")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
