#!/usr/bin/env python3
"""
Project validation script for IR-SRGAN-OTSR
Checks for common issues and validates the project setup
"""

import os
import sys
import importlib
import yaml
import torch
import numpy as np
from pathlib import Path

def check_imports():
    """Check if all required packages can be imported"""
    print("Checking imports...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'cv2', 'PIL', 'yaml', 
        'tqdm', 'wandb', 'pulp', 'skimage', 'matplotlib'
    ]
    
    failed_imports = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        return False
    
    return True

def check_files():
    """Check if all required files exist"""
    print("\nChecking required files...")
    
    required_files = [
        'SRGAN_model.py',
        'networks.py',
        'RRDBNet_arch.py',
        'discriminator_vgg_arch.py',
        'loss.py',
        'edge_loss.py',
        'create_dataset.py',
        'utils.py',
        'base_model.py',
        'lr_scheduler.py',
        'module_util.py',
        'linePromgram.py',
        'opt.yml',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_config():
    """Validate configuration file"""
    print("\nValidating configuration...")
    
    try:
        with open('opt.yml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['datasets', 'network_G', 'network_D', 'path', 'train']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing section: {section}")
                return False
            print(f"✓ {section}")
        
        # Check specific values
        if config['scale'] != 4:
            print(f"⚠ Scale is {config['scale']}, expected 4")
        
        if config['datasets']['train']['batch_size'] <= 0:
            print("✗ Invalid batch size")
            return False
        
        print("✓ Configuration is valid")
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_model_creation():
    """Test if the model can be created without errors"""
    print("\nTesting model creation...")
    
    try:
        # Import required modules
        from SRGAN_model import SRGANModel
        import yaml
        
        # Load config
        with open('opt.yml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = SRGANModel(config, train=False)
        print("✓ Model created successfully")
        
        # Test with dummy data
        dummy_lr = torch.randn(1, 3, 32, 32)
        dummy_hr = torch.randn(1, 3, 128, 128)
        
        model.feed_data(dummy_hr, dummy_lr)
        model.test()
        
        visuals = model.get_current_visuals()
        assert 'SR' in visuals
        assert 'GT' in visuals
        assert 'LQ' in visuals
        
        print("✓ Model forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_edge_loss():
    """Test edge loss functionality"""
    print("\nTesting edge loss...")
    
    try:
        from edge_loss import sobel
        
        # Test with dummy tensor
        dummy_tensor = torch.randn(2, 3, 64, 64)
        edge_result = sobel(dummy_tensor)
        
        assert edge_result.shape == dummy_tensor.shape
        print("✓ Edge loss test passed")
        return True
        
    except Exception as e:
        print(f"✗ Edge loss test failed: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        import utils
        
        # Test tensor2img
        dummy_tensor = torch.randn(1, 3, 64, 64)
        img = utils.tensor2img(dummy_tensor)
        assert img.shape == (64, 64, 3)
        
        # Test PSNR calculation
        img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        psnr = utils.calculate_psnr(img1, img2)
        assert isinstance(psnr, float)
        
        # Test SSIM calculation
        ssim = utils.calculate_ssim(img1, img2)
        assert isinstance(ssim, float)
        
        print("✓ Utility functions test passed")
        return True
        
    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        return False

def check_cuda():
    """Check CUDA availability and compatibility"""
    print("\nChecking CUDA...")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available - training will be slow on CPU")
        return True
    
    print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return True

def check_dataset_structure():
    """Check if dataset directories exist"""
    print("\nChecking dataset structure...")
    
    dataset_dirs = [
        'datasets/train/HR',
        'datasets/train/LR', 
        'datasets/val/HR',
        'datasets/val/LR'
    ]
    
    missing_dirs = []
    for dir_path in dataset_dirs:
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"✓ {dir_path} ({file_count} images)")
        else:
            print(f"✗ {dir_path}")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\nMissing directories: {', '.join(missing_dirs)}")
        print("Create these directories and add your images")
    
    return len(missing_dirs) == 0

def main():
    """Main validation function"""
    print("IR-SRGAN-OTSR Project Validation")
    print("=" * 50)
    
    checks = [
        ("Package imports", check_imports),
        ("Required files", check_files),
        ("Configuration", check_config),
        ("Model creation", test_model_creation),
        ("Edge loss", test_edge_loss),
        ("Utility functions", test_utils),
        ("CUDA availability", check_cuda),
        ("Dataset structure", check_dataset_structure)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} check failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All checks passed! Your project is ready to use.")
        return True
    else:
        print("⚠ Some checks failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
