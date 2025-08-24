# HIT-UAV Dataset SRGAN Training

This guide explains how to use the HIT-UAV drone dataset to train an SRGAN (Super-Resolution Generative Adversarial Network) model for image super-resolution.

## Overview

The HIT-UAV dataset contains high-quality drone images that are perfect for training super-resolution models. The dataset includes:
- **Training images**: 2008 images in `hit-uav/images/train/`
- **Validation images**: 287 images in `hit-uav/images/val/`
- **Test images**: 571 images in `hit-uav/images/test/`

## Files Created

1. **`hit_uav_dataset.py`** - Custom dataset loader for HIT-UAV images
2. **`train_hit_uav.py`** - Training script specifically for HIT-UAV dataset
3. **`hit_uav_config.yml`** - Configuration file for training parameters
4. **`test_hit_uav_dataset.py`** - Script to test dataset loading and visualize samples

## Quick Start

### 1. Test Dataset Loading

First, test that the dataset can be loaded correctly:

```bash
cd src
python test_hit_uav_dataset.py
```

This will:
- Verify the dataset paths exist
- Create dataloaders
- Load a sample batch
- Visualize some LR-HR pairs
- Save visualization as `hit_uav_samples.png`

### 2. Start Training

Once the dataset test passes, start training:

```bash
python train_hit_uav.py
```

## Training Configuration

The training uses the following key parameters:

- **Scale factor**: 4x (LR: 32x32 → HR: 128x128)
- **Batch size**: 16
- **Learning rate**: 1e-4 for both Generator and Discriminator
- **Total iterations**: 20,000 (reduced for faster training)
- **Loss weights**:
  - Pixel loss (L1): 0.05
  - Feature loss (L2): 0.7
  - GAN loss: 1.0
  - Edge loss (Sobel): 0.01

## Dataset Processing

The dataset loader automatically:

1. **Loads high-resolution images** from the HIT-UAV dataset
2. **Creates LR-HR pairs** by downsampling HR images using bicubic interpolation
3. **Applies data augmentation** during training:
   - Random cropping to 128x128 patches
   - Random horizontal flipping
   - Random rotation
4. **Normalizes images** to [-1, 1] range
5. **Splits data** into training (80%) and validation (20%) sets

## Model Architecture

The SRGAN model uses:

- **Generator**: RRDBNet (Residual in Residual Dense Block Network)
  - 23 RRDB blocks
  - 64 feature channels
  - 4x upscaling

- **Discriminator**: VGG-style discriminator
  - 128x128 input size
  - 64 feature channels

## Training Process

The training includes:

1. **Generator training** with multiple loss components:
   - Pixel-wise L1 loss
   - Perceptual loss using VGG features
   - Adversarial loss from discriminator
   - Edge enhancement loss using Sobel filters

2. **Discriminator training** with WGAN-QC loss

3. **Validation** every 500 iterations to monitor PSNR and SSIM

4. **Model checkpointing** every 2000 iterations

5. **Best model saving** based on validation PSNR

## Expected Results

With the HIT-UAV dataset, you can expect:

- **Training time**: ~2-4 hours on a single GPU
- **Final PSNR**: 25-30 dB on validation set
- **Final SSIM**: 0.7-0.8 on validation set

## Monitoring Training

The training script integrates with Weights & Biases (wandb) for experiment tracking:

- Loss curves for all components
- Validation PSNR/SSIM over time
- Generated sample images
- Model checkpoints

## Output Files

Training will create:

- `experiments/hit_uav_srgan/` - Model checkpoints and logs
- `G_best.pth` - Best generator model
- `D_best.pth` - Best discriminator model
- `G_final.pth` - Final generator model
- `D_final.pth` - Final discriminator model

## Customization

You can modify the training parameters in `hit_uav_config.yml`:

- **Batch size**: Adjust based on GPU memory
- **Learning rate**: Modify for different convergence behavior
- **Loss weights**: Balance between different loss components
- **Training iterations**: Increase for better results (but longer training)

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size in config
2. **Dataset not found**: Check paths in `hit_uav_dataset.py`
3. **Import errors**: Ensure all dependencies are installed

### Dependencies:

```bash
pip install torch torchvision pillow matplotlib tqdm wandb pyyaml
```

## Next Steps

After training:

1. **Test the model** on new drone images
2. **Fine-tune** on specific drone image types
3. **Deploy** for real-time super-resolution
4. **Compare** with other SR methods

## Dataset Information

The HIT-UAV dataset contains drone images with various objects:
- Person detection
- Car detection  
- Bicycle detection
- Other vehicle detection

This makes it excellent for training super-resolution models that need to preserve fine details in aerial imagery.
