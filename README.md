# IR-SRGAN-OTSR: Infrared Image Super-Resolution with GAN and Edge Loss

This repository contains implementations of Super-Resolution Generative Adversarial Networks (SRGAN) with edge loss enhancement for infrared image super-resolution. The project focuses on 4x image super-resolution with improved edge preservation using Sobel edge detection.

## 🚀 Features

- **RRDBNet Generator**: 23 residual dense blocks for high-quality image generation
- **VGG Discriminator**: VGG-based discriminator for adversarial training
- **Edge Loss**: Sobel edge detection for enhanced edge preservation
- **WGAN-QC Loss**: Wasserstein GAN with Quadratic Cost for stable training
- **Multi-scale Training**: 4x super-resolution capability
- **Comprehensive Loss Functions**: Pixel loss (L1), Feature loss (L2), GAN loss, Edge loss
- **Wandb Integration**: Experiment tracking and visualization
- **GPU/CPU Support**: Device-agnostic implementation
- **Error-Free Implementation**: Comprehensive error handling and validation

## 📁 Project Structure

```
ir_srgan_otsr/
├── src/                           # Main SRGAN implementation with edge loss
│   ├── train.py                   # Main training script
│   ├── test.py                    # Model testing script
│   ├── inference.py               # Single image inference
│   ├── setup.py                   # Project setup script
│   ├── validate_project.py        # Project validation script
│   ├── opt.yml                    # Configuration file
│   ├── requirements.txt           # Dependencies
│   ├── README.md                  # Implementation documentation
│   ├── SRGAN_model.py             # Main SRGAN model implementation
│   ├── networks.py                # Network definitions
│   ├── RRDBNet_arch.py            # RRDB generator architecture
│   ├── discriminator_vgg_arch.py  # VGG discriminator architecture
│   ├── loss.py                    # Loss functions (GAN, WGAN-QC, etc.)
│   ├── edge_loss.py               # Sobel edge detection loss
│   ├── create_dataset.py          # Dataset loader
│   ├── utils.py                   # Utility functions
│   ├── base_model.py              # Base model class
│   ├── lr_scheduler.py            # Learning rate schedulers
│   ├── module_util.py             # Module utilities
│   ├── linePromgram.py            # WGAN-QC optimization
│   ├── wb_util.py                 # Wandb utilities
│   └── val_imgs/                  # Validation images
├── IKC/                           # Iterative Kernel Correction implementation
│   ├── train_SFTMD.py             # SFTMD training script
│   ├── train_SFTMD.yml            # SFTMD configuration
│   ├── F_model.py                 # Kernel estimation model
│   ├── sftmd_arch.py              # SFTMD architecture
│   └── ...                        # Other IKC-related files
├── gan_project/                   # Project organization and experiments
│   ├── IKC_/                      # IKC implementation
│   ├── OTSR_dev/                  # OTSR development
│   ├── KernelGAN_Local/           # Local KernelGAN implementation
│   └── get_training_imgs/         # Data preparation utilities
├── SOTA/                          # State-of-the-art implementations
│   ├── ESRGAN/                    # ESRGAN implementation
│   ├── RealSR/                    # RealSR implementation
│   └── ZSSR/                      # ZSSR implementation
├── Kernel_GAN/                    # KernelGAN implementation
├── calculate_eval.ipynb           # Evaluation and metrics calculation
├── run_models.ipynb               # Model running and comparison
├── GetLrHr.ipynb                  # Data preparation and LR/HR generation
├── kernel_generator.ipynb         # Kernel generation utilities
├── model_validation/              # Model validation datasets
├── README.md                      # This file
├── .gitignore                     # Git ignore patterns
```

## 🛠️ Quick Start

### 1. Setup

```bash
# Clone the repository
git clone <repository-url>
cd ir_srgan_otsr

# Navigate to main implementation
cd src

# Run setup script (installs dependencies and creates directories)
python setup.py

# Validate project setup
python validate_project.py
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Wandb (optional but recommended)

```bash
wandb login
```

## 📊 Usage

### Training

1. **Prepare your dataset** in the following structure:
```
datasets/
├── train/
│   ├── HR/          # High-resolution training images
│   └── LR/          # Low-resolution training images (optional, will be generated)
└── val/
    ├── HR/          # High-resolution validation images
    └── LR/          # Low-resolution validation images (optional, will be generated)
```

2. **Configure training parameters** in `opt.yml`:
```yaml
# Key parameters you might want to adjust:
train:
  niter: 40001              # Total training iterations
  batch_size: 32            # Batch size
  lr_G: 1e-4               # Generator learning rate
  lr_D: 1e-4               # Discriminator learning rate
  pixel_weight: 5e-2       # Pixel loss weight
  feature_weight: 0.7      # Feature loss weight
  gan_weight: 1.0          # GAN loss weight
  edge_weight: 1e-2        # Edge loss weight
```

3. **Start training**:
```python
from train import main

# Training parameters
HR_train = "datasets/train/HR"
HR_val = "datasets/val/HR"
ymlpath = "opt.yml"
val_results_path = "val_results"
trained_model_path = "trained_models"
note = "Training with edge loss"

# Start training
main(HR_train, HR_val, ymlpath, val_results_path, trained_model_path, note)
```

### Inference

```python
from inference import super_resolve_image

# Single image super-resolution
super_resolve_image(
    model_path="trained_models/40000_G.pth",
    input_path="input_image.jpg",
    output_path="output_image.jpg"
)
```

### Testing

```python
from test import test_model

# Test model on validation dataset
test_model(
    model_path="trained_models/40000_G.pth",
    test_data_path="datasets/val/HR",
    output_path="test_results"
)
```

## 🔧 Configuration

### Key Training Parameters

- **Model**: RRDBNet with 23 blocks, 64 features
- **Scale**: 4x super-resolution
- **Batch Size**: 32 (adjust based on GPU memory)
- **Learning Rate**: 1e-4 for both generator and discriminator
- **Loss Weights**: 
  - Pixel (0.05): L1 pixel loss
  - Feature (0.7): VGG feature loss
  - GAN (1.0): WGAN-QC adversarial loss
  - Edge (0.01): Sobel edge loss
- **Training Iterations**: 40,001

### Loss Functions

1. **Pixel Loss**: L1 loss between generated and ground truth images
2. **Feature Loss**: L2 loss on VGG features for perceptual quality
3. **GAN Loss**: WGAN-QC for stable adversarial training
4. **Edge Loss**: Sobel edge detection loss for enhanced edge preservation

## 📈 Results

The model achieves improved edge preservation and perceptual quality compared to baseline SRGAN implementations:

- **Enhanced edge sharpness** through Sobel edge loss
- **Stable training** with WGAN-QC loss
- **Better perceptual quality** with VGG feature loss
- **Improved convergence** with learning rate scheduling

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in `opt.yml`
2. **Import errors**: Run `python setup.py` to install dependencies
3. **Configuration errors**: Run `python validate_project.py` to check setup
4. **Dataset not found**: Ensure dataset directories exist and contain images

### Validation

Run the validation script to check your setup:
```bash
python validate_project.py
```

This will check:
- Package imports
- Required files
- Configuration validity
- Model creation
- Edge loss functionality
- Utility functions
- CUDA availability
- Dataset structure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run validation: `python validate_project.py`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on ESRGAN architecture
- Uses VGG19 for feature extraction
- Implements WGAN-QC for stable training
- Edge loss inspired by Sobel edge detection

## 📞 Contact

For questions and support, please open an issue on GitHub.

## 🔄 Recent Updates

- **Fixed missing `rank` attribute** in SRGAN model
- **Improved edge loss implementation** with better device handling
- **Enhanced error handling** in training script
- **Added comprehensive validation** and setup scripts
- **Fixed DataParallel import** issues
- **Improved configuration validation**
- **Added better documentation** and examples
