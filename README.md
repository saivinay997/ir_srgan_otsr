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

## 📁 Project Structure

```
ir_srgan_otsr/
├── src_ir_with_edgeloss_3c/     # Main SRGAN implementation with edge loss
│   ├── train.py                 # Main training script
│   ├── test.py                  # Model testing script
│   ├── inference.py             # Single image inference
│   ├── opt.yml                  # Configuration file
│   ├── requirements.txt         # Dependencies
│   ├── README.md               # Implementation documentation
│   ├── SRGAN_model.py          # Main SRGAN model implementation
│   ├── networks.py             # Network definitions
│   ├── RRDBNet_arch.py         # RRDB generator architecture
│   ├── discriminator_vgg_arch.py # VGG discriminator architecture
│   ├── loss.py                 # Loss functions (GAN, WGAN-QC, etc.)
│   ├── edge_loss.py            # Sobel edge detection loss
│   ├── create_dataset.py       # Dataset loader
│   ├── utils.py                # Utility functions
│   ├── base_model.py           # Base model class
│   ├── lr_scheduler.py         # Learning rate schedulers
│   ├── module_util.py          # Module utilities
│   ├── linePromgram.py         # WGAN-QC optimization
│   ├── wb_util.py              # Wandb utilities
│   └── val_imgs/               # Validation images
├── IKC/                        # Iterative Kernel Correction implementation
│   ├── train_SFTMD.py          # SFTMD training script
│   ├── train_SFTMD.yml         # SFTMD configuration
│   ├── F_model.py              # Kernel estimation model
│   ├── sftmd_arch.py           # SFTMD architecture
│   └── ...                     # Other IKC-related files
├── gan_project/                # Project organization and experiments
│   ├── IKC_/                   # IKC implementation
│   ├── OTSR_dev/               # OTSR development
│   ├── KernelGAN_Local/        # Local KernelGAN implementation
│   └── get_training_imgs/      # Data preparation utilities
├── SOTA/                       # State-of-the-art implementations
│   ├── ESRGAN/                 # ESRGAN implementation
│   ├── RealSR/                 # RealSR implementation
│   └── ZSSR/                   # ZSSR implementation
├── Kernel_GAN/                 # KernelGAN implementation
├── calculate_eval.ipynb        # Evaluation and metrics calculation
├── run_models.ipynb            # Model running and comparison
├── GetLrHr.ipynb              # Data preparation and LR/HR generation
├── kernel_generator.ipynb      # Kernel generation utilities
└── model_validation/           # Model validation datasets
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ir_srgan_otsr
```

2. Install dependencies for the main implementation:
```bash
cd src_ir_with_edgeloss_3c
pip install -r requirements.txt
```

3. Set up Wandb (optional but recommended):
```bash
wandb login
```

## 📊 Usage

### Training

1. Prepare your dataset in the following structure:
```
datasets/
├── train/
│   ├── HR/          # High-resolution training images
│   └── LR/          # Low-resolution training images
└── val/
    ├── HR/          # High-resolution validation images
    └── LR/          # Low-resolution validation images
```

2. Configure training parameters in `src_ir_with_edgeloss_3c/opt.yml`

3. Start training:
```python
from train import main

# Training parameters
HR_train = "path/to/train/HR"
HR_val = "path/to/val/HR"
ymlpath = "opt.yml"
val_results_path = "val_results"
trained_model_path = "trained_models"
note = "Training with edge loss"

# Start training
main(HR_train, HR_val, ymlpath, val_results_path, trained_model_path, note)
```

### Inference

```python
from inference import inference

# Load trained model and perform inference
inference("path/to/model.pth", "path/to/input/image.jpg", "path/to/output")
```

### Testing

```python
from test import test

# Test model on validation dataset
test("path/to/model.pth", "path/to/test/dataset", "path/to/results")
```

## 🔧 Configuration

Key training parameters in `opt.yml`:

- **Model**: RRDBNet with 23 blocks, 64 features
- **Scale**: 4x super-resolution
- **Batch Size**: 32
- **Learning Rate**: 1e-4 for both generator and discriminator
- **Loss Weights**: Pixel (0.05), Feature (0.7), GAN (1.0), Edge (0.01)
- **Training Iterations**: 40,001

## 📈 Results

The model achieves improved edge preservation and perceptual quality compared to baseline SRGAN implementations. Key improvements:

- Enhanced edge sharpness through Sobel edge loss
- Stable training with WGAN-QC loss
- Better perceptual quality with VGG feature loss
- Improved convergence with learning rate scheduling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on ESRGAN architecture
- Uses VGG19 for feature extraction
- Implements WGAN-QC for stable training
- Edge loss inspired by Sobel edge detection

## 📞 Contact

For questions and support, please open an issue on GitHub.
