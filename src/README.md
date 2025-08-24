# SRGAN with Edge Loss for Image Super-Resolution

This project implements a Super-Resolution Generative Adversarial Network (SRGAN) with edge loss enhancement for 4x image super-resolution. The model uses a Residual in Residual Dense Block (RRDB) architecture with Sobel edge detection loss for improved edge preservation.

## Features

- **RRDBNet Generator**: 23 residual dense blocks for high-quality image generation
- **VGG Discriminator**: VGG-based discriminator for adversarial training
- **Edge Loss**: Sobel edge detection for enhanced edge preservation
- **WGAN-QC Loss**: Wasserstein GAN with Quadratic Cost for stable training
- **Multi-scale Training**: 4x super-resolution capability
- **Comprehensive Loss Functions**: Pixel loss (L1), Feature loss (L2), GAN loss, Edge loss
- **Wandb Integration**: Experiment tracking and visualization
- **GPU/CPU Support**: Device-agnostic implementation

## Project Structure

```
src_ir_with_edgeloss_3c/
├── train.py                 # Main training script
├── test.py                  # Model testing script
├── inference.py             # Single image inference
├── opt.yml                  # Configuration file
├── requirements.txt         # Dependencies
├── README.md               # This file
├── SRGAN_model.py          # Main SRGAN model implementation
├── networks.py             # Network definitions
├── RRDBNet_arch.py         # RRDB generator architecture
├── discriminator_vgg_arch.py # VGG discriminator architecture
├── loss.py                 # Loss functions (GAN, WGAN-QC, etc.)
├── edge_loss.py            # Sobel edge detection loss
├── create_dataset.py       # Dataset loader
├── utils.py                # Utility functions
├── base_model.py           # Base model class
├── lr_scheduler.py         # Learning rate schedulers
├── module_util.py          # Module utilities
├── linePromgram.py         # WGAN-QC optimization
├── wb_util.py              # Wandb utilities
└── val_imgs/               # Validation images
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd src_ir_with_edgeloss_3c
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Wandb (optional but recommended):
```bash
wandb login
```

## Configuration

The training configuration is defined in `opt.yml`. Key parameters:

- **Model**: RRDBNet with 23 blocks, 64 features
- **Scale**: 4x super-resolution
- **Batch Size**: 32
- **Learning Rate**: 1e-4 for both generator and discriminator
- **Loss Weights**: Pixel (0.05), Feature (0.7), GAN (1.0), Edge (0.01)
- **Training Iterations**: 40,001

## Training

### Prepare Dataset

Organize your dataset in the following structure:
```
datasets/
├── train/
│   ├── HR/          # High-resolution training images
│   └── LR/          # Low-resolution training images
└── val/
    ├── HR/          # High-resolution validation images
    └── LR/          # Low-resolution validation images
```

### Start Training

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

Or run directly:
```bash
python train.py
```

## Testing

Test your trained model on a test dataset:

```bash
python test.py --model_path trained_models/40000_G.pth --test_data path/to/test/images --output_path test_results
```

## Inference

Perform super-resolution on a single image:

```bash
python inference.py --model_path trained_models/40000_G.pth --input_path input_image.jpg --output_path output_image.png
```

## Model Architecture

### Generator (RRDBNet)
- **Input**: Low-resolution image (3 channels)
- **Architecture**: 
  - Initial convolution (3→64)
  - 23 RRDB blocks
  - 2x upsampling layers
  - Final convolution (64→3)
- **Output**: High-resolution image (4x scale)

### Discriminator (VGG-based)
- **Input**: High-resolution image (3 channels)
- **Architecture**: 5 convolutional blocks with batch normalization
- **Output**: Real/fake classification

### Loss Functions
1. **Pixel Loss**: L1 loss between generated and ground truth images
2. **Feature Loss**: L2 loss on VGG features for perceptual quality
3. **GAN Loss**: WGAN-QC loss for adversarial training
4. **Edge Loss**: Sobel edge detection loss for edge preservation

## Training Process

1. **Data Loading**: Images are loaded and preprocessed to 128x128 HR and 32x32 LR patches
2. **Forward Pass**: Generator creates super-resolution images
3. **Loss Calculation**: Multiple loss functions are computed
4. **Backward Pass**: Gradients are computed and applied
5. **Validation**: PSNR and SSIM metrics are calculated on validation set
6. **Logging**: Results are logged to Wandb for monitoring

## Results

The model typically achieves:
- **PSNR**: 25-30 dB
- **SSIM**: 0.7-0.8
- **Visual Quality**: High perceptual quality with preserved edges

## Monitoring Training

Training progress can be monitored through:
- **Wandb Dashboard**: Real-time metrics and image visualization
- **Console Output**: PSNR, SSIM, and loss values
- **Saved Models**: Checkpoints every 5000 iterations
- **Validation Images**: Sample results saved during training

## Tips for Better Results

1. **Data Quality**: Use high-quality training images
2. **Data Augmentation**: Enable flip and rotation in config
3. **Learning Rate**: Adjust based on your dataset size
4. **Loss Weights**: Fine-tune loss weights for your specific use case
5. **Training Time**: Train for at least 40,000 iterations
6. **Hardware**: Use GPU for faster training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `opt.yml`
2. **Poor Quality**: Increase training iterations or adjust loss weights
3. **Training Instability**: Check learning rates and loss weights
4. **Wandb Issues**: Ensure proper login and internet connection

### Performance Optimization

1. **Mixed Precision**: Enable for faster training on compatible GPUs
2. **Data Loading**: Increase `n_workers` for faster data loading
3. **Model Parallelism**: Use multiple GPUs for larger models

## Citation

If you use this code in your research, please cite:

```bibtex
@article{srgan_edge_loss,
  title={SRGAN with Edge Loss for Image Super-Resolution},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the original SRGAN paper
- RRDB architecture from ESRGAN
- WGAN-QC implementation for stable training
- Edge loss enhancement for better detail preservation
