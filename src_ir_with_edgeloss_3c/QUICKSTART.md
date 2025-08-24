# Quick Start Guide - SRGAN with Edge Loss

This guide will help you get started with the SRGAN project quickly.

## Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- At least 8GB RAM

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up Wandb (optional but recommended):**
```bash
wandb login
```

## Quick Start

### 1. Prepare Your Data

Place your high-resolution images in the `imgsResizedHR` directory (or modify the path in `run_training.py`).

### 2. Start Training

Run the simple training script:
```bash
python run_training.py
```

This will:
- Use default parameters from `opt.yml`
- Train for 40,001 iterations
- Save models every 5,000 iterations
- Log results to Wandb
- Save validation results

### 3. Monitor Training

- **Console**: Watch PSNR and SSIM metrics
- **Wandb**: View real-time dashboard with images and metrics
- **Local Files**: Check `val_results` and `trained_models` directories

### 4. Test Your Model

After training, test your model:
```bash
python test.py --model_path trained_models/40000_G.pth --test_data imgsResizedHR --output_path test_results
```

### 5. Use for Inference

Generate super-resolution for a single image:
```bash
python inference.py --model_path trained_models/40000_G.pth --input_path your_image.jpg --output_path result.png
```

## Advanced Usage

### Custom Training

Modify `opt.yml` for custom parameters:
- Change batch size, learning rate, loss weights
- Adjust training iterations
- Modify network architecture

### Dataset Organization

Use the dataset setup script:
```bash
# Create LR-HR pairs
python setup_dataset.py --mode create_pairs --hr_dir your_hr_images --lr_dir your_lr_images

# Split dataset
python setup_dataset.py --mode split --data_dir your_dataset

# Validate dataset structure
python setup_dataset.py --mode validate --data_dir your_dataset
```

### Demo Mode

Showcase your model:
```bash
python demo.py --model_path trained_models/40000_G.pth --input_path demo_image.jpg --output_dir demo_results
```

## Expected Results

With proper training, you should achieve:
- **PSNR**: 25-30 dB
- **SSIM**: 0.7-0.8
- **Visual Quality**: High perceptual quality with preserved edges

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `opt.yml`
   - Use smaller image patches

2. **Poor Quality Results**
   - Increase training iterations
   - Check data quality
   - Adjust loss weights

3. **Training Instability**
   - Reduce learning rate
   - Check loss weight balance
   - Ensure proper data preprocessing

### Performance Tips

1. **Faster Training**
   - Use GPU with more VRAM
   - Increase batch size if possible
   - Use mixed precision training

2. **Better Quality**
   - Use high-quality training data
   - Train for longer (60k+ iterations)
   - Fine-tune loss weights

## File Structure After Training

```
src_ir_with_edgeloss_3c/
├── trained_models/           # Saved model checkpoints
│   ├── 5000_G.pth
│   ├── 10000_G.pth
│   └── 40000_G.pth
├── val_results/             # Validation results
│   └── img_1/
│       ├── 1000_1.png
│       └── 2000_1.png
└── test_results/            # Test results (after testing)
    ├── test_0000_SR.png
    └── test_0000_GT.png
```

## Next Steps

1. **Experiment**: Try different loss weights and architectures
2. **Optimize**: Fine-tune for your specific use case
3. **Deploy**: Use the trained model in your applications
4. **Extend**: Add new features like different upscaling factors

## Support

- Check the main README.md for detailed documentation
- Review the code comments for implementation details
- Use Wandb for experiment tracking and comparison
