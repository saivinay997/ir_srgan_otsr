import math
import yaml
from torch.utils.data import DataLoader
from create_dataset import ImageDataloader
from SRGAN_model import SRGANModel
import logging
from tqdm.auto import tqdm
import os
import utils
import torch
import wandb
import wb_util
import datetime



curr_time = datetime.datetime.now().strftime("%d-%m_%H-%M")

# Get the logger
# logger = utils.setup_logger(log_file='ir_sr_gan_training.log')

def main(HR_train, HR_val, ymlpath, val_results_path, trained_model_path, note):
    # Load configuration
    with open(ymlpath) as f:
        opt = yaml.safe_load(f)
    
    # Validate paths
    if not os.path.exists(HR_train):
        raise FileNotFoundError(f"Training HR directory not found: {HR_train}")
    if not os.path.exists(HR_val):
        raise FileNotFoundError(f"Validation HR directory not found: {HR_val}")
    
    # Create output directories
    os.makedirs(val_results_path, exist_ok=True)
    os.makedirs(trained_model_path, exist_ok=True)
    
    # neglect the resume state as of now.
    resume_state = None
    seed = opt['train']['manual_seed']
    utils.set_random_seed(seed)
    torch.backends.cudnn.benchmark = True  # Fixed typo: benckmark -> benchmark
    
    # Create dataloader with HR and LR images
    try:
        dataset = ImageDataloader(HR_train, val=False)
        dataloader = DataLoader(dataset, batch_size=opt["datasets"]["train"]["batch_size"], 
                               num_workers=4, shuffle=True, drop_last=True)

        val_dataset = ImageDataloader(HR_val, val=True)
        val_loader = DataLoader(val_dataset, batch_size=opt["datasets"]["train"]["batch_size"], 
                               num_workers=4, shuffle=False, drop_last=True)
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        raise

    # Initialize SRGAN model
    try:
        model = SRGANModel(opt)
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

    ## setup training hyperparameters
    start_epoch = 0
    current_step = 0
    total_epochs = 0
    print(f"Dataset size: {len(dataset)}")
    train_size = int(math.ceil(len(dataset) / opt['datasets']['train']['batch_size']))
    total_iters = int(opt["train"]['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))

    # Initialize wandb
    try:
        wandb.init(
            # Set the project where this run will be logged
            project="sr_gan_training", 
            # We pass a run name (otherwise it'll be randomly assigned, like sunshine-lollypop-10)
            name=f"sobel_edge_loss_RGB_{curr_time}", 
            # Track hyperparameters and run metadata
            config={
                "Note": note,
                "epochs": total_epochs,
                "total_iters": total_iters,
                "batch_size": opt["datasets"]["train"]["batch_size"],
                "pixel_criterion": opt["train"]["pixel_criterion"],
                "feature_criterion": opt["train"]["feature_criterion"],
                "gan_type": opt["train"]["gan_type"],
                "train_parameters": opt["train"]
                # "learning_rate": opt["train"]["lr"]
            })
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
        wandb = None

    print(f"Starting the training from epoch: {start_epoch}. Total epoch: {total_epochs}")
    # logger.info(f"Starting the training from epoch: {start_epoch}. Total epoch: {total_epochs}")
    max_psnr, max_ssim = 0.0, 0.0
    
    try:
        for epoch in tqdm(range(start_epoch, total_epochs + 1), desc="Epochs: "):
            for batch_idx, (hr_imgs, lr_imgs) in enumerate(dataloader):
                current_step += 1
                
                # Check if we've reached the total iterations
                if current_step > total_iters:
                    break
                
                try:
                    # forward pass    
                    model.feed_data(hr_imgs, lr_imgs)
                    # call for optimizer
                    model.optimize_parameters(current_step)

                    model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
                    
                    if current_step % 50 == 0 and val_loader is not None:
                        avg_psnr = val_pix_err_f = val_pix_err_nf = val_mean_color_err = avg_ssim = 0.0
                        _lr_img = _hr_img = _sr_img = None
                        idx = 0
                        
                        # Validation loop
                        for val_hr, val_lr in val_loader:
                            if idx > 4:
                                break 
                            idx += 1
                            img_name = f"img_{idx}"
                            img_dir = os.path.join(val_results_path, img_name)
                            if not os.path.exists(img_dir):
                                os.makedirs(img_dir)
                            
                            model.feed_data(val_hr, val_lr)
                            model.test()

                            visuals = model.get_current_visuals()
                            sr_img  = utils.tensor2img(visuals['SR'])
                            gt_img  =  utils.tensor2img(visuals["GT"])
                            # print(sr_img.shape, gt_img.shape, visuals["LQ"].shape)
                            # log the images to wandb
                            _lr_img = visuals['LQ'].to("cpu").permute(1, 2, 0).numpy()
                            _hr_img = visuals['GT'].to("cpu").permute(1, 2, 0).numpy()
                            _sr_img = visuals['SR'].to("cpu").permute(1, 2, 0).numpy()
                            
                            # print("Starting to save image.")
                            if current_step%100 == 0:
                                save_img_path = os.path.join(img_dir, f"{current_step}_{idx}.png")
                                utils.save_img(sr_img, save_img_path)

                            crop_size = opt["scale"]
                            gt_img = gt_img / 255.
                            sr_img = sr_img / 255.
                            cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
                            cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]
                            avg_psnr += utils.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                            avg_ssim += utils.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                        
                        avg_psnr = avg_psnr / idx
                        avg_ssim = avg_ssim / idx
                        val_pix_err_f /= idx
                        val_pix_err_nf /= idx
                        val_mean_color_err /= idx

                        if current_step % 100 == 0 and wandb is not None:
                            # max_psnr = avg_psnr
                            wb_util.log_image_table(lr_imgs=_lr_img, hr_imgs=_hr_img, sr_imgs=_sr_img, 
                                                    psnr = avg_psnr, ssim = avg_ssim, step=current_step)
                        if avg_psnr > max_psnr:
                            max_psnr = avg_psnr
                            if wandb is not None:
                                wandb.log({"max_psnr":max_psnr})
                        if avg_ssim > max_ssim:
                            max_ssim = avg_ssim
                            if wandb is not None:
                                wandb.log({"max_ssim":max_ssim})
                            
                        # logger.info(f"Epoch: {epoch} | iters: {current_step} | PSNR: {avg_psnr}")
                        # logger.info(f"# Validation # PSNR: {avg_psnr}")
                        # logger.info(f"# Validation # SSIM: {avg_ssim}")

                        if wandb is not None:
                            metrics = {"PSNR": avg_psnr, 
                                       "SSIM": avg_ssim,
                                       "epoch": epoch,
                                       "iters": current_step}
                            wandb.log(metrics)

                except Exception as e:
                    print(f"Error in training step {current_step}: {e}")
                    continue

            if epoch % 50 == 0 and epoch!= 0:
                print(f'Saving models and training states at epoch {epoch}')
                model.save(current_step, trained_model_path)
                # model.save_training_state(epoch, current_step)

        print(f'Saving models and training states at final epoch {epoch}')
        model.save(current_step, trained_model_path)
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        raise
    finally:
        if wandb is not None:
            wandb.finish()

if __name__ == "__main__":
    # Example usage
    HR_train = "path/to/train/HR"
    HR_val = "path/to/val/HR"
    ymlpath = "opt.yml"
    val_results_path = "val_results"
    trained_model_path = "trained_models"
    note = "Training with edge loss"
    
    main(HR_train, HR_val, ymlpath, val_results_path, trained_model_path, note)