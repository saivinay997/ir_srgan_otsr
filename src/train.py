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
logger = utils.setup_logger(log_file='sr_gan_training_01.log')

def main(HR_train, HR_val, ymlpath, val_results_path, trained_model_path):
    with open(ymlpath) as f:
        opt = yaml.safe_load(f)
    # neglect the resume state as of now.
    resume_state = None
    seed = opt['train']['manual_seed']
    utils.set_random_seed(seed)
    torch.backends.cudnn.benckmark = True
    # Create dataloader with HR and LR images
    dataset = ImageDataloader(HR_train)
    dataloader = DataLoader(dataset, batch_size=opt["datasets"]["train"]["batch_size"],  num_workers=4)

    val_dataset = ImageDataloader(HR_val)
    val_loader = DataLoader(val_dataset, batch_size=opt["datasets"]["train"]["batch_size"], num_workers=4)

    #Initialize SRGAN model
    model = SRGANModel(opt)

    ## setup traing hyperparameters
    start_epoch = 0
    current_step = 0
    total_epochs = 0
    print(len(dataset))
    train_size = int(math.ceil(len(dataset) / opt['datasets']['train']['batch_size']))
    total_iters = int(opt["train"]['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))

    wandb.init(
        # Set the project where this run will be logged
        project="super-resolution-gan", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{curr_time}", 
        # Track hyperparameters and run metadata
        config={
        "epochs": total_epochs,
        "total_iters": total_iters,
        "batch_size": opt["datasets"]["train"]["batch_size"],
        # "learning_rate": opt["train"]["lr"]
        })

    print(f"Staring the training from epoch: {start_epoch}. Total epoch: {total_epochs}")
    logger.info(f"Staring the training from epoch: {start_epoch}. Total epoch: {total_epochs}")
    max_psnr = 0.0
    for epoch in tqdm(range(start_epoch, total_epochs + 1), desc="Epochs: "):
        for _, (hr_imgs, lr_imgs) in enumerate(dataloader):
            current_step += 1
            # forward pass    
            model.feed_data(hr_imgs, lr_imgs)
            # call for optimizer
            model.optimize_parameters(current_step)

            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            
            if current_step % 50 == 0 and val_loader is not None:
                avg_psnr = val_pix_err_f = val_pix_err_nf = val_mean_color_err = avg_ssim = 0.0
                _lr_img = _hr_img = _sr_img = None
                idx = 0
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

                    # log the images to wandb
                    _lr_img = visuals['LQ'].to("cpu").numpy()
                    _hr_img = visuals['GT'].to("cpu").numpy()
                    _sr_img = visuals['SR'].to("cpu").numpy()
                    
                    # print("Starting to save image.")
                    save_img_path = os.path.join(img_dir, f"{epoch}_{idx}.png")
                    utils.save_img(sr_img, save_img_path)

                    crop_size = opt["scale"]
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += utils.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    avg_ssim += utils.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
                
                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                val_pix_err_f /= idx
                val_pix_err_nf /= idx
                val_mean_color_err /= idx

                if current_step % 500 == 0 or avg_psnr > max_psnr:
                    max_psnr = avg_psnr
                    wb_util.log_image_table(lr_imgs=_lr_img, hr_imgs=_hr_img, sr_imgs=_sr_img, 
                                            psnr = avg_psnr, ssim = avg_ssim, step=current_step)
                    
                logger.info(f"Epoch: {epoch} | iters: {current_step} | PSNR: {avg_psnr}")
                logger.info(f"# Validation # PSNR: {avg_psnr}")
                logger.info(f"# Validation # SSIM: {avg_ssim}")

                metrics = {"PSNR": avg_psnr, 
                           "SSIM": avg_ssim}
                wandb.log(metrics, step=current_step)

        if epoch % 100 == 0 and epoch!= 0:
            print(f'Saving models and training states at epoch {epoch}')
            model.save(current_step, trained_model_path)
            # model.save_training_state(epoch, current_step)

    print(f'Saving models and training states at epoch {epoch}')
    model.save(current_step, trained_model_path)
