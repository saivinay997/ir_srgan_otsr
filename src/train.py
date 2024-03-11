import math
import yaml
from torch.utils.data import DataLoader
from create_dataset import ImageDataloader
from SRGAN_model import SRGANModel
import logging
from tqdm.auto import tqdm
import os
import utils


logging.basicConfig(filename="srgan_exp_0.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Get the logger
logger = utils.setup_logger(log_file='sr_gan_training_01.log')

def main(HR_train, Lr_train, HR_val, LR_val):
    ymlpath = "./opt.yml"
    with open(ymlpath) as f:
        opt = yaml.safe_load(f)
    # neglect the resume state as of now.
    resume_state = None
    
    # Create dataloader with HR and LR images
    dataset = ImageDataloader(HR_train, Lr_train)
    dataloader = DataLoader(dataset, batch_size=opt["datasets"]["train"]["batch_size"],  num_workers=4)

   

    val_dataset = ImageDataloader(HR_val, LR_val)
    val_loader = DataLoader(val_dataset, batch_size=opt["datasets"]["train"]["batch_size"], num_workers=4)

    val_results_path = r"C:\SaiVinay\SproutsAI\GitHub_\ir_srgan_otsr\src\val_results"

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


    print(f"Staring the training from epoch: {start_epoch}. Total epoch: {total_epochs}")
    logger.info(f"Staring the training from epoch: {start_epoch}. Total epoch: {total_epochs}")

    for epoch in tqdm(range(start_epoch, total_epochs + 1), desc="Epochs: "):
        for _, (hr_imgs, lr_imgs) in enumerate(dataloader):
            current_step += 1
            # forward pass    
            model.feed_data(hr_imgs, lr_imgs)

            # call for optimizer
            model.optimize_parameters(current_step)

            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            
            if epoch % 10 == 0 and val_loader is not None:
                avg_psnr = val_pix_err_f = val_pix_err_nf = val_mean_color_err = avg_ssim = 0.0
                idx = 0
                for val_hr, val_lr in val_loader:
                    if idx > 2:
                        break 
                    idx += 1
                    img_name = f"{epoch}_{idx}"
                    img_dir = os.path.join(val_results_path, img_name)
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    
                    model.feed_data(val_hr, val_lr)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = utils.tensor2img(visuals['SR'])
                    gt_img = utils.tensor2img(visuals["GT"])

                    print("Starting to save image.")
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

                logger.info(f"Epoch: {epoch} | iters: {current_step} | PSNR: {avg_psnr}")
                logger.info(f"# Validation # PSNR: {avg_psnr}")
                logger.info(f"# Validation # SSIM: {avg_ssim}")



        if epoch % 50 == 0 and epoch!= 0:
            print(f'Saving models and training states at epoch {epoch}')
            model.save(current_step)
            # model.save_training_state(epoch, current_step)


    print(f'Saving models and training states at epoch {epoch}')
    model.save(current_step)
