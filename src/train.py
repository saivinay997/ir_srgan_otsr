import yaml
from torch.utils.data import DataLoader
from create_dataset import ImageDataloader
from SRGAN_model import SRGANModel
import logging

logging.basicConfig("srgan_exp_0", filemode=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Get the logger
logger = logging.getLogger(__name__)

def main():
    ymlpath = "C:\SaiVinay\Thesis\2024_THESIS_Code\dev_03_02\opt.yml"
    with open(ymlpath) as f:
        opt = yaml.safe_load(f)
    # neglect the resume state as of now.
    resume_state = None
    hr_img_dir = ""
    lr_img_dir = ""
    # Create dataloader with HR and LR images
    dataset = ImageDataloader(hr_img_dir, lr_img_dir)
    dataloader = DataLoader(dataset, batch_size=16,  num_workers=1)

    #Initialize SRGAN model
    model = SRGANModel()

    ## setup traing hyperparameters
    start_epoch = 0
    current_step = 0
    total_epochs = opt["train"]["total_epochs"]

    logger.info(f"Staring the training from epoch: {start_epoch}")

    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(dataloader):
            current_step += 1
            # forward pass    
            model.feed_data(train_data)

            # call for optimizer
            model.optimize_parameters(current_step)

            if epoch % 10 == 0 and epoch!= 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                # model.save_training_state(epoch, current_step)


    
