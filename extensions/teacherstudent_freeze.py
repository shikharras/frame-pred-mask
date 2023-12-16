from tqdm import tqdm

import os
import numpy as np
import json
import logging

import torch
from torch import nn

import time
import pickle
import argparse

from albumentations.pytorch import ToTensorV2
import albumentations as A

import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image


import torchvision.transforms.functional as F

from modules import SimVP, UNet, SimVPSegmentation

class VideoMaskDataset(torch.utils.data.Dataset):
    """
    Dataset class to load frames and their masks
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.folder_paths = [os.path.join(self.root_dir, i) for i in os.listdir(self.root_dir) if 'video_' in i]

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, index):
        folder_name = self.folder_paths[index]
        mask_path = os.path.join(folder_name, 'mask.npy')
        masks = np.load(mask_path)
        image_filenames = [i for i in os.listdir(folder_name) if i.endswith('.png')]
        # print(folder_name, image_filenames)
        image_filenames.sort(key= lambda i: int(i.lstrip('image_').rstrip('.png')))

        input_images = []
        target_masks = []

        for i, image_filename in enumerate(image_filenames):
            image_path = os.path.join(self.root_dir, folder_name, f"image_{i}.png")
            image = np.array(Image.open(image_path).convert('RGB'))
            if self.transform is not None:
                aug = self.transform(image = image)
                image = aug['image']
                input_images.append(image)

        input_tensor = torch.stack(input_images)
        return input_tensor, torch.from_numpy(masks)

def new_forward(self, x_raw):
    B, T, C, H, W = x_raw.shape
    x = x_raw.reshape(B*T, C, H, W)

    embed, skip = self.enc(x)
    _, C_, H_, W_ = embed.shape

    z = embed.view(B, T, C_, H_, W_)
    hid = self.hid(z)
    hid = hid.reshape(B*T, C_, H_, W_)

    Y = self.dec(hid, skip)
    Y = Y.reshape(B, T, 49, H, W)
    return Y

def train(cfg_dict, train_loader, val_loader):
    """
    Train loop
    :param cfg: Config file path
    """

    NUM_FUTURE_FRAMES = 11
    IMG_SIZE = (240,160)

    lr = cfg_dict["fp_lr"]
    epochs = cfg_dict["fp_epochs"]
    log_step = cfg_dict["fp_log_step"]

    file_identifier = f'st_{epochs}'


    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Torch device: {device}")

    simvp = nn.DataParallel(SimVP(shape_in=(11, 3, 240, 160)))
    FFP_wt_path = os.path.join(cfg_dict["model_root"], "frame_pred_model", cfg_dict["fp_model_name"])
    FFP_state_dict = torch.load(FFP_wt_path,map_location="cpu")
    simvp_dic=FFP_state_dict["simvp_state_dict"]
    simvp.load_state_dict(simvp_dic)
    simvp_seg = SimVPSegmentation(shape_in=(11, 3, IMG_SIZE[0], IMG_SIZE[1]))

    simvp_seg = nn.DataParallel(simvp_seg)


    simvp.module.dec = simvp_seg.module.dec
    import types
    simvp.module.forward = types.MethodType(new_forward, simvp.module)

    for param in simvp.module.enc.parameters():
        param.requires_grad = False

    for param in simvp.module.hid.parameters():
        param.requires_grad = False

    simvp.to(device)

    # Define optimizers and LR Schedulers
    optimizer = torch.optim.Adam(simvp.parameters(), lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    ce_criterion = torch.nn.CrossEntropyLoss()
    mse_criterion = torch.nn.MSELoss()



    # Begin training (restart from checkpoint if possible)
    start_epoch = 0
    if os.path.isfile(os.path.join(cfg_dict["model_root"], "st", "frozen" + cfg_dict["st_model_name"])):
        logging.info("Restarting training from checkpoint")
        checkpoint = torch.load(os.path.join(cfg_dict["model_root"], "st", "frozen" + cfg_dict["st_model_name"]))
        simvp.load_state_dict(checkpoint["simvp_state_dict"])
        optimizer.load_state_dict(checkpoint["simvp_optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["simvp_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]

    # Value trackers
    train_loss_list = []
    test_loss_list = []
    train_metric_list = []
    test_metric_list = []

    best_avg_val_loss = 1000

    # Train loop

    unet = UNet()

    SM_wt_path = os.path.join(cfg_dict["model_root"], "seg_model", cfg_dict["seg_model_name"])
    SM_state_dict=torch.load(SM_wt_path, map_location="cpu")
    unet_dic=SM_state_dict["model_state_dict"]
    unet.load_state_dict(unet_dic)
    unet = unet.to(device)

    unet.eval()

    for epoch in range(start_epoch, epochs):
        logging.info(f"Epoch Number: {epoch}")
        train_pbar = tqdm(train_loader)
        simvp.train()
        train_loss_list = []
        for step, (all_frames, all_masks) in enumerate(train_pbar):
            all_frames, all_masks = all_frames.to(device), all_masks.to(device)
            all_masks = all_masks.type(torch.long)
            #print(all_frames.shape, all_masks.shape)
            teacher_dec_output_list = []
            with torch.no_grad():
                for i in range(11,22):
                    teacher_dec_output_i = unet(all_frames[:, i, :, :, :])
                    teacher_dec_output_list.append(teacher_dec_output_i)
                teacher_dec_output = torch.stack(teacher_dec_output_list, dim=1)

            student_dec_output = simvp(all_frames[:, :11, :, :, :])
            #print(f"Student decoder output shape:  {student_dec_output.shape}")

            distillation_loss = (1/10) * mse_criterion(student_dec_output, teacher_dec_output)

            forecasting_loss = ce_criterion(student_dec_output[:, 10, :, :, :], all_masks[:, 21, :, :])

            logging.info(f"Distillation Loss: {distillation_loss}, Forcasting Loss: {forecasting_loss}")

            total_loss = distillation_loss + forecasting_loss

            train_loss_list.append(total_loss.item())
            train_pbar.set_description('train loss: {:.4f}'.format(total_loss.item()))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

        if (epoch) % log_step == 0:
        # Evaluate and test the model
            with torch.no_grad():
                simvp.eval()
                preds_lst, trues_lst, total_val_loss = [], [], []
                vali_pbar = tqdm(val_loader)
                for i, (all_frames_test, all_masks_test) in enumerate(vali_pbar):
                    if i * all_frames_test.shape[0] > 1000:
                        break


                    all_frames_test, all_masks_test = all_frames_test.to(device), all_masks_test.to(device)
                    all_masks_test = all_masks_test.type(torch.long)

                    pred_future_masks_test = simvp(all_frames_test[:, :11, :, :, :])

                    loss = ce_criterion(pred_future_masks_test[:, 10, :, :, :], all_masks_test[:, 21, :, :])
                    vali_pbar.set_description(
                        'vali loss: {:.4f}'.format(loss.mean().item()))
                    total_val_loss.append(loss.mean().item())


                avg_val_loss = np.average(total_val_loss)

                simvp.train()
                logging.info(f"Epoch: {epoch + 1} | Train Loss: {total_loss} Vali Loss: {avg_val_loss}")


                if (avg_val_loss<best_avg_val_loss):
                    best_avg_val_loss=avg_val_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "simvp_state_dict": simvp.state_dict(),
                            "simvp_optimizer_state_dict": optimizer.state_dict(),
                            "simvp_scheduler_state_dict": lr_scheduler.state_dict()
                        },
                        cfg_dict["model_root"] + "st/" + "/frozen" + f"model_{file_identifier}.pth",
                    )


def load_config(file_path_or_dict='config.json'):
    if type(file_path_or_dict) is str:
        config = dict(json.load(open(file_path_or_dict)))
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default=os.path.join(os.getcwd(), "config.json"),
        type=str,
        required=False,
        help="Training config file path",
    )

    args = parser.parse_args()

    cfg_dict = load_config(args.cfg)

    logging.basicConfig(filename= cfg_dict['log_dir'] + 'st_train_frozen.log',filemode='a',level=logging.INFO)

    logging.info(f"Loaded config file and initialized logging")
    logging.info(f"CFG_Dict: {cfg_dict}")

    train_data_dir = os.path.join(cfg_dict["dataset_root"], "train")
    val_data_dir = os.path.join(cfg_dict["dataset_root"], "val")
    unlabeled_data_dir = os.path.join(cfg_dict["dataset_root"], "unlabeled")

    batch_size = cfg_dict["fp_batch_size"]

    use_unlabaled_count = cfg_dict["fp_use_unlabeled_count"]

    t1 = A.Compose([A.augmentations.transforms.Normalize(mean=[0.5, 0.5,0.5], std=[0.5,0.5,0.5]),ToTensorV2()]) # making it channel-first

    # Create dataloaders for train and test dataset
    train_dataset = VideoMaskDataset(
        root_dir=f'{train_data_dir}', 
        transform=t1)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False)


    val_dataset = VideoMaskDataset(
        root_dir=f'{val_data_dir}',
        transform=t1)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False)

    os.makedirs(cfg_dict["model_root"] + "st/", exist_ok=True)

    train(cfg_dict, train_loader, val_loader)