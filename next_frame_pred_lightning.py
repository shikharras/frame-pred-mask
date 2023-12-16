import os
from tqdm import tqdm
import numpy as np
import json
import logging
import torch
from torch import nn
import time
import pickle
import argparse
import torchvision
import torch.utils.data
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms.functional as F
from modules import SimVP #.  to be added for HPC
#import svp #
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
from modules import FramePredictionModel




class FFPDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, unlabeled_dir, use_unlabeled_count=0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.unlabeled_dir = unlabeled_dir
        unlabeled_folders = []
        if use_unlabeled_count>0:
            unlabeled_folders = [os.path.join(unlabeled_dir, i) for i in os.listdir(unlabeled_dir) if 'video_' in i][:use_unlabeled_count]

        # Get the list of folder names in the root directory
        self.folder_paths = [os.path.join(self.root_dir, i) for i in os.listdir(self.root_dir) if 'video_' in i]
        self.folder_paths.extend(unlabeled_folders)

    def __len__(self):
        # Return the number of folders in the root directory
        return len(self.folder_paths)

    def __getitem__(self, index):
        folder_name = self.folder_paths[index]
        image_filenames = [i for i in os.listdir(folder_name) if i.endswith('.png')]
        # print(folder_name, image_filenames)
        image_filenames.sort(key= lambda i: int(i.lstrip('image_').rstrip('.png')))

        # Load the input images and target images into separate tensors
        input_images = []
        target_images = []
        for i, image_filename in enumerate(image_filenames):
            image_path = os.path.join(self.root_dir, folder_name, f"image_{i}.png")
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                aug = self.transform(image)
            if i < 11:
                input_images.append(aug)
            else:
                target_images.append(aug)

        # Convert the input and target image lists to tensors
        input_tensor = torch.stack(input_images)
        target_tensor = torch.stack(target_images)

        return input_tensor, target_tensor


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
    #len_trainloader=args.len_trainloader
    cfg_dict = config = dict(json.load(open(args.cfg)))
    logging.basicConfig(filename=cfg_dict['log_dir'] + 'fp_train.log', filemode='a', level=logging.INFO)
    logging.info(f"Loaded config file and initialized logging")
    logging.info(f"CFG_Dict: {cfg_dict}")

    train_data_dir = os.path.join(cfg_dict["dataset_root"], "train")
    val_data_dir = os.path.join(cfg_dict["dataset_root"], "val")
    unlabeled_data_dir = os.path.join(cfg_dict["dataset_root"], "unlabeled")

    batch_size = cfg_dict["fp_batch_size"]
    use_unlabeled_count = cfg_dict["fp_use_unlabeled_count"]

    train_dataset = FFPDataset(
        root_dir=f'{train_data_dir}',
        unlabeled_dir=unlabeled_data_dir,
        use_unlabeled_count=cfg_dict["fp_use_unlabeled_count"],
        transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

        )

    val_dataset = FFPDataset(
        root_dir=f'{val_data_dir}',
        unlabeled_dir=unlabeled_data_dir,
        transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=7
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=7
    )

    frame_pred_model = FramePredictionModel(cfg_dict,len(train_loader))

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg_dict["model_root"] + "frame_pred_model/",
        filename=f"model_fp_{cfg_dict['fp_epochs']}" + "-{epoch:02d}-{val_loss:.6f}",
        save_top_k=1,
        mode='min'
    )
    trainer = Trainer(
        max_epochs=cfg_dict['fp_epochs'],
        accelerator="gpu",
        default_root_dir=cfg_dict["model_root"] + "frame_pred_checkpoints/",
        callbacks=[checkpoint_callback],
    )

    # ckpt_path=os.path.join(cfg_dict["model_root"] + "/frame_pred_model/", "model_fp_1000-epoch=65-val_loss=0.001249.ckpt")
    trainer.fit(frame_pred_model, train_loader, val_loader)





