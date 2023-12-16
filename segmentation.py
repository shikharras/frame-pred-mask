import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader, random_split

from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
import albumentations as A

from modules import UNet, SimVP, FramePredictionModel
import random
from torchvision import transforms

import logging
import json


class SegmentationDataset(Dataset):
    """
    Dataset class to load frames and their masks
    """
    def __init__(self, root_dir, transform=None,val=False):
        self.root_dir = root_dir
        self.transform = transform
        self.folder_paths = [os.path.join(self.root_dir, i) for i in os.listdir(self.root_dir) if 'video_' in i]

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        video_path = self.folder_paths[idx]
        frame=[]
        mask_list=[]
        for fn in range(22): ##### can do something better than looping?
            image_path=os.path.join(video_path,'image_{}.png'.format(fn))
            img=np.array(Image.open(image_path))/1.
            mask_path = os.path.join(video_path, 'mask.npy')
            masks = np.load(mask_path)
            mask=masks[fn]
            if self.transform is not None:
                aug = self.transform(image=img,mask=mask)
                img = aug['image']
                mask = aug['mask']
            frame.append(img)
            mask_list.append(mask)
        return frame,mask_list

def train(cfg_dict, train_loader, val_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    lr = cfg_dict["seg_lr"]
    num_epochs = cfg_dict["seg_epochs"]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    old_val_loss=float('inf')

    simvp = FramePredictionModel(cfg_dict)
    simvp_path = os.path.join(cfg_dict["model_root"], "frame_pred_model", cfg_dict["fp_model_name"])
    simvp_state_dict = torch.load(simvp_path,map_location="cpu")["state_dict"]

    simvp.load_state_dict(simvp_state_dict)
    simvp=simvp.to(device)

    simvp.eval()

    logging.info("Loaded simvp model")

    for epoch in range(num_epochs):
        print(f"Epoch Number: {epoch}")
       
        model.train()
        loop = tqdm(enumerate(train_loader),total=len(train_loader))
        for batch_idx, (data, targets) in loop:
            data=torch.stack(data,dim=1)
            targets=torch.stack(targets,dim=1)
            data = data.to(device)
            data = data.type(torch.float)
            targets = targets.to(device)
            targets = targets.type(torch.long)
           
         
            if random.random() > 0.65:
                with torch.no_grad():
                    print("data shape: ", data.shape)
                    pred_future_frames = simvp(data[:,:11,:,:,:])
                    print("pff shape: ", pred_future_frames.shape)
                    target_frames=pred_future_frames #Getting all predicted frames
                    print("target_frames: ", target_frames.shape)
                data[:, 11:, :, :, :] = target_frames



            for i in range(22):
                d=data[:,i,:,:,:].to(device)
                t=targets[:,i,:,:].to(device)

                with torch.cuda.amp.autocast():
                    predictions = model(d)
                    loss = loss_fn(predictions, t)
                #backward prop
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                #update tqdm loop
                loop.set_postfix(loss=loss.item())
        model.eval()
        val_loss=0.0
        with torch.no_grad():
            for (data,targets) in val_loader:
                data=torch.stack(data,dim=1).to(device)
                targets=torch.stack(targets,dim=1).to(device)
                data = data.type(torch.float)
                targets = targets.type(torch.long)
                for i in range(22):
                    d=data[:,i,:,:,:].to(device)
                    t=targets[:,i,:,:].to(device)
                    with torch.cuda.amp.autocast():
                        output=model(d)
                        loss=loss_fn(output,t)
                    val_loss+=loss.item()
            
        if True:     #save all models
            torch.save({
                'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(cfg_dict["model_root"], "seg_model", f"segmentation_model_{epoch}.pth"))
            old_val_loss = val_loss
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] - val loss: {val_loss/len(val_loader):.4f}")


def load_config(file_path_or_dict='config.json'):
    if type(file_path_or_dict) is str:
        config = dict(json.load(open(file_path_or_dict)))
    return config

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default=os.path.join(os.getcwd(), "config.json"),
        type=str,
        required=False,
        help="Training config file path",
    )

    args = parser.parse_args()

    cfg_dict = dict(json.load(open(args.cfg)))

    logging.basicConfig(filename= cfg_dict['log_dir'] + 'seg_train_int.log',filemode='a',level=logging.INFO)

    logging.info(f"Loaded config file and initialized logging")
    logging.info(f"CFG_Dict: {cfg_dict}")

    train_data_dir = os.path.join(cfg_dict["dataset_root"], "train")
    val_data_dir = os.path.join(cfg_dict["dataset_root"], "val")

    batch_size = cfg_dict["seg_batch_size"]

    t1 = A.Compose([A.Resize(160,240),
                A.augmentations.transforms.Normalize(mean=[0.5, 0.5,0.5], std=[0.5,0.5,0.5]),
                # ^ normalizing the image - should this be done
                ToTensorV2(),
                ]) # making it channel-first
    
    train_dataset = SegmentationDataset(train_data_dir ,transform=t1)
    val_dataset= SegmentationDataset(val_data_dir,transform=t1,val=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg_dict["seg_batch_size"], shuffle=True,num_workers=1)
    val_loader=DataLoader(val_dataset,batch_size=cfg_dict["seg_batch_size"],shuffle=False,num_workers=1)

    os.makedirs(cfg_dict["model_root"] + "seg_model/", exist_ok=True)
    train(cfg_dict, train_loader, val_loader)