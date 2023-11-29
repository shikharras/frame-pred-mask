import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam
import torchvision.transforms as traonsforms
#from torchsummary import summary
from torch.utils.data import Dataset,DataLoader, random_split

from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
import albumentations as A

from modules import Unet_model

import logging
import json


class VideoDataset(Dataset):
    """
    Dataset class to load frames and their masks
    """
    def __init__(self, root_dir, transform=None,val=False):
        self.root_dir = root_dir
        self.transform = transform
        self.val=val     #set to true if loading videos from validation set

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if self.val:
            idx+=1000       # Videos 1000-2000 are validation videos
        video_path = os.path.join(self.root_dir, 'video_{}'.format(idx))
        frame=[]
        mask_list=[]
        for fn in range(22): ##### can do something better than looping?
            image_path=os.path.join(video_path,'image_{}.png'.format(fn))
            img=np.array(Image.open(image_path))
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

def check_accuracy(loader, model, device):
    """
    Helper function to check accuracy of predicted mask and true mask
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x=torch.stack(x,dim=1)
            y=torch.stack(y,dim=1)
            x = x.to(device)
            y = y.to(device)
            for i in range(22):
                d=x[:,i,:,:,:]
                t=y[:,i,:,:]
                softmax = nn.Softmax(dim=1)
                preds = torch.argmax(softmax(model(d)),axis=1)

                num_correct += (preds == t).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * t).sum()) / ((preds + t).sum() + 1e-8)

    logging.info(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    logging.info(f"Dice score: {dice_score/len(loader)}")

def train(cfg_dict, train_loader, val_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet_model().to(device)
    lr = cfg_dict["seg_lr"]
    num_epochs = cfg_dict["seg_epochs"]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    old_val_loss=float('inf')

    for epoch in range(num_epochs):
        """
        Training Loop
        """
        model.train()
        loop = tqdm(enumerate(train_loader),total=len(train_loader))
        for batch_idx, (data, targets) in loop:
            data=torch.stack(data,dim=1)
            targets=torch.stack(targets,dim=1)
            data = data.to(device)
            targets = targets.to(device)
            targets = targets.type(torch.long)
            #forward pass
            for i in range(22):
                d=data[:,i,:,:,:].to(device)
                t=targets[:,i,:,:].to(device)
                #### why is the shape of t 7??? is it batch size??

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
                targets = targets.type(torch.long)
                for i in range(22):
                    d=data[:,i,:,:,:].to(device)
                    t=targets[:,i,:,:].to(device)
                    with torch.cuda.amp.autocast():
                        output=model(d)
                        loss=loss_fn(output,t)
                    val_loss+=loss.item()
            check_accuracy(val_loader, model, device)
            
        if val_loss<old_val_loss:     #save model if validation loss decreases
            torch.save({
                'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(cfg_dict["model_root"], "seg_model", "segmentation_model.pth"))
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

    cfg_dict = load_config(args.cfg)

    logging.basicConfig(filename= cfg_dict['log_dir'] + 'seg_train.log',filemode='a',level=logging.INFO)

    logging.info(f"Loaded config file and initialized logging")
    logging.info(f"CFG_Dict: {cfg_dict}")

    train_data_dir = os.path.join(cfg_dict["dataset_root"], "train")
    val_data_dir = os.path.join(cfg_dict["dataset_root"], "val")

    batch_size = cfg_dict["seg_batch_size"]

    t1 = A.Compose([A.Resize(160,240),
                A.augmentations.transforms.Normalize(mean=[0.5, 0.5,0.5], std=[0.5,0.5,0.5]),
                # ^ normalizing the image - should this be done
                ToTensorV2()]) # making it channel-first
    
    train_dataset = VideoDataset(train_data_dir ,transform=t1)
    val_dataset= VideoDataset(val_data_dir,transform=t1,val=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg_dict["seg_batch_size"], shuffle=True,num_workers=1)
    val_loader=DataLoader(val_dataset,batch_size=cfg_dict["seg_batch_size"],shuffle=False,num_workers=1)

    os.makedirs(cfg_dict["model_root"] + "seg_model/", exist_ok=True)
    train(cfg_dict, train_loader, val_loader)





