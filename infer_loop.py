# importing libraries
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchmetrics

import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

import json
import math
import logging

import argparse

import torchvision.utils as vutils
from torchvision import transforms


from torchvision import transforms
from torch.utils.data import DataLoader

from modules import SimVP, UNet, FramePredictionModel


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.folder_names = [f for f in os.listdir(self.root_dir) if not f.startswith(".")]
        self.folder_names.sort(key= lambda i: int(i.lstrip('video_')))

    def __len__(self):
        # Return the number of folders in the root directory
        return len(self.folder_names)

    def __getitem__(self, index):
        # Get the folder name corresponding to the given index
        folder_name = self.folder_names[index]
        print(folder_name)

        # Get the list of image filenames in the folder
        image_filenames = [i for i in os.listdir(os.path.join(self.root_dir, folder_name))
                           if i.endswith('.png')]
        
        image_filenames.sort(key= lambda i: int(i.lstrip('image_').rstrip('.png')))

        # Load the input images and target images into separate tensors
        input_images = []
        target_images = []
        for i, image_filename in enumerate(image_filenames):
            image_path = os.path.join(self.root_dir, folder_name, f"image_{i}.png")
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            input_images.append(image)

        input_tensor = torch.stack(input_images)

        return input_tensor

class ValDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get the list of folder names in the root directory
        self.folder_names = [f for f in os.listdir(self.root_dir) if not f.startswith(".") ]
        self.folder_names.sort(key= lambda i: int(i.lstrip('video_')))

    def __len__(self):
        # Return the number of folders in the root directory
        return len(self.folder_names)

    def __getitem__(self, index):
        # Get the folder name corresponding to the given index
        folder_name = self.folder_names[index]
        print(folder_name)

        # Get the list of image filenames in the folder
        image_filenames = [i for i in os.listdir(os.path.join(self.root_dir, folder_name))
                           if i.endswith('.png')]
        image_filenames.sort(key= lambda i: int(i.lstrip('image_').rstrip('.png')))

        # Load the input images and target images into separate tensors
        input_images = []
        target_images = []
        for i, image_filename in enumerate(image_filenames):
            image_path = os.path.join(self.root_dir, folder_name, f"image_{i}.png")
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            input_images.append(image)

        mask_path = os.path.join(self.root_dir, folder_name, f"mask.npy")
        masks = np.load(mask_path)
        mask=masks[21]
        input_tensor = torch.stack(input_images)

        return input_tensor, mask
    

def test(cfg_dict, test_dataloader, device, simvp, unet, t2):
    answer_masks=[]
    i=0
    with torch.no_grad():
        #for data , mask_r in test_dataloader:
        for data in test_dataloader:
            i+=1
            data=data.to(device)
            pred_future_frames = simvp(data)
            logging.info(f"pred_future_frames shape:  {pred_future_frames.shape}")
            target_frames=pred_future_frames[:,10,:,:,:]
            target_frames=t2(target_frames) ### called t1 in segmentation.py
            softmax = nn.Softmax(dim=1)
            predicted_mask = torch.argmax(softmax(unet(target_frames)),axis=1).squeeze(0)
            answer_masks.append(predicted_mask)
            logging.info(f"predicted_mask shape: {i, predicted_mask.shape}")
    answer_masks_torch = torch.stack(answer_masks,dim=0).to("cpu")
    logging.info(f"answer_masks shape: {answer_masks_torch.shape}, {type(answer_masks_torch)}")
    torch.save(answer_masks, os.path.join(cfg_dict['model_root'], 'leaderboard_2_team_16.pt'))

    

def evaluate(cfg_dict, val_dataloader, device, simvp, unet, t2):
    # validation
    simvp.eval()
    unet.eval()
    answer_masks=[]
    true_mask = []
    i=0
    with torch.no_grad():
        for data, mask_r in val_dataloader:
            i+=1
            data=data.to(device)
            pred_future_frames = simvp(data[:,:11,:,:,:])
            target_frames=pred_future_frames[:,10,:,:,:]
            target_frames=t2(target_frames) ### called t1 in segmentation.py
            softmax = nn.Softmax(dim=1)
            predicted_mask = torch.argmax(softmax(unet(target_frames)),axis=1).squeeze(0)
            answer_masks.append(predicted_mask)
            mask_r = mask_r.squeeze(0)
            true_mask.append(mask_r)
    answer_masks=torch.stack(answer_masks,dim=0).to("cpu")

    true_mask=torch.stack(true_mask,dim=0).to("cpu")

    logging.info(f"answer_masks shape: {answer_masks.shape}")

    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
    jacc_score = jaccard(answer_masks, true_mask)
    logging.info(f"Evaluation Jaccard Score: {jacc_score}")

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default=os.path.join(os.getcwd(), "config.json"),
        type=str,
        required=False,
        help="Training config file path",
    )
    parser.add_argument(
        "--test",
        action='store_true',
    )

    args = parser.parse_args()

    cfg_dict = dict(json.load(open(args.cfg)))

    logging.basicConfig(filename= cfg_dict['log_dir'] + 'infer_loop.log',filemode='a',level=logging.INFO)

    logging.info(f"Loaded config file and initialized logging")
    logging.info(f"CFG_Dict: {cfg_dict}")

    device="cuda" if torch.cuda.is_available() else "cpu"


    t1 = transforms.Compose([
        transforms.ToTensor(),
    ])

    t2=transforms.Compose([transforms.Resize((160,240)),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),])

    simvp = FramePredictionModel(cfg_dict)

    simvp_path = os.path.join(cfg_dict["model_root"], "frame_pred_model", cfg_dict["fp_model_name"])
    simvp_state_dict = torch.load(simvp_path,map_location="cpu")["state_dict"]

    simvp.load_state_dict(simvp_state_dict)

    simvp=simvp.to(device)

    for i in range(100):
        
        unet = UNet()

        logging.info(f"Now loading model: {str(i)}, {cfg_dict['seg_model_name'] + str(i)}")
        fname = cfg_dict['seg_model_name'].split('.')[0]
        ext = cfg_dict['seg_model_name'].split('.')[1]
        SM_wt_path = os.path.join(cfg_dict["model_root"], "seg_model", fname + '_' + str(i) + '.' + ext)

        SM_state_dict=torch.load(SM_wt_path, map_location="cpu")

        unet_dic=SM_state_dict["model_state_dict"]

        unet.load_state_dict(unet_dic)

        unet=unet.to(device)

        #t2=transforms.Compose([transforms.Resize((160,240))])
        val_data_dir = os.path.join(cfg_dict["dataset_root"], "val")

        val_dataset = ValDataset(root_dir=f'{val_data_dir}', transform=t1)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        evaluate(cfg_dict, val_dataloader, device, simvp, unet, t2)


    if (args.test):
        test_dataset_path = cfg_dict["test_dataset_root"]
        test_dataset = TestDataset(root_dir=f'{test_dataset_path}', transform=t1)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test(cfg_dict, test_dataloader, device, simvp, unet, t2)