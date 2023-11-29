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

import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image


import torchvision.transforms.functional as F

from modules import SimVP
from metrics import combined_metric



class MovingObjectsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, unlabeled_dir, use_unlabeled_count=0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        unlabeled_folders = []
        if use_unlabeled_count>0:
            unlabeled_folders = [i for i in os.listdir(unlabeled_dir) if 'video_' in i][:use_unlabeled_count]

        # Get the list of folder names in the root directory
        self.folder_names = [i for i in os.listdir(self.root_dir) if 'video_' in i]
        self.folder_names.extend(unlabeled_folders)

    def __len__(self):
        # Return the number of folders in the root directory
        return len(self.folder_names)

    def __getitem__(self, index):
        # Get the folder name corresponding to the given index
        folder_name = self.folder_names[index]

        # Get the list of image filenames in the folder
        image_filenames = [i for i in os.listdir(os.path.join(self.root_dir, folder_name))
                           if i.endswith('.png')]
        # print(folder_name, image_filenames)
        image_filenames.sort(key= lambda i: int(i.lstrip('image_').rstrip('.png')))

        # Load the input images and target images into separate tensors
        input_images = []
        target_images = []
        for i, image_filename in enumerate(image_filenames):
            image_path = os.path.join(self.root_dir, folder_name, f"image_{i}.png")
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            if i < 11:
                # print(f"{image_filename} going in input")
                input_images.append(image)
            else:
                # print(f"{image_filename} going in target")
                target_images.append(image)
        
        # Convert the input and target image lists to tensors
        input_tensor = torch.stack(input_images)
        target_tensor = torch.stack(target_images)

        return input_tensor, target_tensor


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(3,3))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])



def visualize(past_frames, true_future_frames, pred_future_frames,
              past_frames_test, true_future_frames_test, pred_future_frames_test,num_future_frame, save_dir):

    logging.info("Visualizing")
    past_frames = torchvision.utils.make_grid(past_frames[0], num_future_frame)
    true_future_frames = torchvision.utils.make_grid(true_future_frames[0], num_future_frame)
    pred_future_frames = torchvision.utils.make_grid(pred_future_frames[0], num_future_frame)
    past_frames_test = torchvision.utils.make_grid(past_frames_test[0], num_future_frame)
    true_future_frames_test = torchvision.utils.make_grid(true_future_frames_test[0], num_future_frame)
    pred_future_frames_test = torchvision.utils.make_grid(pred_future_frames_test[0], num_future_frame)

    plt.imsave(
        save_dir + "past_train.png",
        past_frames.cpu().permute(1, 2, 0).numpy()
    )
    plt.imsave(
        save_dir + "true_future_train.png",
        true_future_frames.cpu().permute(1, 2, 0).numpy(),
    )
    plt.imsave(
        save_dir + "pred_future_train.png",
        pred_future_frames.detach().cpu().permute(1, 2, 0).numpy(),
    )
    plt.imsave(
       save_dir + "past_frames_val.png",
        past_frames_test.cpu().permute(1, 2, 0).numpy(),
    )
    plt.imsave(
        save_dir + "true_future_val.png",
        true_future_frames_test.cpu().permute(1, 2, 0).numpy(),
    )
    plt.imsave(
        save_dir + "pred_future_val.png",
        pred_future_frames_test.detach().cpu().permute(1, 2, 0).numpy(),
    )

    # show(past_frames)
    # show(true_future_frames)
    # show(pred_future_frames)
    # show(past_frames_test)
    # show(true_future_frames_test)
    # show(pred_future_frames_test)

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

    file_identifier = f'fp_{epochs}'


    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Torch device: {device}")

    model = SimVP(shape_in=(11, 3, IMG_SIZE[0], IMG_SIZE[1]))

    #Parallel training: https://stackoverflow.com/questions/54216920/how-to-use-multiple-gpus-in-pytorch
    model = nn.DataParallel(model)
    model.to(device)

    # Define optimizers and LR Schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[10, 20, 30, 40], gamma=0.5
    )

    criterion = torch.nn.MSELoss()



    # Begin training (restart from checkpoint if possible)
    start_epoch = 0
    if os.path.isfile(cfg_dict["model_root"] + "frame_pred_model/" + f"/model_{file_identifier}.pth"):
        print("Restarting training...")
        checkpoint = torch.load(cfg_dict["model_root"] + "frame_pred_model/" + f"/model_{file_identifier}.pth")
        model.load_state_dict(checkpoint["simvp_state_dict"])
        optimizer.load_state_dict(checkpoint["simvp_optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    # Value trackers
    train_loss_list = []
    test_loss_list = []
    train_metric_list = []
    test_metric_list = []

    best_avg_val_loss = 1000
    # Train loop

    for epoch in range(start_epoch, epochs):
        print("Epoch Number: ", epoch)
        train_pbar = tqdm(train_loader)
        model.train()
        train_loss_list = []
        for step, (past_frames, true_future_frames) in enumerate(train_pbar):
            past_frames, true_future_frames = past_frames.to(device), true_future_frames.to(device)
            pred_future_frames = model(past_frames)
            train_loss = criterion(pred_future_frames, true_future_frames)
            train_loss_list.append(train_loss.item())
            train_pbar.set_description('train loss: {:.4f}'.format(train_loss.item()))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        lr_scheduler.step()

        if (epoch) % log_step == 0:
        # Evaluate and test the model
            with torch.no_grad():
                model.eval()
                preds_lst, trues_lst, total_val_loss = [], [], []
                vali_pbar = tqdm(val_loader)
                for i, (past_frames_test, true_future_frames_test) in enumerate(vali_pbar):
                    if i * past_frames_test.shape[0] > 1000:
                        break

                    past_frames_test, true_future_frames_test = past_frames_test.to(device), true_future_frames_test.to(device)
                    pred_future_frames_test = model(past_frames_test)
                    list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                        pred_future_frames_test, true_future_frames_test], [preds_lst, trues_lst]))

                    loss = criterion(pred_future_frames_test, true_future_frames_test)
                    vali_pbar.set_description(
                        'vali loss: {:.4f}'.format(loss.mean().item()))
                    total_val_loss.append(loss.mean().item())


                avg_val_loss = np.average(total_val_loss)
                preds = np.concatenate(preds_lst, axis=0)
                trues = np.concatenate(trues_lst, axis=0)
                mse, mae, ssim, psnr = combined_metric(preds, trues, 0, 1, True)
                print('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
                model.train()
                print("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(epoch + 1, train_loss, avg_val_loss))

                visualize(past_frames, true_future_frames, pred_future_frames,
                          past_frames_test, true_future_frames_test, pred_future_frames_test,NUM_FUTURE_FRAMES, cfg_dict["log_dir"])

                if (avg_val_loss<best_avg_val_loss):
                    torch.save(
                        {
                            "epoch": epoch,
                            "simvp_state_dict": model.state_dict(),
                            "simvp_optimizer_state_dict": optimizer.state_dict()
                        },
                        cfg_dict["model_root"] + "frame_pred_model/" + f"/model_{file_identifier}.pth",
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

    logging.basicConfig(filename= cfg_dict['log_dir'] + 'fp_train.log',filemode='a',level=logging.DEBUG)

    logging.info(f"Loaded config file and initialized logging")
    logging.info(f"CFG_Dict: {cfg_dict}")

    train_data_dir = os.path.join(cfg_dict["dataset_root"], "train")
    val_data_dir = os.path.join(cfg_dict["dataset_root"], "val")
    unlabeled_data_dir = os.path.join(cfg_dict["dataset_root"], "unlabeled")

    batch_size = cfg_dict["fp_batch_size"]

    use_unlabaled_count = cfg_dict["fp_use_unlabeled_count"]


    # Create dataloaders for train and test dataset
    train_dataset = MovingObjectsDataset(
        root_dir=f'{train_data_dir}', 
        unlabeled_dir=unlabeled_data_dir, 
        use_unlabeled_count=10,
        transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False)


    val_dataset = MovingObjectsDataset(
        root_dir=f'{val_data_dir}',
        unlabeled_dir=unlabeled_data_dir,
        transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False)

    os.makedirs(cfg_dict["model_root"] + "frame_pred_model/", exist_ok=True)

    train(cfg_dict, train_loader, val_loader)

