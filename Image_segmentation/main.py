import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import re
import os
from model import ResNetUNet
from train import train_model
from dataset import SimDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Regular expression for numerical sorting
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Paths to the image and mask directories
imgs_train_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/train/images/'
imgs_val_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/val/images/'
masks_train_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/train/inst_masks/'
masks_val_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/val/inst_masks/'

# Function to get all file paths from a folder
def get_file_paths(folder):
    return sorted([os.path.join(folder, filename) for filename in os.listdir(folder)], key=numericalSort)

# Get file paths for images and masks
imgs_train_paths = get_file_paths(imgs_train_path)
imgs_val_paths = get_file_paths(imgs_val_path)
masks_train_paths = get_file_paths(masks_train_path)
masks_val_paths = get_file_paths(masks_val_path)


# Define the transformations (COMENTAR)
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
])



# Create the datasets
train_set = SimDataset(imgs_train_paths, masks_train_paths, transform=trans)
val_set = SimDataset(imgs_val_paths, masks_val_paths, transform=trans)

# Create the dataloaders
batch_size = 25

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

# Function to reverse the normalization (COMENTAR)
# def reverse_transform(inp):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     #inp = np.clip(inp, 0, 1)
#     inp = np.clip(inp, 0, 255)
#     #inp = (inp * 255).astype(np.uint8)
#     inp = inp.astype(np.uint8)
#     return inp




num_class = 1
model = ResNetUNet(num_class).to(device)


optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, device, num_epochs=1)