import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.utils
import torch.nn as nn
from torchvision import models
from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from tqdm import tqdm
import os
from dataset import SimDataset
from model import ResNetUNet
import re
import config

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Paths to the image and mask directories
imgs_train_path = config.imgs_train_path 
imgs_test_path =  config.imgs_test_path 
masks_train_path =  config.masks_train_path 
masks_test_path = config.masks_test_path 

# Function to get all file paths from a folder
def get_file_paths(folder):
    return sorted([os.path.join(folder, filename) for filename in os.listdir(folder)], key=numericalSort)

# Get file paths for images and masks
imgs_train_paths = get_file_paths(imgs_train_path)
imgs_test_paths = get_file_paths(imgs_test_path)
masks_train_paths = get_file_paths(masks_train_path)
masks_test_paths = get_file_paths(masks_test_path)

# Define the transformations
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
])

# Create the datasets
train_set = SimDataset(imgs_train_paths, masks_train_paths, transform=trans)
test_set = SimDataset(imgs_test_paths, masks_test_paths, transform=trans)

# Create the dataloaders
batch_size = 25

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0) 
}

# Function to reverse the normalization
def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))  # Cambia las dimensiones a HWC
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # Desnormaliza
    inp = np.clip(inp, 0, 1)  # Asegúrate de que los valores estén en el rango [0, 1]
    inp = (inp * 255).astype(np.uint8)  # Convierte a rango [0, 255] y a tipo uint8
    return inp

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

# Load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNetUNet(n_class=1)
model.load_state_dict(torch.load('best_model_big.pth'))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Select the first three images from the test set
num_images_to_display = 3
images, true_masks = next(iter(dataloaders['test']))
images = images[:num_images_to_display]
true_masks = true_masks[:num_images_to_display]

# Predict masks
with torch.no_grad():
    inputs = images.to(device)
    outputs = model(inputs)
    pred_masks = torch.sigmoid(outputs)

# Plot the images, true masks, and predicted masks
for i in range(num_images_to_display):
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(reverse_transform(images[i].cpu()))
    plt.axis('off')

    # True Mask
    plt.subplot(1, 3, 2)
    plt.title('True Mask')
    true_mask = true_masks[i].cpu().numpy().squeeze()  # Convert tensor to numpy array and remove channel dimension
    plt.imshow(true_mask, cmap='gray')
    plt.axis('off')

    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    pred_mask = pred_masks[i].cpu().numpy().squeeze()  # Convert tensor to numpy array and remove channel dimension
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')

    plt.show()



