import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.utils
import torch.nn as nn
from torchvision import models

# Load the images and masks
imgs_train_path = '/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/PanNuke_dataset/FinalData/Train/imgs_train.npy'
imgs_test_path = '/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/PanNuke_dataset/FinalData/Test/imgs_test.npy'
masks_train_path = '/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/PanNuke_dataset/FinalData/Train/masks_train.npy'
masks_test_path = '/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/PanNuke_dataset/FinalData/Test/masks_test.npy'

# Load the data from the .npy files
imgs_train = np.load(imgs_train_path)
imgs_test = np.load(imgs_test_path)
masks_train = np.load(masks_train_path)
masks_test = np.load(masks_test_path)

# Extract the last channel of the masks
masks_train = masks_train[:, :, :, -1]
masks_test = masks_test[:, :, :, -1]

#first_index_train = 0
#first_index_test = 0

# Get the initial and normalized images and masks
#initial_train_image = imgs_train[first_index_train]
#train_mask = masks_train[first_index_train]

#initial_test_image = imgs_test[first_index_test]
#test_mask = masks_test[first_index_test]

# Plot the initial image and mask side by side for training set
#plt.figure(figsize=(10, 5))

#plt.subplot(1, 2, 1)
#plt.title('Train Initial Image')
#plt.imshow(initial_train_image.astype(np.uint8))
#plt.axis('off')

#plt.subplot(1, 2, 2)
#plt.title('Train Mask')
#plt.imshow(train_mask, cmap='gray')
#plt.axis('off')

#plt.show()

# Ensure the number of images and masks are the same
#print("Number of training images matches number of training masks:", imgs_train.shape[0] == masks_train.shape[0])
#print("Number of testing images matches number of testing masks:", imgs_test.shape[0] == masks_test.shape[0])

# Print the structure of the images and masks
#print("Shape of training images:", imgs_train.shape)
#print("Shape of training masks:", masks_train.shape)
#print("Shape of testing images:", imgs_test.shape)
#print("Shape of testing masks:", masks_test.shape)

# Define the transformations
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
])

# Define the custom Dataset class
class SimDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.input_images = images
        self.target_masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]

# Create the datasets
train_set = SimDataset(imgs_train, masks_train, transform=trans)
test_set = SimDataset(imgs_test, masks_test, transform=trans)

# Create the dataloaders
batch_size = 25

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

# Function to reverse the normalization
def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    #inp = np.clip(inp, 0, 1)
    inp = np.clip(inp, 0, 255)
    #inp = (inp * 255).astype(np.uint8)
    inp = inp.astype(np.uint8)
    return inp

# Get a batch of training data
#inputs, masks = next(iter(dataloaders['train']))

# Select an example index
#example_index = 3

# Get the initial image (before normalization) - assuming we have access to the original data
#initial_image = imgs_train[example_index]

# Get the normalized image
#normalized_image = inputs[example_index]

# Get the corresponding mask
#mask = masks[example_index]

# Plot the initial image, normalized image, and mask side by side
#plt.figure(figsize=(15, 5))

# Initial image
#plt.subplot(1, 3, 1)
#plt.title('Initial Image')
#plt.imshow(initial_image.astype(np.uint8))

# Normalized image (reversed transformation)
#plt.subplot(1, 3, 2)
#plt.title('Normalized Image (Reversed)')
#plt.imshow(reverse_transform(normalized_image).astype(np.uint8))

# Corresponding mask
#plt.subplot(1, 3, 3)
#plt.title('Mask')
#plt.imshow(mask, cmap='gray')

#plt.show()

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)  # Adjust input channels to 3 for RGB images
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        # Adjust output channels to n_class for single-channel masks
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetUNet(n_class=6)
model = model.to(device)

# check keras-like model summary using torchsummary
from torchsummary import summary
summary(model, input_size=(3, 224, 224))