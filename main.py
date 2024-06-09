import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.utils

# Custom transformation including normalization
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

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

# Normalize training and testing images
imgs_train = [trans(img).numpy() for img in imgs_train]
imgs_test = [trans(img).numpy() for img in imgs_test]

# Extract the last channel of the masks
masks_train = masks_train[:, :, :, -1]
masks_test = masks_test[:, :, :, -1]


# Function to display an image and its mask
def display_image_and_mask(images, masks, index, title_prefix):
    image = images[index]
    mask = masks[index]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(f'{title_prefix} Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'{title_prefix} Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()

# Select the first index for train and test
first_index_train = 0
first_index_test = 0

# Display the first image and mask from the training set
display_image_and_mask(imgs_train, masks_train, first_index_train, 'Train')

# Display the first image and mask from the test set
display_image_and_mask(imgs_test, masks_test, first_index_test, 'Test')


sys.exit()

# Ensure the number of images and masks are the same
#print("Number of training images matches number of training masks:", imgs_train.shape[0] == masks_train.shape[0])
#print("Number of testing images matches number of testing masks:", imgs_test.shape[0] == masks_test.shape[0])

# Print the structure of the images and masks
#print("Shape of training images:", imgs_train.shape)
#print("Shape of training masks:", masks_train.shape)
#print("Shape of testing images:", imgs_test.shape)
#print("Shape of testing masks:", masks_test.shape)

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

# Define the transformations
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
])

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
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp

# Get a batch of training data
inputs, masks = next(iter(dataloaders['train']))

# Select an example index
example_index = 3

# Get the initial image (before normalization) - assuming we have access to the original data
initial_image = imgs_train[example_index]

# Get the normalized image
normalized_image = inputs[example_index]

# Get the corresponding mask
mask = masks[example_index]

# Plot the initial image, normalized image, and mask side by side
plt.figure(figsize=(15, 5))

# Initial image
plt.subplot(1, 3, 1)
plt.title('Initial Image')
plt.imshow(initial_image)

# Normalized image (reversed transformation)
plt.subplot(1, 3, 2)
plt.title('Normalized Image (Reversed)')
plt.imshow(reverse_transform(normalized_image))

# Corresponding mask
plt.subplot(1, 3, 3)
plt.title('Mask')
plt.imshow(mask, cmap='gray')

plt.show()


