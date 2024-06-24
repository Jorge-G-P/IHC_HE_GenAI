import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from IHC_HE_GenAI.Image_segmentation.model2 import UNet
from IHC_HE_GenAI.Image_segmentation.train2 import train_model, validate_model
from skimage.measure import label, regionprops
from skimage.color import label2rgb

# Regular expression for numerical sorting
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Paths to the image and mask directories
imgs_train_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/train/images/'
imgs_test_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/test/images/'
masks_train_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/train/inst_masks/'
masks_test_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/test/inst_masks/'

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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet
])

# Define the custom Dataset class with lazy loading
class SimDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        mask = np.array(mask) / 255.0  # Normalize mask to 0 and 1
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        return [image, mask]

# Create the datasets
train_set = SimDataset(imgs_train_paths, masks_train_paths, transform=trans)
test_set = SimDataset(imgs_test_paths, masks_test_paths, transform=trans)

# Create the dataloaders
batch_size = 25

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

# Initialize the model, criterion and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1
save_dir = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/Model_results/'  # Define el directorio donde se guardarán los resultados
train_losses, val_losses = train_model(model, dataloaders, criterion, optimizer, num_epochs, device)

# Validate the model
validate_model(model, dataloaders['test'], criterion, device)

# Function to reverse the normalization
def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp

# Function to display an image and its mask
def display_image_and_mask(images, masks, index, title_prefix):
    image = reverse_transform(images[index])
    mask = masks[index].squeeze()  # Remove channel dimension

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

# Function to detect and display cells in the mask
def detect_and_display_cells(image, mask):
    mask = mask.squeeze()  # Remove channel dimension
    labeled_mask = label(mask)
    image_label_overlay = label2rgb(labeled_mask, image=image, bg_label=0)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Labeled Cells')
    plt.imshow(image_label_overlay)
    plt.axis('off')

    plt.show()

    return labeled_mask

# Select the first index for train and test
first_index_train = 0
first_index_test = 0

# Get a batch of training data
train_images, train_masks = next(iter(dataloaders['train']))
test_images, test_masks = next(iter(dataloaders['test']))

# Display the first image and mask from the training set
display_image_and_mask(train_images, train_masks, first_index_train, 'Train')

# Display the first image and mask from the test set
display_image_and_mask(test_images, test_masks, first_index_test, 'Test')

# Select an example index
example_index = 3

# Get the initial image, normalized image, and corresponding mask
initial_image = reverse_transform(train_images[example_index])
normalized_image = reverse_transform(train_images[example_index])
mask = train_masks[example_index].squeeze()  # Remove channel dimension

# Plot the initial image, normalized image, and mask side by side
plt.figure(figsize=(15, 5))

# Initial image
plt.subplot(1, 3, 1)
plt.title('Initial Image')
plt.imshow(initial_image)

# Normalized image (reversed transformation)
plt.subplot(1, 3, 2)
plt.title('Normalized Image (Reversed)')
plt.imshow(normalized_image)

# Corresponding mask
plt.subplot(1, 3, 3)
plt.title('Mask')
plt.imshow(mask, cmap='gray')

plt.show()

# Save the best model weights
torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
print('Model saved as best_model.pth')









# import os
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms

# # Regular expression for numerical sorting
# numbers = re.compile(r'(\d+)')
# def numericalSort(value):
#     parts = numbers.split(value)
#     parts[1::2] = map(int, parts[1::2])
#     return parts

# # Paths to the image and mask directories
# imgs_train_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/train/images/'
# imgs_test_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/test/images/'
# masks_train_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/train/inst_masks/'
# masks_test_path = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/datasplit/test/inst_masks/'

# # Function to get all file paths from a folder
# def get_file_paths(folder):
#     return sorted([os.path.join(folder, filename) for filename in os.listdir(folder)], key=numericalSort)

# # Get file paths for images and masks
# imgs_train_paths = get_file_paths(imgs_train_path)
# imgs_test_paths = get_file_paths(imgs_test_path)
# masks_train_paths = get_file_paths(masks_train_path)
# masks_test_paths = get_file_paths(masks_test_path)

# # Define the transformations
# trans = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
# ])

# # Define the custom Dataset class with lazy loading
# class SimDataset(Dataset):
#     def __init__(self, image_paths, mask_paths, transform=None):
#         self.image_paths = image_paths
#         self.mask_paths = mask_paths
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = Image.open(self.image_paths[idx]).convert("RGB")
#         mask = Image.open(self.mask_paths[idx]).convert("L")
#         if self.transform:
#             image = self.transform(image)
#         mask = np.array(mask) / 255.0  # Normalize mask to 0 and 1
#         mask = np.expand_dims(mask, axis=0)  # Add channel dimension
#         return [image, mask]

# # Create the datasets
# train_set = SimDataset(imgs_train_paths, masks_train_paths, transform=trans)
# test_set = SimDataset(imgs_test_paths, masks_test_paths, transform=trans)

# # Create the dataloaders
# batch_size = 25

# dataloaders = {
#     'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
#     'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
# }

# # Function to reverse the normalization
# def reverse_transform(inp):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     inp = (inp * 255).astype(np.uint8)
#     return inp

# # Function to display an image and its mask
# def display_image_and_mask(images, masks, index, title_prefix):
#     image = reverse_transform(images[index])
#     mask = masks[index].squeeze()  # Remove channel dimension

#     plt.figure(figsize=(10, 5))

#     plt.subplot(1, 2, 1)
#     plt.title(f'{title_prefix} Image')
#     plt.imshow(image)
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title(f'{title_prefix} Mask')
#     plt.imshow(mask, cmap='gray')
#     plt.axis('off')

#     plt.show()

# # Select the first index for train and test
# first_index_train = 0
# first_index_test = 0

# # Get a batch of training data
# train_images, train_masks = next(iter(dataloaders['train']))
# test_images, test_masks = next(iter(dataloaders['test']))

# # Display the first image and mask from the training set
# display_image_and_mask(train_images, train_masks, first_index_train, 'Train')

# # Display the first image and mask from the test set
# display_image_and_mask(test_images, test_masks, first_index_test, 'Test')

# # Select an example index
# example_index = 3

# # Get the initial image, normalized image, and corresponding mask
# initial_image = reverse_transform(train_images[example_index])
# normalized_image = reverse_transform(train_images[example_index])
# mask = train_masks[example_index].squeeze()  # Remove channel dimension

# # Plot the initial image, normalized image, and mask side by side
# plt.figure(figsize=(15, 5))

# # Initial image
# plt.subplot(1, 3, 1)
# plt.title('Initial Image')
# plt.imshow(initial_image)

# # Normalized image (reversed transformation)
# plt.subplot(1, 3, 2)
# plt.title('Normalized Image (Reversed)')
# plt.imshow(normalized_image)

# # Corresponding mask
# plt.subplot(1, 3, 3)
# plt.title('Mask')
# plt.imshow(mask, cmap='gray')

# plt.show()
