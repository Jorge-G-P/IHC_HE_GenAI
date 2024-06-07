import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

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

# Convert lists to numpy arrays
imgs_train = np.array(imgs_train)
imgs_test = np.array(imgs_test)

# Extract the last channel of the masks
masks_train = masks_train[:, :, :, -1]
masks_test = masks_test[:, :, :, -1]

# Ensure the number of images and masks are the same
#print("Number of training images matches number of training masks:", imgs_train.shape[0] == masks_train.shape[0])
#print("Number of testing images matches number of testing masks:", imgs_test.shape[0] == masks_test.shape[0])

# Print the structure of the images and masks
#print("Shape of training images:", imgs_train.shape)
#print("Shape of training masks:", masks_train.shape)
#print("Shape of testing images:", imgs_test.shape)
#print("Shape of testing masks:", masks_test.shape)

# Randomly select an index
idx = np.random.randint(len(imgs_train))

# Get the random image and its corresponding mask
random_img = imgs_train[idx]
random_mask = masks_train[idx]  # Mask is already the last channel

def reverse_normalize_old(inp): #PENDENT
    mean = np.array([0.485, 0.456, 0.406])
    print(mean)
    std = np.array([0.229, 0.224, 0.225])
    print(std)
    # Convert the input to a numpy array if it's a tensor
    if isinstance(inp, torch.Tensor):
        inp = inp.numpy()
    # Ensure the input has the correct shape and datatype
    print("Input shape:", inp.shape)
    #print(inp)
    inp = inp.transpose((1, 2, 0))
    print("Transposed shape:", inp.shape)
    #print(inp)
    inp = inp * std + mean
    print("After normalization shape:", inp.shape)
    #print(inp)
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    print('old')

    return inp


# Plot the random image and its mask side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(random_img)
plt.title('Random Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(random_mask, cmap='gray')
plt.title('Corresponding Mask')
plt.axis('off')

plt.show()


