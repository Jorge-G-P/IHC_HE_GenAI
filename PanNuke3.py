#Show all masks for an image

import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file containing images
images = np.load('/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/PanNuke_dataset/Fold 1/images/fold1/images.npy', mmap_mode='r')

# Load the .npy file containing masks
masks = np.load('/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/PanNuke_dataset/Fold 1/masks/fold1/masks.npy', mmap_mode='r')

# Visualize the first image and its corresponding mask
first_image = images[1]
first_mask = np.sum(masks[1], axis=-1)  # Sum across channels to collapse into a single-channel mask

# Get unique values in the mask for class 1
unique_values_mask_1 = np.unique(first_mask)
print(f"Unique values in mask 1: {unique_values_mask_1}")

# Binarize the mask: all non-zero values become 1 (white), zero values remain 0 (black)
first_mask = np.where(first_mask > 1, 1, 0)

# Plotting the image and mask
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(first_image.astype(np.uint8))
axes[0].set_title('Image')

axes[1].imshow(first_mask, cmap='gray')
axes[1].set_title('Mask')

plt.show()