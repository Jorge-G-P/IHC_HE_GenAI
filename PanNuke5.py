import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files containing images and masks
images = np.load('/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/PanNuke_dataset/Fold 1/images/fold1/images.npy', mmap_mode='r')
masks = np.load('/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/PanNuke_dataset/Fold 1/masks/fold1/masks.npy', mmap_mode='r')

# Get the first mask for class 1
mask_1 = masks[1][:, :, 0]

# Get unique values in the mask for class 1
unique_values_mask_1 = np.unique(mask_1)
print(f"Unique values in mask 1: {unique_values_mask_1}")

mask_1 = np.where(mask_1 == 3, 1, 0)

# Binarize the mask: all non-zero values become 1 (white), zero values remain 0 (black)
binary_mask = np.where(mask_1 > 0, 1, 0)

# Plot the mask using only black and white
plt.imshow(mask_1)#, cmap='gray')
plt.title('Mask 1 (Black and White)')
plt.axis('off')  # Hide the axis
plt.show()