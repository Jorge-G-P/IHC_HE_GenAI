import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files containing images and masks
images = np.load('/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/PanNuke_dataset/Fold 1/images/fold1/images.npy', mmap_mode='r')
masks = np.load('/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/PanNuke_dataset/Fold 1/masks/fold1/masks.npy', mmap_mode='r')

# Get the first image and its corresponding masks
first_image = images[1]
first_mask = masks[1]

# Extract individual class masks
class_masks = first_mask[:, :, :5]

# Create a composite mask initialized to -1 (background)
composite_mask = np.full(class_masks.shape[:2], -1)

# Iterate over each class mask and set the values in the composite mask
for i in range(5):
    composite_mask[class_masks[:, :, i] > 0] = i

# Plotting the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(first_image.astype(np.uint8))
axes[0].set_title('Image')

axes[1].imshow(composite_mask, cmap='tab10')
axes[1].set_title('Composite Mask (Class Indexed)')

combined_mask = first_mask[:, :, 5]
axes[2].imshow(combined_mask, cmap='gray')
axes[2].set_title('Provided Combined Mask (6th Channel)')

plt.show()