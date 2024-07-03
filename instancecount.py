import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

def plot_mask_with_centroids(image_path):
    # Load the image
    mask_image = imread(image_path)

    # Convert to grayscale if the image is in color
    if len(mask_image.shape) == 3:
        mask_image_gray = rgb2gray(mask_image)
    else:
        mask_image_gray = mask_image

    # Threshold the image to create a binary mask
    thresh = threshold_otsu(mask_image_gray)
    binary_mask = mask_image_gray > thresh

    # Label connected regions
    labeled_mask, num_features = morphology.label(binary_mask, connectivity=2, return_num=True)

    # Calculate centroids
    properties = measure.regionprops(labeled_mask)
    centroids = [prop.centroid for prop in properties]

    # Plot the mask and centroids
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(binary_mask, cmap='gray')

    for centroid in centroids:
        ax.plot(centroid[1], centroid[0], 'ro')

    ax.set_title(f"Mask with Centroids (Total: {len(centroids)})")
    ax.axis('off')
    plt.show()

    # Print the total number of centroids
    print(f"Total number of centroids: {len(centroids)}")

#Replace with the actual path to one of your saved mask images
mask_path = '/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/R-CNN_Josep/Test/Separated_masks/1_2.png'

#Plot the mask with centroids and count them
plot_mask_with_centroids(mask_path)