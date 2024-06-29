import os
import random
import matplotlib.pyplot as plt
from skimage import io

# Define the paths
image_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/ENDONUKE/dataset/Josep_resized_images"
txt_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/ENDONUKE/dataset/Josep_resized_txt"
output_image_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/Pipeline/Crops_Images"
output_txt_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/Pipeline/Crops_txt"

# Function to read coordinates from txt file
def read_coordinates(txt_file):
    coordinates = []
    with open(txt_file, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 2:
                x_coordinate = int(parts[0])
                y_coordinate = int(parts[1])
                coordinates.append((x_coordinate, y_coordinate))
    return coordinates

# Function to perform specific crops and identify centroids within each crop
def specific_crops_with_centroids(image, coordinates, crop_size=(256, 256)):
    crops = []
    h, w, _ = image.shape
    crop_h, crop_w = crop_size
    
    # Crop from the right top corner
    right_top_crop = image[0:crop_h, w - crop_w:w]
    right_top_centroids = [(x - (w - crop_w), y) for x, y in coordinates if (w - crop_w) <= x < w and 0 <= y < crop_h]
    crops.append((right_top_crop, right_top_centroids))

    # Crop from the left top corner
    left_top_crop = image[0:crop_h, 0:crop_w]
    left_top_centroids = [(x, y) for x, y in coordinates if 0 <= x < crop_w and 0 <= y < crop_h]
    crops.append((left_top_crop, left_top_centroids))

    # Crop from the middle bottom part
    start_x = (w - crop_w) // 2
    middle_bottom_crop = image[h - crop_h:h, start_x:start_x + crop_w]
    middle_bottom_centroids = [(x - start_x, y - (h - crop_h)) for x, y in coordinates if start_x <= x < start_x + crop_w and (h - crop_h) <= y < h]
    crops.append((middle_bottom_crop, middle_bottom_centroids))

    return crops

# Collect all image and txt file pairs
image_txt_pairs = []
for root, dirs, files in os.walk(image_path):
    for file in files:
        if file.endswith(".png"):
            img_path = os.path.join(root, file)
            txt_file = file.replace(".png", ".txt")
            txt_file_path = os.path.join(txt_path, txt_file)
            if os.path.exists(txt_file_path):
                image_txt_pairs.append((img_path, txt_file_path))

# Select 3 random pairs
random_pairs = random.sample(image_txt_pairs, 1740) #Increased random sample to all images

# Plot the images with original and cropped coordinates
for img_path, txt_file_path in random_pairs:
    image = io.imread(img_path)
    coordinates = read_coordinates(txt_file_path)
    
    # Perform specific crops and get centroids in crops
    crops = specific_crops_with_centroids(image, coordinates)
            
    # Plot cropped images with centroids within the crop
    crop_titles = ["Right Top Crop", "Left Top Crop", "Middle Bottom Crop"]
    for i, (crop, crop_centroids) in enumerate(crops):
        
        # Save cropped image
        crop_image_name = f"{os.path.basename(img_path).replace('.png', '')}_crop{i + 1}.png"
        crop_image_path = os.path.join(output_image_path, crop_image_name)
        io.imsave(crop_image_path, crop)
        
        # Save adjusted coordinates to txt file
        crop_txt_name = f"{os.path.basename(img_path).replace('.png', '')}_crop{i + 1}.txt"
        crop_txt_path = os.path.join(output_txt_path, crop_txt_name)
        
        # Save adjusted coordinates to txt file
        with open(crop_txt_path, 'w') as file:
            for (x, y) in crop_centroids:
                file.write(f"{x} {y}\n")
