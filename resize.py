import os
from skimage import io, transform
import numpy as np

import os
import shutil
from skimage import io, transform, img_as_ubyte
import numpy as np

# Define the paths
source_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/ENDONUKE/dataset/Original_images"
txt_source_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/ENDONUKE/dataset/Original_txt"
target_image_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/ENDONUKE/dataset/Josep_resized_images"
target_txt_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/ENDONUKE/dataset/Josep_resized_txt"

# Define the target size
target_size = (400, 400)

# Function to resize images
def resize_image(image_path, target_size):
    image = io.imread(image_path)
    resized_image = transform.resize(image, target_size, anti_aliasing=True)
    return img_as_ubyte(resized_image)

# Function to save the resized image
def save_image(image, path):
    io.imsave(path, image)

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

# Function to save coordinates to txt file
def save_coordinates(coordinates, txt_file):
    with open(txt_file, 'w') as file:
        for x, y in coordinates:
            file.write(f"{x} {y}\n")

# Function to resize coordinates
def resize_coordinates(coordinates, original_size, target_size):
    x_ratio = target_size[1] / original_size[1]
    y_ratio = target_size[0] / original_size[0]
    resized_coords = [(int(x * x_ratio), int(y * y_ratio)) for x, y in coordinates]
    return resized_coords

# Create the target directories if they don't exist
os.makedirs(target_image_path, exist_ok=True)
os.makedirs(target_txt_path, exist_ok=True)

# Walk through the directory and process images and txt files
for root, dirs, files in os.walk(source_path):
    for file in files:
        if file.endswith(".png"):
            # Construct full file paths
            image_path = os.path.join(root, file)
            txt_file = file.replace(".png", ".txt")
            txt_file_path = os.path.join(txt_source_path, txt_file)
            
            # Ensure the corresponding txt file exists
            if not os.path.exists(txt_file_path):
                continue
            
            # Read the image
            image = io.imread(image_path)
            original_size = image.shape[:2]
            
            # Check if the image is already 400x400
            if original_size == target_size:
                # Copy the image and txt file directly to the target paths
                relative_path = os.path.relpath(image_path, source_path)
                new_image_path = os.path.join(target_image_path, relative_path)
                new_txt_path = os.path.join(target_txt_path, relative_path.replace(".png", ".txt"))
                
                new_image_dir = os.path.dirname(new_image_path)
                os.makedirs(new_image_dir, exist_ok=True)
                
                shutil.copy2(image_path, new_image_path)
                shutil.copy2(txt_file_path, new_txt_path)
                
                print(f"Copied image and txt directly: {new_image_path}, {new_txt_path}")
            else:
                # Resize the image
                resized_image = resize_image(image_path, target_size)
                
                # Create the new image path while maintaining the directory structure
                relative_path = os.path.relpath(image_path, source_path)
                new_image_path = os.path.join(target_image_path, relative_path)
                new_image_dir = os.path.dirname(new_image_path)
                os.makedirs(new_image_dir, exist_ok=True)
                
                # Save the resized image
                save_image(resized_image, new_image_path)
                print(f"Resized and saved image: {new_image_path}")
                
                # Read, resize, and save the coordinates
                coordinates = read_coordinates(txt_file_path)
                resized_coordinates = resize_coordinates(coordinates, original_size, target_size)
                
                # Create the new txt path while maintaining the directory structure
                new_txt_path = os.path.join(target_txt_path, relative_path.replace(".png", ".txt"))
                new_txt_dir = os.path.dirname(new_txt_path)
                os.makedirs(new_txt_dir, exist_ok=True)
                
                # Save the resized coordinates
                save_coordinates(resized_coordinates, new_txt_path)
                print(f"Resized and saved coordinates: {new_txt_path}")

print("All images and coordinates have been processed and saved.")

