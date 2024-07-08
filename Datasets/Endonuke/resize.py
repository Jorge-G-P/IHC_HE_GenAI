import os
import numpy as np
import shutil
from skimage import io, transform, img_as_ubyte


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

