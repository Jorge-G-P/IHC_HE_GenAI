import os
import matplotlib.pyplot as plt
from skimage import io, transform
import numpy as np
import shutil
from crop import read_coordinates, specific_crops_with_centroids, copy_txt_files, copy_images, delete_images_without_txt
from resize import resize_image, save_image, read_coordinates, save_coordinates, resize_coordinates
from config import path_to_endonuke_data_folder


# Define the paths
endonuke_dataset = path_to_endonuke_data_folder

#Group txt files
copy_txt_files(endonuke_dataset)

#Copy images
copy_images(endonuke_dataset)

#Delete images without txt
delete_images_without_txt(endonuke_dataset)

#Path to new folders
original_images = os.path.join(endonuke_dataset, 'clean_images')
original_txt = os.path.join(endonuke_dataset, 'txt_files')

# Construct paths for new folders
resized_images = os.path.join(endonuke_dataset, 'resized_images')
crop_images = os.path.join(endonuke_dataset, 'crop_images')
resized_txt = os.path.join(endonuke_dataset, 'resized_txt')
crop_txt = os.path.join(endonuke_dataset, 'crop_txt')
    
# Create directories if they don't exist
os.makedirs(resized_images, exist_ok=True)
os.makedirs(crop_images, exist_ok=True)
os.makedirs(resized_txt, exist_ok=True)
os.makedirs(crop_txt, exist_ok=True)
os.makedirs(original_images, exist_ok=True)
os.makedirs(original_txt, exist_ok=True)

# Define the target size
target_size = (400, 400)

# Walk through the directory and process images and txt files to resize into 400x400
for root, dirs, files in os.walk(original_images):
    for file in files:
        if file.endswith(".png"):
            # Construct full file paths
            image_path = os.path.join(root, file)
            txt_file = file.replace(".png", ".txt")
            txt_file_path = os.path.join(original_txt, txt_file)
            
            # Ensure the corresponding txt file exists
            if not os.path.exists(txt_file_path):
                continue
            
            # Read the image
            image = io.imread(image_path)
            original_size = image.shape[:2]
            
            # Check if the image is already 400x400
            if original_size == target_size:
                # Copy the image and txt file directly to the target paths
                relative_path = os.path.relpath(image_path, original_images)
                new_image_path = os.path.join(resized_images, relative_path)
                new_txt_path = os.path.join(resized_txt, relative_path.replace(".png", ".txt"))
                
                new_image_dir = os.path.dirname(new_image_path)
                os.makedirs(new_image_dir, exist_ok=True)
                
                shutil.copy2(image_path, new_image_path)
                shutil.copy2(txt_file_path, new_txt_path)
                
                print(f"Copied image and txt directly: {new_image_path}, {new_txt_path}")
            else:
                # Resize the image
                resized_image = resize_image(image_path, target_size)
                
                # Create the new image path while maintaining the directory structure
                relative_path = os.path.relpath(image_path, original_images)
                new_image_path = os.path.join(resized_images, relative_path)
                new_image_dir = os.path.dirname(new_image_path)
                os.makedirs(new_image_dir, exist_ok=True)
                
                # Save the resized image
                save_image(resized_image, new_image_path)
                print(f"Resized and saved image: {new_image_path}")
                
                # Read, resize, and save the coordinates
                coordinates = read_coordinates(txt_file_path)
                resized_coordinates = resize_coordinates(coordinates, original_size, target_size)
                
                # Create the new txt path while maintaining the directory structure
                new_txt_path = os.path.join(resized_txt, relative_path.replace(".png", ".txt"))
                new_txt_dir = os.path.dirname(new_txt_path)
                os.makedirs(new_txt_dir, exist_ok=True)
                
                # Save the resized coordinates
                save_coordinates(resized_coordinates, new_txt_path)
                print(f"Resized and saved coordinates: {new_txt_path}")

print("All images and coordinates have been processed and saved.")

#Crop resized images
# Collect all image and txt file pairs
image_txt_pairs = []
for root, dirs, files in os.walk(resized_images):
    for file in files:
        if file.endswith(".png"):
            img_path = os.path.join(root, file)
            txt_file = file.replace(".png", ".txt")
            txt_file_path = os.path.join(resized_txt, txt_file)
            if os.path.exists(txt_file_path):
                image_txt_pairs.append((img_path, txt_file_path))

# Plot the images with original and cropped coordinates
for img_path, txt_file_path in image_txt_pairs:
    image = io.imread(img_path)
    coordinates = read_coordinates(txt_file_path)
    
    # Perform specific crops and get centroids in crops
    crops = specific_crops_with_centroids(image, coordinates)
            
    for i, (crop, crop_centroids) in enumerate(crops):
        
        # Save cropped image
        crop_image_name = f"{os.path.basename(img_path).replace('.png', '')}_crop{i + 1}.png"
        crop_image_path = os.path.join(crop_images, crop_image_name)
        io.imsave(crop_image_path, crop)
        
        # Save adjusted coordinates to txt file
        crop_txt_name = f"{os.path.basename(img_path).replace('.png', '')}_crop{i + 1}.txt"
        crop_txt_path = os.path.join(crop_txt, crop_txt_name)
        
        # Save adjusted coordinates to txt file
        with open(crop_txt_path, 'w') as file:
            for (x, y) in crop_centroids:
                file.write(f"{x} {y}\n")
