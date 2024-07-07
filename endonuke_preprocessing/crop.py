import os
import random
import matplotlib.pyplot as plt
from skimage import io
import shutil

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

def copy_txt_files(data_path):
    # Create a new directory named txt_files inside data_path
    txt_files_dir = os.path.join(data_path, 'txt_files')
    os.makedirs(txt_files_dir, exist_ok=True)
    
    # Define the path to the source folder containing txt files
    source_folder = os.path.join(data_path, 'dataset', 'labels', 'bulk')
    
    # Iterate through all subdirectories inside the bulk folder
    for root, dirs, files in os.walk(source_folder):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Iterate through files in each subdirectory
            for file in os.listdir(dir_path):
                if file.endswith('.txt'):
                    src_file = os.path.join(dir_path, file)
                    dst_file = os.path.join(txt_files_dir, file)
                    shutil.copy(src_file, dst_file)
                    print(f"Copied {src_file} to {dst_file}")

def copy_images(path):
    # Construct paths to source and destination directories
    dataset_images_dir = os.path.join(path, 'dataset', 'images')
    clean_images_dir = os.path.join(path, 'clean_images')
    
    # Create clean_images directory outside dataset if it doesn't exist
    os.makedirs(clean_images_dir, exist_ok=True)
    
    # Iterate over files in dataset/images and copy them to clean_images
    for filename in os.listdir(dataset_images_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            src_file = os.path.join(dataset_images_dir, filename)
            dst_file = os.path.join(clean_images_dir, filename)
            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")

def delete_images_without_txt(path):
    txt_files_dir = os.path.join(path, 'txt_files')
    clean_images_dir = os.path.join(path, 'clean_images')
    
    # Check if txt_files and clean_images directories exist
    if not os.path.exists(txt_files_dir) or not os.path.exists(clean_images_dir):
        print("Error: txt_files or clean_images directory not found.")
        return
    
    # List all .txt files in txt_files directory
    txt_files = [f[:-4] for f in os.listdir(txt_files_dir) if f.endswith('.txt')]  # Remove .txt extension
    
    # List all image files in clean_images directory
    image_files = [f for f in os.listdir(clean_images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    
    # Delete images without corresponding .txt files
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]  # Get filename without extension
        
        # Check if corresponding .txt file exists
        if image_name not in txt_files:
            image_path = os.path.join(clean_images_dir, image_file)
            os.remove(image_path)
            print(f"Deleted {image_path}")