import os
from PIL import Image

# Define the folder path containing the images
folder_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/Endonuke_Preprocessing/data/crop_images"

# Initialize variables
image_count = 0
image_sizes = set()

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    # Check if the file is an image (you can expand this to include more extensions)
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Count the image
        image_count += 1
        
        # Open the image to get its size
        try:
            with Image.open(file_path) as img:
                image_sizes.add(img.size)
        except OSError as e:
            print(f"Skipping {file_name}: {e}")

# Print results
print(f"Total number of images: {image_count}")
print("Different image sizes found:")
for size in image_sizes:
    print(size)