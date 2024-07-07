import os
import random
import matplotlib.pyplot as plt
from skimage import io

# Define the paths
image_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/Endonuke_Preprocessing/data/crop_images"
txt_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/Endonuke_Preprocessing/data/crop_txt"

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

# Function to plot image with marked coordinates
def plot_image_with_coordinates(image_path, coordinates, title="Image"):
    image = io.imread(image_path)
    plt.imshow(image)
    for (x, y) in coordinates:
        plt.plot(x, y, 'ro')  # Red dot for the coordinate
    plt.title(title)
    plt.axis('off')
    plt.show()

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
random_pairs = random.sample(image_txt_pairs, 3)

# Plot the random images with coordinates
for img_path, txt_file_path in random_pairs:
    coordinates = read_coordinates(txt_file_path)
    plot_image_with_coordinates(img_path, coordinates, title=f"Random Image: {os.path.basename(img_path)}")

# Plot the specified images with coordinates
specific_images = ["1767", "1773", "1638"]
for img_name in specific_images:
    img_path = os.path.join(image_path, f"{img_name}.png")
    txt_file_path = os.path.join(txt_path, f"{img_name}.txt")
    if os.path.exists(img_path) and os.path.exists(txt_file_path):
        coordinates = read_coordinates(txt_file_path)
        plot_image_with_coordinates(img_path, coordinates, title=f"Specified Image: {img_name}.png")
    else:
        print(f"Missing image or txt file for {img_name}.png")

print("Displayed 3 random images with coordinates and specified images 1767, 1773, 1638 with coordinates.")