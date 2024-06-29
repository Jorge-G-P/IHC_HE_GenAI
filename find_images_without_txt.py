import os

# Define the paths
image_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/Pipeline/Crops_Images"
txt_path = "/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/Pipeline/Crops_txt"

# Collect all image filenames (without extension)
image_files = set()
for root, _, files in os.walk(image_path):
    for file in files:
        if file.endswith(".png"):
            image_files.add(os.path.splitext(file)[0])

# Collect all txt filenames (without extension)
txt_files = set()
for root, _, files in os.walk(txt_path):
    for file in files:
        if file.endswith(".txt"):
            txt_files.add(os.path.splitext(file)[0])

# Find images without a corresponding txt file
images_without_txt = image_files - txt_files

# Find txt files without a corresponding image
txts_without_image = txt_files - image_files

# Print the count of image files and txt files
print(f"Total number of image files: {len(image_files)}")
print(f"Total number of txt files: {len(txt_files)}")

# Print results
if images_without_txt:
    print("\nImages without corresponding txt files:")
    for image in images_without_txt:
        print(f"{image}.png")
else:
    print("\nAll images have corresponding txt files.")

if txts_without_image:
    print("\nTxt files without corresponding images:")
    for txt in txts_without_image:
        print(f"{txt}.txt")
else:
    print("\nAll txt files have corresponding images.")
