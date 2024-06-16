import numpy as np
import os
import gc

def save_chunks(output_dir, file_prefix, data, chunk_size=1000):
    num_chunks = len(data) // chunk_size + int(len(data) % chunk_size != 0)
    for i in range(num_chunks):
        chunk = data[i * chunk_size: (i + 1) * chunk_size]
        np.save(os.path.join(output_dir, f'{file_prefix}_chunk_{i}.npy'), chunk)

# Define filenames for the image and mask .npy files
image_files = [
    "/Users/AmaiaZurinagaGutierr/OneDrive - Alira Health/Documents/GitHub/Josep_Model_Branch/split_pannuke.pyFold 1/images/fold1/images.npy", 
    "/Users/AmaiaZurinagaGutierr/OneDrive - Alira Health/Documents/GitHub/Josep_Model_Branch/PanNuke_dataset/Fold 2/images/fold2/images.npy", 
    "/Users/AmaiaZurinagaGutierr/OneDrive - Alira Health/Documents/GitHub/Josep_Model_Branch/PanNuke_dataset/Fold 3/images/fold3/images.npy"
]
mask_files = [
    "/Users/AmaiaZurinagaGutierr/OneDrive - Alira Health/Documents/GitHub/Josep_Model_Branch/PanNuke_dataset/Fold 1/masks/fold1/masks.npy", 
    "/Users/AmaiaZurinagaGutierr/OneDrive - Alira Health/Documents/GitHub/Josep_Model_Branch/PanNuke_dataset/Fold 2/masks/fold2/masks.npy", 
    "/Users/AmaiaZurinagaGutierr/OneDrive - Alira Health/Documents/GitHub/Josep_Model_Branch/PanNuke_dataset/Fold 3/masks/fold3/masks.npy"
]

# Define the directory to save the files
output_dir = '/Users/AmaiaZurinagaGutierr/OneDrive - Alira Health/Documents/GitHub/Josep_Model_Branch/PanNuke_dataset'
os.makedirs(output_dir, exist_ok=True)

# Load and concatenate images and masks using memory-mapped files
all_images = []
all_masks = []

for img_file, mask_file in zip(image_files, mask_files):
    images = np.load(img_file, mmap_mode='r').astype(np.float32)
    masks = np.load(mask_file, mmap_mode='r').astype(np.float32)
    
    # Ensure the number of images and masks are the same for each pair
    assert images.shape[0] == masks.shape[0], f"Number of images and masks must match in {img_file} and {mask_file}"
    
    all_images.append(images)
    all_masks.append(masks)

# Concatenate all images and masks
all_images = np.concatenate(all_images, axis=0)
all_masks = np.concatenate(all_masks, axis=0)

# Free up memory
gc.collect()

# Shuffle the data
indices = np.arange(all_images.shape[0])
np.random.shuffle(indices)

all_images = all_images[indices]
all_masks = all_masks[indices]

# Calculate the number of samples for each subset
num_samples = all_images.shape[0]
num_train = int(num_samples * 0.6)
num_val = int(num_samples * 0.2)
num_test = num_samples - num_train - num_val  # Remaining samples for testing

# Split the data
imgs_train = all_images[:num_train]
masks_train = all_masks[:num_train]

imgs_val = all_images[num_train:num_train + num_val]
masks_val = all_masks[num_train:num_train + num_val]

imgs_test = all_images[num_train + num_val:]
masks_test = all_masks[num_train + num_val:]

# Save each subset into separate .npy files in chunks
save_chunks(output_dir, 'imgs_train', imgs_train)
save_chunks(output_dir, 'masks_train', masks_train)
save_chunks(output_dir, 'imgs_val', imgs_val)
save_chunks(output_dir, 'masks_val', masks_val)
save_chunks(output_dir, 'imgs_test', imgs_test)
save_chunks(output_dir, 'masks_test', masks_test)

print(f"Data has been combined, split, and saved into training, validation, and testing sets in the '{output_dir}' directory.")
