import os
from PIL import Image
import torch.utils.data as data
import random
import glob
import numpy as np
import config

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

class GanDataset(data.Dataset):

    def __init__(self, A_dir, B_dir, subset_percentage=100, patch_size=512, transform=None, target_transform=None, shuffle=False):
        super().__init__()
        self.A_dir = A_dir
        self.B_dir = B_dir
        self.subset_percentage = subset_percentage
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform

        self.A_paths = []
        self.B_paths = []
        for ext in IMG_EXTENSIONS:
            self.A_paths.extend(glob.glob(os.path.join(A_dir, '*' + ext)))
            self.B_paths.extend(glob.glob(os.path.join(B_dir, '*' + ext)))

        self.B_initial_paths = dict()
        for i, path in enumerate(self.B_paths):
            self.B_initial_paths[i] = path

        if shuffle is True:
            np.random.shuffle(self.A_paths)
            np.random.shuffle(self.B_paths)
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.num_patches_per_image = (1024 // self.patch_size) ** 2  # 4 patches per image assuming 1024x1024 images

        if self.subset_percentage is not None:
            if not (0 < self.subset_percentage <= 100):
                raise ValueError("subset_percentage must be in range (0, 100]")
            self.subset_size = int(self.subset_percentage / 100.0 * max(self.A_size, self.B_size) * self.num_patches_per_image)
        else:
            self.subset_size = max(self.A_size, self.B_size) * self.num_patches_per_image
    

    def split_image_into_patches(self, image) -> list:   # returns a list of 4 smaller images of patch_size
        patches = []
        width, height = image.size
        for i in range(0, width, self.patch_size):
            for j in range(0, height, self.patch_size):
                patch = image.crop((i, j, i+self.patch_size, j+self.patch_size))
                patches.append(patch)

        return patches


    def __len__(self):
        # return max(self.A_size, self.B_size) * self.num_patches_per_image
        return self.subset_size
    

    def __getitem__(self, index):

        # Check if index is within the subset size
        if index >= len(self):
            raise IndexError("Index out of range. Dataset subset size reached.")

        image_index = index // self.num_patches_per_image
        patch_index = index % self.num_patches_per_image

        A_path = self.A_paths[image_index % self.A_size]
        B_path = self.B_paths[image_index % self.B_size]

        # Used as extra info for datasets with paired data, if we shuffle data first to train GAN with unpaired data
        for idx, value in self.B_initial_paths.items():
            if B_path == value:
                B_initial_index = idx

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_patches = self.split_image_into_patches(A_img)
        B_patches = self.split_image_into_patches(B_img)

        # random_index = random.randint(0, min(len(A_patches), len(B_patches)) - 1)
        A_patch = A_patches[patch_index]
        B_patch = B_patches[patch_index]

        if self.transform:
            augmentations = self.transform(image = np.array(A_patch), image0 = np.array(B_patch))
            A_patch = augmentations['image']
            B_patch = augmentations['image0']

        return {'A': A_patch, 
                'B': B_patch, 
                'A_img': A_img, 
                'B_img': B_img, 
                'A_index': image_index, 
                'B_index': image_index, 
                'B_initial_index': B_initial_index, 
                'patch_index': patch_index,
                'A_path': A_path, 
                'B_path': B_path,
                }

def main():

    ''' Code below is just to make some tests to the dataset class.
        Not necessary for project '''

    x = config.parent_path / "BCI_dataset/HE/train"
    y = config.parent_path / "BCI_dataset/IHC/test"

    A_paths = []
    A_paths.extend(glob.glob(os.path.join(x, '*')))

    myclass = GanDataset(x, y)
    sample = myclass[0]
    sample2 = myclass[4]

    print(f'Original dataset size: {len(A_paths)}')
    print(f'Dataset size: {len(myclass)}')
    print('\n' f'Index Image A: {sample["A_index"]}', '\n' f'Index Image B: {sample["B_index"]}', '\n' f'Index Image B before shuffle: {sample["B_initial_index"]}')
    print('\n' f'Index Image A2: {sample2["A_index"]}', '\n' f'Index Image B2: {sample2["B_index"]}', '\n' f'Index Image B2 before shuffle: {sample2["B_initial_index"]}')

if __name__=="__main__":
    main()
