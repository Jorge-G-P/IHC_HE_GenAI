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

    def __init__(self, A_dir, B_dir, subset_percentage=100, img_size=1024, patch_size=512, transform=None, shuffle=False):
        super().__init__()
        self.A_dir = A_dir
        self.B_dir = B_dir
        self.subset_percentage = subset_percentage
        self.img_size = img_size
        self.patch_size = patch_size
        self.transform = transform
        self.num_patches_per_image = (self.img_size // self.patch_size) ** 2 

        self.A_paths = []
        self.B_paths = []
        for ext in IMG_EXTENSIONS:
            self.A_paths.extend(glob.glob(os.path.join(A_dir, '*' + ext)))
            self.B_paths.extend(glob.glob(os.path.join(B_dir, '*' + ext)))

        self.A_initial_paths = dict()
        self.B_initial_paths = dict()
        self.A_initial_paths = {i: path for i, path in enumerate(self.A_paths)}
        self.B_initial_paths = {i: path for i, path in enumerate(self.B_paths)}

        if shuffle is True:
            np.random.shuffle(self.A_paths)
            np.random.shuffle(self.B_paths)
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        if self.patch_size > self.img_size:
            raise ValueError("patch_size > img_size. Patch size must be equal or lower than image size or len(dataset) will be zero")        

        if self.subset_percentage is not None:
            if not (0 < self.subset_percentage <= 100):
                raise ValueError("subset_percentage must be in range [0, 100]")
            self.subset_size = int(self.subset_percentage / 100.0 * min(self.A_size, self.B_size) * self.num_patches_per_image)
        else:
            self.subset_size = min(self.A_size, self.B_size) * self.num_patches_per_image
    

    def split_image_into_patches(self, image) -> list:   # returns a list of 4 smaller images of patch_size
        
        """     
                +----+----+
                | 0  |  2 |
                +----+----+
                | 1  |  3 |
                +----+----+
            The split_image_into_patches crops are according to this representation:
            - Patch 0 is in the top-left corner.
            - Patch 1 is in the bottom-left corner.
            - Patch 2 is in the top-right corner.
            - Patch 3 is in the bottom-right corner.
        """
        patches = []
        width, height = image.size
        for i in range(0, width, self.patch_size):
            for j in range(0, height, self.patch_size):
                patch = image.crop((i, j, i+self.patch_size, j+self.patch_size))
                patches.append(patch)

        return patches


    def __len__(self):
        return self.subset_size
    

    def __getitem__(self, index):

        # Check if index is within the subset size
        if index >= len(self):
            raise IndexError("Index out of range. Dataset subset size reached")

        if self.A_size == 0 or self.B_size == 0:
            raise ZeroDivisionError("Not possible to get dataset sample because at least one of them is empty")

        image_index = index // self.num_patches_per_image
        patch_index = index % self.num_patches_per_image

        A_path = self.A_paths[image_index % self.A_size]
        B_path = self.B_paths[image_index % self.B_size]

        # Useful for paired datasets, if we shuffle data first to train GAN with data unpaired
        A_initial_index = next((idx for idx, value in self.A_initial_paths.items() if A_path == value), None)
        B_initial_index = next((idx for idx, value in self.B_initial_paths.items() if B_path == value), None)

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_patches = self.split_image_into_patches(A_img)
        B_patches = self.split_image_into_patches(B_img)

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
                'A_initial_index': A_initial_index, 
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

    myclass = GanDataset(x, y, img_size=600, patch_size=200, subset_percentage=80, shuffle=False)
    print(myclass.img_size)
    print(myclass.patch_size)
    print(myclass.num_patches_per_image)

    sample = myclass[0]
    sample2 = myclass[10]
    
    print(f'Original dataset size: {len(A_paths)}')
    print(f'Dataset size: {len(myclass)}')
    print('\n' f'Index Image A: {sample["A_index"]}', '\n' f'Index Image B: {sample["B_index"]}', '\n' f'Index Image A before shuffle: {sample["A_initial_index"]}', '\n' f'Index Image B before shuffle: {sample["B_initial_index"]}')
    print('\n' f'Index Image A2: {sample2["A_index"]}', '\n' f'Index Image B2: {sample2["B_index"]}', '\n' f'Index Image A2 before shuffle: {sample2["A_initial_index"]}', '\n' f'Index Image B2 before shuffle: {sample2["B_initial_index"]}')

if __name__=="__main__":
    main()

