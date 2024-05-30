import os
from PIL import Image
import torch.utils.data as data
import random
import glob
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

class GanDataset(data.Dataset):

    def __init__(self, A_dir, B_dir, patch_size=512, transform=None, target_transform=None, shuffle=False):
        super().__init__()
        self.A_dir = A_dir
        self.B_dir = B_dir
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

        # np.random.shuffle(self.A_paths)
        # np.random.shuffle(self.B_paths)
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
    

    def split_image_into_patches(self, image) -> list:   # returns a list of 4 smaller images of patch_size (512*512)
        patches = []
        width, height = image.size
        for i in range(0, width, self.patch_size):
            for j in range(0, height, self.patch_size):
                patch = image.crop((i, j, i+self.patch_size, j+self.patch_size))
                patches.append(patch)

        return patches


    def __len__(self):
        return max(self.A_size, self.B_size)
    

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        for idx, value in self.B_initial_paths.items():
            if B_path == value:
                B_initial_index = idx

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_patches = self.split_image_into_patches(A_img)
        B_patches = self.split_image_into_patches(B_img)
        random_index = random.randint(0, min(len(A_patches), len(B_patches)) - 1)
        A_patch = A_patches[random_index]
        B_patch = B_patches[random_index]

        if self.transform:
            A_patch = self.transform(image=np.array(A_patch))['image']
            B_patch = self.transform(image=np.array(B_patch))['image']

        return {'A': A_patch, 'B': B_patch, 'A_img': A_img, 'B_img': B_img, 'A_index': index, 'B_index': index, 'B_initial_index': B_initial_index, 'patch_index': random_index}



def main():

    ''' Code below is just to make some tests to the dataset class.
        Not necessary for project '''

    x = r'/home/jotapv98/coding/MyProjects/JOAO_HE_IHC/BCI_dataset/HE/train'
    y = r'/home/jotapv98/coding/MyProjects/JOAO_HE_IHC/BCI_dataset/IHC/test'

    A_paths = []
    A_paths.extend(glob.glob(os.path.join(x, '*')))
    print(len(A_paths))

    myclass = GanDataset(x, y)
    sample = myclass[0]
    sample2 = myclass[4]

    A_img = sample['A_img']
    A_crop = sample['A']
    B_img = sample['B_img']
    B_crop = sample['B']
    A_index = sample['A_index']
    B_index = sample['B_index']
    B_initial_index = sample['B_initial_index']
    patch_index = sample['patch_index']

    save_dir = r'/home/jotapv98/coding/MyProjects/GenAI_HE_IHC/data'
    # A_img.save(os.path.join(save_dir, f'[dataset.py]_A_index_{A_index}.jpg'))
    # A_crop.save(os.path.join(save_dir, f'[dataset.py]_A_crop_index_{A_index}_patch_{patch_index}.jpg'))
    # B_img.save(os.path.join(save_dir, f'[dataset.py]_B_index_{B_initial_index}.jpg'))
    # B_crop.save(os.path.join(save_dir, f'[dataset.py]_B_crop_index_{B_initial_index}_patch_{patch_index}.jpg'))

    print(f'Dataset size: {len(myclass)}')
    # print(myclass.A_size, myclass.B_size)
    print('\n' f'Index Image A: {sample["A_index"]}', '\n' f'Index Image B: {sample["B_index"]}', '\n' f'Index Image B before shuffle: {sample["B_initial_index"]}')
    print('\n' f'Index Image A2: {sample2["A_index"]}', '\n' f'Index Image B2: {sample2["B_index"]}', '\n' f'Index Image B2 before shuffle: {sample2["B_initial_index"]}')


if __name__=="__main__":
    main()
