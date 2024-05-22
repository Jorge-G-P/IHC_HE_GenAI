import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self,root_IHC,root_HE,transform=None):
        self.root_IHC = root_IHC
        self.root_HE = root_HE
        self.transform = transform

        self.IHC_images = os.listdir(root_IHC)
        self.HE_images = os.listdir(root_HE)
        self.length_dataset = max(len(self.root_HE),len(self.IHC_images))
        self.IHC_len = len(self.IHC_images)
        self.HE_len = len(self.HE_images)
    
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self,index):
        IHC_img = self.IHC_images[index % self.IHC_len]
        HE_img = self.HE_images[index % self.HE_len]

        IHC_path = os.path.join(self.root_IHC,IHC_img)
        HE_path = os.path.join(self.root_HE,HE_img)
        IHC_img = np.array(Image.open(IHC_path).convert("RGB"))
        HE_img = np.array(Image.open(HE_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image = IHC_img, image0 = HE_img)
            IHC_img = augmentations['image']
            HE_img = augmentations['image0']
        return IHC_img,HE_img