import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# Define the custom Dataset class
class SimDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = np.array(mask) / 255.0  # Normalize mask to 0 and 1
        if self.transform:
            image = self.transform(image)
        #mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        return [image, mask]