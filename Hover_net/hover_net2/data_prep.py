import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
repo_path = dir_path.parent
parent_path = repo_path.parent


path_pannuke = parent_path / "Datasets/Pannuke/data/"

image_paths = [
    parent_path / "Datasets/Pannuke/data/Fold 1/images/fold1/images.npy",
    parent_path / "Datasets/Pannuke/data/Fold 2/images/fold2/images.npy",
    parent_path / "Datasets/Pannuke/data/Fold 3/images/fold3/images.npy"]


mask_paths = [
    parent_path / "Datasets/Pannuke/data/Fold 1/masks/fold1/masks.npy",
    parent_path / "Datasets/Pannuke/data/Fold 2/masks/fold2/masks.npy",
    parent_path / "Datasets/Pannuke/data/Fold 3/masks/fold3/masks.npy"
]


out_dir = parent_path / "Datasets/Pannuke/data/"

directory_name_train = parent_path / "Datasets/Pannuke/data/train/"
directory_name_val = parent_path / "Datasets/Pannuke/data/val/"
directory_name_test = parent_path / "Datasets/Pannuke/data/test/"





images_list = []


for image_path in image_paths:
    images = np.load(image_path, allow_pickle=True)
    images_list.append(images)

images = np.concatenate(images_list, axis=0)


masks_list = []


for mask_path in mask_paths:
    masks = np.load(mask_path, allow_pickle=True)
    masks_list.append(masks)


masks = np.concatenate(masks_list, axis=0)
print(images.shape)
print(masks.shape)


def flat_for(a, f):
    a = a.reshape(-1)
    for i, v in enumerate(a):
        a[i] = f(v)



def map_inst(inst):
    seg_indexes = np.unique(inst)
    new_indexes = np.array(range(0, len(seg_indexes)))
    dict = {}
    for seg_index, new_index in zip(seg_indexes, new_indexes):
        dict[seg_index] = new_index

    flat_for(inst, lambda x: dict[x])


# A helper function to transform PanNuke format to HoverNet data format
def transform(images, masks, path, out_dir, start, finish):

    fold_path = out_dir / path
    try:
        os.mkdir(fold_path)
    except FileExistsError:
        pass
    
    start = int(images.shape[0]*start)
    finish = int(images.shape[0]*finish)
    
    for i in tqdm(range(start, finish)):
        np_file = np.zeros((256,256,5), dtype='int16')

        # add rgb channels to array
        img_int = np.array(images[i],np.int16)
        for j in range(3):
            np_file[:,:,j] = img_int[:,:,j]

        # convert inst and type format for mask
        msk = masks[i]

        inst = np.zeros((256,256))
        for j in range(5):
            #copy value from new array if value is not equal 0
            inst = np.where(msk[:,:,j] != 0, msk[:,:,j], inst)
        map_inst(inst)

        types = np.zeros((256,256))
        for j in range(5):
            # write type index if mask is not equal 0 and value is still 0
            types = np.where((msk[:,:,j] != 0) & (types == 0), j+1, types)

        # add padded inst and types to array
        np_file[:,:,3] = inst
        np_file[:,:,4] = types

        np.save(fold_path / f'{i}.npy', np_file)


transform(images, masks, 'train', out_dir=out_dir, start=0, finish=0.6)
transform(images, masks, 'val', out_dir=out_dir, start=0.6, finish=0.8)
transform(images, masks, 'test', out_dir=out_dir, start=0.8, finish=1)



img_dir = directory_name_test / "images_png"
img_dir.mkdir(exist_ok=True)


for npy_file in directory_name_test.glob('*.npy'):
    data = np.load(npy_file)
    
    img_array = data[:, :, :3] 

    if img_array.dtype != np.uint8:
        img_array = (img_array / img_array.max() * 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    img.save(img_dir / f"{npy_file.stem}.png")

