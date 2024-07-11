import os
import numpy as np
import cv2
import shutil
from PIL import Image
from tqdm import tqdm
from Datasets.Pannuke.config import (
    image_paths, mask_paths, out_dir, directory_name_train, zip_file_name_train, directory_name_val, zip_file_name_val, directory_name_test, zip_file_name_test)


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

    fold_path = out_dir+path
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

        np.save(fold_path + '/' + '%d.npy' % (i), np_file)


transform(images, masks, 'train', out_dir=out_dir, start=0, finish=0.6)
transform(images, masks, 'val', out_dir=out_dir, start=0.6, finish=0.8)
transform(images, masks, 'test', out_dir=out_dir, start=0.8, finish=1)


# Create a zip file of the directory
shutil.make_archive(zip_file_name_train.replace('.zip', ''), 'zip', directory_name_train)

# Create a zip file of the directory
shutil.make_archive(zip_file_name_val.replace('.zip', ''), 'zip', directory_name_val)

# Create a zip file of the directory
shutil.make_archive(zip_file_name_test.replace('.zip', ''), 'zip', directory_name_test)
