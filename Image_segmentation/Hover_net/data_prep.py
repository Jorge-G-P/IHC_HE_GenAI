import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn
import cv2
from PIL import Image
from tqdm import tqdm



images = np.load('/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/PanNuke_dataset/Fold 1/images/fold1/images.npy',allow_pickle=True)
masks = np.load('/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/PanNuke_dataset/Fold 1/masks/fold1/masks.npy',allow_pickle=True)

print(images.shape)
print(masks.shape)

# data_dir = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/PanNuke_dataset/'  # ubicación de los datos extraídos
# output_dir = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/Folds/'  # ubicación para guardar los datos de salida

out_dir = "/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/PanNuke_dataset/"

# A helper function to map 2d numpy array
def flat_for(a, f):
    a = a.reshape(-1)
    for i, v in enumerate(a):
        a[i] = f(v)


# A helper function to unique PanNuke instances indexes to [0..N] range where 0 is background
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


transform(images, masks, 'train', out_dir=out_dir, start=0, finish=0.8)
transform(images, masks, 'val', out_dir=out_dir, start=0.8, finish=1)

import shutil
import os

# Define the directory and zip file names
directory_name_train = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/PanNuke_dataset/train/'
zip_file_name_train = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/PanNuke_dataset/train.zip'

# Create a zip file of the directory
shutil.make_archive(zip_file_name_train.replace('.zip', ''), 'zip', directory_name_train)

# Remove the original directory
#shutil.rmtree(directory_name)

directory_name_val = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/PanNuke_dataset/val/'
zip_file_name_val = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/PanNuke_dataset/val.zip'

# Create a zip file of the directory
shutil.make_archive(zip_file_name_val.replace('.zip', ''), 'zip', directory_name_val)