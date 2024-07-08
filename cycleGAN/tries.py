import os
from pathlib import Path
from PIL import Image
import config
import glob
from HE_IHC_dataset import GanDataset

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
repo_path = dir_path.parent
parent_path = repo_path.parent

CHECKPOINT_GEN_HE = parent_path / "pretrained-models/genHE_epochs.pth.tar"
original = config.ENDONUKE_ORIGINAL
cropped = config.ENDONUKE_CROPPED
data1 = parent_path / "endonuke_dataset/data/dataset/images"
data2 = parent_path / "endonuke_dataset/data/dataset/images_context"

def test_dataset_class():
    myclass = GanDataset(
        config.ENDONUKE_CROPPED,
        config.ENDONUKE_CROPPED,
        config.SUBSET_PERCENTAGE,
        config.IMG_ORIGINAL_SIZE,
        config.PATCHES_SIZE,
        transform=config.transforms, 
        shuffle=config.SHUFFLE_DATASET
    )

    x = glob.glob(os.path.join(data1, '*'))
    print(myclass.img_size)
    print(myclass.patch_size)
    print(myclass.num_patches_per_image)

    sample = myclass[0]
    sample2 = myclass[10]

    print(f'Original dataset size: {len(x)}')
    print(f'Dataset size: {len(myclass)}')
    print('\n' f'Index Image A: {sample["A_index"]}', '\n' f'Index Image B: {sample["B_index"]}', '\n' f'Index Image A before shuffle: {sample["A_initial_index"]}', '\n' f'Index Image B before shuffle: {sample["B_initial_index"]}')
    print('\n' f'Index Image A2: {sample2["A_index"]}', '\n' f'Index Image B2: {sample2["B_index"]}', '\n' f'Index Image A2 before shuffle: {sample2["A_initial_index"]}', '\n' f'Index Image B2 before shuffle: {sample2["B_initial_index"]}')

def check_endonuke_size():
    x = glob.glob(os.path.join(data1, '*'))
    y = glob.glob(os.path.join(data2, '*'))

    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    total = 0
    for path in x:
        with Image.open(path) as img:
            width, height = img.size
        total += 1
        if width == 200 and height == 200:
            A += 1
        elif width == 400 and height == 400:
            B += 1
        elif width == 600 and height == 600:
            C += 1
        elif width == 1200 and height == 1200:
            D += 1
        else:
            E += 1
            print(f"Img {path} has width {width} and height {height}")
    print(f"\nIn path: {data1}")
    print(f"200*200: {A}\n", f"400*400: {B}\n", f"600*600: {C}\n", f"1200*1200: {D}\n", f"Other: {E}\n", f"Total: {total}")

    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    total = 0
    for path in y:
        with Image.open(path) as img:
            width, height = img.size
        total += 1
        if width == 200 and height == 200:
            A += 1
        elif width == 400 and height == 400:
            B += 1
        elif width == 600 and height == 600:
            C += 1
        elif width == 1200 and height == 1200:
            D += 1
        else:
            E += 1
            print(f"Img {path} has width {width} and height {height}")
    print(f"\nIn path: {data2}")
    print(f"200*200: {A}\n", f"400*400: {B}\n", f"600*600: {C}\n", f"1200*1200: {D}\n", f"Other: {E}\n", f"Total: {total}")

test_dataset_class()
