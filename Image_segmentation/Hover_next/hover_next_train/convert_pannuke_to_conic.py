import os
import argparse
import numpy as np
from scipy.ndimage import label

# quick and dirty script to convert pannuke masks to lizard/conic format
# define path_to_pannuke to point to the folder containing the pannuke masks
# the folder should contain 3 subfolders: fold1, fold2, fold3 with the respective masks.npy files
# the script will create a labels.npy file in each of the 3 subfolders


def process_chunk(y_chunk):
    y_inst_chunk = np.stack(
        [label(np.sum(y_[..., :-1], axis=-1).astype(int)) for y_ in y_chunk]
    )
    y_cls_chunk = np.max((y_chunk[..., :-1] > 0) * np.arange(1, 6), axis=-1)
    # Asegurarse de que las formas sean las mismas antes de apilarlas
    if y_inst_chunk.shape == y_cls_chunk.shape:
        y_lab_chunk = np.stack([y_inst_chunk, y_cls_chunk], axis=-1)
    else:
        raise ValueError("Las formas de y_inst_chunk y y_cls_chunk no coinciden.")
    return y_lab_chunk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default='\\Users\\amaia\\Documents\\GitHub\\IHC_HE_GenAI\\Image_segmentation\\dataset\\PN\\',
        
        help="specify the path to the pannuke dataset folder",
    )
    path_to_pannuke = parser.parse_args().path
    for F in ["Fold 1", "Fold 2", "Fold 3"]:
        for f in ["fold1", "fold2", "fold3"]:
            mask_path = os.path.join(path_to_pannuke, F, "masks", f, "masks.npy")
            if not os.path.exists(mask_path):
                print(f"File not found: {mask_path}")
                continue

            print(f"Processing: {mask_path}")
            y = np.load(mask_path)

            chunk_size = 100  # Ajusta esto según la memoria disponible
            num_chunks = y.shape[0] // chunk_size + (1 if y.shape[0] % chunk_size != 0 else 0)
            y_lab = []

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, y.shape[0])
                y_chunk = y[start_idx:end_idx]
                y_lab_chunk = process_chunk(y_chunk)
                y_lab.append(y_lab_chunk)

            y_lab = np.concatenate(y_lab, axis=0)
            labels_path = os.path.join(path_to_pannuke, F, "images", f, "labels.npy")
            np.save(labels_path, y_lab)

    print("done")

    