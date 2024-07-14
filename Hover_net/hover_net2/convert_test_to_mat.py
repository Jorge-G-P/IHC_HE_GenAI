import os
import numpy as np
from pathlib import Path
import scipy.io as sio
from tqdm import tqdm
from PIL import Image

# Definir la ruta del directorio de prueba
parent_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
test_dir = parent_path / "Datasets/Pannuke/data/test"
img_dir = test_dir / "images_png"
mat_dir = test_dir / "mat"

# Crear los directorios si no existen
img_dir.mkdir(exist_ok=True)
mat_dir.mkdir(exist_ok=True)

# Función para mapear las instancias
def flat_for(a, f):
    a = a.reshape(-1)
    for i, v in enumerate(a):
        a[i] = f(v)

def map_inst(inst):
    seg_indexes = np.unique(inst)
    new_indexes = np.array(range(0, len(seg_indexes)))
    mapping = {seg_index: new_index for seg_index, new_index in zip(seg_indexes, new_indexes)}
    flat_for(inst, lambda x: mapping[x])
    return inst, mapping

def calculate_centroids(inst_map):
    centroids = []
    for label in np.unique(inst_map):
        if label == 0:
            continue
        mask = inst_map == label
        coords = np.column_stack(np.where(mask))
        centroid = coords.mean(axis=0)
        centroids.append(centroid)
    return np.array(centroids)

# Convertir y guardar archivos .npy como .mat
for npy_file in tqdm(test_dir.glob('*6320.npy'), desc="Converting .npy to .mat"):
    data = np.load(npy_file)
    
    img_array = data[:, :, :3]  # RGB channels

    # Convertir la imagen a uint8 si es necesario
    if img_array.dtype != np.uint8:
        img_array = (img_array / img_array.max() * 255).astype(np.uint8)
    
    # Guardar la imagen como PNG
    img = Image.fromarray(img_array)
    img.save(img_dir / f"{npy_file.stem}.png")

    inst = data[:, :, 3]
    inst, mapping = map_inst(inst)
    inst_uid = np.unique(inst)
    inst_centroid = calculate_centroids(inst)
    
    # Verificar si los datos están vacíos
    if inst.size == 0 or inst_uid.size == 0 or inst_centroid.size == 0:
        print(f"Skipping {npy_file.stem} due to empty data.")
        continue

    # Guardar la anotación en formato .mat con la estructura correcta
    mat_save_path = mat_dir / f'{npy_file.stem}.mat'
    sio.savemat(mat_save_path, {
        'inst_map': inst,
        'inst_uid': inst_uid,
        'inst_centroid': inst_centroid
    })
    print(f"Saved {mat_save_path} with keys: {list(sio.loadmat(mat_save_path).keys())}")
    print(np.unique(inst))



# from scipy.io import loadmat

# # Rutas a los archivos .mat
# file_path = r'C:\Users\amaia\Documents\GitHub\IHC_HE_GenAI\Datasets\Pannuke\data\test\mat\6320.mat'
# file_path2 = r'C:\Users\amaia\Documents\GitHub\IHC_HE_GenAI\Results\hover_results\mat\6320.mat'

# # Función para verificar si los archivos .mat están vacíos
# def check_mat_file(file_path):
#     data = loadmat(file_path)
#     print(f"File: {file_path}")
#     print("Keys:", data.keys())
#     for key in data.keys():
#         if not key.startswith('__'):
#             print(f"  {key} is empty: {data[key].size == 0}")

# # Verificar los archivos
# check_mat_file(file_path)
# check_mat_file(file_path2)