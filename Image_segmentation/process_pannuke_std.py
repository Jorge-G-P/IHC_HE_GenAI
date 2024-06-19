import numpy as np
import cv2
from PIL import Image
import os
from tqdm import trange
import matplotlib as mpl
from skimage.segmentation import find_boundaries

mpl.rcParams['figure.dpi'] = 300

data_dir = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/PanNuke_dataset/'  # ubicación de los datos extraídos
output_dir = '/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Image_segmentation/dataset/Folds/'  # ubicación para guardar los datos de salida

os.chdir(data_dir)
folds = os.listdir(data_dir)

def get_boundaries(raw_mask):
    '''
    Para extraer los límites de las instancias desde el archivo de groundtruth
    '''
    bdr = np.zeros(shape=raw_mask.shape)
    for i in range(raw_mask.shape[-1]-1):  # porque el último canal es el fondo
        bdr[:,:,i] = find_boundaries(raw_mask[:,:,i], connectivity=1, mode='thick', background=0)
    bdr = np.sum(bdr, axis=-1)
    return bdr.astype(np.uint8)

def get_instance_mask(mask):
    """
    Genera una máscara de instancias donde cada objeto está representado en blanco (255).
    
    Args:
        mask (np.array): Máscara semántica con los objetos etiquetados.

    Returns:
        np.array: Máscara de instancias donde cada objeto está en blanco (255).
    """
    instance_mask = np.zeros_like(mask, dtype=np.uint8)
    
    for label in np.unique(mask):
        if label == 0:
            continue  # Ignorar fondo
        
        # Crear una máscara binaria para el objeto actual
        binary_mask = (mask == label).astype(np.uint8) * 255
        
        # Agregar al resultado final
        instance_mask += binary_mask
    
    return instance_mask

for i, j in enumerate(folds):
    
    # Obtener rutas
    print('Cargando datos para {}, Espere...'.format(j))
    img_path = data_dir + j + '/images/fold{}/images.npy'.format(i+1)
    type_path = data_dir + j + '/images/fold{}/types.npy'.format(i+1)
    mask_path = data_dir + j + '/masks/fold{}/masks.npy'.format(i+1)
    print(40*'=', '\n', j, 'Comenzando\n', 40*'=')
    
    # Cargar archivos numpy
    masks = np.load(file=mask_path, mmap_mode='r')  # modo de solo lectura
    images = np.load(file=img_path, mmap_mode='r')  # modo de solo lectura
    types = np.load(file=type_path) 
    
    # Crear directorios para guardar imágenes
    try:
        os.mkdir(output_dir + j)
        os.mkdir(output_dir + j + '/images')
        os.mkdir(output_dir + j + '/sem_masks')
        os.mkdir(output_dir + j + '/inst_masks')
    except FileExistsError:
        pass
        
    
    for k in trange(images.shape[0], desc='Escribiendo archivos para {}'.format(j), total=images.shape[0]):
        
        raw_image =  images[k,:,:,:].astype(np.uint8)
        raw_mask = masks[k,:,:,:]
        sem_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8)
        # Intercambiar los canales 0 y 5 para que el fondo esté en el canal 0
        sem_mask = np.where(sem_mask == 5, 6, sem_mask)
        sem_mask = np.where(sem_mask == 0, 5, sem_mask)
        sem_mask = np.where(sem_mask == 6, 0, sem_mask)

        tissue_type = types[k]
        
        # Obtener las máscaras de instancias
        instance_mask = get_instance_mask(sem_mask)
        
        # Guardar imágenes y máscaras
        Image.fromarray(sem_mask).save(output_dir + '{}/sem_masks/sem_{}_{}_{:05d}.png'.format(j, tissue_type, i+1, k)) 
        Image.fromarray(instance_mask).save(output_dir + '{}/inst_masks/inst_{}_{}_{:05d}.png'.format(j, tissue_type, i+1, k)) 
        Image.fromarray(raw_image).save(output_dir + '{}/images/img_{}_{}_{:05d}.png'.format(j, tissue_type, i+1, k))