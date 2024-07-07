import os
import numpy as np
import torch
from PIL import Image
import subprocess
from torchvision import transforms
from celldet import CellDetectionMetric
from import_gan import pretrained_generator, process_image
from DicDataset import DicDataset

data_folder = r'C:\Users\amaia\Documents\GitHub\IHC_HE_GenAI\Pipeline\endonuke_data\data'
run_infer_path = r'C:\Users\amaia\Documents\GitHub\IHC_HE_GenAI\Image_segmentation\Hover_net\hover_net2\run_infer.py'

#Create paths to folders
crop_images_folder = os.path.join(data_folder, 'crop_images')
crop_txt_folder = os.path.join(data_folder, 'crop_txt')
gan_results_folder = os.path.join(data_folder, 'Results')
results_hover_folder = os.path.join(data_folder, 'results_hover')

if not os.path.exists(results_hover_folder):
    os.makedirs(results_hover_folder)

if not os.path.exists(gan_results_folder):
    os.makedirs(gan_results_folder)

# Load the pre-trained generator
generatorHE = pretrained_generator()
transform = transforms.ToTensor()

# Process and save each image in the crop_images_folder
for filename in os.listdir(crop_images_folder):
    if filename.endswith(".png"):  # Add more extensions if needed
        image_path = os.path.join(crop_images_folder, filename)
        process_image(image_path, generatorHE, transform, gan_results_folder)
        print(f"Processed and saved: {filename}")

# Define the command to be executed
command = [
    'python', run_infer_path,
    '--model_path=C:\\Users\\amaia\\Documents\\GitHub\\IHC_HE_GenAI\\Image_segmentation\\Hover_net\\hover_net2\\checkpoint\\01\\net_epoch=1.tar',
    '--model_mode=fast',
    'tile',
    '--input_dir=C:\\Users\\amaia\\Documents\\GitHub\\IHC_HE_GenAI\\Pipeline\\endonuke_data\\data\\Results',
    '--output_dir=C:\\Users\\amaia\\Documents\\GitHub\\IHC_HE_GenAI\\Pipeline\\endonuke_data\\data\\results_hover',
    '--draw_dot'
]

# Execute the command (creates a subfolder)
subprocess.run(command)

# In the created subfolder, search for the 'jason' files
json_hover_folder = os.path.join(results_hover_folder, 'json')

dataset = DicDataset(crop_txt_folder, json_hover_folder)

# Instantiate the metric
metric = CellDetectionMetric(num_classes=1, thresholds=0.5)

real_centroids = []
predicted_centroids = []

for i in range(len(dataset)):
    dict_from_txt, dict_from_json, name = dataset[i]
    real_centroids.append(dict_from_txt)
    predicted_centroids.append(dict_from_json)
    if i%200==0 and i!=0:
        metric.update(predicted_centroids, real_centroids)
        real_centroids = []
        predicted_centroids = []
        print(i)

metric.update(predicted_centroids, real_centroids)
results = metric.compute()
print(results)