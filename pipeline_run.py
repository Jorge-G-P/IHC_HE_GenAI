import os
import subprocess
from Pipeline.celldet import CellDetectionMetric
# import cycleGAN.config
from Pipeline.DicDataset import DicDataset
import Pipeline.config as config
# datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, datasets_dir)
# from Datasets.Endonuke.preprocessing import preprocess_endonuke
# dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
# repo_path = dir_path.parent
# parent_path = repo_path.parent
# print(repo_path)
# print(parent_path)


# if config.PREPROCESS_ENDONUKE:
#     preprocess_endonuke()

#Create paths to folders
crop_images_folder = os.path.join(config.path_to_endonuke_data_folder, 'crop_images')
crop_txt_folder = os.path.join(config.path_to_endonuke_data_folder, 'crop_txt')


os.makedirs(config.parent_path / "IHC_HE_GenAI/Results/gan_results", exist_ok=True)
os.makedirs(config.parent_path / "IHC_HE_GenAI/Results/hover_results", exist_ok=True)
gan_results_folder = config.parent_path / "IHC_HE_GenAI/Results/gan_results"
hover_results_folder = config.parent_path / "IHC_HE_GenAI/Results/hover_results"
results_folder = config.parent_path / "IHC_HE_GenAI/Results"

# # Load the pre-trained generator
# # generatorHE = pretrained_generator()
# generatorHE = Generator(img_channels=3, num_residuals=config.N_RES_BLOCKS).to(config.DEVICE)
# load_model_weights(config.path_to_genHE_weights, generatorHE)
# print(generatorHE)
# transform = transforms.ToTensor()

# # Process and save each image in the crop_images_folder
# for filename in os.listdir(crop_images_folder):
#     if filename.endswith(".png"):  # Add more extensions if needed
#         image_path = os.path.join(crop_images_folder, filename)
#         process_image(image_path, generatorHE, transform, gan_results_folder)
#         print(f"Processed and saved: {filename}")

# Define the command to be executed
command = [
    'python', config.path_to_Hover_net_run_infer,
    '--model_path='+str(config.path_to_Hover_net_weights),
    '--model_mode=fast',
    'tile',
    '--input_dir='+str(gan_results_folder),
    '--output_dir='+str(hover_results_folder),
    '--draw_dot'
]

# Execute the command (creates a subfolder)
subprocess.run(command)

# In the created subfolder, search for the 'jason' files
json_hover_folder = os.path.join(hover_results_folder, 'json')

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


results_path = os.path.join(results_folder, 'results.txt')
with open(results_path, 'w') as f:
    f.write(str(results))
