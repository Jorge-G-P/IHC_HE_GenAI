import torch
import Pipeline.config as config
from PIL import Image 
import torch
import numpy as np
import os


def load_model_weights(checkpoint_file, model):
    try:
        print(f"=> Loading checkpoint {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
    except Exception as e:
        print(f"=> Failed to load checkpoint {checkpoint_file}: {str(e)}")
        raise

# def pretrained_generator():
#     gen_HE = Generator(img_channels=3, num_residuals=6).to(config.DEVICE)
#     load_model_weights("C:/Users/amaia/Documents/GitHub/IHC_HE_GenAI/Pipeline/checkpoint/genHE_200_epochs_20240628.pth.tar", gen_HE) #Put correct path to model weights
#     return gen_HE


def process_image(image_path, generatorHE, transform, output_dir):
    img = Image.open(image_path).convert('RGB')
    image_tensor = transform(img).unsqueeze(0).to(config.DEVICE)  # Add batch dimension and move to device

    converted_img = generatorHE(image_tensor)
    converted_img = converted_img.squeeze(0).cpu().detach()  # Remove batch dimension and move to CPU

    image_np = converted_img.numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    image_np = np.transpose(image_np, (1, 2, 0))

    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    image_pil = image_pil.resize((256, 256))  # Resize to 256x256

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    image_pil.save(output_path)