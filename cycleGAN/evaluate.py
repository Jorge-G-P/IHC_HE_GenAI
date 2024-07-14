import torch
from tqdm import tqdm
import numpy as np
import os
import config
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.utils import save_image
from pytorch_fid.inception import InceptionV3
from scipy import linalg

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_activations(images, model, batch_size=50, dims=2048, device='cpu'):
    model.eval()

    if batch_size > len(images):
        batch_size = len(images)

    dataloader = torch.utils.data.DataLoader(
        images, batch_size=batch_size, shuffle=False, drop_last=False
    )

    pred_arr = np.empty((len(images), dims))
    start_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)[0]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx : start_idx + pred.shape[0]] = pred
            start_idx += pred.shape[0]

    return pred_arr

def calculate_activation_statistics(images, model, batch_size=50, dims=2048, device='cpu'):
    act = get_activations(images, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_fid_score(real_images, generated_images, device, batch_size=50, dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(real_images, model, batch_size, dims, device)
    m2, s2 = calculate_activation_statistics(generated_images, model, batch_size, dims, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def evaluate_fid_scores(generator_HE, generator_IHC, dataloader, device, fid_batch_size):
    generator_HE.eval()
    generator_IHC.eval()

    real_HE_images = []
    fake_HE_images = []
    real_IHC_images = []
    fake_IHC_images = []

    with torch.no_grad():
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            ihc_images = batch['A'].to(device)
            he_images = batch['B'].to(device)

            fake_he_images = generator_HE(ihc_images)
            fake_ihc_images = generator_IHC(he_images)

            real_HE_images.append(he_images)
            fake_HE_images.append(fake_he_images)
            real_IHC_images.append(ihc_images)
            fake_IHC_images.append(fake_ihc_images)
    
    real_HE_images = torch.cat(real_HE_images, dim=0)
    fake_HE_images = torch.cat(fake_HE_images, dim=0)
    real_IHC_images = torch.cat(real_IHC_images, dim=0)
    fake_IHC_images = torch.cat(fake_IHC_images, dim=0)

    fid_he = calculate_fid_score(real_HE_images, fake_HE_images, device, fid_batch_size)
    fid_ihc = calculate_fid_score(real_IHC_images, fake_IHC_images, device, fid_batch_size)

    return fid_he, fid_ihc


def calculate_incremental_activation_statistics(model, dataloader, generator, key, device, batch_size, dims=2048):
    model.eval()
    mu = np.zeros(dims)
    sigma = np.zeros((dims, dims))
    n = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=True):
            images = batch[key].to(device)
            fake_images = generator(images)

            activations = get_activations(fake_images, model, batch_size, dims, device)

            n_batch = activations.shape[0]
            n += n_batch
            mu += activations.sum(axis=0)
            sigma += np.dot(activations.T, activations)

    mu /= n
    sigma = sigma / n - np.outer(mu, mu)
    return mu, sigma



def evaluate_fid_scores_2(generator_HE, generator_IHC, dataloader, device, fid_batch_size):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)

    # Use identity function for real HE images (from key 'B')
    mu_real_HE, sigma_real_HE = calculate_incremental_activation_statistics(model, dataloader, lambda x: x, 'B', device, fid_batch_size)
    # Generate fake HE images from real IHC images (from key 'A')
    mu_fake_HE, sigma_fake_HE = calculate_incremental_activation_statistics(model, dataloader, generator_HE, 'A', device, fid_batch_size)
    
    # Use identity function for real IHC images (from key 'A')
    mu_real_IHC, sigma_real_IHC = calculate_incremental_activation_statistics(model, dataloader, lambda x: x, 'A', device, fid_batch_size)
    # Generate fake IHC images from real HE images (from key 'B')
    mu_fake_IHC, sigma_fake_IHC = calculate_incremental_activation_statistics(model, dataloader, generator_IHC, 'B', device, fid_batch_size)

    fid_he = calculate_frechet_distance(mu_real_HE, sigma_real_HE, mu_fake_HE, sigma_fake_HE)
    fid_ihc = calculate_frechet_distance(mu_real_IHC, sigma_real_IHC, mu_fake_IHC, sigma_fake_IHC)

    return fid_he, fid_ihc

# import torch
# from tqdm import tqdm
# import numpy as np
# from torch.nn.functional import adaptive_avg_pool2d
# from torchvision.models import inception_v3
# import torchvision.transforms as transforms
# from scipy import linalg

# # Predefined transform for Inception V3
# transform_inception = transforms.Compose([
#     transforms.Resize((299, 299)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)

#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)

#     assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
#     assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

#     diff = mu1 - mu2

#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

#     if np.iscomplexobj(covmean):
#         covmean = covmean.real

#     tr_covmean = np.trace(covmean)

#     return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# def get_activations(images, model, batch_size=50, dims=2048, device='cpu'):
#     model.eval()

#     if batch_size > len(images):
#         batch_size = len(images)

#     pred_arr = np.empty((len(images), dims))
#     start_idx = 0

#     with torch.no_grad():
#         for start in range(0, len(images), batch_size):
#             end = start + batch_size
#             batch = images[start:end].to(device)
#             pred = model(batch)[0]

#             if pred.size(2) != 1 or pred.size(3) != 1:
#                 pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

#             pred = pred.squeeze(3).squeeze(2).cpu().numpy()
#             pred_arr[start:start + pred.shape[0]] = pred

#     return pred_arr

# def calculate_activation_statistics(images, model, batch_size=50, dims=2048, device='cpu'):
#     act = get_activations(images, model, batch_size, dims, device)
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma

# def calculate_fid_score(real_images, generated_images, device, batch_size=50, dims=2048):
#     model = inception_v3(pretrained=True, transform_input=False).to(device)
#     model.eval()

#     real_images = torch.stack([transform_inception(img) for img in real_images])
#     generated_images = torch.stack([transform_inception(img) for img in generated_images])

#     m1, s1 = calculate_activation_statistics(real_images, model, batch_size, dims, device)
#     m2, s2 = calculate_activation_statistics(generated_images, model, batch_size, dims, device)
#     fid_value = calculate_frechet_distance(m1, s1, m2, s2)

#     return fid_value

# def evaluate_fid_scores(generator_HE, generator_IHC, dataloader, device, fid_batch_size):
#     generator_HE.eval()
#     generator_IHC.eval()

#     real_HE_images = []
#     fake_HE_images = []
#     real_IHC_images = []
#     fake_IHC_images = []

#     with torch.no_grad():
#         loop = tqdm(dataloader, leave=True)
#         for batch in loop:
#             ihc_images = batch['A'].to(device)
#             he_images = batch['B'].to(device)

#             fake_he_images = generator_HE(ihc_images)
#             fake_ihc_images = generator_IHC(he_images)

#             real_HE_images.append(he_images)
#             fake_HE_images.append(fake_he_images)
#             real_IHC_images.append(ihc_images)
#             fake_IHC_images.append(fake_ihc_images)
    
#     real_HE_images = torch.cat(real_HE_images, dim=0)
#     fake_HE_images = torch.cat(fake_HE_images, dim=0)
#     real_IHC_images = torch.cat(real_IHC_images, dim=0)
#     fake_IHC_images = torch.cat(fake_IHC_images, dim=0)

#     fid_he = calculate_fid_score(real_HE_images, fake_HE_images, device, fid_batch_size)
#     fid_ihc = calculate_fid_score(real_IHC_images, fake_IHC_images, device, fid_batch_size)

#     return fid_he, fid_ihc

# def calculate_incremental_activation_statistics(model, dataloader, generator, key, device, batch_size, dims=2048):
#     model.eval()
#     mu = np.zeros(dims)
#     sigma = np.zeros((dims, dims))
#     n = 0

#     with torch.no_grad():
#         for batch in tqdm(dataloader, leave=True):
#             images = batch[key].to(device)
#             fake_images = generator(images)

#             activations = get_activations(fake_images, model, batch_size, dims, device)

#             n_batch = activations.shape[0]
#             n += n_batch
#             mu += activations.sum(axis=0)
#             sigma += np.dot(activations.T, activations)

#     mu /= n
#     sigma = sigma / n - np.outer(mu, mu)
#     return mu, sigma

# def evaluate_fid_scores_2(generator_HE, generator_IHC, dataloader, device, fid_batch_size):
#     model = inception_v3(pretrained=True, transform_input=False).to(device)

#     # Use identity function for real HE images (from key 'B')
#     mu_real_HE, sigma_real_HE = calculate_incremental_activation_statistics(model, dataloader, lambda x: x, 'B', device, fid_batch_size)
#     # Generate fake HE images from real IHC images (from key 'A')
#     mu_fake_HE, sigma_fake_HE = calculate_incremental_activation_statistics(model, dataloader, generator_HE, 'A', device, fid_batch_size)
    
#     # Use identity function for real IHC images (from key 'A')
#     mu_real_IHC, sigma_real_IHC = calculate_incremental_activation_statistics(model, dataloader, lambda x: x, 'A', device, fid_batch_size)
#     # Generate fake IHC images from real HE images (from key 'B')
#     mu_fake_IHC, sigma_fake_IHC = calculate_incremental_activation_statistics(model, dataloader, generator_IHC, 'B', device, fid_batch_size)

#     fid_he = calculate_frechet_distance(mu_real_HE, sigma_real_HE, mu_fake_HE, sigma_fake_HE)
#     fid_ihc = calculate_frechet_distance(mu_real_IHC, sigma_real_IHC, mu_fake_IHC, sigma_fake_IHC)

#     return fid_he, fid_ihc
