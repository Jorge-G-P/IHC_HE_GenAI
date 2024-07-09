# [AIDL24] AI-Driven Histopathology: Transforming IHC to H&E Stained Images for Enhanced Nuclei Segmentation Using CycleGAN and HoverNet

This repository contains 



### About
Final Project for the UPC [Artificial Intelligence with Deep Learning Postgraduate Course](https://www.talent.upc.edu/ing/estudis/formacio/curs/310402/postgraduate-course-artificial-intelligence-deep-learning/) 2023-2024 edition, authored by:

* [Amaia](linkedin??)
* [João Pedro Vieira](www.linkedin.com/in/joão-pedro-vieira-1369a51b6)
* [Jorge Garcia]()
* [Josep Baradat]()

Advised by [Oscar Pina]()

## Table of Contents <a name="toc"></a>

- [1. Introduction](#1_intro)
    - [1.1. Motivation](#11_motivation)
    - [1.2. Objectives](#12_objectives)
- [2. Corpora](#2_corpora)
    - [2.1. BCI Dataset](#21_bcidataset)
        - [2.1.1. Pre-processing](#211_preprocess)   
    - [2.2. Endonuke Dataset](#22_endonukedataset)
    - [2.3. Pannuke Dataset](#23_pannukeedataset) 
- [3. Methodology??](#3_methodology)
    - [3.1. Time costs](#31_timecosts)
- [4. Environment Requirements](#4_envreqs)
    - [4.1. Software](#41_software)
    - [4.2. Hardware](#42_hardware)
- [5. Experiment's design and results](#5_experimentsdesignandresults)    
    - [5.1. cycleGAN](#51_cycleGAN)
        - [Data preprocessing](#511_datapreprocessing)
        - [Model architecture](#512_modelarchitecture)
        - [Training configuration](#513_trainingconfiguration)
        - [Fine_tuning procedure](#514_finetuningprocedure)
        - [Test results](#515_testresults)
    - [5.2. HoverNet](#52_hovernet)
        - [Data preprocessing](#521_datapreprocessing)
        - [Model architecture??](#522_modelarchitecture)
        - [Training configuration](#523_trainingconfiguration)
        - [Test results](#524_testresults)
    - [5.3. Pipeline ensemble](#53_gans)
        - [Data preprocessing](#531_datapreprocessing)
        - [Ensemble](#532_ensemble)
        - [Test results](#533_testresults)
- [6. How to Run](#6_howtorun)
- [7. Conclusions and future work](#7_conclusionsandfuturework) 
- [8. Acknowledgements](#8_acknowledgements)
 
## 1. Introduction <a name="1_intro"></a>


### 1.1. Motivation <a name="11_motivation"></a>


### 1.2. Objectives <a name="12_objectives"></a>


## 2. Corpora <a name="2_corpora"></a>

We have trained and tested two different models for our final pipeline. First, we needed to perform a medical image-to-image translation task using a cycleGAN architecture. Then, an instance segmentation of cells nuclei task for medical images. 

For that we used the [BCI Dataset](https://bci.grand-challenge.org/) obtained from the Grand Challenge, the [Endonuke Dataset](https://endonuke.ispras.ru/) and the [Pannuke Dataset]().
These 3 datasets are explained more in detail in section XXX.


### 2.1. BCI Dataset  <a name="21_bcidataset"></a>

For training our cycleGAN model, we used the [BCI dataset](https://bci.grand-challenge.org/) obtained from the Grand Challenge. This dataset is specifically designed for medical imaging tasks and is well-suited for our project's objectives. 
It proposes a breast cancer immunohistochemical (BCI) benchmark attempting to synthesize IHC data directly with the paired hematoxylin and eosin (HE) stained images. 

The original dataset contains 9746 images (4873 pairs), 3896 pairs for train and 977 for test, covering a variety of HER2 expression levels, each with a resolution of 1024x1024 pixels. Some sample HE-IHC image pairs are shown below:

![BCI dataset example](readme_images/BCIdatasetpreview.png)


#### 2.1.1. Pre-processing  <a name="211_preprocess"></a>

- #### Dataset Resolution

Due to the high resolution of the original dataset and in order to accelerate the training process, we built the dataset class with a dynamic cropping feature to be able to work with smaller images. Before feeding them to the network, they are divided into smaller patches, yielding a total of 38984 images (19492 pairs) with a resolution of 512x512 pixels. The dataset is subsequently split into training, validation, and test sets, adhering to a ratio of 60%, 20%, and 20%, respectively.

![BCI crops](readme_images/BCI_crops.png)


- #### Dataset Reduction

We used just 50% of our training and validation sets, to reduce training costs, while keeping a representative sample size for good generalization. Finally, the model was trained with 15592 images (7796 pairs).

- #### Data Augmentation

In order to increase diversity on the training set, we applied several transformations to it, using Albumentations library.
For further reduction of the training time, we resized the images to 256*256. To avoid overfitting, techniques like flips and rotations were applied. Also, we applied normalization of mean and standard deviation and finally we converted the images to tensors so they can be processed by the model.


### 2.2. Endonuke Dataset  <a name="22_endonukedataset"></a> 

### 2.3. Pannuke Dataset  <a name="22_pannukedataset"></a> 



## 3. Methodology <a name="3_-_methodology"></a>

Under this section we present all the GAN versions implemented. We approach to the proble with our own variation of implementation of the technique and methodology first introduced in [Frid-Adar et al.](https://arxiv.org/abs/1803.01229) in 2018.

### 3.1. Time costs  <a name="31_timecosts"></a>

Since we lack from any medical expertise for assessing the quality of the generated images, we have implemented several metrics to measure traits of our output pictures.

- #### Peak Signal-to-Noise Ratio (PSNR)

This metric is used to measure the quality of a given image (noise), which underwent some transformation, compared to the its original (signal). In our case, the original picture is the real batch of images feeded into our network and the noise is represented by a given generated image.

- #### Structural Similarity (SSIM)

SSIM aims to predict the perceived the quality of a digital image. It is a perception based model that computes the degradation in an image comparison as in the preceived change in the structural information. This metric captures the perceptual changes in traits such as luminance and contrast.

- #### Multi-Scale Gradient Magnitude Similarity Deviation (MS GMSD)

MS-GMSD works on a similar version as SSIM, but it also accounts for different scales for computing luminance and incorporates chromatic distortion support.

- #### Mean Deviation Similarity Index (MDSI)

MDSI computes the joint similarity map of two chromatic channels through standard deviation pooling, which serves as an estimate of color changes. 

- #### Haar Perceptural Similarity Index (HaarPSI)

HaarPSI works on the Haar wavelet decomposition and assesses local similarities between two images and the relative importance of those local regions. This metric is the current state-of-the-art as for the agreement with human opinion on image quality. 

- #### Measure Assessment

Measure | Bar | 
:------: | :------:|
PSNR   | Context dependant, generally the higher the better.  | 
SSIM   |  Ranges from 0 to 1, being 1 the best value.     | 
MS-GMSD |  Ranges from 0 to 1, being 1 the best value.    |  
MDSI   |   Ranges from 0 to inf, being 0 the best value.    |
HaarPSI |   Ranges from 0 to 1, being 1 the best value.   |



## 4. Environment Requirements <a name="4_env_reqs"></a>
### 4.1. Software  <a name="41_software"></a>

We selected PyTorch as the framework for our AI project development due to its robust capabilities in scientific computing and deep learning. Our project leverages several key libraries and tools to enhance its functionality and performance:

**Image Transformations and Augmentations:** We utilized both Torchvision and Albumentations packages for standard augmentations and image transformations, ensuring diverse and effective training data.

**Dataset Preprocessing:** Custom functions and classes were developed for specific preprocessing needs, complemented by the skimage library for additional image processing utilities.

**Architectures and Models:** Our implementations include advanced models such as CycleGAN for image-to-image translation tasks and HoverNet for nuclear segmentation in biomedical images.

**Metrics and Evaluation:** For evaluating the quality of the synthetic images generated by CycleGAN, we employed the Fréchet Inception Distance (FID) metric, ensuring rigorous and accurate assessment of our model's performance.

This combination of powerful libraries and custom solutions has enabled the development of a robust and efficient AI system tailored to our project's requirements.


### 4.2. Hardware  <a name="42_hardware"></a> 

On an initial phase we started by training our models locally on our laptops but as the project architecture expanded, we rapidly looked for another approach due to the high computing demands.

- **Google Cloud Platform**

To start, we utilized a VM from Google Cloud Platform (GCP) with an Ubuntu Image, equipped with 1 NVIDIA T4 GPU, and a machine type of n1-standard-4 (4 vCPUs, 15 GB memory). As the computational demands increased for model training and to expedite the process, we upgraded to a VM from GCP with an Ubuntu Image, featuring 1 NVIDIA L4 GPU and a machine type of g2-standard-8 (8 vCPUs, 32 GB memory).

To leverage GPU acceleration, we employed CUDA, significantly enhancing our processing capabilities. We used Google Cloud Buckets to store and import raw dataset files to the VM. Additionally, we utilized the gcloud SDK for seamless data import/export to and from the VM.

For accessing the VM and conducting our work, we established an SSH connection and utilized Visual Studio Code with the remote-ssh extension. This setup provided an efficient and flexible environment for developing and training our AI models.


## 5. Experiment's Design and Results <a name="5_experimentsdesignandresults"></a>

### 5.1. cycleGAN  <a name="51_cyclegan"></a> 
- [Data preprocessing]<a name="511_datapreprocessing"></a>
- [Model architecture](512_modelarchitecture)<a name="512_modelarchitecture"></a>
- [Training configuration](513_trainingconfiguration)
- [Fine_tuning procedure](514_finetuningprocedure)
- [Test results](515_testresults)
### 5.2. Pannuke Dataset  <a name="43-pannukedataset"></a> 
### 5.3. Endonuke Dataset  <a name="43-endonukedataset"></a> 



## 6. How to Run <a name="6_howtorun"></a>



## 7. Conclusions and Future Work  <a name="7_conclusionsandfuturework"></a>

* **Training GANs** proved to be a **hard task**.
    * Requires a vest amount of **resources**.
    * **Training process** is **not straightforward**.

* **SNGAN outperformed** DCGAN, ACGAN and WGAN.
    * Even though **after huge amount of experimentation** metrics were still far from initial goal.

* On the **GAN training parametrization**:
    * **Batch size** is among the most relevant parameters to reduce training times and improve image quality. The reasonale behind this effect could come from the _Discriminator_ having less examples to generalize its classification of real and fake images.
    * The number of **training epochs** also affects the quality of the generated images. Longer traning usually ends up producing better images even though the two losses did not converge.
    * Another parameter tweak that comes handy when training these architectures is the **size of the latent vector**. With higher sizes the quality of images did not improve, but it did reduce the training time.
    * **Label smoothing** has another critical change that was done in our GANs. It did produce better images and also it did stabilize the training. Mathematically, the class probabilities of the discriminator are, in general, lower when using this technique and thus, it balances the performance of the _Discriminator_ and the _Generator_.
    * **Spectral normalization**, which deals with exploding gradients, did also increase the quality of the generated images. It gave out a new architecture purely based on a DCGAN.
    * **Different learning rates**, more specifically with higher values for the _Discriminator_, did stabilize training and also increased the quality of the images. The explanation behind this behavior is that setting bigger steps for optimizing the loss function of the _Discriminator_  makes this agent more imprecise at the classification task whereas the smaller steps for the _Generator_ gives it a more precise approach to image generation.

* **Different metrics** are sensible to **different aspects** of image quality.
    * Best practice to **use a set** of them to assess the generated images.
    * **Include a metric** based on **human perception**.
 
* Good results for a **lack** of **resources**.
    * Fine-tuned **EfficientNet** achieves **high accuracy** with **reduced dataset**.
    * Dataset with **sysnthetic images** does **not improve accuracy**.
    * **Balanced dataset** with **synthetic images** and no augmentations achieves **good results**.

## 8. Acknowledgements <a name="8_acknowledgements"></a>

We would like to thank all the staff from the Prostgraduate Course on Artificial Intelligence with Deep Learning for all the effort and care that they took and showed preparing the materials and lecture which provided us with the tools to build this project.

We would like to give a special thanks to Oscar, our advisor, who provided very helpful advise and spent numerous hours revising our material and searching tricks and tips for our project.
