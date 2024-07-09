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
- [2. Tools and technologies](#2_toolstechnologies)
    - [2.1. Software](#21_software)
    - [2.2. Hardware](#22_hardware) 
- [3. Methodology??](#3_methodology)
    - [3.1. Time costs](#31_timecosts)
- [4. Data Overview](#4_dataoverview)
    - [4.1. Biological context](#41_biologicalcontext)
    - [4.2. BCI dataset](#42_bcidataset)
    - [4.3. Pannuke dataset](#43_pannukedataset)
    - [4.4. Endonuke dataset](#44_endonukedataset) 
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

Over the last decade Deep Neural Networks have produced unprecedented performance on a number of tasks,

On the other hand, since their introduction by [Goodfellowet al.](https://papers.nips.cc/paper/5423-generative-adversarial-nets), Generative Adversarial Networks (GANs) have become the defacto standard for high quality image synthesis. There are two general ways in which GANs have been used in medical imaging. The first is focused on the generative aspect and the second one is on the discriminative aspect. Focusing on the first one, GANs can help in exploring and discovering the underlying structure of training data and learning to generate new images. This property makes GANs very promising in coping with data scarcity and patient privacy.


### 1.1. Motivation <a name="11_motivation"></a>

Breast cancer is a leading cause of death for women. Histopathological checking is a gold standard to identify breast cancer.  To achieve this, the tumor materials are first made into hematoxylin and eosin (HE) stained slices (Figure 1). Then, the diagnosis is performed by pathologists by observing the HE slices under the microscope or analyzing the digitized whole slide images (WSI).

For diagnosed breast cancer, it is essential to formulate a precise treatment plan by checking the expression of specific proteins, such as human epidermal growth factor receptor 2 (HER2). The routine evaluation of HER2 expression is conducted with immunohistochemical techniques (IHC). An IHC-stained slice is shown in Figure 1. Intuitively, the higher the level of HER2 expression, the darker the color of the IHC image (Figure 2).

??foto


However, there are certain limitations in assessing the level of HER2 expression by IHC technology: 1)The preparation of IHC-stained slices is expensive. 2)Tumors are heterogeneous, however, IHC staining is usually performed on only one pathological slice in clinical applications, which may not entirely reflect the status of the tumor.

Therefore, we hope to be able to directly generate IHC images based on HE images.  In this way, we can save the cost of IHC staining, and can generate IHC images of multiple pathological tissues of the same patient to comprehensively assess HER2 expression levels.


### 1.2. Objectives <a name="12_objectives"></a>

The main purpose of this project is to demonstrate the potential solution to the problem of insufficiency data volume in the medical domain. The proposed solution consists of using GANs for synthetic medical data augmentation for improving a CNN-based classifier's performance. To tackle this task, it can be further broken down into the following sub-objectives:
- Explore, clean and process the data that will be used for training and evaluating the implemented Deep Neural Networks.
- Research, develop, implement and train a classifier model. This classifier will be based on a scaled up CNN whose function will be to detect malign dermathological lesions from the different augmented images.
- Perform classifier performance tests. In order to establish a solid base evaluation model to compare with, there wil be necessary to undertake several experiments for fine tuning appropriately the model to our data.
- Research, develop, implement and train a series of GANs-based models to be able to demonstrate how much we can improve the performance of the classifier.
- Carried out a series of experiments comparing the performance of the classifier using standard augmentation over the training data with respect to the performance obtained using the synthetic data from the differents GANs.
- Draw final conclusions from all the experiments conducted and the different improvements attempted.

## 2. Tools and technologies <a name="2_toolstechnologies"></a>

We have trained and tested two different models for our final pipeline. First, our task was a medical image-to-image translation task using a cycleGAN architecture and second an instance segmentation of cells nuclei of medical images. 

For that we used the [BCI Dataset](https://bci.grand-challenge.org/) obtained from the Grand Challenge, the [Endonuke Dataset](https://endonuke.ispras.ru/) and the [Pannuke Dataset]().
These 3 datasets are explained more in detail in section XXX.

<p align="center">
  <img src="Data/images-sagan/data-tree-background.png">
</p>

### 2.1. BCI Dataset  <a name="21_bcidataset"></a>

For training our model, we used the [BCI dataset](https://bci.grand-challenge.org/) obtained from the Grand Challenge. This dataset is specifically designed for medical imaging tasks and is well-suited for our project's objectives.

It proposes a breast cancer immunohistochemical (BCI) benchmark attempting to synthesize IHC data directly with the paired hematoxylin and eosin (HE) stained images. BCI dataset contains **9746 images (4873 pairs), 3896 pairs for train and 977 for test**, covering a variety of HER2 expression levels. 

Some sample HE-IHC image pairs are shown below:

![BCI dataset example](readme_images/BCIdatasetpreview.png)
 

### 2.2. Endonuke Dataset  <a name="22_hardware"></a> 

To enhance performance and efficiency, we carefully selected and configured our hardware resources in alignment with the demands of our model and the size of our dataset.



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

We selected PyTorch as framwork for our scientific computing package to develop our project. Regarding the image transformations used for standard augmentations, we have selected both Torchvision and Albumentation packages. To approach the imbalance dataset issue we used Torchsampler’s Imbalanced Dataset Sampler library. For visualization, we also used both classical Pyplot and Seaborn packages. For the dataset preprocessing, we made use of the modules available in Scikit-Learn library. Some of the GANs-based implementations developed make use of YAML as the preferred language for defining its configuration parameters files. Lastly, the package Pytorch Image Quality Assessment (PIQA) is used to generate the metrics that evaluate the quality of the synthetic images. And finally, for the model we made use of lukemelas EfficientNet architecture. 
 

### 4.2. Hardware  <a name="42_hardware"></a> 

- **Google Cloud Platform**

To start, we utilized a VM from Google Cloud Platform (GCP) with an Ubuntu Image, equipped with 1 NVIDIA L4 GPU, and a machine type of n1-standard-4 (4 vCPUs, 15 GB memory). As the computational demands increased for model training and to expedite the process, we upgraded to a VM from GCP with an Ubuntu Image, featuring 1 NVIDIA L4 GPU and a machine type of g2-standard-8 (8 vCPUs, 32 GB memory).

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
