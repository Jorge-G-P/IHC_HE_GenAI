# AIDL21: GAN-based synthetic medical image augmentation

This repository contains 



### About
Final Project for the UPC [Artificial Intelligence with Deep Learning Postgraduate Course](https://www.talent.upc.edu/ing/estudis/formacio/curs/310402/postgraduate-course-artificial-intelligence-deep-learning/) 2020-2021 online edition, authored by:

* [Amaia](linkedin??)
* []()
* []()
* []()

Advised by [Oscar]()

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

Histopathological imaging plays a **crucial role** in medical diagnosis and research. It provides a detailed view of the biological tissues at a microscopic level, enabling the identification of diseases such as cancer. Two common staining techniques used in histopathology are **Immunohistochemistry (IHC)** and **Hematoxylin and Eosin (HE)** staining.

**IHC staining** is a special staining process used to detect specific antigens in tissues with the help of antibodies. It is particularly useful in the identification of abnormal cells such as those found in cancerous tumors. On the other hand, **HE staining** is the most widely used staining technique in medical diagnosis. It uses hematoxylin, which stains nuclei blue, and eosin, which stains the cytoplasm and extracellular matrix pink. This results in a high contrast image that allows pathologists to distinguish different tissue structures and cell types.

However, the process of staining is time-consuming and requires expert knowledge to interpret the results. Moreover, each staining technique provides different information, and often, pathologists need to review slides under both stains for accurate diagnosis. This is where **Deep Learning** can make a significant impact.

In this project, we propose a novel approach that uses **CycleGAN**, a type of Generative Adversarial Network (GAN), to translate **IHC stained images to HE staining**. CycleGAN has shown remarkable results in image-to-image translation tasks, even when there are no paired examples in the training set. By training a CycleGAN model on unpaired IHC and HE stained images, we aim to generate synthetic HE images from IHC input. This could potentially save a significant amount of time and resources in the staining process.

After that, another deep learning model was employed to calculate **cell centroids** from the translated HE images. The location of cell centroids can provide valuable information about the spatial distribution of cells, which is often an important factor in disease diagnosis.

By leveraging deep learning to translate between different stainings and calculate cell centroids, we aim to enhance the efficiency and accuracy of histopathological imaging analysis. This could lead to faster and more accurate diagnoses, ultimately improving patient outcomes. We believe that our project will contribute significantly to the field of digital pathology and demonstrate the transformative potential of deep learning in medical imaging.

--------------------------------

Breast cancer is a leading cause of death for women. Histopathological checking is a gold standard to identify breast cancer.  To achieve this, the tumor materials are first made into hematoxylin and eosin (HE) stained slices (Figure 1). Then, the diagnosis is performed by pathologists by observing the HE slices under the microscope or analyzing the digitized whole slide images (WSI).

For diagnosed breast cancer, it is essential to formulate a precise treatment plan by checking the expression of specific proteins, such as human epidermal growth factor receptor 2 (HER2). The routine evaluation of HER2 expression is conducted with immunohistochemical techniques (IHC). An IHC-stained slice is shown in Figure 1. Intuitively, the higher the level of HER2 expression, the darker the color of the IHC image (Figure 2).

??foto


However, there are certain limitations in assessing the level of HER2 expression by IHC technology: 1)The preparation of IHC-stained slices is expensive. 2)Tumors are heterogeneous, however, IHC staining is usually performed on only one pathological slice in clinical applications, which may not entirely reflect the status of the tumor.

Therefore, we hope to be able to directly generate IHC images based on HE images.  In this way, we can save the cost of IHC staining, and can generate IHC images of multiple pathological tissues of the same patient to comprehensively assess HER2 expression levels.


### 1.2. Objectives <a name="12_objectives"></a>

The main purpose of this project is to elaborate a method in which the more scarce IHC images can be 

In order to achieve it, a pipeline

To tackle this task, it haas been further broken down into the following sub-objectives:
- Explore, clean and process the data that will be used for training and evaluating the implemented Deep Neural Networks.
- Research, develop, implement and train a classifier model. This classifier will be based on a scaled up CNN whose function will be to detect malign dermathological lesions from the different augmented images.
- Perform classifier performance tests. In order to establish a solid base evaluation model to compare with, there wil be necessary to undertake several experiments for fine tuning appropriately the model to our data.
- Research, develop, implement and train a series of GANs-based models to be able to demonstrate how much we can improve the performance of the classifier.
- Carried out a series of experiments comparing the performance of the classifier using standard augmentation over the training data with respect to the performance obtained using the synthetic data from the differents GANs.
- Draw final conclusions from all the experiments conducted and the different improvements attempted.

## 2. Tools and technologies <a name="2_toolstechnologies"></a>

For training and testing our models, we have used the dataset provided by the healthcare organization for informatics in medical imaging, the [Society for Imaging Informatics in Medicine (SIIM)](https://siim.org/) joined by the [International Skin Imaging Collaboration (ISIC)](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main). 

After some filtering for rebalancing the missing or non-labeled images and cutting off the excess of benign lesions, we finish with a total number of 22,922 images split in Train (16,045 observations) and Test (6,877 observations).

<p align="center">
  <img src="Data/images-sagan/data-tree-background.png">
</p>

### 2.1. Software  <a name="21_software"></a>


We selected PyTorch as framwork for our scientific computing package to develop our project. Regarding the image transformations used for standard augmentations, we have selected both Torchvision and Albumentation packages. To approach the imbalance dataset issue we used Torchsampler’s Imbalanced Dataset Sampler library. For visualization, we also used both classical Pyplot and Seaborn packages. For the dataset preprocessing, we made use of the modules available in Scikit-Learn library. Some of the GANs-based implementations developed make use of YAML as the preferred language for defining its configuration parameters files. Lastly, the package Pytorch Image Quality Assessment (PIQA) is used to generate the metrics that evaluate the quality of the synthetic images. And finally, for the model we made use of lukemelas EfficientNet architecture. 
 




### 2.2. Hardware  <a name="22_hardware"></a> 

To be able to feed our dataset into the classifier, we must first condition it to the network and to our resource’s limitations.

- **Students' personal laptops**
-
-
-
-

- **Google Cloud Platform**

To launch the instance Cloud Deep Learning VM Image was used. We created a Linux VM, with a n1-highmem-2 machine and 1 NVIDIA Tesla k80 GPU. In addition to the Instance, we created a bucket in order to upload the Images from the different datasets (the reduced one, and the ones with the GANs) to then move them to the persistent disk. We firstly implemented our code using the Jupyter Notebook function, but since the training process took a long time and Google Cloud Shell would eventually log off, we switched to SSH and launched our script from the terminal.



## 3. Methodology <a name="3_-_methodology"></a>

One-hour meetings were held weekly between the team and the advisor. Besides from that, another two two-hour meetings were held weekly by the team without the advisor.
Moreover two sprints were done during the development procedure. The last week before the critical review, and three weeks before the final presentation. During those sprints, the amount of time spent by each student on the project was roughly doubled.

### 3.1. Time costs  <a name="31_timecosts"></a>

??Gantt-diagram


--------------------??métricas??
- #### ?? (FID)




- #### Measure Assessment

Measure | Bar | 
:------: | :------:|
PSNR   | Context dependant, generally the higher the better.  | 
SSIM   |  Ranges from 0 to 1, being 1 the best value.     | 
MS-GMSD |  Ranges from 0 to 1, being 1 the best value.    |  
MDSI   |   Ranges from 0 to inf, being 0 the best value.    |
HaarPSI |   Ranges from 0 to 1, being 1 the best value.   |



## 4. Data overview <a name="4_dataoverview"></a>
### 4.1. Biological Context  <a name="41_biologicalcontext"></a>




#### CSV files

As mentioned before, every image comes with a JSON files with relevant information regarding the patient and the skin spot. This files were all put into a CSV file where each column stands for a field from the JSON file. 
In addition to the initial fields, we added “dcm_name” to store the name of the image the data belongs to, and “target” which is set to 0 if the skin spot is benign and to 1 in case it is malignant.

#### Dataset reduction

We reduced the dataset to 5K images to diminish the training cost, keeping the malignant/benign ratio so the results can be escalated to the complete dataset.

Given the size of the image’s directory, we modified it so it only contained the images that were going to be fed into the network, in order not to use more storage than necessary in GCP. We did this through a series of automated functions in python.

#### Data augmentation

We applied several transformations to avoid overfitting when training our network with the reduced dataset. To do that, we have used albumentations’ library due to the large number of augmentations it has available.

The images input size was variable and with a bigger resolution, we resized them to 128x128 to fit the synthetically generated images. Furthermore, we applied techniques involving changing the image’s contrast and brightness scale and rotations. We also normalized its mean and standard deviation and finally we converted the images ton tensors so they can be feed into our model.

Here are some of the images before and after applying the transformations.






### 4.2. BCI Dataset  <a name="42_bcidataset"></a> 

The GANs were trained using Google Colab. This work environment provided us an easy way to work in teams and to access to GPUs. The Classifier also started as a Google Colab project, however, due to its high computing demands, we were forced to port it to Google Cloud to avoid the time limit of the Colab.  

### 4.3. Pannuke Dataset  <a name="43_pannukedataset"></a> 

### 4.4. Endonuke Dataset  <a name="44_endonukedataset"></a> 

## 5. Experiment's Design and Results <a name="5_experimentsdesignandresults"></a>



### 5.1. cycleGAN  <a name="51_cyclegan"></a> 
- Data preprocessing<a name="511_datapreprocessing"></a>
- Model architecture<a name="512_modelarchitecture"></a>
- Training configuration<a name="513_modelarchitecture"></a>
- Fine_tuning procedure<a name="514_modelarchitecture"></a>
- Test results<a name="515_modelarchitecture"></a>
### 5.2. Pannuke Dataset  <a name="52-pannukedataset"></a>
- Data preprocessing<a name="521_datapreprocessing"></a>
- Model architecture<a name="522_modelarchitecture"></a>
- Training configuration<a name="523_modelarchitecture"></a>
- Test results<a name="524_modelarchitecture"></a> 
### 5.3. Endonuke Dataset  <a name="53-endonukedataset"></a> 
- Data preprocessing<a name="531_datapreprocessing"></a>
- Ensemble<a name="532_emsemble"></a>
- Test results<a name="533_modelarchitecture"></a>


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

We would like to thank all the staff from the Prostgraduate Course on Artificial Intelligence with Deep Learning for all the effort and care that they took and showed preparing the materials and lectures which provided us with the tools to build this project.

We would like to give a special thanks to Oscar, our advisor, who provided very helpful advise and spent numerous hours revising our material and searching tricks and tips for our project.


## Refs??
Essam H. Houssein, Marwa M. Emam, Abdelmgeid A. Ali, Ponnuthurai Nagaratnam Suganthan,
Deep and machine learning techniques for medical imaging-based breast cancer: A comprehensive review,
Expert Systems with Applications,
Volume 167,
2021,
114161,
ISSN 0957-4174,
https://doi.org/10.1016/j.eswa.2020.114161.

??Ejemplo cite: [Frid-Adar et al.](https://arxiv.org/abs/1803.01229)