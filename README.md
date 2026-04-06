# Tea Leaf Disease Classification using m-EfficientNetB0+ECA

This repository contains the implementation of a deep learning model for the multi-class classification of tea leaf diseases. The project utilizes a modified version of the EfficientNet-B0 architecture, enhanced with **Efficient Channel Attention (ECA)** modules to improve performance while maintaining computational efficiency.

## 🌿 Project Overview
Agricultural productivity for tea crops is often threatened by various diseases and pests. This project provides an automated computer vision solution to identify and classify these issues from images of tea leaves. 

The core contribution is the **m-EfficientNetB0+ECA** model, which replaces standard Squeeze-and-Excitation (SE) blocks with ECA modules to capture cross-channel interactions more effectively without dimensionality reduction.

## 📊 Dataset: teaLeafBD
The model is trained and evaluated on the **teaLeafBD** dataset, comprising 5,276 images across 7 classes:

1.  **Tea algal leaf spot**
2.  **Brown Blight**
3.  **Gray Blight**
4.  **Helopeltis**
5.  **Red spider**
6.  **Green mirid bug**
7.  **Healthy leaf**

![Example of Tea Leaf Diseases](image_9ac348.jpg)

### Data Split
* **Training:** 3,798 images
* **Validation:** 423 images
* **Testing:** 1,055 images

## 🏗️ Architecture: m-EfficientNetB0+ECA
* **Base Model:** EfficientNet-B0
* **Enhancement:** Efficient Channel Attention (ECA)
* **Key Modification:** Stage 7 and Stage 8 were modified, and the Stage 9 Conv1x1 input channels were fixed to 192.
* **Head:** Adaptive Average Pooling followed by a Dropout layer and a Linear classifier for 7 classes.

![Model Architecture Diagram](image_9ac3fd.png)

## ⚙️ Requirements
* Python 3.x
* PyTorch
* Torchvision
* TorchInfo
* Matplotlib
* Scikit-learn

## 🚀 Usage

### 1. Training
Open `final.ipynb` and run the cells sequentially. The script will:
* Set up data loaders with augmentations (RandomResizedCrop, Flips, Rotation, ColorJitter).
* Initialize the modified EfficientNet-B0 model.
* Train for up to 100 epochs using the Adam optimizer and a `ReduceLROnPlateau` scheduler.
* Save the best performing weights to `m_efficientnet_b0_eca_best.pth`.

![Training Loss and Accuracy Curves](image_9ac361.png)

### 2. Evaluation
The notebook includes an evaluation section that loads the saved weights and computes:
* Test Accuracy
* Macro-averaged Precision, Recall, and F1-Score
* Confusion Matrix visualization

![Confusion Matrix](image_9ac385.png)

## 📈 Results
The model is designed to optimize the trade-off between parameter count and classification accuracy. By using ECA modules instead of SE blocks, the architecture achieves high sensitivity to disease features with lower computational overhead.

![Comparison of Performance Metrics](image_9ac3e3.png)
