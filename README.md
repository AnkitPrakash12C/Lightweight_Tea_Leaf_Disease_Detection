# Tea Leaf Disease Classification using m-EfficientNetB0+ECA

This repository contains the implementation of a deep learning model for the multi-class classification of tea leaf diseases. The project utilizes a modified version of the EfficientNet-B0 architecture, enhanced with **Efficient Channel Attention (ECA)** modules to improve performance while maintaining computational efficiency.

## 🌿 Project Overview
Agricultural productivity for tea crops is often threatened by various diseases and pests. This project provides an automated computer vision solution to identify and classify these issues from images of tea leaves. 

The core contribution is the **m-EfficientNetB0+ECA** model, which replaces standard Squeeze-and-Excitation (SE) blocks with ECA modules to capture cross-channel interactions more effectively without dimensionality reduction.

## 📊 Dataset: teaLeafBD
The model is trained and evaluated on the **teaLeafBD** dataset, comprising 5,276 images across 7 classes.

<p align="center">
  <img src="image_9ac348.jpg" alt="Dataset Sample Grid" width="600px">
  <br>
  <em>Dataset Sample Grid: Visual characteristics of the 7 tea leaf classes.</em>
</p>

### Data Split
* **Training:** 3,798 images
* **Validation:** 423 images
* **Testing:** 1,055 images

<p align="center">
  <img src="Picture2.png" alt="Dataset Distribution Histogram" width="500px">
  <br>
  <em>Dataset Distribution Histogram: Class balance across the dataset.</em>
</p>

## 🏗️ Architecture: m-EfficientNetB0+ECA
* **Base Model:** EfficientNet-B0
* **Enhancement:** Efficient Channel Attention (ECA)
* **Key Modification:** Stage 7 and Stage 8 were modified, and the Stage 9 Conv1x1 input channels were fixed to 192.

<p align="center">
  <img src="Picture7.png" alt="Methodology Flow Chart" width="700px">
  <br>
  <em>Methodology Flow Chart: Structural diagram of the m-EfficientNetB0+ECA architecture.</em>
</p>

## 🚀 Usage

### 1. Training
The training loop utilizes the Adam optimizer and a `ReduceLROnPlateau` scheduler over 100 epochs.

<p align="center">
  <img src="Picture4.png" alt="Training Progress Curves" width="600px">
  <br>
  <em>Training Progress Curves: Loss and Accuracy metrics over 100 epochs.</em>
</p>

### 2. Evaluation
The model is evaluated using a confusion matrix to analyze misclassification patterns between similar disease types.

<p align="center">
  <img src="Picture8.png" alt="Confusion Matrix" width="500px">
  <br>
  <em>Confusion Matrix: Heatmap of predicted vs. true labels on the test set.</em>
</p>

## 📈 Results
The final performance is summarized via precision, recall, and F1-score metrics for each category.

<p align="center">
  <img src="Picture10.png" alt="Performance Summary Bar Chart" width="600px">
  <br>
  <em>Performance Summary: Classification metrics across all disease categories.</em>
</p>
