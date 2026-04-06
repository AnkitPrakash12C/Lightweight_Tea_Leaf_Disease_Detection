🌿 Project Overview
Agricultural productivity for tea crops is often threatened by various diseases and pests. This project provides an automated computer vision solution to identify and classify these issues from images of tea leaves.

The core contribution is the m-EfficientNetB0+ECA model, which replaces standard Squeeze-and-Excitation (SE) blocks with ECA modules to capture cross-channel interactions more effectively without dimensionality reduction.

📊 Dataset: teaLeafBD
The model is trained and evaluated on the teaLeafBD dataset, comprising 5,276 images across 7 classes:

Tea algal leaf spot

Brown Blight

Gray Blight

Helopeltis

Red spider

Green mirid bug

Healthy leaf

Data Split
Training: 3,798 images

Validation: 423 images

Testing: 1,055 images

🏗️ Architecture: m-EfficientNetB0+ECA
Base Model: EfficientNet-B0

Enhancement: Efficient Channel Attention (ECA)

Key Modification: Stage 7 and Stage 8 were modified, and the Stage 9 Conv1x1 input channels were fixed to 192.

Head: Adaptive Average Pooling followed by a Dropout layer and a Linear classifier for 7 classes.

⚙️ Requirements
Python 3.x

PyTorch

Torchvision

TorchInfo (for model summaries)

Matplotlib

Scikit-learn

🚀 Usage
1. Training
Open final.ipynb and run the cells sequentially. The script will:

Set up data loaders with augmentations (RandomResizedCrop, Flips, Rotation, ColorJitter).

Initialize the modified EfficientNet-B0 model.

Train for up to 100 epochs using the Adam optimizer and a ReduceLROnPlateau scheduler.

Save the best performing weights to m_efficientnet_b0_eca_best.pth.

2. Evaluation
The notebook includes an evaluation section that loads the saved weights and computes:

Test Accuracy

Macro-averaged Precision, Recall, and F1-Score

Confusion Matrix visualization

📈 Results
The model is designed to optimize the trade-off between parameter count and classification accuracy. By using ECA modules instead of SE blocks, the architecture achieves high sensitivity to disease features with lower computational overhead.
