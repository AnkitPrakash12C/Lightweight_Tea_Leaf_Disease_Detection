Tea Leaf Disease Classification using m-EfficientNetB0+ECA
This project implements a deep learning model for the multi-class classification of tea leaf diseases. It utilizes a modified version of the EfficientNet-B0 architecture, replacing standard Squeeze-and-Excitation (SE) blocks with Efficient Channel Attention (ECA) modules to improve feature extraction efficiency and model performance.

📊 Dataset: teaLeafBD
The model is trained on the teaLeafBD dataset, which contains 5,276 images categorized into 7 distinct classes:

Tea algal leaf spot

Brown Blight

Gray Blight

Helopeltis

Red spider

Green mirid bug

Healthy leaf

Data Distribution
Total images: 5,276

Training set: 3,798 images

Validation set: 423 images

Testing set: 1,055 images

🏗️ Model Architecture: m-EfficientNetB0+ECA
The core of the project is a customized EfficientNet-B0. Key modifications include:

ECA Integration: All Squeeze-and-Excitation (SE) modules in the original architecture are replaced with Efficient Channel Attention (ECA) modules.

Structural Modifications: The architecture's Stage 7 and Stage 8 are modified, and Stage 9 (Conv1x1) input channels are fixed to 192.

Global Average Pooling (GAP): Used within the ECA module for dimensionality reduction before channel-wise weight computation.

Output Layer: A final linear layer adjusted to output predictions for the 7 disease classes.

⚙️ Training Configuration
Optimizer: Adam

Loss Function: CrossEntropyLoss

Learning Rate: 0.001

Scheduler: ReduceLROnPlateau (factor=0.5, patience=15) to dynamically adjust the learning rate based on validation loss.

Batch Size: 16

Epochs: Up to 100 (with early saving of the best model).

Data Augmentation: includes Random Resized Crop, Horizontal/Vertical Flips, Rotation, and Color Jitter to improve generalization.

🚀 Getting Started
Prerequisites
Python 3.x

PyTorch / Torchvision

Scikit-learn

Numpy / Matplotlib / Seaborn

Setup & Usage
Dataset Path: Ensure the teaLeafBD dataset is available locally. Update the DATA_DIR variable in the notebook to point to your dataset location.

Training: Run the training cells in final.ipynb. The script will automatically detect if a CUDA-enabled GPU is available.

Best Model: The training loop saves the weights of the model with the lowest validation loss as m_efficientnet_b0_eca_best.pth.

Evaluation: The final cell loads the best model and reports accuracy, precision, recall, and F1-score on the held-out test set.

📈 Results
The final model performance is evaluated using macro-averaged metrics to ensure balanced performance across all 7 disease categories. Test results typically include:

Test Loss

Test Accuracy

Macro Precision, Recall, and F1-Score
