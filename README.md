Gemini said
A Lightweight Deep Learning Model for Tea Leaf Disease Detection
This project introduces m-EfficientNetB0-ECA, a streamlined convolutional neural network optimized for real-time, offline disease diagnosis on resource-constrained Edge AI hardware such as smartphones and drones. By redesigning the EfficientNetB0 backbone, the model achieves high diagnostic precision while maintaining a compact storage footprint.

Project Overview
Tea productivity is frequently threatened by fungal diseases and pests like Gray Blight, Brown Blight, and Red Spider mites. While deep learning offers automated solutions, traditional "heavyweight" models are often unsuitable for field deployment. This project bridges that gap with an architecture that balances a 91.94% accuracy rate with a model size of just 4.27 MB.

Key Innovations
1. Structural Pruning
The architecture eliminates redundant deep layers identified as unnecessary for leaf-texture analysis:

Stage 7: Reduced from 4 blocks to a single block.

Stage 8: Completely removed to prevent overfitting and reduce parameter count.

2. Efficient Channel Attention (ECA)
Standard Squeeze-and-Excitation (SE) blocks are replaced with ECA modules. ECA captures local cross-channel interactions using a 1D convolution without dimensionality reduction, preserving vital spatial features while lowering computational weight.

3. Manual Channel Rewiring
To ensure information continuity post-pruning, Stage 9 was manually rewired to accept 192 input channels instead of the original 320. This optimization ensures a continuous data stream directly into the classification head.

Technical Specifications
Framework: PyTorch 2.4.1

Backbone: Modified EfficientNetB0

Input Size: 256 x 256 pixels

Optimizer: Adam (Initial LR: 0.001)

Scheduler: ReduceLROnPlateau (Patience: 15 epochs, Factor: 0.5)

Loss Function: Cross-Entropy Loss

Dataset: teaLeafBD (5,276 images across 7 classes)

Performance Metrics
Metric	Value
Accuracy	91.94%
Macro F1-Score	90.51%
Model Size	4.27 MB
Inference Speed	~20 FPS (0.049s per image)
Parameters	~1.18 Million
Installation & Setup
Requirements
Python 3.10+

PyTorch

Torchvision

Scikit-learn

NumPy

Matplotlib

Setup
Clone the repository and install the dependencies.

Ensure the teaLeafBD dataset is structured using ImageFolder format (Train/Test subdirectories).

Place the pre-trained weights (m_efficientnet_b0_eca_best.pth) in the root directory.

Usage
Training the Model
The model is trained for 100 epochs with dynamic learning rate adjustments. Data augmentation includes 20° rotations, horizontal/vertical flips, and color jittering to simulate field-level variability.

Running Inference
Python
import torch
from model_script import create_m_efficientnet_b0_eca

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_m_efficientnet_b0_eca(num_classes=7).to(device)
model.load_state_dict(torch.load("m_efficientnet_b0_eca_best.pth"))
model.eval()

# Run prediction on a 256x256 preprocessed image
# output = model(input_tensor)
