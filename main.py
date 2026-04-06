import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. Setup Parameters ---
dataset_dir = 'teaLeafBD/teaLeafBD'  # Path to the main dataset directory
image_height = 228
image_width = 228
batch_size = 32
validation_split = 0.2 # Use 20% of data for validation
seed_value = 42        # For reproducibility

# --- 2. Load and Split Dataset ---
print("Loading training dataset...")
train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=validation_split,
    subset="training",
    seed=seed_value,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    label_mode='categorical' # Use categorical for multi-class classification
)

print("Loading validation dataset...")
val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=validation_split,
    subset="validation",
    seed=seed_value,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    label_mode='categorical'
)

# --- Get Class Names ---
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nFound classes: {class_names}")
print(f"Number of classes: {num_classes}\n")

# --- 3. Configure Dataset for Performance ---
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. Basic Preprocessing & Augmentation Layers ---
# Rescaling layer (Normalization)
rescale_layer = layers.Rescaling(1./255)

# Simple augmentation layers (example - can customize based on paper)
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    # Add more augmentation as needed (e.g., RandomZoom, RandomContrast)
  ]
)

# --- 5. Build a Simple CNN Model ---
model = keras.Sequential([
    # Input layer expects images of shape (image_height, image_width, 3)
    layers.InputLayer(input_shape=(image_height, image_width, 3)),

    # Apply rescaling
    rescale_layer,

    # Apply augmentation (only during training)
    # data_augmentation, # Uncomment this if you add the layer

    # Convolutional Block 1
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Block 2
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Block 3
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5), # Regularization
    layers.Dense(num_classes, activation='softmax') # Output layer
])

model.summary()

# --- 6. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Use categorical_crossentropy for multi-class
    metrics=['accuracy']
)

# --- 7. Train the Model ---
epochs = 15 # Adjust as needed
print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
print("Training finished.")

# --- 8. Evaluate the Model (Optional: using the validation set here for simplicity) ---
print("\nEvaluating model...")
loss, accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# --- 9. Plot Training History (Optional) ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print("\nMaking predictions on a batch of validation data...")
image_batch, label_batch = next(iter(val_ds)) # Get one batch
predictions = model.predict(image_batch)

# Show predictions for the first few images in the batch
plt.figure(figsize=(10, 10))
for i in range(min(9, batch_size)): # Show up to 9 images
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    predicted_class_index = np.argmax(predictions[i])
    actual_class_index = np.argmax(label_batch[i])
    predicted_class_name = class_names[predicted_class_index]
    actual_class_name = class_names[actual_class_index]
    plt.title(f"Pred: {predicted_class_name}\nActual: {actual_class_name}")
    plt.axis("off")
plt.tight_layout()
plt.show()
