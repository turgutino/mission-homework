# quick_start.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import random

# =========================
# Dataset Configuration
# =========================
dataset_dir = "./casting_512x512"
batch_size = 32
img_height = 224
img_width = 224


if not os.path.exists(dataset_dir):
    raise FileNotFoundError("Dataset not found! Please place 'casting_512x512' folder in the same directory.")


train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")

# =========================
# Performance Optimization
# =========================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =========================
# Build a Simple CNN Model
# =========================
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# Train the Model
# =========================
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# =========================
# Save the Model
# =========================
model.save("quick_start_model.h5")
print("Model saved as 'quick_start_model.h5'!")

# =========================
# Visualize Training Results
# =========================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')
plt.savefig("quick_start_training_history.png")
plt.show()

print("Training complete! Results and model have been saved.")


for images, labels in val_ds.take(1):
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 10))
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        img = images[i].numpy().astype("uint8")

        true_label = int(labels[i])
        pred_label = int(predicted_classes[i])

        color = "green" if true_label == pred_label else "red"
        plt.imshow(img)
        plt.title(f"True: {true_label}, Pred: {pred_label}", color=color, fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("quick_start_predictions.png")
    plt.show()

print("Sample predictions visualized and saved as 'quick_start_predictions.png'!")
