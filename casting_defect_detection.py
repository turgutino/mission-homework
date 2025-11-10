# =====================================================
# CASTING DEFECT DETECTION - CPU TRAINING PIPELINE
# =====================================================
# Author: Turqut Sofuyev
# =====================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# =====================================================
# Check Device (CPU or GPU)
# =====================================================
print("Checking available devices...")
devices = tf.config.list_physical_devices()
print("All Devices:", devices)
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

if not tf.config.list_physical_devices('GPU'):
    print("⚙️ No GPU detected — running on CPU mode only.")
else:
    print("GPU is available!")

# =====================================================
# Dataset Configuration
# =====================================================
dataset_dir = "./casting_512x512"
batch_size = 32
img_height = 299
img_width = 299
seed = 42

if not os.path.exists(dataset_dir):
    raise FileNotFoundError("Dataset not found! Please make sure './casting_512x512/' exists in project directory.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")

# =====================================================
# Data Pipeline Optimization & Augmentation
# =====================================================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=seed),
    layers.RandomRotation(0.1, seed=seed),
    layers.RandomZoom(0.1, seed=seed),
    layers.RandomContrast(0.2, seed=seed)
])

# =====================================================
# Transfer Learning Model - Xception
# =====================================================
base_model = keras.applications.Xception(
    weights="imagenet",
    input_shape=(img_height, img_width, 3),
    include_top=False
)
base_model.trainable = False

inputs = keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = keras.applications.xception.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

# =====================================================
# Model Compilation
# =====================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =====================================================
# Training Configuration & Callbacks
# =====================================================
epochs = 20

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# =====================================================
# Train the Model
# =====================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr]
)

# =====================================================
# Save Model
# =====================================================
model.save("xception_transfer_model_cpu.h5")
print("Model saved as 'xception_transfer_model_cpu.h5'!")

# =====================================================
# Training Visualization
# =====================================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
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

plt.tight_layout()
plt.savefig("training_curves_cpu.png")
plt.show()

# =====================================================
# Evaluation & Confusion Matrix
# =====================================================
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype(int).flatten())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_cpu.png")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# =====================================================
# Final Summary
# =====================================================
print("\nTraining complete (CPU mode).")
print("Results saved:")
print(" - xception_transfer_model_cpu.h5")
print(" - training_curves_cpu.png")
print(" - confusion_matrix_cpu.png")
