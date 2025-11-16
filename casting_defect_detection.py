# =====================================================
# CASTING DEFECT DETECTION - FAST CPU TRAINING PIPELINE
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

print("Checking available devices...")
print("Devices:", tf.config.list_physical_devices())

# =====================================================
# Dataset Configuration
# =====================================================
dataset_dir = "./casting_512x512"
batch_size = 16              # Faster on CPU
img_height = 224             # Speeds up MobileNetV2
img_width = 224
seed = 42

if not os.path.exists(dataset_dir):
    raise FileNotFoundError("Dataset not found!")

# =====================================================
# Load Dataset
# =====================================================
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

# Get class names BEFORE mapping
class_names = train_ds.class_names
print("Classes:", class_names)

# =====================================================
# Safe Preprocess (OUTSIDE model)
# =====================================================
preprocess_layer = layers.Lambda(
    tf.keras.applications.mobilenet_v2.preprocess_input,
    name="preprocess"
)

def preprocess(image, label):
    image = preprocess_layer(image)
    return image, label

# Now map & prefetch datasets
train_ds = train_ds.map(preprocess).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)

# =====================================================
# FAST MODEL — MobileNetV2 (10x faster than Xception)
# =====================================================
base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    input_shape=(img_height, img_width, 3),
    include_top=False
)
base_model.trainable = False  # Freeze backbone

inputs = keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# TRAINING
# =====================================================
epochs = 15

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)

# =====================================================
# SAVE SAFE MODEL
# =====================================================
model.save("mobilenetv2_casting.keras")
print("Model saved as 'mobilenetv2_casting.keras' successfully!")

# =====================================================
# EVALUATION & CONFUSION MATRIX
# =====================================================
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).numpy().astype(int).flatten())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.savefig("confusion_matrix.png")
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))
