import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "mobilenetv2_casting.keras"
DATASET_DIR = "./casting_512x512"
IMG_SIZE = (224, 224)
EPSILON = 0.05
BATCH = 1

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded:", MODEL_PATH)

# -----------------------------
# Load validation dataset
# -----------------------------
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH
)

class_names = val_ds.class_names
print("Classes:", class_names)

# Take 1 image
for images, labels in val_ds.take(1):
    original_images = images
    original_labels = labels
    break

# Preprocess for MobileNetV2
image_for_attack = preprocess_input(
    tf.cast(original_images, tf.float32)
)

# -----------------------------
# FGSM Attack
# -----------------------------
def fgsm_attack(model, image, label, epsilon):
    with tf.GradientTape() as tape:
        tape.watch(image)

        label = tf.reshape(label, (-1, 1))   # fix shape mismatch

        pred = model(image)
        loss = tf.keras.losses.binary_crossentropy(label, pred)

    gradient = tape.gradient(loss, image)
    noise = epsilon * tf.sign(gradient)
    adv_image = image + noise
    return adv_image, noise


adv_img_preprocessed, noise = fgsm_attack(
    model, image_for_attack, original_labels, EPSILON
)

# -----------------------------
# Reverse preprocess so image is visible
# -----------------------------
def reverse_preprocess(x):
    x = (x + 1) * 127.5
    return np.clip(x, 0, 255).astype(np.uint8)

adv_img_vis = reverse_preprocess(adv_img_preprocessed.numpy()[0])
orig_img_vis = original_images[0].numpy().astype("uint8")

# -----------------------------
# Predictions
# -----------------------------
orig_pred = model.predict(image_for_attack)
adv_pred = model.predict(adv_img_preprocessed)

orig_class = int(orig_pred > 0.5)
adv_class = int(adv_pred > 0.5)

orig_name = class_names[orig_class]
adv_name = class_names[adv_class]

attack_status = "SUCCESS" if orig_class != adv_class else "FAILED"

# -----------------------------
# Save results
# -----------------------------
os.makedirs("attack_results", exist_ok=True)

plt.figure(figsize=(12, 6))
plt.suptitle(
    f"Epsilon: {EPSILON} | Original: {orig_name} | "
    f"Adversarial: {adv_name} | Attack: {attack_status}",
    fontsize=12, fontweight="bold"
)

plt.subplot(1, 2, 1)
plt.imshow(orig_img_vis)
plt.title(f"Original ({orig_name})")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(adv_img_vis)
plt.title(f"Adversarial ({adv_name})")
plt.axis("off")

plt.savefig("attack_results/attack_output.png", dpi=130)
plt.show()

with open("attack_results/attack_result.txt", "w") as f:
    f.write(f"Epsilon: {EPSILON}\n")
    f.write(f"Original: {orig_name}\n")
    f.write(f"Adversarial: {adv_name}\n")
    f.write(f"Attack: {attack_status}\n")

print("Results saved to attack_results/")
