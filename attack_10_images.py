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

# Prepare output folder
os.makedirs("attack_10_results", exist_ok=True)

# -----------------------------
# FGSM Attack Function
# -----------------------------
def fgsm_attack(model, image, label, epsilon):
    with tf.GradientTape() as tape:
        tape.watch(image)

        label = tf.reshape(label, (-1, 1))  # binary label must be shape (1,1)

        pred = model(image)
        loss = tf.keras.losses.binary_crossentropy(label, pred)

    gradient = tape.gradient(loss, image)
    noise = epsilon * tf.sign(gradient)
    adv_image = image + noise
    return adv_image, noise


# Reverse preprocess for visualization
def reverse_preprocess(x):
    x = (x + 1) * 127.5
    return np.clip(x, 0, 255).astype(np.uint8)


# -----------------------------
# Process first 10 images
# -----------------------------
counter = 0
for images, labels in val_ds:
    for i in range(len(images)):
        if counter >= 10:
            break

        original_img = images[i:i+1]
        original_label = labels[i:i+1]

        # Preprocess for MobileNetV2
        image_for_attack = preprocess_input(
            tf.cast(original_img, tf.float32)
        )

        # ---- Attack ----
        adv_img_preprocessed, noise = fgsm_attack(
            model, image_for_attack, original_label, EPSILON
        )

        # ---- Reverse preprocess (for display) ----
        orig_img_vis = original_img[0].numpy().astype("uint8")
        adv_img_vis = reverse_preprocess(adv_img_preprocessed.numpy()[0])

        # ---- Predictions ----
        orig_pred = model.predict(image_for_attack)
        adv_pred = model.predict(adv_img_preprocessed)

        orig_class = int(orig_pred > 0.5)
        adv_class = int(adv_pred > 0.5)

        orig_name = class_names[orig_class]
        adv_name = class_names[adv_class]

        attack_status = "SUCCESS" if orig_class != adv_class else "FAILED"

        # ---- Save image result ----
        plt.figure(figsize=(12, 6))
        plt.suptitle(
            f"IMG {counter+1} | Epsilon: {EPSILON} | "
            f"Original: {orig_name} | Adversarial: {adv_name} | Attack: {attack_status}",
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

        plt.savefig(f"attack_10_results/attack_{counter+1}.png", dpi=120)
        plt.close()

        # Save txt log
        with open(f"attack_10_results/attack_{counter+1}.txt", "w") as f:
            f.write(f"Epsilon: {EPSILON}\n")
            f.write(f"Original: {orig_name}\n")
            f.write(f"Adversarial: {adv_name}\n")
            f.write(f"Attack: {attack_status}\n")

        print(f"[{counter+1}/10] Saved attack result for image {counter+1}")

        counter += 1

    if counter >= 10:
        break

print("\nAttack on first 10 images completed successfully!")
