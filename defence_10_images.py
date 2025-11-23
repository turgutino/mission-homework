import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = "mobilenetv2_casting.keras"
ATTACK_DIR = "./attack_10_results"
IMG_SIZE = (224, 224)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded model.")

# Output folders
os.makedirs("defense_10_results/blurred", exist_ok=True)

# Collect all adversarial images
adv_images = sorted([f for f in os.listdir(ATTACK_DIR) if f.endswith(".png")])

print(f"Found {len(adv_images)} adversarial images.")

for idx, img_name in enumerate(adv_images, start=1):

    img_path = os.path.join(ATTACK_DIR, img_name)

    # Load adversarial image
    adv = cv2.imread(img_path)
    adv = cv2.resize(adv, IMG_SIZE)
    adv = adv.astype("float32") / 255.0

    # Model prediction before defense
    adv_pred = model.predict(np.expand_dims(adv, 0))[0][0]

    # ---- Smoothing Defense ----
    adv_uint8 = (adv * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(adv_uint8, (5, 5), 0)
    blurred_norm = blurred.astype("float32") / 255.0

    # Prediction after defense
    def_pred = model.predict(np.expand_dims(blurred_norm, 0))[0][0]

    # Save blurred defense image
    cv2.imwrite(f"defense_10_results/blurred/def_{idx}.png", blurred)

    # Log in text file
    with open(f"defense_10_results/def_{idx}.txt", "w") as f:
        f.write(f"Adversarial Prediction: {adv_pred}\n")
        f.write(f"Defense Prediction: {def_pred}\n")

    print(f"[{idx}/10] Defense applied to: {img_name}")

print("\nDefense on 10 images completed successfully!")
