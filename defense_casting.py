import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("mobilenetv2_casting.keras")

# -------------------------------------------------------------
# Load the original clean image
# (Replace this with any image from def_front folder)
# -------------------------------------------------------------
img_path = "casting_512x512/def_front/cast_def_0_0.jpeg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_input = np.expand_dims(img, axis=0)

# Predict on the original clean image
orig_pred = model.predict(img_input)
print("Original Prediction:", orig_pred)

# -------------------------------------------------------------
# Load the adversarial image generated from FGSM attack
# -------------------------------------------------------------
adv_path = "attack_results/attack_output.png"  # your FGSM output
adv = cv2.imread(adv_path)
adv = cv2.resize(adv, (224, 224))
adv = adv.astype("float32") / 255.0
adv = np.expand_dims(adv, axis=0)

# Predict on the adversarial image
adv_pred = model.predict(adv)
print("Adversarial Prediction:", adv_pred)

# -------------------------------------------------------------
# Apply SMOOTHING DEFENSE (Gaussian Blur)
# -------------------------------------------------------------

adv_np = adv[0]
adv_np = (adv_np * 255).astype(np.uint8)

# Gaussian smoothing (reduces adversarial noise)
blurred = cv2.GaussianBlur(adv_np, (5, 5), 0)

# Convert back to float32 and restore batch dimension
blurred = blurred.astype("float32") / 255.0
blurred = np.expand_dims(blurred, axis=0)

# Prediction after smoothing defense
def_pred = model.predict(blurred)
print("Defense Prediction (Smoothing):", def_pred)

# -------------------------------------------------------------
# Save blurred image for visualization
# -------------------------------------------------------------
cv2.imwrite("attack_results/defense_blurred.png", blurred[0] * 255)

print("\nDefense completed successfully. Blurred image saved to attack_results/defense_blurred.png")
