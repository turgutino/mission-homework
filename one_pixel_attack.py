import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import os
import shutil

# Constants
MODEL_PATH = "mobilenetv2_casting.keras"
DATASET_DIR = "./casting_512x512"
IMG_SIZE = (224, 224)

# --- Load Model ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Prepare Dataset ---
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=1
)

# ============================
# 1) FIND BOUNDARY IMAGES
# ============================

boundary_images = []
index = 0

for images, labels in val_ds:
    img = images[0].numpy()
    x = preprocess_input(np.expand_dims(img, axis=0))
    pred = float(model.predict(x, verbose=0)[0][0])

    if 0.30 <= pred <= 0.70:
        boundary_images.append((index, img, int(labels[0]), pred))

    index += 1

if len(boundary_images) == 0:
    print("No boundary images found. Check dataset or model behavior.")
    exit()

print("\nBoundary images:")
for idx, _, lbl, p in boundary_images:
    print(f"Index: {idx}, Label: {lbl}, Prediction: {p}")

# --- Choose the first boundary image ---
chosen_index, orig, orig_label, orig_pred = boundary_images[0]

print(f"\nChosen image index: {chosen_index}")
print(f"Original Prediction: {orig_pred}\n")

# ============================
# 2) ONE-PIXEL ATTACK
# ============================

def predict(img):
    x = preprocess_input(np.expand_dims(img, axis=0))
    return float(model.predict(x, verbose=0)[0][0])

def fitness(params):
    img = orig.copy()
    
    x, y, r, g, b = params
    x = int(np.clip(x, 0, IMG_SIZE[1]-1))
    y = int(np.clip(y, 0, IMG_SIZE[0]-1))

    img[y, x, 0] = r
    img[y, x, 1] = g
    img[y, x, 2] = b

    pred = predict(img)

    target = 1.0 if orig_pred < 0.5 else 0.0
    return abs(pred - target)

bounds = [
    (0, IMG_SIZE[1] - 1),
    (0, IMG_SIZE[0] - 1),
    (0, 255),
    (0, 255),
    (0, 255)
]

print("Starting differential evolution...")
result = differential_evolution(
    fitness,
    bounds,
    maxiter=50,
    popsize=15,
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=True,
    workers=1,
    disp=True
)

print(f"Optimization completed. Success: {result.success}")

x, y, r, g, b = result.x
x, y = int(x), int(y)

print(f"Modified pixel: ({x}, {y})")
print(f"New RGB values: ({r:.1f}, {g:.1f}, {b:.1f})")

adv = orig.copy()
adv[y, x, 0] = r
adv[y, x, 1] = g
adv[y, x, 2] = b

adv_pred = predict(adv)

print(f"Original prediction: {orig_pred:.4f}")
print(f"Adversarial prediction: {adv_pred:.4f}")

# ============================
# 3) SAVE RESULTS
# ============================

DIR = "one_pixel_success_result"
if os.path.exists(DIR):
    shutil.rmtree(DIR)
os.makedirs(DIR)

with open(f"{DIR}/result.txt", "w") as f:
    f.write("ONE-PIXEL ATTACK RESULTS\n")
    f.write("========================\n")
    f.write(f"Image Index: {chosen_index}\n")
    f.write(f"Original Label: {orig_label}\n")
    f.write(f"Original Prediction: {orig_pred:.4f}\n")
    f.write(f"Adversarial Prediction: {adv_pred:.4f}\n\n")
    f.write(f"Modified Pixel: ({x}, {y})\n")
    f.write(f"RGB Values: ({r:.1f}, {g:.1f}, {b:.1f})\n")
    f.write(f"Attack Successful: {abs(orig_pred - adv_pred) > 0.2}\n")

# ============================
# IMAGE VISUALIZATION
# ============================

def save_clear_images():
    # Original Image (convert from [-1,1] to [0,255] if needed)
    orig_vis = ((orig + 1.0) * 127.5).astype(np.uint8)

    # Adversarial Image
    adv_vis = orig_vis.copy()
    adv_vis[y, x, 0] = np.clip(r, 0, 255)
    adv_vis[y, x, 1] = np.clip(g, 0, 255)
    adv_vis[y, x, 2] = np.clip(b, 0, 255)

    # Difference Mask
    diff = np.zeros_like(orig_vis)
    diff[y, x] = [255, 0, 0]

    # Zoomed Section
    zoom_size = 20
    y1, y2 = max(0, y-zoom_size), min(IMG_SIZE[0], y+zoom_size)
    x1, x2 = max(0, x-zoom_size), min(IMG_SIZE[1], x+zoom_size)

    orig_zoom = orig_vis[y1:y2, x1:x2]
    adv_zoom = adv_vis[y1:y2, x1:x2]

    # Plot Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0,0].imshow(orig_vis)
    axes[0,0].set_title(f"Original\nPrediction: {orig_pred:.4f}")
    axes[0,0].axis('off')

    axes[0,1].imshow(adv_vis)
    axes[0,1].set_title(f"Adversarial (1 Pixel)\nPrediction: {adv_pred:.4f}")
    axes[0,1].axis('off')

    axes[0,2].imshow(orig_vis)
    axes[0,2].imshow(diff, alpha=0.5)
    axes[0,2].set_title(f"Modified Pixel at ({x}, {y})")
    axes[0,2].axis('off')

    axes[1,0].imshow(orig_zoom)
    axes[1,0].set_title("Original (Zoom)")
    axes[1,0].axis('off')

    axes[1,1].imshow(adv_zoom)
    axes[1,1].set_title("Adversarial (Zoom)")
    axes[1,1].axis('off')

    diff_zoom = np.abs(orig_zoom.astype(float) - adv_zoom.astype(float))
    axes[1,2].imshow(diff_zoom)
    axes[1,2].set_title("Difference (Zoom)")
    axes[1,2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{DIR}/detailed_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(adv_vis)
    plt.title(f"Adversarial Image (Pixel {x},{y})")
    plt.axis('off')
    plt.savefig(f"{DIR}/adv_clear.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("All images saved successfully.")

save_clear_images()

print("\nONE-PIXEL ATTACK COMPLETED")
print(f"Prediction changed: {orig_pred:.4f} → {adv_pred:.4f}")
print(f"Difference: {abs(orig_pred - adv_pred):.4f}")
print(f"Success: {abs(orig_pred - adv_pred) > 0.2}")
