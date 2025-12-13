import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, exposure, morphology
from scipy import ndimage as ndi

# Step 0. Load input image
img = io.imread("Dataset Intensity/1.png")

# Convert to grayscale (original image is nearly grayscale, this ensures robustness)
if img.ndim == 3:
    img = color.rgb2gray(img)

img = img.astype(np.float32)
img /= img.max()

plt.imshow(img, cmap="gray")
plt.title("Step 0: Original image")
plt.axis("off")
plt.show()

# Step 1. Crop central region of interest (ROI)
h, w = img.shape
crop_ratio = 0.5   # Take central 50% region

h0 = int(h * (1 - crop_ratio) / 2)
h1 = int(h * (1 + crop_ratio) / 2)
w0 = int(w * (1 - crop_ratio) / 2)
w1 = int(w * (1 + crop_ratio) / 2)

img_crop = img[h0:h1, w0:w1]

plt.imshow(img_crop, cmap="gray")
plt.title("Step 1: Center cropped ROI")
plt.axis("off")
plt.show()

# Step 2. Median filtering (non-Gaussian, edge-preserving)
# Use median filtering to suppress noise while preserving thin boundaries
img_med = filters.median(img_crop, morphology.disk(2))

plt.imshow(img_med, cmap="gray")
plt.title("Step 2: Median filtered")
plt.axis("off")
plt.show()

# Step 3. Adaptive local thresholding
thresh = filters.threshold_local(
    img_med,
    block_size=31,
    offset=-0.01
)

binary_raw = img_med > thresh

plt.imshow(binary_raw, cmap="gray")
plt.title("Step 3: Raw binary (adaptive threshold)")
plt.axis("off")
plt.show()

# Step 4. Morphological smoothing (opening + closing)
binary_smooth = morphology.opening(binary_raw, morphology.disk(2))
binary_smooth = morphology.closing(binary_smooth, morphology.disk(2))
binary_smooth = ndi.binary_fill_holes(binary_smooth)

plt.imshow(binary_smooth, cmap="gray")
plt.title("Step 4: Smoothed binary")
plt.axis("off")
plt.show()

# Step 5. Keep only the central connected component
# Connected-component labeling
labels, num = ndi.label(binary_smooth)

# Identify the label at the image center
center_y = binary_smooth.shape[0] // 2
center_x = binary_smooth.shape[1] // 2
center_label = labels[center_y, center_x]

# If the center belongs to background, return an empty mask
if center_label == 0:
    binary_center = np.zeros_like(binary_smooth)
else:
    binary_center = labels == center_label

plt.imshow(binary_center, cmap="gray")
plt.title("Step 5: Central connected component only")
plt.axis("off")
plt.show()

# Step 6. Resize to 250×250 and final smoothing
from skimage.transform import resize

# Resize binary mask to a fixed resolution (250×250)
binary_resized = resize(
    binary_center.astype(np.float32),
    (250, 250),
    order=0,              # nearest-neighbor to preserve binary structure
    anti_aliasing=False,
    preserve_range=True
)

# Final light smoothing to remove interpolation artifacts
binary_resized = filters.gaussian(binary_resized, sigma=0.8) > 0.5
binary_resized = morphology.opening(binary_resized, morphology.disk(1))
binary_resized = ndi.binary_fill_holes(binary_resized)

plt.imshow(binary_resized, cmap="gray")
plt.title("Step 6: Resized (250*250) and smoothed binary")
plt.axis("off")
plt.show()