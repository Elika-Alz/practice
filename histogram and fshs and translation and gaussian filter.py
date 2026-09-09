# -*- coding: utf-8 -*-
"""
Image and Video Processing for Autonomous Driving - TRACK A
Exercise 2 - Histogram Stretching, Translation and Gaussian Filtering
"""

import os
import numpy as np
import imageio.v3 as iio
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
from skimage import color

# =====================================================
# Paths
# =====================================================

root = r"C:\Users\Elika\Desktop\IVPAD"
input_root = os.path.join(root, "Images")
output_root = os.path.join(root, "123456_Alizadeh_Elika_Outputs")

if not os.path.exists(output_root):
    os.makedirs(output_root)

# =====================================================
# 2a - load and compute the histogram of the original
# =====================================================

img = iio.imread(os.path.join(input_root, "pout.tif"))
print("original:", img.shape, img.dtype, img.min(), img.max())

# np.arange(257) gives 257 BIN EDGES -> 256 bins, one per grey level.
# The second returned value (the edges) is not needed, so it is
# caught in '_'.
n_org, _ = np.histogram(img, np.arange(257))

print("histogram length:", len(n_org), "| total pixels:", n_org.sum())

# =====================================================
# 2a - Full-Scale Histogram Stretching
# =====================================================

# FSHS is the linear point operation y = a*x + b chosen so that
# min(y) = 0 and max(y) = K-1 = 255. Solving the two conditions gives
#
#   a = (K-1) / (max(x) - min(x))
#   b = -(K-1)*min(x) / (max(x) - min(x))
#
# which factorises into the two-step form used below:
#   step 1: (x - min) / (max - min)   -> normalises to [0, 1]
#   step 2: * 255                     -> expands to the full range
#
# The image is converted to float32 first: the formula contains a
# division, and integer arithmetic in uint8 would truncate it and
# could also wrap around.

img_fshs = img.astype(np.float32)
img_fshs = (img_fshs - img_fshs.min()) / (img_fshs.max() - img_fshs.min()) * 255.0
img_fshs = np.clip(img_fshs, 0, 255)
img_fshs = img_fshs.astype(np.uint8)

n_fshs, _ = np.histogram(img_fshs, np.arange(257))

print("stretched:", img_fshs.dtype, img_fshs.min(), img_fshs.max())

# ---- single figure: 2 images on top, 2 histograms below ----

plt.figure()

ax1 = plt.subplot(2, 2, 1)
ax1.imshow(img, cmap="gray", clim=[0, 255])
ax1.set_title("Original image")

ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_fshs, cmap="gray", clim=[0, 255])
ax2.set_title("Stretched image (FSHS)")

# the histogram panels do NOT share axes with the image panels:
# their axes mean grey level and pixel count, not rows and columns.
ax3 = plt.subplot(2, 2, 3)
ax3.bar(np.arange(256), n_org)
ax3.set_title("Original histogram")

ax4 = plt.subplot(2, 2, 4)
ax4.bar(np.arange(256), n_fshs)
ax4.set_title("Stretched histogram")

plt.savefig(os.path.join(output_root, "ExA_Ex2_fig1_fshs.png"))

# =====================================================
# 2b - translation of +60 rows and -100 columns
#      applied to the STRETCHED image
# =====================================================

# ndi.shift takes the displacement as (rows, columns).
# m increases downward and n increases rightward, so
#   +60 rows     -> the image moves DOWN by 60 pixels
#   -100 columns -> the image moves LEFT by 100 pixels
# The uncovered border is filled with the default constant value 0.

img_translated = ndi.shift(img_fshs, (60, -100))

print("shapes:", img_fshs.shape, img_translated.shape)

plt.figure()

ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img_fshs, cmap="gray", clim=[0, 255])
ax1.set_title("Stretched image")

ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_translated, cmap="gray", clim=[0, 255])
ax2.set_title("Translated (+60 rows, -100 cols)")

plt.savefig(os.path.join(output_root, "ExA_Ex2_fig2_translation.png"))

# =====================================================
# 2c - Gaussian filter with sigma = 3
#      applied to the TRANSLATED image
# =====================================================

# The second argument of ndi.gaussian_filter is sigma, the standard
# deviation of the Gaussian, NOT a window size. The kernel size is
# derived from sigma automatically.

sigma = 3
img_gauss = ndi.gaussian_filter(img_translated, sigma)

print("filtered:", img_gauss.dtype, img_gauss.min(), img_gauss.max())

plt.figure()

ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img_translated, cmap="gray", clim=[0, 255])
ax1.set_title("Input (translated)")

ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_gauss, cmap="gray", clim=[0, 255])
ax2.set_title("Gaussian filter, sigma = 3")

plt.savefig(os.path.join(output_root, "ExA_Ex2_fig3_gaussian.png"))

# -----------------------------------------------------
# COMMENT (2c): how does a Gaussian filter differ from a
# box filter in terms of frequency response, and why does
# it produce smoother results?
# -----------------------------------------------------
#
# FREQUENCY RESPONSE.
# Both are low-pass filters: they attenuate the high-frequency
# components of the image, which correspond to rapid intensity
# variations such as fine detail, sharp edges and noise, while
# passing the low-frequency components, which correspond to slow,
# gradual variations.
#
# The difference lies in HOW they attenuate. The box filter has an
# abrupt cut-off in the spatial domain: a pixel inside the window
# counts with full weight, and a pixel one step outside counts zero.
# An abrupt cut-off in one domain produces oscillations in the other,
# so the frequency response of the box filter does not decrease
# monotonically. It has side-lobes: it suppresses certain spatial
# frequencies completely, while allowing neighbouring frequencies to
# leak through, sometimes with inverted sign. Those leaked components
# are exactly what produces the blocking and unnatural artefacts
# visible in a box-filtered image.
#
# The Gaussian has no abrupt cut-off. Its weights decay smoothly and
# never stop abruptly, and its Fourier transform is itself a Gaussian.
# Its frequency response is therefore a smooth bell that decreases
# monotonically, with no side-lobes and no ripples: high frequencies
# are attenuated progressively rather than being chopped off, so
# nothing leaks back through.
#
# WHY THE RESULT IS SMOOTHER.
# Two reasons, and they are independent.
#
# First, the weights decrease with distance from the centre, so a
# pixel far from the centre contributes less than a nearby one. The
# transition between the filtered and unfiltered content is gradual
# rather than a hard cut at the window boundary. The box filter, by
# contrast, gives a pixel at the very edge of the window the same
# weight as the pixel next door.
#
# Second, the Gaussian is isotropic: its weight depends only on the
# squared distance i^2 + j^2 from the centre, and not at all on
# direction, so it behaves identically for horizontal, vertical and
# diagonal features. The box filter is square and therefore reaches
# further along its diagonals than along its axes, so it smooths some
# orientations more than others. That directional bias is a further
# source of unnatural appearance.

# =====================================================
# Save all outputs
# =====================================================

iio.imwrite(os.path.join(output_root, "ExA_Ex2_fshs.jpg"), img_fshs)
iio.imwrite(os.path.join(output_root, "ExA_Ex2_translated.jpg"), img_translated)
iio.imwrite(os.path.join(output_root, "ExA_Ex2_gaussian.jpg"), img_gauss)

# the histograms are matplotlib figures, not image arrays,
# so they are saved with plt.savefig and not with iio.imwrite
plt.figure()
plt.bar(np.arange(256), n_org)
plt.title("Original histogram")
plt.savefig(os.path.join(output_root, "ExA_Ex2_hist_original.png"))

plt.figure()
plt.bar(np.arange(256), n_fshs)
plt.title("Stretched histogram")
plt.savefig(os.path.join(output_root, "ExA_Ex2_hist_stretched.png"))

plt.show()
