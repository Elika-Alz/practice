# -*- coding: utf-8 -*-
"""TRACK B - Exercise 3 - RGB Channels, Grayscale and Histogram"""

import os
import numpy as np
import imageio.v3 as iio
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
from skimage import color

# ---------------- paths ----------------
root = r"C:\Users\Elika\Desktop\IVPAD"
input_root = os.path.join(root, "Images")
output_root = os.path.join(root, "123456_Alizadeh_Elika_Outputs")
if not os.path.exists(output_root):
    os.makedirs(output_root)

# ---------------- load ----------------
img = iio.imread(os.path.join(input_root, "colori.jpg"))
print(img.shape, img.dtype, img.min(), img.max())
H, W, C = img.shape

# ---------------- 3a: extract the three channels ----------------
# R, G, B is the storage order in the file, and Python counts from 0
img_red = img[:, :, 0]
img_green = img[:, :, 1]
img_blue = img[:, :, 2]

plt.figure()
ax1 = plt.subplot(1, 3, 1)
ax1.imshow(img_red, cmap="gray", clim=[0, 255])
ax1.set_title("Red channel")
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_green, cmap="gray", clim=[0, 255])
ax2.set_title("Green channel")
ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
ax3.imshow(img_blue, cmap="gray", clim=[0, 255])
ax3.set_title("Blue channel")
plt.savefig(os.path.join(output_root, "ExB_Ex3_fig1_channels.png"))

# ---------------- 3a: single-channel RGB images ----------------
# black is (0,0,0), so an array of zeros is a black colour image.
# dtype=np.uint8 is required: without it np.zeros gives float64 and
# matplotlib would clip every pixel to full saturation.
only_red = np.zeros(img.shape, dtype=np.uint8)
only_green = np.zeros(img.shape, dtype=np.uint8)
only_blue = np.zeros(img.shape, dtype=np.uint8)
only_red[:, :, 0] = img_red          # slice on the LEFT assigns
only_green[:, :, 1] = img_green
only_blue[:, :, 2] = img_blue

plt.figure()
ax1 = plt.subplot(1, 3, 1)
ax1.imshow(only_red)                  # colour -> no cmap, no clim
ax1.set_title("Only red")
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax2.imshow(only_green)
ax2.set_title("Only green")
ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
ax3.imshow(only_blue)
ax3.set_title("Only blue")
plt.savefig(os.path.join(output_root, "ExB_Ex3_fig2_single_channel.png"))

# ---------------- 3b: grayscale, two methods ----------------
# float32 before summing: uint8 addition would overflow
img_gray_mean = (img_red.astype(np.float32) + img_green + img_blue) / 3
img_gray_mean = img_gray_mean.astype(np.uint8)

# rgb2gray returns float in [0,1]; converting to uint8 is also what
# makes the histograms below work, since np.histogram counts integers
img_gray_weighted = color.rgb2gray(img)
img_gray_weighted = (img_gray_weighted * 255).astype(np.uint8)

# ---------------- 3b: histogram of each version ----------------
n_mean, _ = np.histogram(img_gray_mean, np.arange(257))
n_weighted, _ = np.histogram(img_gray_weighted, np.arange(257))
print("total pixels:", n_mean.sum(), n_weighted.sum())

plt.figure()
ax1 = plt.subplot(2, 2, 1)
ax1.imshow(img_gray_mean, cmap="gray", clim=[0, 255])
ax1.set_title("Arithmetic mean")
ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_gray_weighted, cmap="gray", clim=[0, 255])
ax2.set_title("Weighted luminosity")
ax3 = plt.subplot(2, 2, 3)           # no sharex with the images
ax3.bar(np.arange(256), n_mean)
ax3.set_title("Histogram - mean")
ax4 = plt.subplot(2, 2, 4)
ax4.bar(np.arange(256), n_weighted)
ax4.set_title("Histogram - weighted")
plt.savefig(os.path.join(output_root, "ExB_Ex3_fig3_grayscale.png"))

# ---------------- 3c: 100x100 crop, BOTTOM-LEFT ----------------
# m increases downward, n rightward -> bottom-left = last rows,
# first columns. Start included, end excluded, so H-100:H is 100 rows.
img_crop = img[H - 100:H, 0:100, :]
print("crop:", img_crop.shape)

plt.figure()
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img)
ax1.set_title("Original")
ax2 = plt.subplot(1, 2, 2)           # different size -> no sharex
ax2.imshow(img_crop)
ax2.set_title("Crop bottom-left")
plt.savefig(os.path.join(output_root, "ExB_Ex3_fig4_crop.png"))

# COMMENT (3c): why is weighted luminosity more accurate than the mean?
#
# The eye is not equally sensitive to the three primaries: the green
# cones are in the middle of the visible band and contribute most to
# perceived brightness, the blue cones are at the edge and contribute
# least.
#
# The arithmetic mean gives every channel a weight of 1/3, so blue
# regions come out too bright and green regions too dark.
#
# The weighted formula Y = 0.30R + 0.59G + 0.11B reflects the actual
# sensitivity of the eye. The weights sum to 1, so white maps to
# white and the output range is preserved. The difference is visible
# in the two histograms above, which distribute the grey levels
# differently.

# ---------------- save ----------------
iio.imwrite(os.path.join(output_root, "ExB_Ex3_red.jpg"), img_red)
iio.imwrite(os.path.join(output_root, "ExB_Ex3_green.jpg"), img_green)
iio.imwrite(os.path.join(output_root, "ExB_Ex3_blue.jpg"), img_blue)
iio.imwrite(os.path.join(output_root, "ExB_Ex3_only_red.jpg"), only_red)
iio.imwrite(os.path.join(output_root, "ExB_Ex3_only_green.jpg"), only_green)
iio.imwrite(os.path.join(output_root, "ExB_Ex3_only_blue.jpg"), only_blue)
iio.imwrite(os.path.join(output_root, "ExB_Ex3_gray_mean.jpg"), img_gray_mean)
iio.imwrite(os.path.join(output_root, "ExB_Ex3_gray_weighted.jpg"), img_gray_weighted)
iio.imwrite(os.path.join(output_root, "ExB_Ex3_crop.jpg"), img_crop)

plt.show()
