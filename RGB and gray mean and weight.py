# -*- coding: utf-8 -*-
"""
Image and Video Processing for Autonomous Driving - TRACK A
Exercise 1 - RGB Image: Channels, Grayscale and Histogram
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
# Load
# =====================================================

img = iio.imread(os.path.join(input_root, "test_dog.jpg"))
print("original:", img.shape, img.dtype, img.min(), img.max())

H, W, C = img.shape

plt.figure()
plt.imshow(img)
plt.title("Original image")
plt.savefig(os.path.join(output_root, "ExA_Ex1_fig0_original.png"))

# =====================================================
# 1a - extract the R, G, B channels using slicing
# =====================================================

img_red = img[:, :, 0]
img_green = img[:, :, 1]
img_blue = img[:, :, 2]

print("single channel:", img_red.shape, img_red.dtype)

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

plt.savefig(os.path.join(output_root, "ExA_Ex1_fig1_channels.png"))

# =====================================================
# 1a - reconstruct three RGB images with only one
#      channel active at a time
# =====================================================

only_red = np.zeros(img.shape, dtype=np.uint8)
only_green = np.zeros(img.shape, dtype=np.uint8)
only_blue = np.zeros(img.shape, dtype=np.uint8)

only_red[:, :, 0] = img_red
only_green[:, :, 1] = img_green
only_blue[:, :, 2] = img_blue

plt.figure()

ax1 = plt.subplot(1, 3, 1)
ax1.imshow(only_red)
ax1.set_title("Only red")

ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax2.imshow(only_green)
ax2.set_title("Only green")

ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
ax3.imshow(only_blue)
ax3.set_title("Only blue")

plt.savefig(os.path.join(output_root, "ExA_Ex1_fig2_single_channel_rgb.png"))

# =====================================================
# 1b - grayscale conversion, two methods
# =====================================================

# arithmetic mean of the three channels.
# The channels are converted to float32 BEFORE being summed: adding
# uint8 arrays would overflow (200 + 200 + 200 wraps around instead of
# giving 600) and the result would be completely wrong.
img_gray_mean = (img_red.astype(np.float32) + img_green + img_blue) / 3
img_gray_mean = img_gray_mean.astype(np.uint8)

# standard weighted luminosity formula.
# rgb2gray returns a float array in [0, 1], so it is rescaled to
# [0, 255] and converted to uint8 to match the other version.
img_gray_weighted = color.rgb2gray(img)
img_gray_weighted = (img_gray_weighted * 255).astype(np.uint8)

print("mean     :", img_gray_mean.dtype, img_gray_mean.min(), img_gray_mean.max())
print("weighted :", img_gray_weighted.dtype, img_gray_weighted.min(), img_gray_weighted.max())

plt.figure()

ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img_gray_mean, cmap="gray", clim=[0, 255])
ax1.set_title("Grayscale - arithmetic mean")

ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_gray_weighted, cmap="gray", clim=[0, 255])
ax2.set_title("Grayscale - weighted luminosity")

plt.savefig(os.path.join(output_root, "ExA_Ex1_fig3_grayscale.png"))

# -----------------------------------------------------
# COMMENT (1b): why does the weighted luminosity formula
# produce a visually more accurate grayscale than the
# arithmetic mean?
# -----------------------------------------------------
#
# Because the human eye is not equally sensitive to the three
# primary colours. The retina contains three types of cone, tuned to
# short, medium and long wavelengths. The medium-wavelength cones,
# responsible for green, lie in the middle of the visible band and
# contribute the most to the perceived brightness of a scene, while
# the short-wavelength cones, responsible for blue, lie at the edge
# of the band and contribute far less.
#
# The arithmetic mean assigns a weight of 1/3 to every channel, so it
# treats one unit of blue as contributing as much brightness as one
# unit of green. This does not match perception: in the resulting
# image, blue regions appear too bright and green regions too dark
# relative to how bright they actually look to an observer.
#
# The weighted luminosity formula uses coefficients that reflect the
# actual sensitivity of the eye, approximately
#
#       Y = 0.30*R + 0.59*G + 0.11*B
#
# with green weighted most and blue least. The coefficients sum to 1,
# so the output range is preserved: a white pixel maps to white and a
# black pixel to black. The resulting grayscale therefore matches the
# perceived brightness of the original colour image much more
# closely than the arithmetic mean does.

# =====================================================
# 1c - 100x100 colour crop, TOP-RIGHT corner
# =====================================================

# The origin is the top-left corner, m (rows) increases downward and
# n (columns) increases rightward. The TOP-RIGHT corner therefore
# corresponds to small row indices and large column indices:
#   rows    : 0 ... 100          (the first 100 rows)
#   columns : W-100 ... W        (the last 100 columns)
# The final ':' keeps all three channels, so the crop stays in colour.

img_crop = img[0:100, W - 100:W, :]

print("crop:", img_crop.shape)

plt.figure()

ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img)
ax1.set_title("Original")

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(img_crop)
ax2.set_title("100x100 crop - top-right corner")

plt.savefig(os.path.join(output_root, "ExA_Ex1_fig4_crop.png"))

# =====================================================
# Save all outputs
# =====================================================

iio.imwrite(os.path.join(output_root, "ExA_Ex1_red_channel.jpg"), img_red)
iio.imwrite(os.path.join(output_root, "ExA_Ex1_green_channel.jpg"), img_green)
iio.imwrite(os.path.join(output_root, "ExA_Ex1_blue_channel.jpg"), img_blue)

iio.imwrite(os.path.join(output_root, "ExA_Ex1_only_red.jpg"), only_red)
iio.imwrite(os.path.join(output_root, "ExA_Ex1_only_green.jpg"), only_green)
iio.imwrite(os.path.join(output_root, "ExA_Ex1_only_blue.jpg"), only_blue)

iio.imwrite(os.path.join(output_root, "ExA_Ex1_gray_mean.jpg"), img_gray_mean)
iio.imwrite(os.path.join(output_root, "ExA_Ex1_gray_weighted.jpg"), img_gray_weighted)

iio.imwrite(os.path.join(output_root, "ExA_Ex1_crop_top_right.jpg"), img_crop)

plt.show()
