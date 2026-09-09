# -*- coding: utf-8 -*-
"""TRACK B - Exercise 2 - Histogram Stretching, Scaling and Box Filtering"""

import os
import numpy as np
import imageio.v3 as iio
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
from skimage import color
from skimage.transform import rescale, resize

# ---------------- paths ----------------
root = r"C:\Users\Elika\Desktop\IVPAD"
input_root = os.path.join(root, "Images")
output_root = os.path.join(root, "123456_Alizadeh_Elika_Outputs")
if not os.path.exists(output_root):
    os.makedirs(output_root)

# ---------------- 2a: load, histogram, FSHS ----------------
img = iio.imread(os.path.join(input_root, "granelli.jpg"))
print(img.shape, img.dtype, img.min(), img.max())

# 257 bin EDGES -> 256 bins, one per grey level
n_org, _ = np.histogram(img, np.arange(257))

# y = (K-1)/(max-min) * (x - min): subtract min, divide by span, x255
img_fshs = img.astype(np.float32)
img_fshs = (img_fshs - img_fshs.min()) / (img_fshs.max() - img_fshs.min()) * 255.0
img_fshs = np.clip(img_fshs, 0, 255)
img_fshs = img_fshs.astype(np.uint8)

n_fshs, _ = np.histogram(img_fshs, np.arange(257))
print("stretched:", img_fshs.min(), img_fshs.max())

plt.figure()
ax1 = plt.subplot(2, 2, 1)
ax1.imshow(img, cmap="gray", clim=[0, 255])
ax1.set_title("Original")
ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_fshs, cmap="gray", clim=[0, 255])
ax2.set_title("Stretched (FSHS)")
ax3 = plt.subplot(2, 2, 3)          # no sharex with the images
ax3.bar(np.arange(256), n_org)
ax3.set_title("Original histogram")
ax4 = plt.subplot(2, 2, 4)
ax4.bar(np.arange(256), n_fshs)
ax4.set_title("Stretched histogram")
plt.savefig(os.path.join(output_root, "ExB_Ex2_fig1_fshs.png"))

# ---------------- 2b: upscale x3, NN and bilinear ----------------
# normalise FIRST: on a uint8 input rescale returns uint8 for
# order=0 but float [0,1] for order=1, so one panel would be black.
# On a float input both return float in [0,1].
img_norm = img_fshs.astype(np.float32) / 255.0

img_nn = rescale(img_norm, scale=3, order=0)     # nearest-neighbour
img_bl = rescale(img_norm, scale=3, order=1)     # bilinear

print("input:", img_norm.shape, "| upscaled:", img_nn.shape)

plt.figure()
# both results are the same size, so they can share axes
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img_nn, cmap="gray", clim=[0, 1])
ax1.set_title("Upscale x3 - nearest neighbour")
ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_bl, cmap="gray", clim=[0, 1])
ax2.set_title("Upscale x3 - bilinear")
plt.savefig(os.path.join(output_root, "ExB_Ex2_fig2_upscale.png"))

# COMMENT (2b): visual differences
#
# Upscaling by 3 means one input pixel must fill nine output
# positions, so eight of every nine values are estimated.
#
# NEAREST-NEIGHBOUR copies the closest pixel, so each input pixel
# becomes a 3x3 block of identical values. No new intensity is
# created; edges stay sharp but the replication shows as BLOCKY
# artefacts and diagonal edges as a staircase.
#
# BILINEAR averages the four surrounding pixels, creating
# intermediate values. Transitions are gradual, diagonal edges look
# smooth, but edges are slightly BLURRED and fine details smeared.
#
# Neither adds information: interpolation only decides how the gaps
# between known samples are filled.

# ---------------- 2c: 9x9 box filter on the bilinear result ----------------
# kernel built manually: ones divided by the number of elements.
# The weights must sum to 1 so the average intensity is preserved;
# forgetting the division would make the image 81x too bright.
side = 9
box_filter = np.ones((side, side), dtype=np.float32) / (side * side)
print("box sum =", box_filter.sum())

img_box = ndi.correlate(img_bl, box_filter)

plt.figure()
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img_bl, cmap="gray", clim=[0, 1])
ax1.set_title("Input (bilinear upscaled)")
ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_box, cmap="gray", clim=[0, 1])
ax2.set_title("9x9 box filter")
plt.savefig(os.path.join(output_root, "ExB_Ex2_fig3_box.png"))

# COMMENT (2c): nearest-neighbour vs bilinear - quality and cost
#
# VISUAL QUALITY. Nearest-neighbour uses ONE pixel, the closest, so
# it preserves the original values exactly and keeps edges perfectly
# sharp - but each input pixel is replicated as a block, giving
# blocky artefacts and staircased diagonal edges. Bilinear uses FOUR
# pixels as a weighted average, creating intermediate values, so
# transitions are gradual and diagonals look smooth, at the cost of
# slightly blurred edges and smeared fine details.
#
# COMPUTATIONAL COST. Nearest-neighbour needs one rounding and one
# memory access per output pixel; bilinear needs four accesses plus
# several multiplications, so it is noticeably more expensive.
#
# CHOICE. Bilinear for photographs, where smooth appearance matters
# most. Nearest-neighbour when the values themselves carry meaning
# and must not be altered - segmentation masks or label images, where
# an average of class 2 and class 4 would be meaningless.
#
# Neither recovers lost detail: bilinear fills the gaps more
# plausibly, not more correctly.

# ---------------- save ----------------
iio.imwrite(os.path.join(output_root, "ExB_Ex2_fshs.jpg"), img_fshs)
iio.imwrite(os.path.join(output_root, "ExB_Ex2_upscale_nn.jpg"),
            (img_nn * 255).astype(np.uint8))
iio.imwrite(os.path.join(output_root, "ExB_Ex2_upscale_bl.jpg"),
            (img_bl * 255).astype(np.uint8))
iio.imwrite(os.path.join(output_root, "ExB_Ex2_box_filtered.jpg"),
            (img_box * 255).astype(np.uint8))

# histograms are figures, so plt.savefig, not iio.imwrite
plt.figure()
plt.bar(np.arange(256), n_org)
plt.title("Original histogram")
plt.savefig(os.path.join(output_root, "ExB_Ex2_hist_original.png"))

plt.figure()
plt.bar(np.arange(256), n_fshs)
plt.title("Stretched histogram")
plt.savefig(os.path.join(output_root, "ExB_Ex2_hist_stretched.png"))

plt.show()
