# -*- coding: utf-8 -*-
"""TRACK B - Exercise 4 - Watershed Pipeline"""

import os
import numpy as np
import imageio.v3 as iio
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
from skimage import color
from skimage.segmentation import watershed, mark_boundaries
from skimage.feature import peak_local_max
from skimage.color import label2rgb

# ---------------- paths ----------------
root = r"C:\Users\Elika\Desktop\IVPAD"
input_root = os.path.join(root, "Images")
output_root = os.path.join(root, "123456_Alizadeh_Elika_Outputs")
if not os.path.exists(output_root):
    os.makedirs(output_root)

# ---------------- 4a: load, normalise, Gaussian blur ----------------
img = iio.imread(os.path.join(input_root, "granelli_riso.tif"))
print(img.shape, img.dtype, img.min(), img.max())

img = img.astype(np.float32) / 255.0

sigma = 2
img_blur = ndi.gaussian_filter(img, sigma)

# COMMENT (4a): justification of sigma
#
# sigma = 2 removes variations at the scale of one or two pixels -
# the noise that would create spurious local minima in the gradient -
# while preserving the boundaries between the grains, which span tens
# of pixels. A Gaussian kernel is about 6*sigma wide, here roughly 13
# pixels, comparable to the noise scale and not to the grain size.
#
# It is a compromise: a larger sigma would blur the boundaries of
# touching grains so they merge into one region; a smaller one would
# leave noise in the gradient and cause over-segmentation.

# ---------------- 4b: Sobel gradient magnitude ----------------
# Sobel = smoothed derivative: it factorises into the difference
# [-1,0,1] across the edge and the smoothing [1,2,1] along it
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
sobel_y = sobel_x.T
print("sobel sums:", sobel_x.sum(), sobel_y.sum())     # derivative -> 0

img_dx = ndi.correlate(img_blur, sobel_x)
img_dy = ndi.correlate(img_blur, sobel_y)

# one kernel alone is blind to edges parallel to its own direction,
# so both are needed; the magnitude combines them and is always >= 0
img_grad = np.sqrt(img_dx ** 2 + img_dy ** 2)
print("gradient:", img_grad.min(), img_grad.max())

plt.figure()
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img, cmap="gray", clim=[0, 1])
ax1.set_title("Original")
ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_grad, cmap="gray")        # not in [0,1] -> no clim
ax2.set_title("Sobel gradient magnitude")
plt.savefig(os.path.join(output_root, "ExB_Ex4_fig1_gradient.png"))

# ---------------- 4c: marker-controlled Watershed ----------------
# every local minimum starts a basin, so markers are chosen
# explicitly. peak_local_max finds MAXIMA, so it is applied to -grad:
# the valleys of the gradient are the peaks of its negative.
coords = peak_local_max(-img_grad, min_distance=30)

markers = np.zeros(img_grad.shape, dtype=int)
markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

labels = watershed(img_grad, markers=markers)
print("regions:", labels.max())          # should be tens, not thousands

# labels holds region IDs, not intensities, so it is visualised
overlay = label2rgb(labels, image=img, bg_label=0, alpha=0.4)
boundaries = mark_boundaries(img, labels)

plt.figure()
ax1 = plt.subplot(1, 3, 1)
ax1.imshow(img, cmap="gray", clim=[0, 1])
ax1.set_title("Original")
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax2.imshow(overlay)                       # colour -> no cmap, no clim
ax2.set_title("Regions overlaid")
ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
ax3.imshow(boundaries)
ax3.set_title("Boundaries")
plt.savefig(os.path.join(output_root, "ExB_Ex4_fig2_watershed.png"))

# COMMENT (4c): role of the low-pass filtering, and what if it were skipped
#
# The Watershed reads the gradient magnitude as a topographic
# surface: grain interiors are valleys (intensity varies slowly),
# grain boundaries are ridges (intensity changes abruptly). It floods
# from the local minima and builds a dam where two basins meet, so
# EVERY LOCAL MINIMUM GENERATES ONE REGION.
#
# That is why the blur is necessary. The gradient is a derivative,
# and derivatives amplify rapid variations - which is exactly what
# noise is. On an unfiltered image the gradient contains a very large
# number of spurious local minima caused by noise and texture rather
# than by real boundaries.
#
# IF THE BLUR WERE SKIPPED, each spurious minimum would start its own
# basin and the result would be severe OVER-SEGMENTATION: thousands
# of meaningless fragments instead of roughly one region per grain,
# with the true boundaries lost among them.
#
# The Gaussian suppresses these fine variations so that only genuine
# boundaries form ridges. Here it is combined with marker-controlled
# watershed, where only selected minima may start a basin.

# ---------------- save ----------------
# the gradient is not in [0,1]; it is never negative, so dividing by
# the maximum is enough
img_grad_save = img_grad / img_grad.max()
iio.imwrite(os.path.join(output_root, "ExB_Ex4_gradient.jpg"),
            (img_grad_save * 255).astype(np.uint8))
iio.imwrite(os.path.join(output_root, "ExB_Ex4_blurred.jpg"),
            (img_blur * 255).astype(np.uint8))
iio.imwrite(os.path.join(output_root, "ExB_Ex4_overlay.jpg"),
            (overlay * 255).astype(np.uint8))
iio.imwrite(os.path.join(output_root, "ExB_Ex4_boundaries.jpg"),
            (boundaries * 255).astype(np.uint8))

plt.show()
