# -*- coding: utf-8 -*-
"""
Image and Video Processing for Autonomous Driving - TRACK A
Exercise 4 - Handcrafted Segmentation: Watershed Pipeline
"""

import os
import numpy as np
import imageio.v3 as iio
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
from skimage import color
from skimage.segmentation import watershed, mark_boundaries
from skimage.feature import peak_local_max
from skimage.color import label2rgb

# =====================================================
# Paths
# =====================================================

root = r"C:\Users\Elika\Desktop\IVPAD"
input_root = os.path.join(root, "Images")
output_root = os.path.join(root, "123456_Alizadeh_Elika_Outputs")

if not os.path.exists(output_root):
    os.makedirs(output_root)

# =====================================================
# 4a - load, normalise, Gaussian low-pass filter
# =====================================================

img = iio.imread(os.path.join(input_root, "dowels.tif"))
print("original:", img.shape, img.dtype, img.min(), img.max())

img = img.astype(np.float32) / 255.0

sigma = 2
img_blur = ndi.gaussian_filter(img, sigma)

# -----------------------------------------------------
# COMMENT (4a): justification of the choice of sigma
# -----------------------------------------------------
#
# sigma = 2 was chosen so that the smoothing removes variations at
# the scale of one or two pixels - that is, the noise and fine
# texture that would otherwise create spurious local minima in the
# gradient and cause severe over-segmentation - while preserving the
# boundaries between the objects, which span tens of pixels.
#
# A Gaussian kernel is effectively about 6*sigma wide, since beyond
# three standard deviations on each side the weights are negligible.
# With sigma = 2 that is roughly 13 pixels, which is comparable to
# the scale of the noise to be removed rather than to the size of the
# objects to be separated.
#
# The choice is a compromise in both directions. A larger sigma would
# start to blur the boundaries between adjacent objects, so that
# touching objects would merge into a single catchment basin and be
# segmented as one region. A smaller sigma would leave too much noise
# in the image, so the gradient would contain many spurious local
# minima and the Watershed would produce a large number of
# meaningless regions.

# =====================================================
# 4b - gradient magnitude and phase, Sobel filters
# =====================================================

# The Sobel operator is a smoothed first-order derivative. It
# factorises as the outer product of a centred difference [-1, 0, 1]
# in one direction and a smoothing kernel [1, 2, 1] in the
# perpendicular direction, so it differentiates across the edge while
# averaging along it.

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

# the vertical kernel is the transpose of the horizontal one
sobel_y = sobel_x.T

# the coefficients of a derivative kernel must sum to 0, so that a
# uniform region produces exactly zero response
print("sobel_x sum =", sobel_x.sum(), "| sobel_y sum =", sobel_y.sum())

img_dx = ndi.correlate(img_blur, sobel_x)
img_dy = ndi.correlate(img_blur, sobel_y)

# gradient magnitude: sqrt(fx^2 + fy^2).
# A single derivative kernel is blind to edges parallel to its own
# direction of differentiation, so both partial derivatives are
# required. The magnitude combines them into one orientation
# independent measure of intensity variation, and being a square root
# of squares it is always non-negative.
img_grad = np.sqrt(img_dx ** 2 + img_dy ** 2)

# gradient phase: the orientation of the intensity change
img_phase = np.arctan(img_dx / img_dy)

print("gradient magnitude:", img_grad.min(), img_grad.max())

plt.figure()

ax1 = plt.subplot(1, 3, 1)
ax1.imshow(img, cmap="gray", clim=[0, 1])
ax1.set_title("Original")

# no clim on the derivative panels: the gradient magnitude is not
# in [0, 1], so matplotlib is left to auto-scale the display
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_grad, cmap="gray")
ax2.set_title("Sobel gradient magnitude")

ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
ax3.imshow(img_phase, cmap="gray")
ax3.set_title("Gradient phase")

plt.savefig(os.path.join(output_root, "ExA_Ex4_fig1_gradient.png"))

# =====================================================
# 4c - marker-controlled Watershed
# =====================================================

# The Watershed interprets the gradient magnitude as a topographic
# surface and floods it starting from local minima, so EVERY local
# minimum generates one region. Applied directly, this produces
# thousands of meaningless regions.
#
# To control this, a set of markers is selected explicitly: only
# those points are allowed to start a catchment basin. The markers
# are the local minima of the gradient, which correspond to the
# smooth interiors of the objects. peak_local_max finds local MAXIMA,
# so it is applied to -img_grad: the valleys of the gradient are the
# peaks of its negative.
#
# min_distance sets the smallest allowed separation between two
# markers and is therefore the main control over over-segmentation.

coords = peak_local_max(-img_grad, min_distance=30)

markers = np.zeros(img_grad.shape, dtype=int)
markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

print("number of markers:", markers.max())

labels = watershed(img_grad, markers=markers)

print("number of regions:", labels.max())

# labels is a map of region identifiers, not intensities, so it
# cannot be displayed directly. Two standard visualisations are used:
#   label2rgb       - colours each region and blends it over the image
#   mark_boundaries - draws a line along every region boundary
# Both return COLOUR images, so they are displayed with no cmap and
# no clim.

overlay = label2rgb(labels, image=img, bg_label=0, alpha=0.4)
boundaries = mark_boundaries(img, labels)

plt.figure()

ax1 = plt.subplot(1, 3, 1)
ax1.imshow(img, cmap="gray", clim=[0, 1])
ax1.set_title("Original")

ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax2.imshow(overlay)
ax2.set_title("Watershed regions overlaid")

ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
ax3.imshow(boundaries)
ax3.set_title("Watershed boundaries")

plt.savefig(os.path.join(output_root, "ExA_Ex4_fig2_watershed.png"))

# -----------------------------------------------------
# COMMENT (4c): the role of the low-pass filtering step
# before computing the gradient, and what would happen
# if it were skipped
# -----------------------------------------------------
#
# The Watershed algorithm interprets its input as a topographic
# surface: low values are valleys and high values are ridges. When
# the input is the gradient magnitude, the smooth interiors of the
# objects become valleys, because the intensity varies slowly there,
# and the boundaries between objects become ridges, because the
# intensity changes abruptly there. The algorithm then simulates
# flooding: water rises from each local minimum, and wherever two
# basins would merge a dam is built. Each catchment basin becomes one
# region and the dams become the segmentation boundaries.
#
# The essential consequence is that EVERY LOCAL MINIMUM OF THE
# GRADIENT GENERATES ONE REGION. This is exactly why the low-pass
# filtering step is necessary.
#
# The gradient is a derivative operation, and derivatives amplify
# rapid intensity variations. Noise consists precisely of rapid
# variations, so a derivative responds to noise very strongly. On an
# unfiltered image the gradient therefore contains a very large
# number of small spurious local minima, produced by sensor noise and
# fine texture rather than by real object boundaries.
#
# IF THE LOW-PASS FILTERING WERE SKIPPED, each of those spurious
# minima would start its own catchment basin, and the result would be
# severe OVER-SEGMENTATION: the image would be divided into thousands
# of tiny meaningless fragments instead of approximately one region
# per object, and the genuine object boundaries would be completely
# lost among them. The segmentation would be unusable.
#
# The Gaussian blur suppresses these fine variations before the
# gradient is computed, so that only the genuine boundaries produce
# ridges in the topographic surface. Its parameter sigma is a
# compromise, as discussed in 4a: too small and the noise survives,
# too large and the boundaries of adjacent objects merge.
#
# In this implementation the smoothing is combined with a second
# defence, marker-controlled watershed, in which only a selected set
# of minima is allowed to start a basin. The two together reduce the
# result from thousands of regions to approximately one per object.

# =====================================================
# Save all outputs
# =====================================================

# the gradient magnitude is not in [0, 1], so it must be rescaled
# before being converted to uint8. It is never negative, so dividing
# by the maximum is sufficient.
img_grad_save = img_grad / img_grad.max()
iio.imwrite(os.path.join(output_root, "ExA_Ex4_gradient_magnitude.jpg"),
            (img_grad_save * 255).astype(np.uint8))

iio.imwrite(os.path.join(output_root, "ExA_Ex4_blurred.jpg"),
            (img_blur * 255).astype(np.uint8))

iio.imwrite(os.path.join(output_root, "ExA_Ex4_watershed_overlay.jpg"),
            (overlay * 255).astype(np.uint8))

iio.imwrite(os.path.join(output_root, "ExA_Ex4_watershed_boundaries.jpg"),
            (boundaries * 255).astype(np.uint8))

plt.show()
