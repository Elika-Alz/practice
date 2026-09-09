# -*- coding: utf-8 -*-
"""TRACK B - Exercise 1 - Gamma Correction, Rotation and Median Filtering"""

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

# ---------------- 1a: load, normalise, gamma ----------------
img = iio.imread(os.path.join(input_root, "angiogramma.jpg"))
print(img.shape, img.dtype, img.min(), img.max())

img = img.astype(np.float32) / 255.0

# y = (K-1)*(x/(K-1))^gamma; already normalised, so just x^gamma.
# No clip needed: [0,1] to a positive power stays in [0,1].
img_gamma_03 = img ** 0.3
img_gamma_30 = img ** 3.0

plt.figure()
ax1 = plt.subplot(1, 3, 1)
ax1.imshow(img, cmap="gray", clim=[0, 1])
ax1.set_title("Original")
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_gamma_03, cmap="gray", clim=[0, 1])
ax2.set_title("Gamma = 0.3")
ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
ax3.imshow(img_gamma_30, cmap="gray", clim=[0, 1])
ax3.set_title("Gamma = 3.0")
plt.savefig(os.path.join(output_root, "ExB_Ex1_fig1_gamma.png"))

# COMMENT (1a): effect of gamma < 1 and gamma > 1, plus an application
#
# GAMMA < 1 (0.3). For values in [0,1] an exponent below one acts as
# a root, which INCREASES them, so the image gets BRIGHTER. The
# transfer curve bulges above the diagonal: steep near 0, flat near
# 1, so contrast is ENHANCED IN THE DARK REGIONS and compressed in
# the bright ones.
# Application: an UNDEREXPOSED image where the information is in the
# shadows - medical imaging such as an angiogram or an X-ray, where
# the diagnostic detail sits in the dark part of the range.
#
# GAMMA > 1 (3.0). An exponent above one means more multiplication,
# and numbers below one get SMALLER, so the image gets DARKER. The
# curve sags below the diagonal: flat near 0, steep near 1, so
# contrast is ENHANCED IN THE BRIGHT REGIONS.
# Application: an OVEREXPOSED image where detail is lost in the
# highlights - a washed-out aerial or satellite photograph, where
# compressing the high values makes terrain features visible again.
#
# In both cases the endpoints are fixed, since 0^g = 0 and 1^g = 1:
# gamma REDISTRIBUTES the mid-tones rather than shifting the whole
# range, unlike a linear brightness change which needs clipping.

# ---------------- 1b: 45 deg rotation, NN and bilinear ----------------
# a 45 deg rotation maps output pixels to non-integer source
# positions, so the values must be interpolated.
# order=0 -> nearest-neighbour, order=1 -> bilinear.
# The default is order=3, so 'order' must be passed explicitly.
img_rot_nn = ndi.rotate(img, 45, order=0)
img_rot_bl = ndi.rotate(img, 45, order=1)

# ndi.rotate enlarges the canvas so nothing is cut off
print("original:", img.shape, "| rotated:", img_rot_nn.shape)

plt.figure()
# the two rotated results are the same size as each other, so they
# can share axes; neither matches the original
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img_rot_nn, cmap="gray", clim=[0, 1])
ax1.set_title("45 deg - nearest neighbour")
ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_rot_bl, cmap="gray", clim=[0, 1])
ax2.set_title("45 deg - bilinear")
plt.savefig(os.path.join(output_root, "ExB_Ex1_fig2_rotation.png"))

# COMMENT (1b): visual differences, edges and fine details
#
# NEAREST-NEIGHBOUR copies the closest pixel, so no new intensity is
# ever created. EDGES stay perfectly sharp, but because whole pixels
# are replicated a diagonal edge becomes a visible STAIRCASE - very
# noticeable after a 45 deg rotation, where every original horizontal
# or vertical edge becomes diagonal. FINE DETAILS keep their values
# but their positions are quantised, so they look ragged.
#
# BILINEAR averages the four surrounding pixels, creating
# intermediate values. Transitions become gradual, so diagonal edges
# look SMOOTH and the staircase disappears, but edges are slightly
# BLURRED and fine details are smeared.
#
# COST: nearest-neighbour needs one rounding and one memory access
# per pixel; bilinear needs four accesses and several
# multiplications, so it is more expensive.
#
# Neither adds information: bilinear looks better but is not more
# accurate. Nearest-neighbour is required for label images, where an
# averaged class value would be meaningless.

# ---------------- 1c: median filter, window 7 ----------------
# second argument is the WINDOW SIZE (7x7), not a sigma.
# There is no kernel: the median filter is non-linear.
img_median = ndi.median_filter(img_rot_bl, 7)

plt.figure()
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img_rot_bl, cmap="gray", clim=[0, 1])
ax1.set_title("Rotated (bilinear), unfiltered")
ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_median, cmap="gray", clim=[0, 1])
ax2.set_title("Median filter 7x7")
plt.savefig(os.path.join(output_root, "ExB_Ex1_fig3_median.png"))

# COMMENT (1c): why is the median good for impulsive noise while
# preserving edges?
#
# Impulsive (salt-and-pepper) noise replaces isolated pixels with
# EXTREME values, the minimum or the maximum, so the corrupted pixels
# are OUTLIERS.
#
# An averaging filter computes a weighted sum, so an outlier enters
# with its full magnitude and there is nothing to cancel it. The
# corruption is not removed but SPREAD over the whole window.
#
# The median SORTS the values and takes the middle one. An outlier
# lies at one END of the sorted list, exactly where the median never
# looks, so it contributes nothing. With eight neighbours around 90
# and one pixel at 255, the median is 90 - the correct local value -
# while the mean would be about 123.
#
# EDGES are preserved by the same mechanism: when the window
# straddles an edge the median reports whichever side has more
# pixels, so the output is always a real value from one side, never a
# blend. The transition stays sharp, whereas averaging produces a
# different intermediate value at each position and turns the step
# into a ramp.
#
# In short: the median is SELECTED, not computed, so it is always a
# value that was actually present in the window.

# ---------------- save ----------------
iio.imwrite(os.path.join(output_root, "ExB_Ex1_gamma_03.jpg"),
            (img_gamma_03 * 255).astype(np.uint8))
iio.imwrite(os.path.join(output_root, "ExB_Ex1_gamma_30.jpg"),
            (img_gamma_30 * 255).astype(np.uint8))
iio.imwrite(os.path.join(output_root, "ExB_Ex1_rot_nn.jpg"),
            (img_rot_nn * 255).astype(np.uint8))
iio.imwrite(os.path.join(output_root, "ExB_Ex1_rot_bilinear.jpg"),
            (img_rot_bl * 255).astype(np.uint8))
iio.imwrite(os.path.join(output_root, "ExB_Ex1_median.jpg"),
            (img_median * 255).astype(np.uint8))

plt.show()
