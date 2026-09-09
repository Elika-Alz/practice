# -*- coding: utf-8 -*-
"""
Image and Video Processing for Autonomous Driving - TRACK A
Exercise 3 - Point Operations and Unsharp Masking
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
# 3a - load and normalise to float32 in [0, 1]
# =====================================================

img = iio.imread(os.path.join(input_root, "spina_dorsale.jpg"))
print("original:", img.shape, img.dtype, img.min(), img.max())

# Normalisation to float32 in [0, 1] is required for two reasons.
# First, uint8 cannot represent negative values or values above 255,
# so any intermediate result outside that range would wrap around and
# corrupt the pixel; float32 has no such limit and also keeps
# fractional values. Second, gamma correction is a power law, and any
# number between 0 and 1 raised to a positive power stays between
# 0 and 1, so the operation cannot leave the valid range.

img = img.astype(np.float32) / 255.0

# =====================================================
# 3a - negative transformation
# =====================================================

# On a normalised image the maximum is 1, so the negative is 1 - x.
# (On a uint8 image it would be 255 - x.)

img_negative = 1.0 - img

# =====================================================
# 3a - Gamma Correction with gamma = 0.4 and gamma = 2.5
# =====================================================

# The general formula is y = (K-1) * ( x / (K-1) )^gamma.
# Because the image has already been normalised to [0, 1], the
# division and the multiplication by K-1 are already accounted for,
# and the operation reduces to the single expression y = x^gamma.
#
# No clipping is needed: for an input in [0, 1] and a positive
# exponent the output is guaranteed to remain in [0, 1].

img_gamma_04 = img ** 0.4
img_gamma_25 = img ** 2.5

print("gamma 0.4:", img_gamma_04.min(), img_gamma_04.max())
print("gamma 2.5:", img_gamma_25.min(), img_gamma_25.max())

plt.figure()

ax1 = plt.subplot(1, 4, 1)
ax1.imshow(img, cmap="gray", clim=[0, 1])
ax1.set_title("Original")

ax2 = plt.subplot(1, 4, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_negative, cmap="gray", clim=[0, 1])
ax2.set_title("Negative")

ax3 = plt.subplot(1, 4, 3, sharex=ax1, sharey=ax1)
ax3.imshow(img_gamma_04, cmap="gray", clim=[0, 1])
ax3.set_title("Gamma = 0.4")

ax4 = plt.subplot(1, 4, 4, sharex=ax1, sharey=ax1)
ax4.imshow(img_gamma_25, cmap="gray", clim=[0, 1])
ax4.set_title("Gamma = 2.5")

plt.savefig(os.path.join(output_root, "ExA_Ex3_fig1_point_operations.png"))

# =====================================================
# 3b - Unsharp Masking with a 15x15 box filter
#      applied to the ORIGINAL image
# =====================================================

# Unsharp Masking consists of three steps:
#   1. blur the original image
#   2. subtract the blurred image from the original -> the MASK
#   3. add the mask back to the original
#
# The mask isolates the detail because blurring is precisely the
# operation that removes fine structure: whatever the blurring
# destroyed must be the detail. In a uniform region the original and
# the blurred version are identical and the mask is zero; at an edge
# the mask is negative on the dark side and positive on the bright
# side, so adding it back pushes the two sides of the edge apart.

# ---- step 1: blur, with a manually built 15x15 box kernel ----
side = 15
box_filter = np.ones((side, side), dtype=np.float32) / (side * side)

# the coefficients of a smoothing kernel must sum to 1, so that the
# average intensity level of the image is preserved
print("box kernel:", box_filter.shape, "sum =", box_filter.sum())

img_blur = ndi.correlate(img, box_filter)

# ---- step 2: the mask ----
g_mask = img - img_blur

print("mask range:", g_mask.min(), g_mask.max())

# ---- step 3: add the mask back (k = 1 -> Unsharp Masking) ----
k = 1.0
img_um = img + k * g_mask

print("unsharp before clipping:", img_um.min(), img_um.max())
img_um = np.clip(img_um, 0, 1)

plt.figure()

ax1 = plt.subplot(1, 4, 1)
ax1.imshow(img, cmap="gray", clim=[0, 1])
ax1.set_title("Original")

ax2 = plt.subplot(1, 4, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_blur, cmap="gray", clim=[0, 1])
ax2.set_title("Blurred (15x15 box)")

# the mask contains negative values, so no clim is set here:
# clim=[0,1] would render the whole negative half as black
ax3 = plt.subplot(1, 4, 3, sharex=ax1, sharey=ax1)
ax3.imshow(g_mask, cmap="gray")
ax3.set_title("Mask")

ax4 = plt.subplot(1, 4, 4, sharex=ax1, sharey=ax1)
ax4.imshow(img_um, cmap="gray", clim=[0, 1])
ax4.set_title("Unsharp Masking result")

plt.savefig(os.path.join(output_root, "ExA_Ex3_fig2_unsharp_masking.png"))

# =====================================================
# 3c - Laplacian Sharpening, kernel WITH diagonals
# =====================================================

# The Laplacian is the sum of the second derivatives along both axes.
# Including the four diagonal neighbours gives a 3x3 kernel in which
# all eight neighbours have the same coefficient and the centre has
# the opposite sign with eight times the magnitude.
#
# The sign convention used here has a POSITIVE centre, so the
# sharpened image is obtained by ADDING the response:
#       g = f + gamma * laplacian(f)
# (with the opposite convention, +1 around a -8 centre, the response
# would have to be subtracted instead.)

laplacian = np.array([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]], dtype=np.float32)

# the coefficients of a derivative kernel must sum to 0, so that a
# uniform region produces exactly zero response
print("laplacian kernel sum =", laplacian.sum())

img_lapl = ndi.correlate(img, laplacian)

print("laplacian response:", img_lapl.min(), img_lapl.max())

gamma = 0.15
img_ls = img + gamma * img_lapl

print("laplacian sharpening before clipping:", img_ls.min(), img_ls.max())
img_ls = np.clip(img_ls, 0, 1)
print("laplacian sharpening after clipping :", img_ls.min(), img_ls.max())

plt.figure()

ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img_um, cmap="gray", clim=[0, 1])
ax1.set_title("Unsharp Masking")

ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
ax2.imshow(img_ls, cmap="gray", clim=[0, 1])
ax2.set_title("Laplacian Sharpening (gamma = 0.15)")

plt.savefig(os.path.join(output_root, "ExA_Ex3_fig3_sharpening_comparison.png"))

# -----------------------------------------------------
# COMMENT (3c): justification of the gamma value
# -----------------------------------------------------
#
# gamma = 0.15 was chosen as a compromise between a visible
# enhancement and the preservation of the intensity range.
#
# The Laplacian response is signed and is unbounded relative to the
# image range: with the diagonal kernel the centre coefficient is 8,
# so on data normalised to [0, 1] the response can reach values
# several times larger than the range of the image itself. This is
# confirmed by the printed range of img_lapl above.
#
# If gamma were large, the term gamma * laplacian would push a large
# number of pixels above 1 or below 0. The clipping that necessarily
# follows would then flatten precisely the strong edges that the
# operation is meant to enhance, producing visible bright and dark
# halos along every boundary - the opposite of the intended effect.
#
# If gamma were too small, the added term would be negligible and the
# result would be indistinguishable from the original.
#
# With gamma = 0.15 the printed range before clipping shows that only
# a small fraction of pixels falls outside [0, 1], while the
# enhancement of edges and fine details is clearly visible. Note also
# that the diagonal kernel requires a smaller gamma than the
# non-diagonal one, since its centre coefficient is 8 rather than 4
# and it therefore produces roughly twice the response.

# -----------------------------------------------------
# COMMENT (3c): what is the conceptual difference between
# the two sharpening methods?
# -----------------------------------------------------
#
# Structurally the two methods are the SAME operation: the original
# image plus a scaled detail map,
#
#       g = f + k * detail
#
# The difference lies entirely in how that detail map is obtained.
#
# UNSHARP MASKING obtains it INDIRECTLY, BY SUBTRACTION. The image is
# blurred with a smoothing filter and the blurred version is
# subtracted from the original. Since blurring is what removes fine
# structure, whatever the blurring destroyed must be the detail. A
# consequence is that the method is explicitly SCALE-DEPENDENT: the
# size of the blurring kernel, here 15x15, determines which spatial
# scale of detail is extracted, so a larger kernel produces a mask
# containing coarser structure and a smaller one a mask containing
# only the finest structure.
#
# LAPLACIAN SHARPENING obtains it DIRECTLY, BY DIFFERENTIATION. A
# second-derivative operator is applied, which by construction gives
# zero response in uniform regions and a large signed response at
# discontinuities. There is no blurring step and no scale parameter:
# the operator is fixed at 3x3, so it always responds to the finest
# scale present in the image.
#
# The practical consequences follow from that difference.
#
# Unsharp Masking is more FLEXIBLE, because the scale of the
# enhancement can be tuned through the blurring parameter, and more
# ROBUST TO NOISE, because its first step is a smoothing operation
# which suppresses noise before the detail is extracted.
#
# Laplacian Sharpening uses a single small kernel and is
# computationally CHEAPER, but being a second derivative it is
# considerably MORE SENSITIVE TO NOISE: differentiation amplifies
# rapid intensity variations, and noise consists precisely of rapid
# variations, so noise is amplified together with the genuine edges.
#
# Finally, one point applies to both methods: neither recovers
# information. They amplify what is already present in the image, so
# detail that was genuinely lost cannot be restored.

# =====================================================
# Save all outputs
# =====================================================

iio.imwrite(os.path.join(output_root, "ExA_Ex3_negative.jpg"),
            (img_negative * 255).astype(np.uint8))

iio.imwrite(os.path.join(output_root, "ExA_Ex3_gamma_04.jpg"),
            (img_gamma_04 * 255).astype(np.uint8))

iio.imwrite(os.path.join(output_root, "ExA_Ex3_gamma_25.jpg"),
            (img_gamma_25 * 255).astype(np.uint8))

iio.imwrite(os.path.join(output_root, "ExA_Ex3_blurred.jpg"),
            (img_blur * 255).astype(np.uint8))

# the mask is SIGNED, so it must be rescaled to [0, 1] before saving:
# subtracting the minimum as well as dividing by the range.
# Converting a negative float directly to uint8 would destroy the
# whole negative half of the mask.
g_mask_save = (g_mask - g_mask.min()) / (g_mask.max() - g_mask.min())
iio.imwrite(os.path.join(output_root, "ExA_Ex3_mask.jpg"),
            (g_mask_save * 255).astype(np.uint8))

iio.imwrite(os.path.join(output_root, "ExA_Ex3_unsharp_masking.jpg"),
            (img_um * 255).astype(np.uint8))

iio.imwrite(os.path.join(output_root, "ExA_Ex3_laplacian_sharpening.jpg"),
            (img_ls * 255).astype(np.uint8))

plt.show()
