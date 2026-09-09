# -*- coding: utf-8 -*-
"""
Image and Video Processing for Autonomous Driving - TRACK A
Exercise 5 - Residual Architectures
"""

import torch
import torch.nn as nn


# =====================================================
# 5a - classic ResBlock
# =====================================================

class ResBlock(nn.Module):
    """
    Classic residual block.

    Input : tensor of shape (B, C, H, W)
    Path  : Conv(C, C, 3, padding=1) -> BatchNorm -> ReLU
            -> Conv(C, C, 3, padding=1) -> BatchNorm
    The residual (skip) connection is added BEFORE the final ReLU.

    The number of input and output channels is the same, and a 3x3
    convolution with padding=1 preserves H and W. The block output
    therefore has exactly the same shape as its input, which is what
    makes the element-wise addition of the skip connection possible.
    """

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x                 # keep the input for the skip connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity         # residual (skip) connection
        out = self.relu(out)         # final ReLU, AFTER the addition

        return out


# =====================================================
# 5b - ConvBlock
# =====================================================

class ConvBlock(nn.Module):
    """
    Conv(x, 64, 7, padding='same') -> BatchNorm -> ReLU
    -> Conv(64, 64, 5, padding='same') -> ReLU -> MaxPool(2, 2)

    Both convolutions use padding='same', so they leave H and W
    unchanged; only the MaxPool halves the spatial dimensions.
    The layers are wrapped in nn.Sequential because the data flows
    straight through with no skip connection.
    """

    def __init__(self, in_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


# =====================================================
# 5b - ResNet-like network
# =====================================================

class ResNetLike(nn.Module):
    """
    ConvBlock(3) -> ConvBlock(64) -> ResBlock(64) -> ResBlock(64)
    -> Flatten -> FC -> FC -> FC(., 10)

    Input: 224 x 224 x 3
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.convblock1 = ConvBlock(3)
        self.convblock2 = ConvBlock(64)
        self.resblock1 = ResBlock(64)
        self.resblock2 = ResBlock(64)

        self.flatten = nn.Flatten()

        # -------------------------------------------------
        # INPUT DIMENSION OF THE FIRST FC LAYER
        # step-by-step calculation
        # -------------------------------------------------
        #
        # The general output-size formula for a convolution or a
        # pooling layer is
        #
        #       H_out = floor( (H_in + 2P - K) / S ) + 1
        #
        # where P is the padding, K the kernel size and S the stride.
        #
        # Two special cases are used here:
        #   - padding='same' with stride 1  -> H and W unchanged
        #   - MaxPool(2, 2): K=2, S=2, P=0  -> H_out = H/2
        #
        # INPUT:                                  (B,  3, 224, 224)
        #
        # ConvBlock(3):
        #   Conv(3, 64, 7, padding='same')     -> (B, 64, 224, 224)
        #       'same' keeps H and W; channels go from 3 to 64
        #   BatchNorm2d(64)                    -> (B, 64, 224, 224)
        #   ReLU                               -> (B, 64, 224, 224)
        #   Conv(64, 64, 5, padding='same')    -> (B, 64, 224, 224)
        #   ReLU                               -> (B, 64, 224, 224)
        #   MaxPool(2, 2): 224 / 2 = 112       -> (B, 64, 112, 112)
        #
        # ConvBlock(64):
        #   Conv(64, 64, 7, padding='same')    -> (B, 64, 112, 112)
        #   BatchNorm2d(64)                    -> (B, 64, 112, 112)
        #   ReLU                               -> (B, 64, 112, 112)
        #   Conv(64, 64, 5, padding='same')    -> (B, 64, 112, 112)
        #   ReLU                               -> (B, 64, 112, 112)
        #   MaxPool(2, 2): 112 / 2 = 56        -> (B, 64,  56,  56)
        #
        # ResBlock(64):
        #   Conv 3x3 with padding=1 preserves H and W, and the number
        #   of channels is unchanged by construction
        #                                      -> (B, 64,  56,  56)
        #
        # ResBlock(64):                        -> (B, 64,  56,  56)
        #
        # Flatten: the batch dimension is kept and the remaining
        # three dimensions are multiplied together:
        #
        #       64 * 56 * 56
        #       64 * 56   = 3584
        #       3584 * 56 = 200704
        #
        #                                      -> (B, 200704)
        #
        # => INPUT DIMENSION OF THE FIRST FC LAYER = 200704
        #
        # Note that only the two MaxPool layers changed the spatial
        # size (224 -> 112 -> 56); every convolution in the network
        # preserves it.
        # -------------------------------------------------

        self.fc1 = nn.Linear(200704, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.convblock1(x)
        out = self.convblock2(out)
        out = self.resblock1(out)
        out = self.resblock2(out)

        out = self.flatten(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        # no activation after the last layer: it produces the raw
        # class scores (logits) expected by nn.CrossEntropyLoss

        return out


# -----------------------------------------------------
# COMMENT (5c): what is the vanishing gradient problem,
# and how do residual (skip) connections address it?
# Why is this particularly important in very deep networks?
# -----------------------------------------------------
#
# THE VANISHING GRADIENT PROBLEM.
# A neural network is trained by backpropagation, which computes the
# gradient of the loss with respect to each weight by applying the
# chain rule backwards through the layers. The gradient that reaches
# an early layer is therefore a PRODUCT of the local derivatives of
# every layer that comes after it.
#
# When those local derivatives are smaller than one, multiplying many
# of them together makes the product shrink exponentially with depth.
# In a very deep network the gradient arriving at the first layers
# becomes almost zero, so those layers receive essentially no update
# and stop learning altogether. This is the vanishing gradient
# problem. It is the reason why, before residual architectures,
# simply stacking more layers made networks perform WORSE rather than
# better - the so-called degradation problem, which is an
# optimisation failure and not merely overfitting.
#
# HOW SKIP CONNECTIONS ADDRESS IT.
# A residual connection adds the input of a block directly to its
# output, so the block computes F(x) + x instead of F(x). When the
# gradient is propagated backwards through that addition, it splits
# into two paths: one through the convolutional layers, and one
# directly along the identity shortcut.
#
# The shortcut path contributes a derivative of exactly 1, which is
# ADDED to the contribution of the convolutional path. Because of
# that additive term, the gradient reaching the earlier layers can no
# longer vanish, however many blocks are stacked: there is always a
# direct route from the loss back to every earlier layer.
#
# There is a second, related benefit. A residual block only has to
# learn the RESIDUAL F(x), that is the difference between the desired
# output and the input, rather than the entire mapping. If the
# optimal behaviour of a block is to leave its input unchanged, it
# only needs to drive F(x) towards zero, which is far easier for an
# optimiser than learning an identity mapping from scratch through
# several convolutions. Consequently, adding depth can no longer make
# the network worse.
#
# WHY THIS MATTERS MOST IN VERY DEEP NETWORKS.
# Because the vanishing effect is EXPONENTIAL in the number of
# layers. With ten layers the shrinking of the gradient is
# negligible; with a hundred layers it is severe enough to stop the
# early layers learning at all. Residual connections are precisely
# what made networks of a hundred layers or more trainable, and they
# are the reason very deep architectures became practical.


# =====================================================
# Forward pass test
# =====================================================

if __name__ == "__main__":

    model = ResNetLike(num_classes=10)

    # a batch of 2 random images of the required input size
    x = torch.rand(2, 3, 224, 224)
    y = model(x)

    print("input shape :", x.shape)
    print("output shape:", y.shape)      # expected: torch.Size([2, 10])

    n_params = sum(p.numel() for p in model.parameters())
    print("number of parameters:", n_params)
