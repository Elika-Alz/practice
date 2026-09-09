# -*- coding: utf-8 -*-
"""TRACK B - Exercise 5 - Residual Architectures (bottleneck)"""

import torch
import torch.nn as nn


# ================= 5a: bottleneck ResBlock =================

class BottleneckResBlock(nn.Module):
    """Conv(C,C//4,1) -> BN -> ReLU -> Conv(C//4,C//4,3,p=1) -> BN
    -> ReLU -> Conv(C//4,C,1) -> BN, skip added BEFORE the final ReLU.

    Squeeze - work - expand: the 1x1 convs change only the channel
    count (they mix across channels, not space), and the 3x3 with
    padding=1 preserves H and W. The block therefore returns the same
    shape it received, which is what makes the addition possible."""

    def __init__(self, channels):
        super().__init__()
        mid = channels // 4              # 64 // 4 = 16

        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x                     # keep the input for the skip

        out = self.conv1(x)              # squeeze: C -> C/4
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)            # work: 3x3 at reduced width
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)            # expand: C/4 -> C
        out = self.bn3(out)

        out = out + identity             # skip connection
        out = self.relu(out)             # final ReLU, after the addition
        return out


# ================= 5b: ConvBlock =================

class ConvBlock(nn.Module):
    """Conv(x,64,7,'same') -> BN -> ReLU -> Conv(64,64,5,'same')
    -> ReLU -> MaxPool(2,2). Both convs keep H and W; only the
    MaxPool halves them."""

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


# ================= 5b: ResNet-like network =================

class ResNetLike(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.convblock1 = ConvBlock(3)
        self.convblock2 = ConvBlock(64)
        self.resblock1 = BottleneckResBlock(64)
        self.resblock2 = BottleneckResBlock(64)
        self.flatten = nn.Flatten()

        # ---- INPUT DIMENSION OF THE FIRST FC LAYER ----
        # H_out = floor((H_in + 2P - K)/S) + 1
        #   padding='same', stride 1        -> unchanged
        #   Conv 3x3 padding=1              -> unchanged
        #   MaxPool(2,2): K=2, S=2, P=0     -> H/2
        #
        # input                     (B,  3, 224, 224)
        # ConvBlock(3):
        #   two convs, 'same'    -> (B, 64, 224, 224)
        #   MaxPool: 224/2=112   -> (B, 64, 112, 112)
        # ConvBlock(64):
        #   two convs, 'same'    -> (B, 64, 112, 112)
        #   MaxPool: 112/2=56    -> (B, 64,  56,  56)
        # BottleneckResBlock(64) -> (B, 64,  56,  56)
        # BottleneckResBlock(64) -> (B, 64,  56,  56)
        # Flatten: 64 * 56 * 56
        #     64 * 56   = 3584
        #     3584 * 56 = 200704   -> (B, 200704)
        #
        # => FIRST FC INPUT DIMENSION = 200704
        # Only the two MaxPools changed the size: 224 -> 112 -> 56.

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
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)              # no activation: raw logits
        return out


# COMMENT (5c): classic vs bottleneck ResBlock
#
# ARCHITECTURE. A CLASSIC ResBlock uses two 3x3 convolutions that
# keep the channel count constant: Conv(C,C,3) -> BN -> ReLU ->
# Conv(C,C,3) -> BN.
#
# A BOTTLENECK ResBlock uses three: a 1x1 that REDUCES C to C/4, a
# 3x3 that works at that reduced width, and a 1x1 that EXPANDS back
# to C so the skip connection still matches. The 1x1 convolutions
# look at a single pixel and mix across CHANNELS only, which is how
# the channel count is changed cheaply. Both take and return the same
# shape (B, C, H, W).
#
# COMPUTATIONAL COST. The saving comes from doing the expensive 3x3
# on a quarter of the channels. With C = 64:
#
#   classic:    2 * (3*3*64*64) = 73728 weights
#   bottleneck: 1*1*64*16 =  1024
#               3*3*16*16 =  2304
#               1*1*16*64 =  1024
#               total     =  4352
#
#   73728 / 4352 = about 17x fewer parameters, for the same shapes.
#
# WHY PREFERRED IN DEEPER NETWORKS. The per-block cost is multiplied
# by the number of blocks, so the classic design becomes
# prohibitively expensive in deep networks. Since the whole benefit
# of residual architectures comes from DEPTH, spending a fixed budget
# on MORE BLOCKS rather than WIDER blocks gives better accuracy - and
# the bottleneck is what makes that trade possible.


# ================= forward pass test =================

if __name__ == "__main__":
    model = ResNetLike(num_classes=10)
    x = torch.rand(2, 3, 224, 224)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)          # expected torch.Size([2, 10])
