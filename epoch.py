# -*- coding: utf-8 -*-
"""
Image and Video Processing for Autonomous Driving - TRACK A
Exercise 6 - Training Loop and Validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =====================================================
# 6a - training function for a single epoch
# =====================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one full training epoch.

    Iterates over the DataLoader, performs the forward pass, computes
    the Cross-Entropy loss, resets the gradients, performs
    backpropagation and updates the weights.

    Returns the average training loss over the epoch.
    """

    # model.train() puts the model in TRAINING mode. This matters for
    # layers that behave differently during training and inference:
    # BatchNorm updates its running statistics, and Dropout is active.
    model.train()

    running_loss = 0.0
    n_batches = 0

    for images, labels in dataloader:

        # move the batch to the device the model lives on
        images = images.to(device)
        labels = labels.to(device)

        # ---- forward pass ----
        outputs = model(images)

        # ---- loss ----
        # nn.CrossEntropyLoss expects the RAW class scores (logits),
        # not probabilities: it applies the softmax internally.
        loss = criterion(outputs, labels)

        # ---- reset the gradients ----
        # PyTorch ACCUMULATES gradients by default, so they must be
        # cleared before each backward pass. Without this line the
        # gradients of all previous batches would be added together
        # and the weight updates would be wrong.
        optimizer.zero_grad()

        # ---- backpropagation ----
        # computes the gradient of the loss with respect to every
        # parameter of the model
        loss.backward()

        # ---- update the weights ----
        # applies the update rule of the optimiser using the
        # gradients just computed
        optimizer.step()

        # loss is a tensor; .item() extracts the plain Python number.
        # Accumulating the tensor itself would keep the whole
        # computational graph alive and waste memory.
        running_loss = running_loss + loss.item()
        n_batches = n_batches + 1

    average_loss = running_loss / n_batches

    return average_loss


# =====================================================
# 6b - validation function for a single epoch
# =====================================================

def validate_one_epoch(model, dataloader, criterion, device):
    """
    Runs one full validation epoch.

    Computes the loss and the classification accuracy over the
    validation DataLoader and returns both.

    Gradient computation is disabled throughout.
    """

    # model.eval() puts the model in EVALUATION mode: BatchNorm uses
    # its stored running statistics instead of the statistics of the
    # current batch, and Dropout is switched off. Without this the
    # validation results would depend on the batch composition and
    # would not be reproducible.
    model.eval()

    running_loss = 0.0
    n_batches = 0
    n_correct = 0
    n_total = 0

    # torch.no_grad() disables the construction of the computational
    # graph. During validation no weights are updated, so gradients
    # are never needed: switching them off saves memory and time, and
    # guarantees that the validation data cannot influence the model.
    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            # ---- forward pass only ----
            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss = running_loss + loss.item()
            n_batches = n_batches + 1

            # ---- classification accuracy ----
            # outputs has shape (B, num_classes) and holds one score
            # per class. torch.max(outputs, 1) takes the maximum
            # along dimension 1, the class dimension, and returns the
            # values and their indices. The INDEX of the largest
            # score is the predicted class, so only that is kept.
            _, predicted = torch.max(outputs, 1)

            n_correct = n_correct + (predicted == labels).sum().item()
            n_total = n_total + labels.size(0)

    average_loss = running_loss / n_batches
    accuracy = n_correct / n_total

    return average_loss, accuracy


# =====================================================
# 6c - training loop over 30 epochs
# =====================================================

def run_training(model, train_loader, val_loader, num_epochs=30):
    """
    Wraps the training and validation functions in a loop and prints
    the epoch number, the training loss, the validation loss and the
    validation accuracy at each epoch.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Cross-Entropy loss, the standard choice for multi-class
    # classification
    criterion = nn.CrossEntropyLoss()

    # Stochastic Gradient Descent with momentum. Momentum accumulates
    # a moving average of past gradients, which damps the
    # oscillations of plain SGD in narrow valleys of the loss surface
    # and accelerates convergence along consistent directions.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(num_epochs):

        train_loss = train_one_epoch(model, train_loader,
                                     criterion, optimizer, device)

        val_loss, val_accuracy = validate_one_epoch(model, val_loader,
                                                    criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"| train loss: {train_loss:.4f} "
              f"| val loss: {val_loss:.4f} "
              f"| val accuracy: {val_accuracy:.4f}")


# -----------------------------------------------------
# COMMENT (6c): what does overfitting look like on loss
# curves, and name two regularisation techniques that
# could mitigate it
# -----------------------------------------------------
#
# WHAT OVERFITTING LOOKS LIKE.
# Overfitting occurs when the model stops learning the general
# structure of the data and begins to memorise the specific training
# examples, including their noise. On the loss curves it has a very
# characteristic signature.
#
# During the first epochs both the training loss and the validation
# loss decrease together: the model is learning genuine structure
# that generalises. At some point the two curves SEPARATE. The
# training loss keeps decreasing, and can be driven arbitrarily close
# to zero, while the validation loss stops decreasing, flattens out
# and then starts to INCREASE again. The validation accuracy
# correspondingly stops improving and begins to fall.
#
# The distinctive feature is therefore the growing GAP between the
# two curves, together with the upturn of the validation loss. The
# epoch at which the validation loss reaches its minimum is the point
# of best generalisation; everything after it is memorisation.
#
# For contrast, if both losses were still high and moving together,
# the problem would be underfitting, not overfitting, and the remedy
# would be the opposite one: more capacity or longer training.
#
# TWO REGULARISATION TECHNIQUES.
#
# 1. DROPOUT. During training, a randomly chosen fraction of the
#    units is switched off at every forward pass, and the remaining
#    activations are rescaled to compensate. The network can
#    therefore never rely on any single unit or on a fixed
#    co-adaptation of a few units, so it is forced to build
#    redundant, more robust representations. Dropout is most
#    effective in the fully connected layers, which contain the
#    majority of the parameters and are the most prone to
#    memorisation. At evaluation time it is switched off, which is
#    exactly what model.eval() does in the validation function above.
#
# 2. DATA AUGMENTATION. The training set is enlarged artificially by
#    applying label-preserving transformations to the images -
#    random rotations, translations, horizontal flips, scaling or
#    changes in brightness and contrast. The model then sees a
#    different version of each image at every epoch, which makes
#    memorising individual examples much harder and teaches the model
#    invariances that genuinely hold in the data. It attacks the
#    cause of overfitting directly, since overfitting is fundamentally
#    a consequence of having too little data relative to the number
#    of parameters.
#
# A third and complementary technique, visible directly on the curves
# described above, is EARLY STOPPING: training is halted at the epoch
# where the validation loss reaches its minimum, before the gap
# between the two curves begins to widen.


# =====================================================
# Self-test with synthetic data
# =====================================================
#
# This block is only a demonstration that the two functions and the
# loop work. It builds a very small random dataset and a tiny model
# so that the code can be executed and verified quickly.

if __name__ == "__main__":

    torch.manual_seed(0)

    # ---- a small synthetic classification dataset ----
    n_train, n_val, n_features, n_classes = 256, 64, 20, 10

    x_train = torch.rand(n_train, n_features)
    y_train = torch.randint(0, n_classes, (n_train,))
    x_val = torch.rand(n_val, n_features)
    y_val = torch.randint(0, n_classes, (n_val,))

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    # shuffle=True for training, so the order of the samples differs
    # at every epoch and the gradient estimates are not correlated
    # with the ordering of the data.
    # shuffle=False for validation, because the order has no effect
    # on the result and a fixed order makes the evaluation
    # reproducible.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ---- a small model, just for the test ----
    demo_model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, n_classes),
    )

    # a short run so the test finishes quickly;
    # the exercise asks for 30 epochs
    run_training(demo_model, train_loader, val_loader, num_epochs=5)
