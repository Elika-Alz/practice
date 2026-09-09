# -*- coding: utf-8 -*-
"""TRACK B - Exercise 6 - Training Loop and Validation"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ================= 6a: training for one epoch =================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Forward pass, Cross-Entropy loss, reset gradients,
    backpropagation, weight update. Returns the average loss."""

    # training mode: BatchNorm updates its statistics, Dropout is on
    model.train()

    running_loss = 0.0
    n_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)                 # forward pass
        loss = criterion(outputs, labels)       # CrossEntropy: needs logits

        optimizer.zero_grad()   # PyTorch accumulates gradients: clear them
        loss.backward()         # backpropagation
        optimizer.step()        # update the weights

        running_loss = running_loss + loss.item()   # .item(): tensor -> number
        n_batches = n_batches + 1

    return running_loss / n_batches


# ================= 6b: validation for one epoch =================

def validate_one_epoch(model, dataloader, criterion, device):
    """Computes loss and classification accuracy.
    Gradient computation is disabled."""

    # evaluation mode: BatchNorm uses its stored statistics,
    # Dropout is switched off
    model.eval()

    running_loss = 0.0
    n_batches = 0
    n_correct = 0
    n_total = 0

    # no gradients are needed here, so the graph is not built:
    # this saves memory and guarantees the model is not updated
    with torch.no_grad():

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss = running_loss + loss.item()
            n_batches = n_batches + 1

            # outputs is (B, num_classes); the INDEX of the largest
            # score is the predicted class
            _, predicted = torch.max(outputs, 1)
            n_correct = n_correct + (predicted == labels).sum().item()
            n_total = n_total + labels.size(0)

    return running_loss / n_batches, n_correct / n_total


# ================= 6c: loop over 30 epochs =================

def run_training(model, train_loader, val_loader, num_epochs=30):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # momentum accumulates a moving average of past gradients, which
    # damps oscillations and speeds up convergence
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader,
                                     criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader,
                                               criterion, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"| train loss: {train_loss:.4f} "
              f"| val loss: {val_loss:.4f} "
              f"| val accuracy: {val_acc:.4f}")


# COMMENT (6c): overfitting on these curves, and how Dropout and data
# augmentation mitigate it
#
# WHAT IT LOOKS LIKE. In the first epochs the training loss and the
# validation loss decrease TOGETHER: the model is learning structure
# that generalises. Then the curves SEPARATE: the training loss keeps
# falling towards zero while the validation loss flattens and then
# starts to RISE, and the validation accuracy stops improving and
# begins to fall.
#
# The signature is therefore the growing GAP between the two losses
# plus the upturn of the validation loss. The epoch of minimum
# validation loss is the point of best generalisation; after it the
# model is memorising the training examples, including their noise.
#
# DROPOUT. During training a random fraction of the units is switched
# off at each forward pass and the remaining activations are rescaled
# to compensate. The network can never rely on a single unit or on a
# fixed co-adaptation of a few units, so it is forced to build
# redundant representations, which generalise better. Equivalently it
# trains an implicit ensemble of thinned sub-networks sharing their
# weights. It is most effective in the fully connected layers, which
# hold most of the parameters, and it is switched off at evaluation
# time - which is what model.eval() does above.
#
# DATA AUGMENTATION. The training set is enlarged with
# label-preserving transformations: rotations, translations, flips,
# small scale changes, brightness variations. A rotated cat is still
# a cat, so the label is unchanged while the pixels are not. The
# model therefore sees a different version of each image at every
# epoch, which makes memorising individual examples much harder, and
# it explicitly learns the invariances that genuinely hold in the
# data. It attacks the cause directly, since overfitting comes from
# having too little data relative to the number of parameters. It is
# applied only to the training set, never to the validation set.
#
# A third technique, visible directly on the curves, is EARLY
# STOPPING: halt training at the epoch of minimum validation loss.


# ================= self-test with synthetic data =================

if __name__ == "__main__":

    torch.manual_seed(0)

    n_train, n_val, n_features, n_classes = 256, 64, 20, 10
    x_train = torch.rand(n_train, n_features)
    y_train = torch.randint(0, n_classes, (n_train,))
    x_val = torch.rand(n_val, n_features)
    y_val = torch.randint(0, n_classes, (n_val,))

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    # shuffle=True for training so the sample order changes every
    # epoch; shuffle=False for validation so the result is reproducible
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Dropout shown here in the fully connected part, where it is
    # most effective
    demo_model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, n_classes),
    )

    run_training(demo_model, train_loader, val_loader, num_epochs=5)
