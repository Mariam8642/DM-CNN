#Running the standard CNN

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from lib.config import config_loader
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
CFG = os.path.join(BASE_PATH, "cfg", "training_config.cfg")
cfg = config_loader(CFG)

SEED = int(cfg.seed_number)
EPOCHS = 20
VALIDATION_FRACTION = 0.20
IMAGE_TREE = "image2d_image2d_binary_tree"

OUTPUT_DIR = cfg.output_directory
WEIGHTS_DIR = cfg.weights_directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataset(input_file, augment=False):
    return mpid_data_binary.MPID_Dataset(
        input_file,
        IMAGE_TREE,
        "cpu",
        plane=0,
        augment=augment
    )


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_events = 0

    with torch.no_grad():
        for x_batch, y_batch, info, nevents in loader:
            x_batch = x_batch.to(device, dtype=torch.float32).reshape(-1, 1, 512, 512)
            y_batch = y_batch.to(device, dtype=torch.float32)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_events += batch_size

    return total_loss / total_events


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_events = 0

    for x_batch, y_batch, info, nevents in loader:
        x_batch = x_batch.to(device, dtype=torch.float32).reshape(-1, 1, 512, 512)
        y_batch = y_batch.to(device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_events += batch_size

    return total_loss / total_events


def plot_loss_curves(train_losses, val_losses, output_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Training loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Normal CNN: training and validation loss")
    plt.xticks(list(epochs))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    set_seed(SEED)
    print("Device:", device)

    train_dataset = create_dataset(cfg.input_train, augment=bool(cfg.augment))
    val_dataset = create_dataset(cfg.input_train, augment=False)

    indices = torch.randperm(
        len(train_dataset),
        generator=torch.Generator().manual_seed(SEED)
    ).tolist()
    val_size = max(1, int(VALIDATION_FRACTION * len(indices)))

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=int(cfg.batch_size_train),
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=int(cfg.batch_size_test),
        shuffle=False,
        num_workers=0
    )

    # Fixed hyperparameters: no Optuna search.
    model = mpid_net_binary.MPID(
        dropout=0.3,
        num_classes=2
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    model_path = os.path.join(WEIGHTS_DIR, "DM-CNN_normal_best.pwf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.5f} | "
            f"Val loss: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation_loss": val_loss,
                    "dropout": 0.3,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-5
                },
                model_path
            )

    plot_loss_curves(
        train_losses,
        val_losses,
        os.path.join(OUTPUT_DIR, "normal_cnn_loss_curve.png")
    )

    with open(os.path.join(OUTPUT_DIR, "normal_cnn_history.json"), "w") as f:
        json.dump(
            {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "best_val_loss": best_val_loss
            },
            f,
            indent=4
        )

    print("Saved best model:", model_path)
    print("Saved loss curve:", os.path.join(OUTPUT_DIR, "normal_cnn_loss_curve.png"))
