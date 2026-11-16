import os
import sys
import json
import random
import numpy as np

# Load the existing training configuration
from lib.config import config_loader

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
CFG = os.path.join(BASE_PATH, "cfg", "training_config.cfg")
cfg = config_loader(CFG)

# Configure GPU before importing torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPUID)

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")  # safe for headless/remote runs (no display needed)
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary

# to run a test
QUICK = os.environ.get("QUICK") == "1"
EPOCHS = 1 if QUICK else 20

# --------------------------------------------------
# SINGLE SET OF HYPERPARAMETERS  (edit these)
# --------------------------------------------------
HYPERPARAMS = {
    "dropout": 0.3,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
}
# --------------------------------------------------

VALIDATION_FRACTION = 0.20
SEED = int(cfg.seed_number)

IMAGE_TREE = "image2d_image2d_binary_tree"

OUTPUT_DIR = cfg.output_directory
WEIGHTS_DIR = cfg.weights_directory

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("GPU available:", torch.cuda.is_available())
print("Epochs:", EPOCHS)
print("Hyperparameters:", HYPERPARAMS)


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------
# DATASET
# --------------------------------------------------

def create_dataset(input_file, augment=False):

    return mpid_data_binary.MPID_Dataset(
        input_file,
        IMAGE_TREE,
        "cpu",
        # plane=int(cfg.plane),
        plane=0,
        augment=augment
    )


# Separate dataset objects so validation images are never augmented.
train_dataset = create_dataset(cfg.input_train, augment=bool(cfg.augment))
val_dataset = create_dataset(cfg.input_train, augment=False)

dataset_size = len(train_dataset)

if dataset_size < 2:
    raise ValueError("Training dataset contains fewer than 2 events.")

# Deterministic 80/20 split
generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(dataset_size, generator=generator).tolist()

val_size = max(1, int(VALIDATION_FRACTION * dataset_size))

val_indices = indices[:val_size]
train_indices = indices[val_size:]

if QUICK:
    train_indices = train_indices[:320]
    val_indices = val_indices[:160]

if len(train_indices) == 0:
    raise ValueError("Training subset is empty.")

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=int(cfg.batch_size_train),
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_subset,
    batch_size=int(cfg.batch_size_test),
    shuffle=False,
    num_workers=0
)

print("Total training events:", dataset_size)
print("Training subset:", len(train_subset))
print("Validation subset:", len(val_subset))


# --------------------------------------------------
# EVALUATION: loss + ROC-AUC
# --------------------------------------------------

def evaluate(model, loader, criterion):

    model.eval()

    all_scores = []
    all_labels = []

    total_loss = 0.0
    total_events = 0

    with torch.no_grad():

        for x_batch, y_batch, info, nevents in loader:

            x_batch = x_batch.to(
                device, dtype=torch.float32
            ).reshape(-1, 1, 512, 512)

            y_batch = y_batch.to(device, dtype=torch.float32)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_events += batch_size

            # Class 0 represents signal (run == 100).
            # Class 1 represents background.
            signal_scores = torch.sigmoid(logits)[:, 0]

            all_scores.extend(signal_scores.cpu().numpy())
            all_labels.extend(y_batch[:, 0].cpu().numpy())

    if total_events == 0:
        raise ValueError("Loader is empty.")

    all_labels = np.asarray(all_labels)
    all_scores = np.asarray(all_scores)

    if len(np.unique(all_labels)) < 2:
        raise ValueError("Dataset must contain both signal and background.")

    auc = roc_auc_score(all_labels, all_scores)
    average_loss = total_loss / total_events

    return average_loss, auc


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0.0
    total_events = 0

    for x_batch, y_batch, info, nevents in loader:

        x_batch = x_batch.to(
            device, dtype=torch.float32
        ).reshape(-1, 1, 512, 512)

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


# --------------------------------------------------
# LOSS CURVE PLOTTING
# --------------------------------------------------

def plot_loss_curves(train_losses, val_losses, out_path):

    epochs_axis = list(range(1, len(train_losses) + 1))

    plt.figure()
    plt.plot(epochs_axis, train_losses, label="Train loss", marker="o")
    plt.plot(epochs_axis, val_losses, label="Val loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model training: loss curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()

    print("Saved loss curve plot:", out_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    set_seed(SEED)

    model = mpid_net_binary.MPID(
        dropout=HYPERPARAMS["dropout"],
        num_classes=2
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=HYPERPARAMS["learning_rate"],
        weight_decay=HYPERPARAMS["weight_decay"]
    )

    train_losses = []
    val_losses = []
    val_aucs = []

    best_val_auc = -float("inf")
    best_model_path = os.path.join(WEIGHTS_DIR, "DM-CNN_best.pwf")

    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_auc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Train loss: {train_loss:.5f} "
            f"| Val loss: {val_loss:.5f} "
            f"| Val AUC: {val_auc:.5f}"
        )

        # Keep the checkpoint with the best validation AUC.
        if val_auc > best_val_auc:
            best_val_auc = val_auc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hyperparameters": HYPERPARAMS,
                    "validation_auc": best_val_auc,
                    "epoch": epoch + 1
                },
                best_model_path
            )

            print(
                f"  -> New best val AUC ({best_val_auc:.5f}), "
                f"checkpoint saved to {best_model_path}"
            )

    print("\nSaved best model:", best_model_path)
    print("Best validation AUC:", best_val_auc)

    # Loss curves + per-epoch history
    plot_loss_curves(
        train_losses,
        val_losses,
        os.path.join(OUTPUT_DIR, "training_loss_curve.png")
    )

    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(
            {
                "hyperparameters": HYPERPARAMS,
                "train_loss": train_losses,
                "val_loss": val_losses,
                "val_auc": val_aucs,
                "best_val_auc": best_val_auc
            },
            f,
            indent=4
        )

    # Reload the best checkpoint before the test evaluation.
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # --------------------------------------------------
    # TEST SET EVALUATION (ONCE)
    # --------------------------------------------------

    print("\nEvaluating model on independent test set...")

    test_dataset = create_dataset(cfg.input_test, augment=False)
    if QUICK:
        idx = torch.randperm(len(test_dataset)).tolist()[:160]
        test_dataset = Subset(test_dataset, idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.batch_size_test),
        shuffle=False,
        num_workers=0
    )

    test_loss, test_auc = evaluate(model, test_loader, criterion)

    print("\nFINAL RESULTS")
    print("Best validation AUC:", best_val_auc)
    print("Independent test AUC:", test_auc)
    print("Independent test loss:", test_loss)

    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(
            {
                "validation_auc": best_val_auc,
                "test_auc": test_auc,
                "test_loss": test_loss,
                "hyperparameters": HYPERPARAMS
            },
            f,
            indent=4
        )