import os
import sys
import json
import random
import numpy as np
import time
import csv

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

# #to run a test
# QUICK = os.environ.get("QUICK") == "1"
# EPOCHS = 1 if QUICK else 20

QUICK = os.environ.get("QUICK") == "1"
EPOCHS = 1 if QUICK else int(os.environ.get("EPOCHS", "5"))

EARLY_STOPPING_PATIENCE = int(
    os.environ.get("EARLY_STOPPING_PATIENCE", "5")
)

MIN_DELTA = float(
    os.environ.get("MIN_DELTA", "0.0")
)

if EPOCHS < 1:
    raise ValueError("EPOCHS must be at least 1.")
if EARLY_STOPPING_PATIENCE < 1:
    raise ValueError("EARLY_STOPPING_PATIENCE must be at least 1.")
if MIN_DELTA < 0:
    raise ValueError("MIN_DELTA must be non-negative.")

# --------------------------------------------------
# Fixed hyperparameters
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
print("Early-stopping patience:", EARLY_STOPPING_PATIENCE)
print("Minimum AUC improvement:", MIN_DELTA)
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
if len(val_indices) == 0:
    raise ValueError("Validation subset is empty.")

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

def plot_validation_auc_curve(val_aucs, out_path):
    epochs_axis = list(range(1, len(val_aucs) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs_axis,
        val_aucs,
        label="Validation AUC",
        marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("Standard CNN: validation AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved validation AUC curve:", out_path)

def save_misclassified_events(
    model,
    loader,
    output_path,
    threshold=0.5
):
    model.eval()
    rows = []

    with torch.no_grad():
        for x_batch, y_batch, info, nevents in loader:
            x_batch = x_batch.to(
                device,
                dtype=torch.float32
            ).reshape(-1, 1, 512, 512)

            y_batch = y_batch.to(
                device,
                dtype=torch.float32
            )

            logits = model(x_batch)

            signal_scores = torch.sigmoid(logits)[:, 0]
            scores = signal_scores.cpu().numpy()

            true_signal = (
                y_batch[:, 0].cpu().numpy() >= 0.5
            )
            predicted_signal = scores >= threshold

            for i in range(len(scores)):
                if true_signal[i] == predicted_signal[i]:
                    continue

                if true_signal[i] and not predicted_signal[i]:
                    classification = "false_negative"
                elif not true_signal[i] and predicted_signal[i]:
                    classification = "false_positive"
                else:
                    classification = "unknown"

                rows.append(
                    {
                        "event_index": len(rows),
                        "true_signal": int(true_signal[i]),
                        "predicted_signal": int(predicted_signal[i]),
                        "signal_score": float(scores[i]),
                        "classification": classification,
                    }
                )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_index",
                "true_signal",
                "predicted_signal",
                "signal_score",
                "classification",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved {len(rows)} misclassified events to "
        f"{output_path}"
    )

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
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = os.path.join(WEIGHTS_DIR, "DM-CNN_standard_best.pwf")

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
        if val_auc > best_val_auc + MIN_DELTA:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hyperparameters": HYPERPARAMS,
                    "validation_auc": best_val_auc,
                    "epoch": best_epoch,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "min_delta": MIN_DELTA,
                },
                best_model_path
            )

            print(
                f"  -> New best val AUC ({best_val_auc:.5f}), "
                f"checkpoint saved to {best_model_path}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"  -> No meaningful improvement for "
                f"{epochs_without_improvement} epoch(s)"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best epoch: {best_epoch}; "
                f"best validation AUC: {best_val_auc:.5f}"
            )
            break

    if not os.path.exists(best_model_path):
        raise RuntimeError("No best model checkpoint was saved.")

    print("\nSaved best model:", best_model_path)
    print("Best epoch:", best_epoch)
    print("Best validation AUC:", best_val_auc)
    # Save training curves.
    loss_curve_path = os.path.join(
        OUTPUT_DIR,
        "training_loss_curve.png"
    )
    # Loss curves + per-epoch history
    plot_loss_curves(
        train_losses,
        val_losses,
        os.path.join(OUTPUT_DIR, "training_loss_curve_standard.png")
    )

    auc_curve_path = os.path.join(
        OUTPUT_DIR,
        "training_validation_auc_curve_standard.png"
    )
    plot_validation_auc_curve(
        val_aucs,
        auc_curve_path
    )
    history_path = os.path.join(
        OUTPUT_DIR,
        "training_history_standard.json"
    )
    with open(history_path, "w") as f:
        json.dump(
            {
                "hyperparameters": HYPERPARAMS,
                "train_loss": train_losses,
                "val_loss": val_losses,
                "val_auc": val_aucs,
                "best_val_auc": best_val_auc,
                "best_epoch": best_epoch,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "min_delta": MIN_DELTA,
            },
            f,
            indent=4
        )
import os
import json
import random
import numpy as np

# Load the existing training configuration.
from lib.config import config_loader

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
CFG = os.path.join(BASE_PATH, "cfg", "training_config.cfg")
cfg = config_loader(CFG)

# Configure GPU before importing torch.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPUID)

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary


# --------------------------------------------------
# EXPERIMENT SETTINGS
# --------------------------------------------------

QUICK = os.environ.get("QUICK", "0") == "1"

# QUICK reduces the default number of epochs, but EPOCHS can still be
# overridden explicitly from the shell.
EPOCHS = int(
    os.environ.get("EPOCHS", "1" if QUICK else "5")
)

EARLY_STOPPING_PATIENCE = int(
    os.environ.get("EARLY_STOPPING_PATIENCE", "5")
)

MIN_DELTA = float(
    os.environ.get("MIN_DELTA", "0.0")
)

if EPOCHS < 1:
    raise ValueError("EPOCHS must be at least 1.")
if EARLY_STOPPING_PATIENCE < 1:
    raise ValueError("EARLY_STOPPING_PATIENCE must be at least 1.")
if MIN_DELTA < 0:
    raise ValueError("MIN_DELTA must be non-negative.")

# --------------------------------------------------
# SINGLE SET OF HYPERPARAMETERS
# --------------------------------------------------

HYPERPARAMS = {
    "dropout": 0.3,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
}

VALIDATION_FRACTION = 0.20
SEED = int(cfg.seed_number)
IMAGE_TREE = "image2d_image2d_binary_tree"

OUTPUT_DIR = cfg.output_directory
WEIGHTS_DIR = cfg.weights_directory

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("GPU available:", torch.cuda.is_available())
print("Epochs:", EPOCHS)
print("Early-stopping patience:", EARLY_STOPPING_PATIENCE)
print("Minimum AUC improvement:", MIN_DELTA)
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
        plane=0,
        augment=augment
    )


# Separate dataset objects so validation images are never augmented.
train_dataset = create_dataset(
    cfg.input_train,
    augment=bool(cfg.augment)
)

val_dataset = create_dataset(
    cfg.input_train,
    augment=False
)

dataset_size = len(train_dataset)

if dataset_size < 2:
    raise ValueError("Training dataset contains fewer than 2 events.")

# Deterministic 80/20 split.
generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(
    dataset_size,
    generator=generator
).tolist()

val_size = max(1, int(VALIDATION_FRACTION * dataset_size))
val_indices = indices[:val_size]
train_indices = indices[val_size:]

if QUICK:
    train_indices = train_indices[:320]
    val_indices = val_indices[:160]

if len(train_indices) == 0:
    raise ValueError("Training subset is empty.")
if len(val_indices) == 0:
    raise ValueError("Validation subset is empty.")

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
# EVALUATION: LOSS + ROC-AUC
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
                device,
                dtype=torch.float32
            ).reshape(-1, 1, 512, 512)

            y_batch = y_batch.to(
                device,
                dtype=torch.float32
            )

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_events += batch_size

            # Class 0 represents signal.
            signal_scores = torch.sigmoid(logits)[:, 0]
            all_scores.extend(signal_scores.cpu().numpy())
            all_labels.extend(y_batch[:, 0].cpu().numpy())

    if total_events == 0:
        raise ValueError("Evaluation loader is empty.")

    all_labels = np.asarray(all_labels)
    all_scores = np.asarray(all_scores)

    if len(np.unique(all_labels)) < 2:
        raise ValueError(
            "Evaluation dataset must contain both signal and background."
        )

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
            device,
            dtype=torch.float32
        ).reshape(-1, 1, 512, 512)

        y_batch = y_batch.to(
            device,
            dtype=torch.float32
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_events += batch_size

    if total_events == 0:
        raise ValueError("Training loader is empty.")

    return total_loss / total_events


# --------------------------------------------------
# PLOTTING
# --------------------------------------------------

def plot_loss_curves(train_losses, val_losses, out_path):
    epochs_axis = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, train_losses, label="Train loss", marker="o")
    plt.plot(epochs_axis, val_losses, label="Validation loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Standard CNN: training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved loss curve:", out_path)


def plot_validation_auc_curve(val_aucs, out_path):
    epochs_axis = list(range(1, len(val_aucs) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs_axis,
        val_aucs,
        label="Validation AUC",
        marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("Standard CNN: validation AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved validation AUC curve:", out_path)


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
    best_epoch = 0
    epochs_without_improvement = 0

    best_model_path = os.path.join(
        WEIGHTS_DIR,
        "DM-CNN_best.pwf"
    )

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_auc = evaluate(
            model,
            val_loader,
            criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Train loss: {train_loss:.5f} "
            f"| Val loss: {val_loss:.5f} "
            f"| Val AUC: {val_auc:.5f}"
        )

        # Save the best checkpoint based on validation AUC.
        if val_auc > best_val_auc + MIN_DELTA:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hyperparameters": HYPERPARAMS,
                    "validation_auc": best_val_auc,
                    "epoch": best_epoch,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "min_delta": MIN_DELTA,
                },
                best_model_path
            )

            print(
                f"  -> New best val AUC ({best_val_auc:.5f}), "
                f"checkpoint saved to {best_model_path}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"  -> No meaningful improvement for "
                f"{epochs_without_improvement} epoch(s)"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best epoch: {best_epoch}; "
                f"best validation AUC: {best_val_auc:.5f}"
            )
            break

    if not os.path.exists(best_model_path):
        raise RuntimeError("No best model checkpoint was saved.")

    print("\nSaved best model:", best_model_path)
    print("Best epoch:", best_epoch)
    print("Best validation AUC:", best_val_auc)

    # Save training curves.
    loss_curve_path = os.path.join(
        OUTPUT_DIR,
        "training_loss_curve_standard.png"
    )
    plot_loss_curves(
        train_losses,
        val_losses,
        loss_curve_path
    )

    auc_curve_path = os.path.join(
        OUTPUT_DIR,
        "training_val_auc_standard.png"
    )
    plot_validation_auc_curve(
        val_aucs,
        auc_curve_path
    )

    history_path = os.path.join(
        OUTPUT_DIR,
        "training_history_standard.json"
    )
    with open(history_path, "w") as f:
        json.dump(
            {
                "hyperparameters": HYPERPARAMS,
                "train_loss": train_losses,
                "val_loss": val_losses,
                "val_auc": val_aucs,
                "best_val_auc": best_val_auc,
                "best_epoch": best_epoch,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "min_delta": MIN_DELTA,
            },
            f,
            indent=4
        )

    print("Saved training history:", history_path)

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

    test_results_path = os.path.join(
        OUTPUT_DIR,
        "test_results_standard.json"
    )
    with open(test_results_path, "w") as f:
        json.dump(
            {
                "validation_auc": best_val_auc,
                "test_auc": test_auc,
                "test_loss": test_loss,
                "best_epoch": best_epoch,
                "hyperparameters": HYPERPARAMS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "min_delta": MIN_DELTA,
            },
            f,
            indent=4
        )
    print("Saved test results:", test_results_path)