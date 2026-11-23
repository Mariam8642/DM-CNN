import os
import json
import random
import csv

import numpy as np

# Load the existing training configuration before importing project modules.
from lib.config import config_loader

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
CFG = os.path.join(BASE_PATH, "cfg", "training_config.cfg")
cfg = config_loader(CFG)

# Configure CUDA before importing torch.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPUID)

import torch
import torch.nn as nn
import torch.optim as optim
import optuna

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

# QUICK limits the dataset, but EPOCHS can still be overridden.
EPOCHS = int(
    os.environ.get("EPOCHS", "1" if QUICK else "50")
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
print("Maximum epochs per trial/final training:", EPOCHS)
print("Early-stopping patience:", EARLY_STOPPING_PATIENCE)
print("Minimum AUC improvement:", MIN_DELTA)


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


# Use separate dataset objects so validation images are never augmented.
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
# VALIDATION: ROC-AUC
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
# TRAINING FUNCTION
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

    plt.figure()
    plt.plot(epochs_axis, train_losses, label="Train loss", marker="o")
    plt.plot(epochs_axis, val_losses, label="Validation loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Final model training: loss curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved loss curve plot:", out_path)


def plot_validation_auc_curve(
    val_aucs,
    out_path,
    title="Final model validation AUC"
):
    epochs_axis = list(range(1, len(val_aucs) + 1))

    plt.figure()
    plt.plot(
        epochs_axis,
        val_aucs,
        label="Validation AUC",
        marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved validation AUC curve:", out_path)


# --------------------------------------------------
# OPTUNA OBJECTIVE
# --------------------------------------------------

def objective(trial):
    set_seed(SEED)

    dropout = trial.suggest_float("dropout", 0.1, 0.7)
    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-2,
        log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-2,
        log=True
    )

    print("\n" + "=" * 50)
    print("TRIAL:", trial.number)
    print("Dropout:", dropout)
    print("Learning rate:", learning_rate)
    print("Weight decay:", weight_decay)
    print("=" * 50)

    model = mpid_net_binary.MPID(
        dropout=dropout,
        num_classes=2
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_auc = -float("inf")
    epochs_without_improvement = 0

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

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.5f} | "
            f"Val loss: {val_loss:.5f} | "
            f"Val AUC: {val_auc:.5f}"
        )

        # Optuna pruning stops the entire trial when it is underperforming.
        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Ordinary early stopping stops this trial after no useful progress.
        if val_auc > best_auc + MIN_DELTA:
            best_auc = val_auc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping trial {trial.number} at epoch "
                f"{epoch + 1}; best validation AUC: {best_auc:.5f}"
            )
            break

    return best_auc


def extract_energy(info, n_events):
    possible_keys = (
        "energy",
        "Energy",
        "event_energy",
        "shower_energy",
        "E",
    )

    values_out = None

    if isinstance(info, dict):
        for key in possible_keys:
            if key in info:
                values = info[key]

                if torch.is_tensor(values):
                    values_out = values.detach().cpu().numpy().tolist()
                elif isinstance(values, np.ndarray):
                    values_out = values.tolist()
                elif isinstance(values, (list, tuple)):
                    values_out = list(values)
                break

    if values_out is None:
        return [float("nan")] * n_events

    values_out = list(values_out)
    if len(values_out) < n_events:
        values_out.extend([float("nan")] * (n_events - len(values_out)))

    return values_out[:n_events]


def save_misclassified_events(model, loader, output_path, threshold=0.5):
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
            true_signal = y_batch[:, 0].cpu().numpy() >= 0.5
            predicted_signal = scores >= threshold
            energies = extract_energy(info, len(scores))

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
                        "energy": energies[i],
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
                "energy",
                "true_signal",
                "predicted_signal",
                "signal_score",
                "classification",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} misclassified events to {output_path}")


# --------------------------------------------------
# RUN OPTUNA
# --------------------------------------------------

if __name__ == "__main__":
    set_seed(SEED)

    study_name = "dmcnn_quick" if QUICK else "dmcnn"
    storage_path = (
        "sqlite:////home/hep/ma2922/Project/DM-CNN/outputs/optuna.db"
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=10,
            interval_steps=2
        )
    )

    print("\nStarting Optuna optimisation...\n")

    study.optimize(
        objective,
        n_trials=1
    )

    print("\nOPTIMISATION FINISHED")
    print("Best validation AUC:", study.best_value)
    print("Best hyperparameters:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    # Save Optuna trial results
    trials_path = os.path.join(
        OUTPUT_DIR,
        "optuna_trials.csv"
    )

    study.trials_dataframe().to_csv(
        trials_path,
        index=False
    )

    print("Saved Optuna trials:", trials_path)

    history_path = os.path.join(
        OUTPUT_DIR,
        "final_training_history_optuna.json"
    )

    with open(history_path, "w") as f:
        json.dump(
            {
                "train_loss": final_train_losses,
                "val_loss": final_val_losses,
                "val_auc": final_val_aucs,
                "best_val_auc": best_final_val_auc,
                "best_epoch": best_epoch,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "min_delta": MIN_DELTA,
            },
            f,
            indent=4
        )

    print("Saved Optuna training history:", history_path)

    # --------------------------------------------------
    # FINAL TRAINING WITH EARLY STOPPING
    # --------------------------------------------------

    print("\nTraining final model with best hyperparameters...")
    set_seed(SEED)

    # Keep validation separate so early stopping is not evaluated on training data.
    final_train_loader = DataLoader(
        train_subset,
        batch_size=int(cfg.batch_size_train),
        shuffle=True,
        num_workers=0
    )

    best_params = study.best_params

    final_model = mpid_net_binary.MPID(
        dropout=best_params["dropout"],
        num_classes=2
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    final_optimizer = optim.Adam(
        final_model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )

    final_train_losses = []
    final_val_losses = []
    final_val_aucs = []

    best_final_val_auc = -float("inf")
    final_epochs_without_improvement = 0
    best_epoch = 0

    best_model_path = os.path.join(
        WEIGHTS_DIR,
        "DM-CNN_optuna_best.pwf"
    )

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            final_model,
            final_train_loader,
            criterion,
            final_optimizer
        )

        val_loss, val_auc = evaluate(
            final_model,
            val_loader,
            criterion
        )

        final_train_losses.append(train_loss)
        final_val_losses.append(val_loss)
        final_val_aucs.append(val_auc)

        print(
            f"Final training epoch {epoch + 1}/{EPOCHS} "
            f"| Train loss: {train_loss:.5f} "
            f"| Val loss: {val_loss:.5f} "
            f"| Val AUC: {val_auc:.5f}"
        )

        if val_auc > best_final_val_auc + MIN_DELTA:
            best_final_val_auc = val_auc
            best_epoch = epoch + 1
            final_epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": final_model.state_dict(),
                    "hyperparameters": best_params,
                    "validation_auc": best_final_val_auc,
                    "epoch": best_epoch,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "min_delta": MIN_DELTA,
                },
                best_model_path
            )

            print(
                f"  -> New best val AUC ({best_final_val_auc:.5f}), "
                f"checkpoint saved to {best_model_path}"
            )
        else:
            final_epochs_without_improvement += 1

        if final_epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping final training at epoch {epoch + 1}; "
                f"best epoch: {best_epoch}, "
                f"best validation AUC: {best_final_val_auc:.5f}"
            )
            break

    if not os.path.exists(best_model_path):
        raise RuntimeError("No best model checkpoint was saved.")

    print("\nSaved best model:", best_model_path)
    print("Best epoch:", best_epoch)
    print("Best epoch val AUC during final training:", best_final_val_auc)

    loss_curve_path = os.path.join(
        OUTPUT_DIR,
        "final_training_loss_curve_optuna.png"
    )

    plot_loss_curves(
        final_train_losses,
        final_val_losses,
        loss_curve_path
    )

    auc_curve_path = os.path.join(
        OUTPUT_DIR,
        "final_training_validation_auc_curve_optuna.png"
    )

    plot_validation_auc_curve(
        final_val_aucs,
        auc_curve_path,
        "Final model validation AUC (Optuna search)"
    )

    with open(
        os.path.join(OUTPUT_DIR, "final_training_history_optuna.json"),
        "w"
    ) as f:
        json.dump(
            {
                "train_loss": final_train_losses,
                "val_loss": final_val_losses,
                "val_auc": final_val_aucs,
                "best_val_auc": best_final_val_auc,
                "best_epoch": best_epoch,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "min_delta": MIN_DELTA,
            },
            f,
            indent=4
        )

    # Reload the best checkpoint before independent test evaluation.
    checkpoint = torch.load(best_model_path, map_location=device)
    final_model.load_state_dict(checkpoint["model_state_dict"])

    # --------------------------------------------------
    # FINAL TEST SET EVALUATION
    # --------------------------------------------------

    print("\nEvaluating final model on independent test set...")

    test_dataset = create_dataset(
        cfg.input_test,
        augment=False
    )

    if QUICK:
        idx = torch.randperm(len(test_dataset)).tolist()[:160]
        test_dataset = Subset(test_dataset, idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg.batch_size_test),
        shuffle=False,
        num_workers=0
    )

    misclassified_path = os.path.join(
        OUTPUT_DIR,
        "misclassified_test_events.csv"
    )

    save_misclassified_events(
        final_model,
        test_loader,
        misclassified_path,
        threshold=0.5
    )

    test_loss, test_auc = evaluate(
        final_model,
        test_loader,
        criterion
    )

    print("\nFINAL RESULTS")
    print("Best validation AUC (Optuna search):", study.best_value)
    print("Best validation AUC (final training):", best_final_val_auc)
    print("Independent test AUC:", test_auc)
    print("Independent test loss:", test_loss)

    with open(
        os.path.join(OUTPUT_DIR, "optuna_test_results.json"),
        "w"
    ) as f:
        json.dump(
            {
                "validation_auc_search": study.best_value,
                "validation_auc_final_training": best_final_val_auc,
                "test_auc": test_auc,
                "test_loss": test_loss,
                "best_params": best_params,
                "best_epoch": best_epoch,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "min_delta": MIN_DELTA,
            },
            f,
            indent=4
        )

    # Save Optuna test results
    test_results_path = os.path.join(
        OUTPUT_DIR,
        "test_results_optuna.json"
    )

    with open(test_results_path, "w") as f:
        json.dump(
            {
                "validation_auc_search": study.best_value,
                "validation_auc_final_training": best_final_val_auc,
                "test_auc": test_auc,
                "test_loss": test_loss,
                "best_params": best_params,
                "best_epoch": best_epoch,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "min_delta": MIN_DELTA,
            },
            f,
            indent=4
        )

    print("Saved Optuna test results:", test_results_path)
