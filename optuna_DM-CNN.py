
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
import optuna

import matplotlib
matplotlib.use("Agg")  # safe for headless/remote runs (no display needed)
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary


# --------------------------------------------------
# EXPERIMENT SETTINGS
# --------------------------------------------------

#to run the whole thing
# N_TRIALS = 25
# EPOCHS = int(cfg.EPOCHS)
# --------------------------------------------------

#to run a test
QUICK = os.environ.get("QUICK") == "1"
N_TRIALS = 2 if QUICK else 1

EPOCHS = 1 if QUICK else 20
# --------------------------------------------------

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
print("Trials:", N_TRIALS)
print("Epochs per trial:", EPOCHS)


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


# Create separate dataset objects so that validation
# images are never augmented.

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

# Deterministic 80/20 split
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

            # Class 0 represents signal (run == 100).
            # Class 1 represents background.
            signal_scores = torch.sigmoid(logits)[:, 0]

            all_scores.extend(
                signal_scores.cpu().numpy()
            )

            all_labels.extend(
                y_batch[:, 0].cpu().numpy()
            )

    if total_events == 0:
        raise ValueError("Validation loader is empty.")

    all_labels = np.asarray(all_labels)
    all_scores = np.asarray(all_scores)

    if len(np.unique(all_labels)) < 2:
        raise ValueError(
            "Validation dataset must contain both signal and background."
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

    return total_loss / total_events

# --------------------------------------------------
# LOSS CURVE PLOTTING
# --------------------------------------------------

def plot_loss_curves(train_losses, val_losses, out_path):
    """
    Plots train and validation loss on the same figure and saves it to disk.
    Safe to call in headless environments (e.g. Condor/cluster jobs) since
    matplotlib is set to the 'Agg' backend above.
    """

    epochs_axis = list(range(1, len(train_losses) + 1))

    plt.figure()
    plt.plot(epochs_axis, train_losses, label="Train loss", marker="o")
    plt.plot(epochs_axis, val_losses, label="Val loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Final model training: loss curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()

    print("Saved loss curve plot:", out_path)

# --------------------------------------------------
# OPTUNA OBJECTIVE
# --------------------------------------------------

def objective(trial):

    set_seed(SEED)

    # Hyperparameter search space

    dropout = trial.suggest_float(
        "dropout",
        0.1,
        0.7
    )

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

    # pooling = trial.suggest_categorical(
    #     "pooling",
    #     ["avg", "max"]
    # )

    print("\n" + "=" * 50)
    print("TRIAL:", trial.number)
    print("Dropout:", dropout)
    print("Learning rate:", learning_rate)
    print("Weight decay:", weight_decay)
    # print("Pooling:", pooling)
    print("=" * 50)

    # Initialise a fresh model for every trial

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

    best_auc = 0.0

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

        trial.report(val_auc, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        best_auc = max(best_auc, val_auc)

    return best_auc


# --------------------------------------------------
# RUN OPTUNA
# --------------------------------------------------

if __name__ == "__main__":

    set_seed(SEED)

    study = optuna.create_study(
        direction="maximize",
        study_name="dmcnn_quick" if QUICK else "dmcnn",
        storage="sqlite:////home/hep/ma2922/Project/DM-CNN/outputs/optuna.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,     # need more completed trials before median is trustworthy
            n_warmup_steps=10,       # don't allow pruning until halfway through (epoch 10 of 20)
            interval_steps=2         # only check every 2 epochs after warmup, not every single one
        )
    )

    print("\nStarting Optuna optimisation...\n")

    study.optimize(
        objective,
        n_trials=N_TRIALS
    )

    # Save results

    print("\nOPTIMISATION FINISHED")

    print("Best validation AUC:", study.best_value)
    print("Best hyperparameters:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    study.trials_dataframe().to_csv(
        os.path.join(OUTPUT_DIR, "optuna_trials.csv"),
        index=False
    )

    with open(
        os.path.join(OUTPUT_DIR, "optuna_best_params.json"),
        "w"
    ) as f:

        json.dump(
            {
                "best_validation_auc": study.best_value,
                "best_params": study.best_params
            },
            f,
            indent=4
        )

    # --------------------------------------------------
    # FINAL TRAINING ON FULL TRAINING DATA
    # --------------------------------------------------

    print("\nTraining final model with best hyperparameters...")

    set_seed(SEED)

    # full_train_dataset = create_dataset(
    #     cfg.input_train,
    #     augment=bool(cfg.augment)
    # )

    full_train_dataset = create_dataset(cfg.input_train, augment=bool(cfg.augment))
    if QUICK:
        idx = torch.randperm(len(full_train_dataset)).tolist()[:320]
        full_train_dataset = Subset(full_train_dataset, idx)


    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=int(cfg.batch_size_train),
        shuffle=True,
        num_workers=0
    )

    best_params = study.best_params

    final_model = mpid_net_binary.MPID(
        dropout=best_params["dropout"],
        num_classes=2,
        # pooling=best_params["pooling"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    final_optimizer = optim.Adam(
        final_model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )

    # for epoch in range(EPOCHS):

    #     loss = train_one_epoch(
    #         final_model,
    #         full_train_loader,
    #         criterion,
    #         final_optimizer
    #     )

    #     print(
    #         f"Final training epoch {epoch + 1}/{EPOCHS} "
    #         f"| Loss: {loss:.5f}"
    #     )

    # # Save final model

    # model_path = os.path.join(
    #     WEIGHTS_DIR,
    #     "DM-CNN_optuna_best.pwf"
    # )

    # torch.save(
    #     {
    #         "model_state_dict": final_model.state_dict(),
    #         "hyperparameters": best_params,
    #         "validation_auc": study.best_value
    #     },
    #     model_path
    # )

    # print("\nSaved model:", model_path)

 # --------------------------------------------------
    # FINAL TRAINING ON FULL TRAINING DATA
    # --------------------------------------------------

    # print("\nTraining final model with best hyperparameters...")

    # set_seed(SEED)

    # full_train_dataset = create_dataset(
    #     cfg.input_train,
    #     augment=bool(cfg.augment)
    # )

    full_train_dataset = create_dataset(cfg.input_train, augment=bool(cfg.augment))
    if QUICK:
        idx = torch.randperm(len(full_train_dataset)).tolist()[:320]
        full_train_dataset = Subset(full_train_dataset, idx)


    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=int(cfg.batch_size_train),
        shuffle=True,
        num_workers=0
    )

    best_params = study.best_params

    final_model = mpid_net_binary.MPID(
        dropout=best_params["dropout"],
        num_classes=2,
        # pooling=best_params["pooling"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    final_optimizer = optim.Adam(
        final_model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )

    # Track loss/AUC per epoch so we can plot curves and keep the best checkpoint.
    final_train_losses = []
    final_val_losses = []
    final_val_aucs = []

    best_final_val_auc = 0.0
    best_model_path = os.path.join(
        WEIGHTS_DIR,
        "DM-CNN_optuna_best.pwf"
    )

    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(
            final_model,
            full_train_loader,
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

        # Save the checkpoint with the best validation AUC seen so far,
        # not just whatever the last epoch happens to produce.
        if val_auc > best_final_val_auc:
            best_final_val_auc = val_auc

            torch.save(
                {
                    "model_state_dict": final_model.state_dict(),
                    "hyperparameters": best_params,
                    "validation_auc": best_final_val_auc,
                    "epoch": epoch + 1
                },
                best_model_path
            )

            print(
                f"  -> New best val AUC ({best_final_val_auc:.5f}), "
                f"checkpoint saved to {best_model_path}"
            )

    print("\nSaved best model:", best_model_path)
    print("Best epoch val AUC during final training:", best_final_val_auc)

    # Plot train vs. val loss curves on the same figure.
    loss_curve_path = os.path.join(OUTPUT_DIR, "final_training_loss_curve.png")
    plot_loss_curves(final_train_losses, final_val_losses, loss_curve_path)

    # Save the per-epoch history alongside the plot for later inspection.
    with open(
        os.path.join(OUTPUT_DIR, "final_training_history.json"),
        "w"
    ) as f:

        json.dump(
            {
                "train_loss": final_train_losses,
                "val_loss": final_val_losses,
                "val_auc": final_val_aucs,
                "best_val_auc": best_final_val_auc
            },
            f,
            indent=4
        )

    # Reload the best checkpoint (by val AUC) before running the final,
    # independent test evaluation, rather than using whatever the last
    # epoch happened to produce.
    checkpoint = torch.load(best_model_path, map_location=device)
    final_model.load_state_dict(checkpoint["model_state_dict"])

    # --------------------------------------------------
    # FINAL TEST SET EVALUATION (ONCE)
    # --------------------------------------------------

    print("\nEvaluating final model on independent test set...")

    # test_dataset = create_dataset(
    #     cfg.input_test,
    #     augment=False
    # )

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
                "best_params": best_params
            },
            f,
            indent=4
        )