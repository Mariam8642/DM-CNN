import os
import json
import csv

import torch
from torch.utils.data import DataLoader

from lib.config import config_loader
from mpid_data import mpid_data_binary
from mpid_net import mpid_net_binary

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

CFG = "/home/hep/ma2922/Project/DM-CNN/cfg/training_config.cfg"
cfg = config_loader(CFG)

IMAGE_TREE = "image2d_image2d_binary_tree"

BEST_MODEL_PATH = "/home/hep/ma2922/Project/DM-CNN/weights/DM-CNN_best.pwf"
#BEST_MODEL_PATH = "/home/hep/ma2922/Project/DM-CNN/weights/DM-CNN_optuna_best.pwf"

# EDIT THIS: path to the run1 LArCV file you want to run inference on
INPUT_FILE = "/vols/sbn/uboone/darkTridents/data/larcv_files/run3_data/run3_NuMI_beamon_larcv_cropped_full_set.root"

OUTPUT_DIR = cfg.output_directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = int(cfg.batch_size_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("GPU available:", torch.cuda.is_available())

# --------------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------------

checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)

hyperparams = checkpoint["hyperparameters"]
print("Loaded checkpoint from:", BEST_MODEL_PATH)
print("Hyperparameters used for this model:", hyperparams)
print("Validation AUC at save time:", checkpoint.get("validation_auc"))
print("Saved at epoch:", checkpoint.get("epoch"))

model = mpid_net_binary.MPID(
    dropout=hyperparams["dropout"],
    num_classes=2
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --------------------------------------------------
# DATASET: run1 data (no augmentation for inference)
# --------------------------------------------------

run_dataset = mpid_data_binary.MPID_Dataset(
    INPUT_FILE,
    IMAGE_TREE,
    "cpu",
    plane=0,
    augment=False,
)

print("dataset length:", len(run_dataset))

run_loader = DataLoader(
    run_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# --------------------------------------------------
# RUN INFERENCE
# --------------------------------------------------

results = []

with torch.no_grad():

    for x_batch, y_batch, info, nevents in run_loader:

        x_batch = x_batch.to(
            device, dtype=torch.float32
        ).reshape(-1, 1, 512, 512)

        logits = model(x_batch)
        signal_scores = torch.sigmoid(logits)[:, 0].cpu().numpy()

        # info = [run_tensor, subrun_tensor, event_tensor]
        run_nums = info[0].cpu().numpy()
        subrun_nums = info[1].cpu().numpy()
        event_nums = info[2].cpu().numpy()

        for i in range(len(signal_scores)):
            results.append(
                {
                    "run": int(run_nums[i]),
                    "subrun": int(subrun_nums[i]),
                    "event": int(event_nums[i]),
                    "signal_score": float(signal_scores[i]),
                    "predicted_signal": int(signal_scores[i] >= 0.5),
                }
            )

print(f"Ran inference on {len(results)} events.")

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "inference_results.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["run", "subrun", "event", "signal_score", "predicted_signal"],
    )
    writer.writeheader()
    writer.writerows(results)

print("Saved inference results to:", csv_path)

n_signal = sum(r["predicted_signal"] for r in results)
print(f"Predicted signal events: {n_signal} / {len(results)}")
