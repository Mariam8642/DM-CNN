#!/usr/bin/env python3
import argparse
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import csv

PROJECT_DIR = Path("/home/hep/ma2922/Project/DM-CNN")
NTUPLE_DIR = Path("/vols/sbn/uboone/darkTridents/data/ntuples")
MODEL_DIR = PROJECT_DIR / "bdt" / "models"
OUTPUT_DIR = PROJECT_DIR / "bdt" / "outputs" / "bdt_outputs"
TREE_NAME = "singlephotonana/vertex_tree"

FEATURES = [
    "sss_candidate_energy",
    "sss3d_shower_energy_max",
    "reco_shower3d_energy_plane0",
    "reco_shower3d_energy_plane1",
    "reco_shower3d_energy_plane2",
    "reco_shower_reclustered_energy_plane0",
    "reco_shower_reclustered_energy_plane1",
    "reco_shower_reclustered_energy_plane2",
    "reco_track_calo_energy_plane0",
    "reco_track_calo_energy_plane1",
    "reco_track_calo_energy_plane2",
    "reco_track_proton_kinetic_energy",
]

SIGNAL_FILES = sorted((NTUPLE_DIR / "dm_masspoints").glob("*.root")) + [
    NTUPLE_DIR / "training" / "dt_overlay_0.05_sp.root",
]

BACKGROUND_FILES = [
    NTUPLE_DIR / "training" / "NCgamma_overlay_production_v2_sp.root",
    NTUPLE_DIR / "training" / "NCeta_overlay_production_v2_sp.root",
    NTUPLE_DIR / "training" / "NCPi0_overlay_production_v2_sp.root",
]

RANDOM_SEED = 42


def list_trees(root_file):
    for key, obj in root_file.items(recursive=True):
        if hasattr(obj, "num_entries") and hasattr(obj, "keys"):
            print(f"TREE: {key}  entries={obj.num_entries}")
            for branch in obj.keys():
                print(f"  {branch}")


def reduce_to_event_values(values):
    """Convert scalar/vector/jagged branch values to one float per event."""
    values = ak.fill_none(values, np.nan)
    values = ak.Array(values)

    if values.ndim == 1:
        return np.asarray(ak.to_numpy(values), dtype=np.float32)

    # For vector or jagged branches, use the largest value in each event.
    values = ak.max(values, axis=-1, mask_identity=True)
    values = ak.fill_none(values, np.nan)
    return np.asarray(ak.to_numpy(values), dtype=np.float32)


def read_features(path):
    with uproot.open(path) as root_file:
        tree = root_file[TREE_NAME]
        arrays = tree.arrays(FEATURES, library="ak")
        columns = [reduce_to_event_values(arrays[name]) for name in FEATURES]
        return np.column_stack(columns)


def build_dataset():
    X_parts, y_parts = [], []

    for path, label in [(p, 1) for p in SIGNAL_FILES] + [(p, 0) for p in BACKGROUND_FILES]:
        print(f"Reading {'signal' if label else 'background'}: {path}")
        X_file = read_features(path)
        y_file = np.full(len(X_file), label, dtype=np.int8)

        valid = np.all(np.isfinite(X_file), axis=1)
        X_parts.append(X_file[valid])
        y_parts.append(y_file[valid])
        print(f"  usable events: {valid.sum()} / {len(valid)}")

    if not X_parts:
        raise RuntimeError("No ntuple events were loaded.")

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def plot_branch(args):
    with uproot.open(args.file) as root_file:
        if not args.tree or not args.branch:
            print("Available trees and branches:")
            list_trees(root_file)
            print("\nRun again with --tree and --branch.")
            return

        tree = root_file[args.tree]
        values = tree[args.branch].array(library="ak")
        values = ak.flatten(values, axis=None)
        values = np.asarray(ak.to_numpy(values), dtype=np.float64)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise ValueError("No finite values found in the selected branch.")

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        positive = values[values > 0]

        plt.figure(figsize=(8, 5))
        plt.hist(
            positive,
            bins=np.logspace(
                np.log10(positive.min()),
                np.log10(positive.max()),
                args.bins + 1
            ),
            histtype="stepfilled",
            alpha=0.75,
        )

        plt.xscale("log")
        plt.xlabel(f"{args.branch} [log scale]")
        plt.ylabel("Entries")
        plt.title(f"{args.branch} distribution")
        plt.grid(alpha=0.25, which="both")
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)


def train_bdt():
    missing = [str(p) for p in SIGNAL_FILES + BACKGROUND_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing ntuple files:\n" + "\n".join(missing))

    X, y = build_dataset()
    print(f"X shape: {X.shape}")
    print(f"Class counts: {np.unique(y, return_counts=True)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=RANDOM_SEED, stratify=y_train
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    predictions = model.predict_proba(X_test)[:, 1]
    labels = (predictions >= 0.5).astype(np.int8)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_DIR / "dark_trident_bdt.json")
    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

    print(f"Saved model: {MODEL_DIR / 'dark_trident_bdt.json'}")
    print(f"Test accuracy: {accuracy_score(y_test, labels):.4f}")
    print(f"Test AUC: {roc_auc_score(y_test, predictions):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-real-data",action="store_true",help="Apply the trained BDT to real data",)
    parser.add_argument("--model",type=Path, default=MODEL_DIR / "dark_trident_bdt.json",help="Trained BDT model",)
    parser.add_argument("--clean-output",type=Path,default=OUTPUT_DIR / "real_data_bdt_cleaned.root",help="Output ROOT file containing selected events",)
    parser.add_argument("--score-output",type=Path,default=OUTPUT_DIR / "real_data_bdt_scores.csv",help="Output CSV containing BDT scores",)
    parser.add_argument("--threshold",type=float,default=0.5,help="BDT score threshold",)
    args = parser.parse_args()

    if args.train_bdt:
        train_bdt()
    elif args.clean_real_data:
        clean_real_data(args)
    elif args.file:
        plot_branch(args)
    else:
        parser.error(
            "Provide a ROOT file for plotting, use --train-bdt, "
            "or use --clean-real-data.")

import csv


# Add this function before main():
def clean_real_data(args):
    """Run the trained BDT on real data and write a BDT-cleaned ROOT file."""
    if not args.file:
        raise ValueError("Provide a real-data ROOT file when using --clean-real-data.")

    input_path = Path(args.file)
    model_path = Path(args.model)
    output_path = Path(args.clean_output)
    score_path = Path(args.score_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input ROOT file not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"BDT model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading BDT model: {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    with uproot.open(input_path) as root_file:
        tree = root_file[args.tree]

        # Read all branches so that the selected events can be written back
        # without losing the original reconstructed-data information.
        all_arrays = tree.arrays(library="ak")
        feature_arrays = tree.arrays(FEATURES, library="ak")

        columns = [reduce_to_event_values(feature_arrays[name]) for name in FEATURES]
        X = np.column_stack(columns)
        valid_features = np.all(np.isfinite(X), axis=1)

        scores = np.full(len(X), np.nan, dtype=np.float32)
        scores[valid_features] = model.predict_proba(X[valid_features])[:, 1]

        # Keep events with a signal probability above the chosen threshold.
        selected = valid_features & (scores >= args.threshold)
        predictions = np.where(np.isfinite(scores), scores >= args.threshold, False)

        # Add the score and selection flag to the output ROOT tree.
        output_arrays = dict(all_arrays)
        output_arrays["bdt_score"] = ak.Array(scores)
        output_arrays["bdt_prediction"] = ak.Array(predictions.astype(np.int8))
        output_arrays = {name: values[selected] for name, values in output_arrays.items()}

        with uproot.recreate(output_path) as output_file:
            output_file[args.tree] = output_arrays

        # Save a compact event-level score file as well.
        event_ids = {}
        for branch_name in ("run", "subrun", "event"):
            if branch_name in all_arrays.fields:
                event_ids[branch_name] = ak.to_numpy(all_arrays[branch_name])

    with score_path.open("w", newline="") as csv_file:
        fieldnames = list(event_ids) + [
            "bdt_score",
            "bdt_prediction",
            "valid_features",
            "selected",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(scores)):
            row = {name: event_ids[name][i] for name in event_ids}
            row.update(
                bdt_score=float(scores[i]) if np.isfinite(scores[i]) else "",
                bdt_prediction=int(predictions[i]),
                valid_features=int(valid_features[i]),
                selected=int(selected[i]),
            )
            writer.writerow(row)

    print(f"Input events: {len(X)}")
    print(f"Events with valid BDT features: {valid_features.sum()}")
    print(f"Events retained: {selected.sum()}")
    print(f"Threshold: {args.threshold}")
    print(f"Saved cleaned ROOT file: {output_path}")
    print(f"Saved BDT scores: {score_path}")

if __name__ == "__main__":
    main()
