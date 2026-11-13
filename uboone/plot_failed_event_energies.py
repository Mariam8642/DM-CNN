#!/usr/bin/env python3
"""Plot energy distributions for events misclassified by a CNN.

Expected CSV columns: energy, label, prediction.
Change the column names below if your file uses different names.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="CSV containing energy, truth label, and prediction")
    parser.add_argument("--energy-column", default="energy")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--output", default="failed_event_energy.png")
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = [args.energy_column, args.label_column, args.prediction_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

    energy = pd.to_numeric(df[args.energy_column], errors="coerce")
    truth = pd.to_numeric(df[args.label_column], errors="coerce")
    prediction = pd.to_numeric(df[args.prediction_column], errors="coerce")
    valid = energy.notna() & truth.notna() & prediction.notna()
    df = df.loc[valid].copy()
    df["energy"] = energy.loc[valid]
    df["truth"] = truth.loc[valid].astype(int)
    df["prediction"] = prediction.loc[valid].astype(int)

    failed = df[df["truth"] != df["prediction"]]
    false_negative = failed[(failed["truth"] == 1) & (failed["prediction"] == 0)]
    false_positive = failed[(failed["truth"] == 0) & (failed["prediction"] == 1)]

    plt.figure(figsize=(8, 5))
    if len(false_negative):
        plt.hist(false_negative["energy"], bins=args.bins, alpha=0.65,
                 label=f"False negatives (n={len(false_negative)})")
    if len(false_positive):
        plt.hist(false_positive["energy"], bins=args.bins, alpha=0.65,
                 label=f"False positives (n={len(false_positive)})")
    if not len(failed):
        raise ValueError("No failed/misclassified events found.")

    plt.xlabel("Event energy")
    plt.ylabel("Number of events")
    plt.title("Energy distribution of failed CNN classifications")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Total events: {len(df)}")
    print(f"Failed events: {len(failed)}")
    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
