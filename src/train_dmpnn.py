#!/usr/bin/env python3
"""End-to-end D-MPNN baseline with Chemprop 1.6 under the same locked folds.

Each of the five outer folds is held out as the test set in turn. Chemprop needs
its own validation set, so fold (k+1) % 5 serves as validation while training
uses the remaining three folds; the held-out fold k is never seen during
training. Five seeds are trained per fold and the per-compound prediction is
the median over seeds.

The outer folds are the ones in the dataset file and are never re-split, so
this baseline is directly comparable with the representation benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from classifiers import metrics

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 30
BATCH_SIZE = 50


def chemprop_command(entry: Path, fallback: str) -> list[str]:
    if entry.exists():
        return [sys.executable, str(entry)]
    return [str(Path(sys.executable).with_name(fallback))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True,
                    help="training.csv with compound_id, canonical_smiles, activity_label, fold")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-cuda", action="store_true",
                    help="run the identical architecture on CPU")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    data = pd.read_csv(args.data)
    required = {"compound_id", "canonical_smiles", "activity_label", "fold"}
    if not required <= set(data.columns):
        raise RuntimeError(f"{args.data} must contain {sorted(required)}")
    if data.compound_id.duplicated().any():
        raise RuntimeError("duplicate compound_id")
    if set(data.fold) != set(range(5)):
        raise RuntimeError("the dataset must carry folds 0-4")

    data = data[["compound_id", "canonical_smiles", "activity_label", "fold"]].rename(
        columns={"canonical_smiles": "smiles", "activity_label": "label"})
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    train_cmd = chemprop_command(
        Path(os.environ.get("CHEMPROP_TRAIN_ENTRY", here / "chemprop_train_entry.py")),
        "chemprop_train")
    predict_cmd = chemprop_command(
        Path(os.environ.get("CHEMPROP_PREDICT_ENTRY", here / "chemprop_predict_entry.py")),
        "chemprop_predict")

    device_args = ["--no_cuda"] if args.no_cuda else ["--gpu", args.gpu]
    log = (out / "training_log.txt").open("a")
    collected = []

    for fold in range(5):
        validation_fold = (fold + 1) % 5
        fold_dir = out / "fold_inputs" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train = data[(data.fold != fold) & (data.fold != validation_fold)]
        validation = data[data.fold == validation_fold]
        test = data[data.fold == fold]
        train.to_csv(fold_dir / "train.csv", index=False)
        validation.to_csv(fold_dir / "validation.csv", index=False)
        test.to_csv(fold_dir / "test.csv", index=False)

        seed_predictions = []
        for seed in SEEDS:
            run_dir = out / "models" / f"fold_{fold}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            predictions = run_dir / "test_predictions.csv"

            command = train_cmd + [
                "--data_path", str(fold_dir / "train.csv"),
                "--separate_val_path", str(fold_dir / "validation.csv"),
                "--dataset_type", "classification", "--target_columns", "label",
                "--smiles_columns", "smiles", "--metric", "auc",
                "--epochs", str(EPOCHS), "--batch_size", str(BATCH_SIZE),
                "--num_workers", "0", "--seed", str(seed), "--pytorch_seed", str(seed),
                "--save_dir", str(run_dir)] + device_args
            log.write(" ".join(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

            command = predict_cmd + [
                "--test_path", str(fold_dir / "test.csv"), "--preds_path", str(predictions),
                "--checkpoint_dir", str(run_dir), "--smiles_columns", "smiles",
                "--num_workers", "0"] + device_args
            log.write(" ".join(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

            frame = pd.read_csv(predictions)
            # Chemprop 1.6 writes the prediction into the original target column.
            column = ("label" if "label" in frame.columns
                      else [c for c in frame.columns
                            if c not in {"compound_id", "smiles", "fold"}][0])
            seed_predictions.append(frame[column].to_numpy(float))

        median = np.median(np.vstack(seed_predictions), axis=0)
        fold_result = test.rename(columns={"smiles": "canonical_smiles",
                                           "label": "true_label"}).copy()
        fold_result["predicted_probability"] = median
        fold_result["predicted_label"] = (median >= 0.5).astype(int)
        collected.append(fold_result)

    log.close()
    oof = pd.concat(collected).sort_values("compound_id")
    if len(oof) != len(data) or oof.compound_id.duplicated().any():
        raise RuntimeError("incomplete out-of-fold assembly")
    oof.to_csv(out / "oof_predictions_dmpnn.csv", index=False)

    per_fold = [{"model": "DMPNN_5seed_median", "fold": fold, "N": len(group),
                 "N_active": int(group.true_label.sum()),
                 "N_inactive": int((group.true_label == 0).sum()),
                 **metrics(group.true_label.to_numpy(),
                           group.predicted_probability.to_numpy(), 0.5)}
                for fold, group in oof.groupby("fold")]
    frame = pd.DataFrame(per_fold)
    frame.to_csv(out / "metrics_per_fold_dmpnn.csv", index=False)

    pooled = metrics(oof.true_label.to_numpy(), oof.predicted_probability.to_numpy(), 0.5)
    summary = {"Model": "DMPNN_5seed_median"}
    for key, value in pooled.items():
        summary[f"{key}_mean"] = frame[key].mean()
        summary[f"{key}_SD"] = frame[key].std(ddof=1)
        summary[f"{key}_pooled"] = value
    pd.DataFrame([summary]).to_csv(out / "metrics_summary_dmpnn.csv", index=False)

    (out / "training_parameters.json").write_text(json.dumps({
        "seeds": SEEDS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "outer_validation": "predefined fold (k+1) % 5",
        "held_out_fold_unused_in_training": True,
        "aggregation": "median over seeds",
        "device": "CPU" if args.no_cuda else f"GPU {args.gpu}",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
