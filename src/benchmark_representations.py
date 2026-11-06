#!/usr/bin/env python3
"""Representation x classifier benchmark under a fixed scaffold five-fold split.

For every representation and classifier, each fold is held out in turn, the
model is fitted on the remaining four folds, and the held-out predictions are
assembled into one out-of-fold vector covering the whole training set. The
decision threshold is then the value maximising MCC on that vector.

The folds come from the dataset file and are never re-split here, so all
28 combinations and the separate D-MPNN baseline see exactly the same folds.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from classifiers import (CLASSIFIERS, active_probability, build_classifier, fit,
                         mcc_optimal_threshold, metrics)
from features_io import REPRESENTATIONS, load_features, read_master

SEED_BASE = 20261008


def run(master: pd.DataFrame, features_root: Path, out_root: Path,
        representations: list[str], classifiers: list[str]) -> None:
    y = master.activity_label.to_numpy()
    folds = master.fold.to_numpy()
    if set(np.unique(folds)) != set(range(5)):
        raise RuntimeError("the dataset must carry folds 0-4")
    out_root.mkdir(parents=True, exist_ok=True)

    audit, summaries = [], []
    for representation in representations:
        x = load_features(master, features_root, representation)
        audit.append({"Representation": representation, "N": len(master),
                      "Dimension": int(x.shape[1])})

        for name in classifiers:
            out = out_root / representation / name
            out.mkdir(parents=True, exist_ok=True)
            started = time.time()

            probability = np.full(len(y), np.nan)
            per_fold = []
            for fold in range(5):
                train = folds != fold
                test = ~train
                model = build_classifier(name, SEED_BASE + fold, y[train])
                fit(model, name, x[train], y[train])
                probability[test] = active_probability(model, x[test])
                per_fold.append({"Representation": representation, "Classifier": name,
                                 "fold": fold, "N_test": int(test.sum()),
                                 **metrics(y[test], probability[test], 0.5)})

            if not np.isfinite(probability).all():
                raise RuntimeError(f"incomplete out-of-fold vector for {representation}/{name}")

            threshold = mcc_optimal_threshold(y, probability)
            oof = master[["canonical_smiles", "activity_label", "fold"]].copy()
            oof["predicted_probability"] = probability
            oof["MCC_optimal_threshold"] = threshold
            oof["predicted_class"] = (probability >= threshold).astype(int)
            oof.to_csv(out / "oof_predictions.csv", index=False)
            pd.DataFrame(per_fold).to_csv(out / "metrics_per_fold.csv", index=False)

            summary = {"Representation": representation, "Classifier": name, "N": len(y),
                       "Feature_dimension": int(x.shape[1]),
                       "OOF_threshold_MCC": threshold,
                       **metrics(y, probability, threshold),
                       "Runtime_seconds": time.time() - started}
            pd.DataFrame([summary]).to_csv(out / "metrics_summary.csv", index=False)
            summaries.append(summary)
            print(representation, name, "done", flush=True)

    pd.DataFrame(audit).to_csv(out_root / "representation_feature_audit.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_root / "all_representation_classifier_metrics.csv",
                                   index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", type=Path, required=True,
                    help="training.csv with canonical_smiles, activity_label and fold")
    ap.add_argument("--features", type=Path, required=True,
                    help="directory holding one subdirectory per representation")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--representations", nargs="+", default=REPRESENTATIONS)
    ap.add_argument("--classifiers", nargs="+", default=CLASSIFIERS)
    args = ap.parse_args()
    run(read_master(args.master), args.features, args.out,
        args.representations, args.classifiers)


if __name__ == "__main__":
    main()
