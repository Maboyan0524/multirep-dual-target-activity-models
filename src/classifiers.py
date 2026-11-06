"""The seven classifier configurations and the metric set.

One fixed configuration per classifier is shared by every representation and
every target, so nothing here is tuned per dataset.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

CLASSIFIERS = ["LR", "RF", "ExtraTrees", "RBF-SVM", "XGBoost", "LightGBM", "MLP"]


def build_classifier(name: str, seed: int, y_train: np.ndarray):
    ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    if name == "LR":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                                    solver="liblinear", max_iter=5000,
                                                    random_state=seed))])
    if name == "RF":
        return RandomForestClassifier(n_estimators=600, max_features="sqrt", min_samples_leaf=2,
                                      class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=600, max_features="sqrt", min_samples_leaf=2,
                                    class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "RBF-SVM":
        # probability=False avoids a second internal Platt calibration layer on
        # top of the outer cross-validation. The monotonic decision score is
        # mapped through expit only to give every model a [0, 1] output; ranking
        # metrics are invariant to that transform and thresholds stay OOF-derived.
        return Pipeline([("scale", StandardScaler()),
                         ("clf", SVC(C=1.0, kernel="rbf", gamma="scale",
                                     class_weight="balanced", probability=False,
                                     random_state=seed))])
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=-1,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              scale_pos_weight=ratio, random_state=seed, n_jobs=4, verbosity=-1)
    if name == "XGBoost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed")
        # Fixed CPU budget shared by every representation; not tuned per target.
        return XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.10, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=ratio,
                             objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", random_state=seed, n_jobs=4)
    if name == "MLP":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4,
                                               learning_rate_init=1e-3, batch_size=64,
                                               early_stopping=True, validation_fraction=0.15,
                                               n_iter_no_change=20, max_iter=300,
                                               random_state=seed))])
    raise ValueError(f"unknown classifier: {name}")


def fit(classifier, name: str, x, y):
    """MLP has no class_weight parameter, so balance it through sample weights."""
    if name == "MLP":
        weight = np.where(y == 1, (y == 0).sum() / max(1, (y == 1).sum()), 1.0)
        classifier.fit(x, y, clf__sample_weight=weight)
    else:
        classifier.fit(x, y)
    return classifier


def active_probability(classifier, x) -> np.ndarray:
    classes = (classifier.classes_ if hasattr(classifier, "classes_")
               else classifier.named_steps["clf"].classes_)
    if hasattr(classifier, "predict_proba") and getattr(classifier, "probability", True):
        positive = int(np.where(np.asarray(classes) == 1)[0][0])
        return classifier.predict_proba(x)[:, positive]
    return expit(classifier.decision_function(x))


def mcc_optimal_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    """Threshold maximising MCC on the pooled out-of-fold predictions.

    Ties are broken toward the lower threshold so the choice is deterministic.
    """
    candidates = np.unique(probability)
    best = max(((matthews_corrcoef(y, (probability >= t).astype(int)), float(t))
                for t in candidates), key=lambda z: (z[0], -z[1]))
    return best[1]


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": roc_auc_score(y, probability),
        "PR_AUC": average_precision_score(y, probability),
        "MCC": matthews_corrcoef(y, predicted),
        "Balanced_Accuracy": balanced_accuracy_score(y, predicted),
        "F1": f1_score(y, predicted, zero_division=0),
        "Precision": precision_score(y, predicted, zero_division=0),
        "Recall": recall_score(y, predicted, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }
