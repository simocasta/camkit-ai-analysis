from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_class_probabilities(probabilities: Any) -> np.ndarray:
    if hasattr(probabilities, "values"):
        array = probabilities.values
    else:
        array = np.asarray(probabilities)
    if array.ndim == 1:
        return array.astype(float)
    return array[:, 1].astype(float)


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    x = safe_logit(y_prob).reshape(-1, 1)
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )
    try:
        # scikit-learn 1.8 deprecates the ``penalty`` constructor argument,
        # while the frozen Python 3.10 environment still needs ``None`` to fit
        # the unpenalised recalibration model.  Suppress only that compatibility
        # warning; it otherwise emits once per bootstrap fit and can flood a
        # deterministic package build with tens of thousands of lines.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            model.fit(x, y_true)
    except Exception:
        return float("nan"), float("nan")
    return float(model.coef_[0][0]), float(model.intercept_[0])


def classification_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    lr_plus = sensitivity / (1.0 - specificity) if np.isfinite(sensitivity) and specificity < 1.0 else float("inf")
    lr_minus = (1.0 - sensitivity) / specificity if np.isfinite(sensitivity) and specificity > 0.0 else float("inf")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "lr_plus": float(lr_plus),
        "lr_minus": float(lr_minus),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "predicted_positive": int(tp + fp),
        "predicted_negative": int(tn + fn),
        "total_positive": int(tp + fn),
        "total_negative": int(tn + fp),
    }


def rank_discrimination_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Average precision and AUC-ROC for any real-valued risk score.

    ``discrimination_metrics`` additionally computes Brier score and calibration,
    which are defined only for a predicted probability. Comparing a probability
    with an unnormalised score — the original CamKIT score runs 0 to 12 — needs
    the rank-based metrics alone. The internal ``auprc`` key is retained for
    compatibility with the frozen result registry, but its value is calculated
    with :func:`sklearn.metrics.average_precision_score` and must be labelled
    average precision (AP) in publication-facing output.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"auprc": float("nan"), "auroc": float("nan")}
    return {
        "auprc": float(average_precision_score(y_true, y_score)),
        "auroc": float(roc_auc_score(y_true, y_score)),
    }


def discrimination_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    metrics = {
        "n": float(len(y_true)),
        "events": float(np.sum(y_true)),
        "prevalence": float(np.mean(y_true)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }
    if len(np.unique(y_true)) < 2:
        metrics["auprc"] = float("nan")
        metrics["auroc"] = float("nan")
    else:
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    return metrics
