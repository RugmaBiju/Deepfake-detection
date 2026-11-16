"""
Shared evaluation metrics, including Equal Error Rate (EER) which standard
sklearn doesn't provide directly but is the conventional metric in the
deepfake/forensics literature alongside AUC.
"""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def compute_eer(y_true, y_prob) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return float(eer)


def full_report(y_true, y_pred, y_prob) -> dict:
    auc = roc_auc_score(y_true, y_prob)
    eer = compute_eer(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    return {
        "auc": auc,
        "eer": eer,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
    }
