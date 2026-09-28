"""
Credit Card Fraud Detection - Evaluation & Utilities
===================================================
Provides model evaluation metrics, threshold tuning utilities,
plotting helpers for Confusion Matrix / ROC / PR-AUC, and joblib persistence.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes all standard fraud detection evaluation metrics.

    Highlights:
    - Precision: Of all flagged transactions, what percentage were real fraud?
    - Recall (Sensitivity): What percentage of actual frauds did we capture?
    - PR-AUC: Area under Precision-Recall curve (the most informative metric
      when the positive class is 0.172%).
    - ROC-AUC: Ranking ability across all possible discrimination thresholds.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = float("nan")
    pr_auc = float("nan")
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
        try:
            pr_auc = average_precision_score(y_true, y_prob)
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_evaluated": int(len(y_true)),
        "actual_fraud_count": int(y_true.sum())
    }


def print_evaluation_report(
    model_name: str,
    metrics: Dict[str, Any],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None
) -> None:
    """
    Prints a formatted evaluation report highlighting fraud detection trade-offs.
    """
    tn = metrics.get("true_negatives", 0)
    fp = metrics.get("false_positives", 0)
    fn = metrics.get("false_negatives", 0)
    tp = metrics.get("true_positives", 0)

    print("\n" + "=" * 65)
    print(f" MODEL PERFORMANCE REPORT: {model_name.upper()}")
    print("=" * 65)
    print(f" Accuracy:       {metrics['accuracy']:.4f}  (NOTE: High accuracy is misleading in 0.17% fraud)")
    print(f" Precision:      {metrics['precision']:.4f}  (Flagged transactions that were truly fraud)")
    print(f" Recall:         {metrics['recall']:.4f}  (Actual frauds detected and blocked)")
    print(f" F1-Score:       {metrics['f1_score']:.4f}  (Harmonic mean of Precision & Recall)")
    print(f" ROC-AUC:        {metrics['roc_auc']:.4f}  (Area under Receiver Operating Characteristic)")
    print(f" PR-AUC:         {metrics['pr_auc']:.4f}  (Primary gold-standard benchmark for imbalanced fraud)")
    print("-" * 65)
    print(" CONFUSION MATRIX BREAKDOWN & BANKING IMPACT:")
    print(f"  [TN] True Negatives:   {tn:>6,}  | Legitimate transactions approved seamlessly.")
    print(f"  [FP] False Positives:  {fp:>6,}  | Legitimate cards flagged (Customer friction / SMS alert).")
    print(f"  [FN] False Negatives:  {fn:>6,}  | MISSED FRAUD (Direct financial loss & chargeback costs).")
    print(f"  [TP] True Positives:   {tp:>6,}  | FRAUD DETECTED (Stolen card transactions blocked).")
    print("=" * 65)

    if y_true is not None and y_pred is not None:
        print("\nCLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, target_names=["Legitimate (0)", "Fraud (1)"], digits=4))


def evaluate_threshold_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Computes Precision, Recall, and F1 across a spectrum of decision thresholds.
    In credit card risk systems, fraud teams often lower thresholds (e.g. 0.20 - 0.35)
    to minimize expensive False Negatives.
    """
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    records = []
    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        records.append({
            "Threshold": th,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "TP (Caught Fraud)": tp,
            "FN (Missed Fraud)": fn,
            "FP (False Alarms)": fp
        })

    return pd.DataFrame(records)


def save_model_artifact(artifact: Dict[str, Any], filepath: str) -> None:
    """
    Persists trained model, scaler, and metadata into a single pickle artifact using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(artifact, filepath)
    print(f"[INFO] Successfully saved model artifact to: {filepath}")


def load_model_artifact(filepath: str) -> Dict[str, Any]:
    """
    Loads saved model bundle from disk.

    Raises:
        FileNotFoundError: If the model file is not present.
    """
    if not os.path.exists(filepath):
        alt_path = os.path.join(os.path.dirname(__file__), "..", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(
                f"Model file not found at '{filepath}'. "
                f"Please run the training pipeline first: python src/train.py"
            )

    artifact = joblib.load(filepath)
    return artifact
