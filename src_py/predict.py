"""
Credit Card Fraud Detection - Prediction Service
===============================================
Performs real-time and batch fraud inference using the saved production model artifact.

Outputs:
- Prediction: "Fraud" (1) vs "Legitimate" (0)
- Fraud Probability: e.g. 94.7%
- Risk Level: "LOW", "MEDIUM", "HIGH"
- Dynamic risk factors highlighting which PCA features deviated most from normal baseline
"""

import os
from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd

from src.preprocessing import REQUIRED_FEATURES
from src.utils import load_model_artifact


# Global in-memory cache to prevent reloading the model on every inference call
_CACHED_MODEL_BUNDLE: Optional[Dict[str, Any]] = None
_CACHED_MODEL_PATH: Optional[str] = None


def get_model(model_path: str = "models/fraud_detection_model.pkl") -> Dict[str, Any]:
    """
    Retrieves the cached model artifact or loads it from disk if not yet cached.
    """
    global _CACHED_MODEL_BUNDLE, _CACHED_MODEL_PATH
    if _CACHED_MODEL_BUNDLE is None or _CACHED_MODEL_PATH != model_path:
        _CACHED_MODEL_BUNDLE = load_model_artifact(model_path)
        _CACHED_MODEL_PATH = model_path
    return _CACHED_MODEL_BUNDLE


def determine_risk_level(prob: float, threshold: float = 0.50) -> str:
    """
    Maps continuous fraud probability to a discrete operational Risk Tier.
    - HIGH: Probability exceeds 70% or significantly exceeds decision threshold
    - MEDIUM: Probability is between 20% and threshold / 70% (Suspicious / requires step-up OTP)
    - LOW: Probability is below 20% (Standard approval)
    """
    if prob >= max(0.70, threshold):
        return "HIGH"
    elif prob >= min(0.20, threshold * 0.7):
        return "MEDIUM"
    else:
        return "LOW"


def identify_anomalies(
    feature_row: pd.Series, top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Identifies the most anomalous features in the transaction.
    Since V1-V28 are orthogonal PCA components with mean ~ 0 and std ~ 1 for normal transactions,
    components with |value| > 2.5 represent significant statistical anomalies.
    """
    anomalies = []
    # Key PCA components historically strongly correlated with fraud in this dataset
    key_fraud_indicators = ["V14", "V10", "V12", "V17", "V16", "V4", "V11"]

    for col in key_fraud_indicators:
        if col in feature_row:
            val = float(feature_row[col])
            # High deviations from 0 indicate unusual transaction patterns
            if abs(val) > 1.8:
                anomalies.append({
                    "feature": col,
                    "value": round(val, 3),
                    "deviation": "Extremely Low" if val < 0 else "Extremely High"
                })

    # Sort by absolute deviation
    anomalies.sort(key=lambda x: abs(x["value"]), reverse=True)
    return anomalies[:top_k]


def predict_transaction(
    transaction_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    model_path: str = "models/fraud_detection_model.pkl",
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Generates a single transaction fraud prediction.

    Parameters:
        transaction_data: Dictionary or Series with transaction features (Time, V1..V28, Amount).
        model_path: Path to serialized model artifact (.pkl).
        threshold: Classification cutoff threshold (0.0 to 1.0).

    Returns:
        Structured result dictionary:
        {
            "prediction": "Fraud" | "Legitimate",
            "predicted_class": 1 | 0,
            "fraud_probability": 0.947,
            "fraud_probability_pct": "94.7%",
            "risk_level": "HIGH" | "MEDIUM" | "LOW",
            "threshold_used": 0.50,
            "anomalies": [...]
        }
    """
    bundle = get_model(model_path)
    model = bundle["model"]
    scaler = bundle.get("scaler")
    expected_cols = bundle.get("feature_names", REQUIRED_FEATURES)

    # 1. Format input into 1-row DataFrame
    if isinstance(transaction_data, dict):
        df_input = pd.DataFrame([transaction_data])
    elif isinstance(transaction_data, pd.Series):
        df_input = pd.DataFrame([transaction_data.to_dict()])
    elif isinstance(transaction_data, pd.DataFrame):
        df_input = transaction_data.iloc[[0]].copy()
    else:
        raise TypeError("transaction_data must be a dict, pd.Series, or pd.DataFrame")

    # 2. Check for missing columns
    missing_cols = [c for c in expected_cols if c not in df_input.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required features: {missing_cols}")

    # Reorder columns to exact training order
    X = df_input[expected_cols].copy()

    # 3. Apply the fitted scaler to 'Amount' and 'Time'
    if scaler is not None:
        scale_cols = [c for c in ["Amount", "Time"] if c in X.columns]
        if scale_cols:
            X[scale_cols] = scaler.transform(X[scale_cols])

    # 4. Predict probability and class
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
    else:
        # Fallback for models without predict_proba
        decision = model.decision_function(X)[0]
        prob = float(1 / (1 + np.exp(-decision)))

    predicted_class = int(prob >= threshold)
    prediction_label = "Fraud" if predicted_class == 1 else "Legitimate"
    risk_level = determine_risk_level(prob, threshold=threshold)
    anomalies = identify_anomalies(df_input.iloc[0])

    return {
        "prediction": prediction_label,
        "predicted_class": predicted_class,
        "fraud_probability": round(prob, 4),
        "fraud_probability_pct": f"{prob * 100:.1f}%",
        "risk_level": risk_level,
        "threshold_used": threshold,
        "anomalies": anomalies,
        "model_name": bundle.get("model_name", "Trained ML Model")
    }


def predict_batch(
    df: pd.DataFrame,
    model_path: str = "models/fraud_detection_model.pkl",
    threshold: float = 0.50
) -> pd.DataFrame:
    """
    Performs vectorised batch inference on an uploaded DataFrame of transactions.

    Returns:
        DataFrame with added columns:
        - 'Fraud_Probability'
        - 'Prediction'
        - 'Risk_Level'
    """
    bundle = get_model(model_path)
    model = bundle["model"]
    scaler = bundle.get("scaler")
    expected_cols = bundle.get("feature_names", REQUIRED_FEATURES)

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Batch file is missing required columns: {missing_cols[:5]}...")

    df_out = df.copy()
    X = df[expected_cols].copy()

    # Scale using training parameters
    if scaler is not None:
        scale_cols = [c for c in ["Amount", "Time"] if c in X.columns]
        if scale_cols:
            X[scale_cols] = scaler.transform(X[scale_cols])

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        decision = model.decision_function(X)
        probs = 1 / (1 + np.exp(-decision))

    preds = (probs >= threshold).astype(int)

    df_out["Fraud_Probability"] = np.round(probs, 4)
    df_out["Prediction"] = np.where(preds == 1, "Fraud", "Legitimate")
    df_out["Risk_Level"] = [determine_risk_level(p, threshold) for p in probs]

    return df_out
