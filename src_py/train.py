"""
Credit Card Fraud Detection - Model Training & Evaluation Pipeline
==================================================================
Trains and benchmarks three robust classification algorithms:
1. Logistic Regression (Linear baseline with class_weight='balanced')
2. Random Forest Classifier (Non-linear ensemble with balanced weights)
3. XGBoost / Gradient Boosting (Gradient boosted trees with imbalance scale)

Selects the champion model based on PR-AUC & F1-Score (rather than raw Accuracy),
extracts feature importance rankings, and exports a production bundle to
'models/fraud_detection_model.pkl'.
"""

import os
import sys
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Try importing XGBoost; gracefully fall back to sklearn GradientBoosting if absent
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Import modular pipeline components
from src.preprocessing import (
    load_data,
    clean_data,
    split_features_target,
    split_train_test,
    fit_and_scale_features,
    apply_smote,
    REQUIRED_FEATURES
)
from src.utils import (
    calculate_metrics,
    print_evaluation_report,
    save_model_artifact
)


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_smote: bool = False
) -> Tuple[Dict[str, Dict[str, Any]], str, Any]:
    """
    Trains multiple candidate models and compares their test set metrics.

    Parameters:
        X_train: Scaled training features.
        y_train: Training labels.
        X_test: Scaled testing features.
        y_test: Testing labels.
        use_smote: Whether to apply SMOTE oversampling on training set.

    Returns:
        (results_dict, best_model_name, best_model_instance)
    """
    # If SMOTE requested, apply strictly post-split
    if use_smote:
        X_fit, y_fit = apply_smote(X_train, y_train, sampling_strategy=0.20, random_state=42)
    else:
        X_fit, y_fit = X_train, y_train

    # Calculate class imbalance ratio for gradient boosting
    neg_count = (y_fit == 0).sum()
    pos_count = (y_fit == 1).sum()
    scale_weight = float(neg_count / max(pos_count, 1))

    # Initialize candidate algorithms
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        )
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    else:
        # Fallback to standard scikit-learn GradientBoosting
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

    results = {}
    print("\n" + "=" * 65)
    print(" STARTING MODEL TRAINING & EVALUATION BENCHMARK")
    print("=" * 65)

    for name, clf in models.items():
        print(f"\n--> Training {name}...")
        clf.fit(X_fit, y_fit)

        # Generate predictions on the untouched test partition
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        metrics = calculate_metrics(y_test.values, y_pred, y_prob)
        results[name] = {
            "model": clf,
            "metrics": metrics,
            "predictions": y_pred,
            "probabilities": y_prob
        }

        print_evaluation_report(name, metrics, y_test.values, y_pred)

    # Champion selection based on PR-AUC (or F1-score if PR-AUC tied)
    best_model_name = max(
        results.keys(),
        key=lambda m: (results[m]["metrics"]["pr_auc"], results[m]["metrics"]["f1_score"])
    )
    best_model = results[best_model_name]["model"]

    print("\n" + "#" * 65)
    print(f" CHAMPION MODEL SELECTED: {best_model_name}")
    print(f" Best PR-AUC:   {results[best_model_name]['metrics']['pr_auc']:.4f}")
    print(f" Best F1-Score: {results[best_model_name]['metrics']['f1_score']:.4f}")
    print(f" Best Recall:   {results[best_model_name]['metrics']['recall']:.4f}")
    print("#" * 65)

    return results, best_model_name, best_model


def extract_feature_importance(model: Any, feature_names: list) -> Dict[str, float]:
    """
    Extracts sorted feature importance weights from tree-based or linear models.
    """
    importances = {}
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        for feat, score in zip(feature_names, raw):
            importances[feat] = float(score)
    elif hasattr(model, "coef_"):
        # For logistic regression: absolute coefficient magnitude
        raw = np.abs(model.coef_[0])
        for feat, score in zip(feature_names, raw):
            importances[feat] = float(score)

    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    return sorted_importances


def run_training_pipeline(
    data_path: str = "data/creditcard.csv",
    output_model_path: str = "models/fraud_detection_model.pkl",
    use_smote: bool = False,
    test_size: float = 0.20
) -> Dict[str, Any]:
    """
    Executes the complete end-to-end training pipeline.
    """
    print(f"[PIPELINE] Starting training workflow...")

    # 1. Load data
    df = load_data(data_path)

    # 2. Clean and deduplicate
    df_clean = clean_data(df, drop_duplicates=True)

    # 3. Separate features and target
    X, y = split_features_target(df_clean, target_col="Class")

    # 4. Stratified split (prevents leakage)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=42
    )

    # 5. Fit scaler ONLY on train set, transform test
    X_train_scaled, X_test_scaled, fitted_scaler = fit_and_scale_features(
        X_train, X_test, columns_to_scale=["Amount", "Time"], scaler_type="robust"
    )

    # 6. Train and benchmark candidate algorithms
    results, champion_name, champion_model = train_and_evaluate_models(
        X_train_scaled, y_train, X_test_scaled, y_test, use_smote=use_smote
    )

    # 7. Extract feature importances
    feature_importances = extract_feature_importance(champion_model, list(X.columns))

    # 8. Package production model bundle
    artifact = {
        "model": champion_model,
        "model_name": champion_name,
        "scaler": fitted_scaler,
        "feature_names": list(X.columns),
        "target_col": "Class",
        "default_threshold": 0.50,
        "recommended_threshold": 0.35,  # Slightly lower threshold minimizes missed fraud
        "metrics": results[champion_name]["metrics"],
        "all_model_metrics": {k: v["metrics"] for k, v in results.items()},
        "feature_importances": feature_importances,
        "trained_date": pd.Timestamp.now().isoformat()
    }

    # 9. Save bundle using joblib
    save_model_artifact(artifact, output_model_path)
    print(f"\n[SUCCESS] Pipeline completed successfully. Production model ready at '{output_model_path}'.\n")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Training Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/creditcard.csv",
        help="Path to creditcard.csv dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/fraud_detection_model.pkl",
        help="Destination path for saved model bundle"
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Enable SMOTE oversampling on training partition"
    )
    args = parser.parse_args()

    # If full dataset not found, check if user has sample data and prompt
    if not os.path.exists(args.data):
        sample_path = "data/sample_transactions.csv"
        if os.path.exists(sample_path):
            print(f"[WARN] Full dataset '{args.data}' not found.")
            print(f"[INFO] You can download the full dataset following 'data/README.md'.")
            print(f"[INFO] Running demonstration training using sample transactions at '{sample_path}'...")
            args.data = sample_path
        else:
            # Let load_data raise actionable FileNotFoundError
            pass

    run_training_pipeline(
        data_path=args.data,
        output_model_path=args.output,
        use_smote=args.smote
    )
