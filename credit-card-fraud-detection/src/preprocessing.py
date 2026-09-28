"""
Credit Card Fraud Detection - Preprocessing Pipeline
===================================================
Handles data loading, validation, deduplication, missing value handling,
stratified train-test splitting, feature scaling, and SMOTE oversampling.

CRITICAL ML INTEGRITY RULES ENFORCED:
1. No Data Leakage: Scalers (RobustScaler) are fit strictly on X_train only,
   then applied (transform) to X_test and future real-world transactions.
2. SMOTE Applied Only Post-Split: Oversampling occurs exclusively on X_train.
   Applying SMOTE before train_test_split would inject synthetic points
   interpolated between train and test distributions into both sets, causing
   severe data leakage and unrealistic optimism.
"""

import os
from typing import Tuple, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE


REQUIRED_FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]
TARGET_COLUMN = "Class"


def load_data(filepath: str = "data/creditcard.csv") -> pd.DataFrame:
    """
    Loads the credit card transactions dataset from the specified CSV filepath.
    If not found, provides an explicit, actionable error message.

    Parameters:
        filepath: Path to the creditcard.csv dataset.

    Returns:
        pd.DataFrame containing the 284,807 transaction records.

    Raises:
        FileNotFoundError: If the dataset is absent at the specified location.
    """
    if not os.path.exists(filepath):
        # Also check alternative common relative paths
        alt_path = os.path.join(os.path.dirname(__file__), "..", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            error_msg = (
                f"\n{'='*70}\n"
                f"DATASET NOT FOUND: '{filepath}'\n"
                f"{'='*70}\n"
                f"Please download the official Credit Card Fraud Detection dataset:\n"
                f"1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                f"2. Or run CLI: kaggle datasets download -d mlg-ulb/creditcardfraud\n"
                f"3. Unzip and place 'creditcard.csv' into the '{os.path.dirname(filepath) or 'data'}' directory.\n"
                f"\nNOTE: To run unit tests or UI demos without the 150MB file,\n"
                f"use the sample file at 'data/sample_transactions.csv'.\n"
                f"{'='*70}\n"
            )
            raise FileNotFoundError(error_msg)

    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Successfully loaded {df.shape[0]:,} transactions with {df.shape[1]} columns.")
    return df


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Validates schema, checks for missing values, and handles duplicates.

    Parameters:
        df: Input DataFrame.
        drop_duplicates: Whether to drop identical transaction rows.
                         The Kaggle dataset contains 1,081 duplicate entries.

    Returns:
        Cleaned pd.DataFrame.
    """
    df_clean = df.copy()

    # 1. Missing Values Check
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        print(f"[WARN] Found {missing_count} missing values. Imputing with median.")
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    else:
        print("[INFO] Zero missing values detected across all columns.")

    # 2. Duplicate Transactions Check
    duplicate_count = df_clean.duplicated().sum()
    print(f"[INFO] Detected {duplicate_count:,} duplicate transaction rows.")
    if drop_duplicates and duplicate_count > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"[INFO] Dropped duplicates. New transaction count: {df_clean.shape[0]:,}")

    return df_clean


def split_features_target(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates feature matrix X and target vector y.

    Parameters:
        df: Transaction DataFrame.
        target_col: Name of the binary target column ('Class').

    Returns:
        (X, y) tuple.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    fraud_count = y.sum()
    legit_count = len(y) - fraud_count
    fraud_pct = (fraud_count / len(y)) * 100

    print(f"[INFO] Class Distribution:")
    print(f"       - Legitimate (0): {legit_count:,} ({100 - fraud_pct:.3f}%)")
    print(f"       - Fraudulent (1): {fraud_count:,} ({fraud_pct:.3f}%)")

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train-test split to ensure identical fraud prevalence
    in both training and testing partitions.

    Parameters:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion allocated to testing (default: 0.20).
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Essential for imbalanced fraud datasets
    )

    print(f"[INFO] Stratified Split Completed:")
    print(f"       - Train set: {X_train.shape[0]:,} records (Frauds: {y_train.sum():,})")
    print(f"       - Test set:  {X_test.shape[0]:,} records (Frauds: {y_test.sum():,})")

    return X_train, X_test, y_train, y_test


def fit_and_scale_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    columns_to_scale: Optional[List[str]] = None,
    scaler_type: str = "robust"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Union[RobustScaler, StandardScaler]]:
    """
    Fits feature scaler strictly on X_train to prevent data leakage.
    PCA features V1-V28 are already standardized/centered; only 'Amount' and 'Time'
    exhibit high variance and heavy skewness. RobustScaler is used by default
    because transaction amounts have extreme outliers.

    Parameters:
        X_train: Training feature DataFrame.
        X_test: Testing feature DataFrame (optional).
        columns_to_scale: List of columns to scale (default: ['Amount', 'Time']).
        scaler_type: 'robust' (RobustScaler) or 'standard' (StandardScaler).

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    if columns_to_scale is None:
        columns_to_scale = [c for c in ["Amount", "Time"] if c in X_train.columns]

    if scaler_type.lower() == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    if columns_to_scale:
        # FIT ONLY ON TRAINING SET
        scaler.fit(X_train[columns_to_scale])
        X_train_scaled[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
        print(f"[INFO] Fitted {scaler.__class__.__name__} strictly on X_train for {columns_to_scale}.")

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = X_test.copy()
        if columns_to_scale:
            # TRANSFORM ONLY USING FITTED PARAMETERS
            X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train_scaled, X_test_scaled, scaler


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) strictly to the
    training data partition.

    DATA LEAKAGE WARNING:
    Never apply SMOTE before train_test_split. If SMOTE is executed on the full
    dataset prior to splitting, synthetic minority points generated from future
    test instances will bleed into the training partition, creating artificial
    correlation and dangerously inflated test metrics.

    Parameters:
        X_train: Scaled training feature matrix.
        y_train: Training target series.
        sampling_strategy: Desired ratio of minority to majority class after resampling.
                           Default 0.20 (20% fraud ratio in training set).
        random_state: Seed for reproducibility.

    Returns:
        (X_resampled, y_resampled)
    """
    print(f"[INFO] Applying SMOTE on training partition (sampling_strategy={sampling_strategy})...")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"[INFO] SMOTE Complete:")
    print(f"       - Original Train: {len(y_train):,} (Frauds: {y_train.sum():,})")
    print(f"       - Resampled Train: {len(y_res):,} (Frauds: {y_res.sum():,})")

    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)
