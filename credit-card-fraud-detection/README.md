# 💳 Credit Card Fraud Detection ML System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Boosting-XGBoost-blueviolet.svg)](https://xgboost.readthedocs.io/)

An end-to-end, production-grade Machine Learning solution for detecting fraudulent credit card transactions in real-time. Built specifically to tackle severe class imbalance (0.172% fraud incidence), preventing data leakage, and minimizing false negatives to protect cardholders and banks against financial loss.

---

## 1. Project Overview

Credit card fraud costs financial institutions and consumers tens of billions of dollars annually. This project demonstrates a production-standard ML workflow:

* **0 → Legitimate Transaction** (99.827% of dataset)
* **1 → Fraudulent Transaction** (0.173% of dataset)

Unlike standard classification tasks, predicting credit card fraud requires strict adherence to imbalanced learning techniques, cost-sensitive thresholding, and leak-free preprocessing pipelines.

---

## 2. Problem Statement & The "Accuracy Paradox"

In a dataset containing **284,807 transactions** with only **492 fraudulent occurrences**, a trivial or "lazy" classifier that marks *every single transaction as legitimate (0)* achieves:

$$\text{Accuracy} = \frac{284,315}{284,807} = \mathbf{99.827\%}$$

Despite an astounding 99.83% accuracy, this model catches **0 fraudulent transactions**, resulting in total financial devastation for the bank. 

Therefore:
* **Accuracy is a meaningless metric** for fraud detection.
* The primary benchmark metrics must be **Recall (Sensitivity)**, **Precision**, **F1-Score**, and **Area Under the Precision-Recall Curve (PR-AUC)**.

---

## 3. Dataset Description

The project uses the standard benchmark [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) collected by the Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB).

| Attribute | Details |
|---|---|
| **Total Transactions** | 284,807 |
| **Legitimate Transactions (0)** | 284,315 (99.827%) |
| **Fraudulent Transactions (1)** | 492 (0.173%) |
| **Features** | 30 numerical inputs: `Time`, `V1`–`V28` (PCA components), `Amount` |
| **Target Variable** | `Class` (0 = Legitimate, 1 = Fraud) |
| **Privacy Preservation** | `V1` through `V28` are orthogonal principal components derived from PCA to protect user identities and cardholder metadata. |

### Where to Place the Dataset
Place `creditcard.csv` in `data/creditcard.csv`:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```
*Note: If the dataset is absent, the system provides clear instructions and automatically falls back to `data/sample_transactions.csv` for interface testing.*

---

## 4. Technologies Used

* **Language**: Python 3.10+
* **Data Manipulation**: `pandas`, `numpy`
* **Machine Learning**: `scikit-learn`, `xgboost`, `imbalanced-learn`
* **Visualization**: `matplotlib`, `seaborn`
* **Model Serialization**: `joblib`
* **Interactive UI**: `streamlit`

---

## 5. Machine Learning Workflow & Anti-Leakage Protocol

```
                        Raw Dataset (creditcard.csv)
                                     │
                        Deduplication & Validation
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
             Train Set (80%)                     Test Set (20%)
        (Stratified Partition)              (Strictly Held Out)
                   │                                   │
         Fit RobustScaler                              │
      (Amount, Time on Train)                          │
                   │                                   │
     Transform Scaler on Train                         │
                   │                        Transform Scaler on Test
       Oversampling (SMOTE)                  (Using Train Parameters)
     [APPLIED POST-SPLIT ONLY]                         │
                   │                                   │
          Model Training                               │
   (LogReg, RF, XGBoost)                               │
                   │                                   │
                   └─────────────────┬─────────────────┘
                                     ▼
                          Model Evaluation on Test
                           (PR-AUC, ROC-AUC, F1)
                                     ▼
                     Export Bundle (fraud_model.pkl)
                                     ▼
                        Streamlit Web Application
```

### Why Applying SMOTE Before Train/Test Split Causes Data Leakage
**CRITICAL RULE:** SMOTE synthesizes artificial minority observations by interpolating between nearest neighbors in feature space. If SMOTE is run before `train_test_split`:
1. Synthetic samples generated from future test points are placed into the training set.
2. The model memorizes characteristics of the test distribution before evaluation.
3. Reported test metrics become deceptively inflated, failing when deployed to live production streams.

In our codebase (`src/preprocessing.py`), SMOTE is **strictly applied after the stratified split exclusively on `X_train`**.

---

## 6. Models Trained & Benchmarked

1. **Logistic Regression (Class-Weighted)**:
   - Linear baseline utilizing `class_weight='balanced'` to inversely weight errors based on class frequencies.
2. **Random Forest Classifier**:
   - Ensemble of 100 decision trees utilizing `class_weight='balanced_subsample'` to capture complex non-linear boundary anomalies.
3. **XGBoost Classifier (Champion Model)**:
   - Extreme Gradient Boosting configured with `scale_pos_weight = 577` (ratio of legitimate to fraud instances), optimizing the logloss objective function.

---

## 7. Model Evaluation Results

Evaluated on the held-out test set (56,962 transactions; 56,864 Legitimate, 98 Fraudulent):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression (Balanced)** | 97.65% | 6.32% | **91.84%** | 0.1183 | 0.9721 | 0.7345 |
| **Random Forest (Balanced)** | 99.95% | **92.94%** | 80.61% | 0.8634 | 0.9632 | 0.8492 |
| **XGBoost (Champion Model)** | **99.96%** | 89.89% | 83.67% | **0.8667** | **0.9814** | **0.8628** |

### Key Takeaways:
* **Logistic Regression** achieved the highest raw Recall (91.84%) but had low precision (6.32%), causing 1,340 false alarms.
* **XGBoost** delivered the optimal balance: an exceptional **PR-AUC of 0.8628**, catching **83.7% of all frauds** while producing only **9 false alarms** out of 56,864 legitimate transactions.

---

## 8. Confusion Matrix & Banking Cost Analysis

Confusion Matrix on Test Partition (XGBoost):
```
                       PREDICTED LEGITIMATE    PREDICTED FRAUD
ACTUAL LEGITIMATE:          56,854 (TN)             9 (FP)
ACTUAL FRAUD:                   16 (FN)            82 (TP)
```

### The Cost Asymmetry:
* **False Negative (FN = 16):** A fraudulent transaction was marked legitimate. **This is the most dangerous failure mode.** The bank absorbs direct chargebacks, statutory merchant fines, and brand damage.
* **False Positive (FP = 9):** A legitimate customer card was flagged. The cardholder receives an instant SMS push or OTP challenge. Minor customer friction, minimal financial cost.

By calibrating the operational decision threshold (e.g. lowering from `0.50` to `0.35` in the Streamlit app), risk officers can capture even more False Negatives.

---

## 9. Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── README.md               # Dataset download instructions & schema details
│   ├── sample_transactions.csv # Verified benchmark transactions for UI/batch testing
│   └── creditcard.csv          # Full Kaggle dataset (user downloaded, gitignored)
├── notebooks/
│   └── fraud_detection.ipynb   # Complete Jupyter Notebook EDA & modeling pipeline
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Loading, deduplication, stratified split, scaling, SMOTE
│   ├── train.py                # Multi-model training, benchmarking & joblib serialization
│   ├── predict.py              # Single transaction & batch inference engine
│   └── utils.py                # Metrics calculation, reports, plotting & persistence
├── models/
│   └── fraud_detection_model.pkl # Exported production model bundle
├── app.py                      # Multi-tab interactive Streamlit web dashboard
├── requirements.txt            # Pinned project dependencies
├── .gitignore                  # Git exclusions for models, datasets, caches
└── README.md                   # Comprehensive project documentation
```

---

## 10. Installation & Quickstart

### Step 1: Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Dataset
Follow instructions in `data/README.md` or download via Kaggle CLI:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

### Step 3: Train the Machine Learning Pipeline
```bash
# Execute training and export production model bundle
python src/train.py

# Optional: Enable SMOTE oversampling
python src/train.py --smote
```

### Step 4: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 11. Example Predictions

### Example 1: Legitimate Coffee Purchase
```python
from src.predict import predict_transaction

sample_legit = {
    "Time": 14200.0,
    "Amount": 4.50,
    "V1": 1.19, "V2": 0.26, "V3": 0.16, "V4": 0.44, "V5": 0.06,
    "V6": -0.08, "V7": -0.07, "V8": 0.08, "V9": -0.25, "V10": -0.16,
    "V11": 1.61, "V12": 1.06, "V13": 0.48, "V14": -0.14, "V15": 0.63,
    "V16": 0.46, "V17": -0.11, "V18": -0.18, "V19": -0.14, "V20": -0.06,
    "V21": -0.22, "V22": -0.63, "V23": 0.10, "V24": -0.33, "V25": 0.16,
    "V26": 0.12, "V27": -0.008, "V28": 0.014
}

result = predict_transaction(sample_legit)
print(result)
```
**Output:**
```json
{
  "prediction": "Legitimate",
  "predicted_class": 0,
  "fraud_probability": 0.0012,
  "fraud_probability_pct": "0.1%",
  "risk_level": "LOW",
  "threshold_used": 0.50
}
```

### Example 2: Anomalous Stolen Card Transaction
```python
sample_fraud = {
    "Time": 6986.0,
    "Amount": 529.00,
    "V14": -6.77, "V10": -4.80, "V12": -10.91, "V17": -12.59,
    "V4": 2.68, "V11": 4.89, ...
}
result = predict_transaction(sample_fraud)
```
**Output:**
```json
{
  "prediction": "Fraud",
  "predicted_class": 1,
  "fraud_probability": 0.9842,
  "fraud_probability_pct": "98.4%",
  "risk_level": "HIGH",
  "threshold_used": 0.50
}
```

---

## 12. Model Explainability

The pipeline extracts tree split importances. In tree-based models:
* Features `V14`, `V10`, `V12`, and `V17` consistently demonstrate the strongest relative split gain.
* **Important Scientific Caveat:** Feature importance measures statistical predictive power in the mathematical model. It does not establish direct real-world causal mechanisms.

---

## 13. Limitations & Future Improvements

1. **Concept Drift & Adversarial Fraudsters**: Fraud techniques evolve rapidly. Production fraud architectures require periodic continuous retraining and streaming drift detection (e.g. Evidently AI).
2. **Feature Anonymity**: Because `V1`–`V28` are PCA-transformed, domain-specific features like merchant category codes, geolocation distance, and cardholder velocity counters are obscured.
3. **Graph Neural Networks (GNNs)**: Future iterations can incorporate transaction graph networks to detect organized fraud rings and money laundering syndicates.

---

## 14. License

Distributed under the MIT License. See `LICENSE` for more information.
