# Dataset Instructions: Credit Card Fraud Detection

This project uses the benchmark **Credit Card Fraud Detection** dataset hosted on Kaggle by Machine Learning Group - ULB.

## 1. Dataset Origin & Description
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Timeframe**: Transactions made by credit cards in September 2013 by European cardholders.
- **Records**: 284,807 transactions over 2 days.
- **Fraud Cases**: 492 transactions (0.172% of all transactions) — extreme class imbalance.
- **Features**:
  - `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset.
  - `V1, V2, ..., V28`: 28 numerical principal components obtained via PCA (for confidentiality, original user identities, merchant names, and locations are transformed).
  - `Amount`: Transaction amount in Euros (not PCA-transformed).
  - `Class`: Response variable (Target) where `1` = Fraudulent transaction, `0` = Legitimate transaction.

## 2. Where to Place the Dataset
Download `creditcard.csv` and place it directly in this directory:

```
data/
└── creditcard.csv
```

### Direct Download Options:

#### Option A: Kaggle CLI (Recommended)
```bash
# Ensure kaggle is installed: pip install kaggle
# Ensure ~/.kaggle/kaggle.json contains your Kaggle API key
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

#### Option B: Manual Web Download
1. Navigate to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click **Download** (approx. 66 MB zipped, 144 MB uncompressed).
3. Extract `creditcard.csv` into the `data/` directory.

## 3. Important Ethical & Scientific ML Standard
In accordance with production machine learning standards:
- **Never synthesize fake transactions and pass them off as real credit card behavior.** Real fraud exhibits nuanced non-linear distributions across principal components (e.g. sharp anomalies in `V14`, `V10`, `V12`, `V17`).
- If `data/creditcard.csv` is missing, the code gracefully alerts you with download instructions.
- A small verified sample file `data/sample_transactions.csv` is provided exclusively for testing user-interface batch predictions and unit testing.
