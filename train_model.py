"""
train_model.py
==============
Run this ONCE before starting app.py to generate all required .pkl files.

Usage:
    python train_model.py

Saves to models/:
    model_rf.pkl        — trained RandomForestClassifier
    scaler.pkl          — fitted StandardScaler
    imputer.pkl         — fitted SimpleImputer
    feature_cols.pkl    — list of feature column names
    feature_importance.pkl — sorted (feature, importance) list
    metrics.pkl         — dict of evaluation metrics
    feature_labels.pkl  — {col: human label} dict
    feature_descriptions.pkl — {col: description} dict
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/companies.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Feature definitions ───────────────────────────────────────────────────────
FEATURE_COLS = [
    "current_ratio", "quick_ratio", "debt_to_equity", "debt_to_assets",
    "profit_margin", "return_on_assets", "interest_coverage",
    "ocf_to_liabilities", "fcf_to_revenue",
]

FEATURE_LABELS = {
    "current_ratio":      "Current Ratio",
    "quick_ratio":        "Quick Ratio",
    "debt_to_equity":     "Debt-to-Equity Ratio",
    "debt_to_assets":     "Debt-to-Assets Ratio",
    "profit_margin":      "Profit Margin",
    "return_on_assets":   "Return on Assets",
    "interest_coverage":  "Interest Coverage Ratio",
    "ocf_to_liabilities": "Operating CF / Liabilities",
    "fcf_to_revenue":     "Free CF / Revenue",
}

FEATURE_DESCRIPTIONS = {
    "current_ratio":      "Current Assets / Current Liabilities — measures short-term liquidity.",
    "quick_ratio":        "(Current Assets - Inventory) / Current Liabilities — conservative liquidity.",
    "debt_to_equity":     "Total Debt / Total Equity — financial leverage indicator.",
    "debt_to_assets":     "Total Liabilities / Total Assets — proportion of assets funded by debt.",
    "profit_margin":      "Net Profit / Revenue — profitability efficiency.",
    "return_on_assets":   "Net Profit / Total Assets — asset utilisation efficiency.",
    "interest_coverage":  "EBIT / Interest Expense — ability to service debt interest.",
    "ocf_to_liabilities": "Operating Cash Flow / Total Liabilities — cash generation vs obligations.",
    "fcf_to_revenue":     "Free Cash Flow / Revenue — free cash flow relative to sales.",
}

# ── Load and engineer features ────────────────────────────────────────────────
def _safe_div(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    return np.where(den.abs() > 1e-9, num / den, fill)

def load_data():
    print(f"[1/6] Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Loaded {len(df):,} rows, {df['company_name'].nunique():,} companies.")

    # Coerce numeric
    raw_cols = [
        "current_assets", "current_liabilities", "inventory",
        "total_liabilities", "total_assets", "total_debt", "total_equity",
        "net_profit", "revenue", "ebit", "interest_expense",
        "operating_cash_flow", "free_cash_flow",
    ]
    for col in raw_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing per company then globally
    df[raw_cols] = df.groupby("company_name")[raw_cols].transform(
        lambda x: x.fillna(x.median())
    )
    df[raw_cols] = df[raw_cols].fillna(df[raw_cols].median())

    # Winsorise
    for col in raw_cols:
        if col in df.columns:
            lo, hi = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lo, hi)

    # Compute ratios
    df["current_ratio"]      = _safe_div(df["current_assets"], df["current_liabilities"])
    df["quick_ratio"]        = _safe_div(df["current_assets"] - df["inventory"], df["current_liabilities"])
    df["debt_to_equity"]     = _safe_div(df["total_debt"], df["total_equity"])
    df["debt_to_assets"]     = _safe_div(df["total_liabilities"], df["total_assets"])
    df["profit_margin"]      = _safe_div(df["net_profit"], df["revenue"])
    df["return_on_assets"]   = _safe_div(df["net_profit"], df["total_assets"])
    df["interest_coverage"]  = _safe_div(df["ebit"], df["interest_expense"])
    df["ocf_to_liabilities"] = _safe_div(df["operating_cash_flow"], df["total_liabilities"])
    df["fcf_to_revenue"]     = _safe_div(df["free_cash_flow"], df["revenue"])

    # Clip ratios
    df["current_ratio"]     = df["current_ratio"].clip(0, 20)
    df["quick_ratio"]       = df["quick_ratio"].clip(0, 20)
    df["debt_to_equity"]    = df["debt_to_equity"].clip(-50, 50)
    df["interest_coverage"] = df["interest_coverage"].clip(-100, 100)
    df["profit_margin"]     = df["profit_margin"].clip(-10, 10)
    df["return_on_assets"]  = df["return_on_assets"].clip(-5, 5)

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    return df

def create_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create distress labels using Altman Z-Score proxy and cash-flow signals.
    Company-year is labelled DISTRESSED (1) if:
      - Altman Z < 1.81 (distress zone), OR
      - negative profit_margin AND interest_coverage < 1.5
    """
    ta  = df["total_assets"].replace(0, np.nan)
    tl  = df["total_liabilities"].replace(0, np.nan)
    wc  = df["current_assets"] - df["current_liabilities"]

    X1 = wc / ta
    X2 = df["net_profit"] / ta
    X3 = df["ebit"] / ta
    X4 = df["total_equity"] / tl.fillna(1e-6)
    X5 = df["revenue"] / ta
    z  = (1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5).fillna(2.0)

    altman_distress = z < 1.81

    cashflow_distress = (
        (df["profit_margin"] < 0) &
        (df["interest_coverage"] < 1.5)
    )

    distressed = (altman_distress | cashflow_distress).astype(int)
    print(f"      Labels: {distressed.sum():,} distressed ({distressed.mean()*100:.1f}%), "
          f"{(1-distressed).sum():,} healthy.")
    return distressed

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    df    = load_data()
    y     = create_labels(df)
    X_raw = df[FEATURE_COLS].values.astype(float)

    print("[2/6] Imputing and scaling ...")
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_imp)

    print("[3/6] Train / test split (80/20, stratified) ...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.20, random_state=42, stratify=y
    )

    print("[4/6] Training Random Forest (200 trees) ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    print("[5/6] Evaluating ...")
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]

    metrics = {
        "Random Forest": {
            "accuracy":  round(accuracy_score(y_te, y_pred) * 100, 2),
            "precision": round(precision_score(y_te, y_pred, zero_division=0) * 100, 2),
            "recall":    round(recall_score(y_te, y_pred, zero_division=0) * 100, 2),
            "f1":        round(f1_score(y_te, y_pred, zero_division=0) * 100, 2),
            "roc_auc":   round(roc_auc_score(y_te, y_prob), 4),
        }
    }
    m = metrics["Random Forest"]
    print(f"      Accuracy {m['accuracy']}%  Precision {m['precision']}%  "
          f"Recall {m['recall']}%  F1 {m['f1']}%  AUC {m['roc_auc']}")

    feat_imp = sorted(
        zip(FEATURE_COLS, rf.feature_importances_.tolist()),
        key=lambda x: -x[1]
    )

    print("[6/6] Saving model artefacts to models/ ...")
    def save(obj, name):
        with open(os.path.join(MODELS_DIR, name), "wb") as f:
            pickle.dump(obj, f)
        print(f"      ✓ {name}")

    save(rf,               "model_rf.pkl")
    save(scaler,           "scaler.pkl")
    save(imputer,          "imputer.pkl")
    save(FEATURE_COLS,     "feature_cols.pkl")
    save(feat_imp,         "feature_importance.pkl")
    save(metrics,          "metrics.pkl")
    save(FEATURE_LABELS,   "feature_labels.pkl")
    save(FEATURE_DESCRIPTIONS, "feature_descriptions.pkl")

    print("\n✅  Training complete! You can now run:  python app.py")
    return metrics

if __name__ == "__main__":
    train()