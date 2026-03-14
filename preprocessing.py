"""
preprocessing.py
================
Responsible for:
  - Loading the CSV dataset
  - Handling missing values and outliers
  - Engineering financial ratios from raw metrics
  - Scaling features for ML consumption
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Raw numeric columns available in the dataset
RAW_COLS = [
    "current_assets", "current_liabilities", "inventory",
    "total_liabilities", "total_assets", "total_debt",
    "total_equity", "net_profit", "revenue", "ebit",
    "interest_expense", "operating_cash_flow", "free_cash_flow", "eps",
]

# Derived ratio definitions: (numerator_col, denominator_col)
RATIO_MAP = {
    "current_ratio":      ("current_assets",     "current_liabilities"),
    "quick_ratio":        (None, None),                  # computed manually
    "debt_to_equity":     ("total_debt",          "total_equity"),
    "debt_to_assets":     ("total_liabilities",   "total_assets"),
    "profit_margin":      ("net_profit",           "revenue"),
    "return_on_assets":   ("net_profit",           "total_assets"),
    "interest_coverage":  ("ebit",                "interest_expense"),
    "ocf_to_liabilities": ("operating_cash_flow", "total_liabilities"),
    "fcf_to_revenue":     ("free_cash_flow",      "revenue"),
}

# Features passed to the ML models
FEATURE_COLS = [
    "current_ratio", "quick_ratio", "debt_to_equity", "debt_to_assets",
    "profit_margin", "return_on_assets", "interest_coverage",
    "ocf_to_liabilities", "fcf_to_revenue",
]


def _safe_div(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
    """Divide two series safely, replacing division-by-zero with fill."""
    return np.where(den.abs() > 1e-9, num / den, fill)


def load_and_preprocess(path: str) -> pd.DataFrame:
    """
    Load CSV, clean data, compute financial ratios.
    Returns a DataFrame with both raw and ratio columns, one row per record.
    """
    df = pd.read_csv(path)

    # Coerce all raw columns to numeric (non-parseable → NaN)
    for col in RAW_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Missing value handling ──────────────────────────────────────────────
    # Strategy: fill with per-company median first, then global median
    df[RAW_COLS] = df.groupby("company_name")[RAW_COLS].transform(
        lambda x: x.fillna(x.median())
    )
    df[RAW_COLS] = df[RAW_COLS].fillna(df[RAW_COLS].median())

    # ── Outlier winsorisation (clip to 1st–99th percentile) ────────────────
    for col in RAW_COLS:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

    # ── Compute financial ratios ────────────────────────────────────────────
    df["current_ratio"] = _safe_div(df["current_assets"], df["current_liabilities"])
    df["quick_ratio"]   = _safe_div(
        df["current_assets"] - df["inventory"], df["current_liabilities"]
    )
    df["debt_to_equity"]     = _safe_div(df["total_debt"],          df["total_equity"])
    df["debt_to_assets"]     = _safe_div(df["total_liabilities"],   df["total_assets"])
    df["profit_margin"]      = _safe_div(df["net_profit"],          df["revenue"])
    df["return_on_assets"]   = _safe_div(df["net_profit"],          df["total_assets"])
    df["interest_coverage"]  = _safe_div(df["ebit"],                df["interest_expense"])
    df["ocf_to_liabilities"] = _safe_div(df["operating_cash_flow"], df["total_liabilities"])
    df["fcf_to_revenue"]     = _safe_div(df["free_cash_flow"],      df["revenue"])

    # Clip ratios to sensible financial bounds
    df["current_ratio"]      = df["current_ratio"].clip(0, 20)
    df["quick_ratio"]        = df["quick_ratio"].clip(0, 20)
    df["debt_to_equity"]     = df["debt_to_equity"].clip(-50, 50)
    df["interest_coverage"]  = df["interest_coverage"].clip(-100, 100)
    df["profit_margin"]      = df["profit_margin"].clip(-10, 10)
    df["return_on_assets"]   = df["return_on_assets"].clip(-5, 5)

    # Fill any NaN ratios with column median
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Extract and z-score-normalise the feature matrix used by ML models.
    Returns (X_scaled, scaler).
    """
    X = df[FEATURE_COLS].values.astype(float)
    # Catch any residual NaN
    col_meds = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_meds, np.where(nan_mask)[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def aggregate_to_company(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multi-year records to one row per company
    (mean of ratios + latest year's raw financials).
    """
    ratio_means = df.groupby("company_name")[FEATURE_COLS].mean().reset_index()
    latest_raw = (
        df.sort_values("fiscal_year")
          .groupby("company_name")[RAW_COLS + ["tradingsymbol", "fiscal_year"]]
          .last()
          .reset_index()
    )
    combined = ratio_means.merge(latest_raw, on="company_name", how="left")
    return combined
