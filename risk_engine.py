"""
risk_engine.py
==============
Implements the financial risk scoring system:
  1. Weighted ratio scoring model  (0–100 scale)
  2. Altman Z-score approximation  (proxy bankruptcy score)
  3. Isolation Forest anomaly flag  (merged from clustering module)
  4. Plain-language XAI explanations

Methodology
-----------
Each financial ratio is mapped to a distress sub-score (0–10, higher = riskier)
using domain-expert thresholds, then weighted and summed to produce a final
0–100 Risk Score.  The Altman Z-Score is computed where possible and used as
a tiebreaker / secondary signal.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from preprocessing import FEATURE_COLS


# ── Ratio sub-scoring thresholds ─────────────────────────────────────────────
# (threshold_low, threshold_high, weight)
# Sub-score: 0 = healthy, 10 = distressed
# Reversed for ratios where higher is better (liquidity, coverage, profitability)

RATIO_CONFIG = {
    # Liquidity — higher is safer
    "current_ratio": {
        "thresholds": [0.5, 1.0, 1.5, 2.0, 2.5],   # danger → safe
        "direction": "higher_is_better",
        "weight": 0.18,
        "label": "Current Ratio",
        "description": "Measures ability to cover short-term obligations",
    },
    "quick_ratio": {
        "thresholds": [0.3, 0.7, 1.0, 1.5, 2.0],
        "direction": "higher_is_better",
        "weight": 0.12,
        "label": "Quick Ratio",
        "description": "Liquidity excluding inventory",
    },
    # Leverage — lower is safer
    "debt_to_equity": {
        "thresholds": [1.0, 2.0, 3.0, 5.0, 8.0],   # safe → danger
        "direction": "lower_is_better",
        "weight": 0.15,
        "label": "Debt-to-Equity",
        "description": "Financial leverage and solvency",
    },
    "debt_to_assets": {
        "thresholds": [0.2, 0.4, 0.6, 0.75, 0.9],
        "direction": "lower_is_better",
        "weight": 0.12,
        "label": "Debt-to-Assets",
        "description": "Proportion of assets financed by debt",
    },
    # Profitability — higher is safer
    "profit_margin": {
        "thresholds": [-0.5, -0.1, 0.0, 0.05, 0.15],
        "direction": "higher_is_better",
        "weight": 0.15,
        "label": "Profit Margin",
        "description": "Net profit as a share of revenue",
    },
    "return_on_assets": {
        "thresholds": [-0.1, -0.02, 0.0, 0.03, 0.08],
        "direction": "higher_is_better",
        "weight": 0.10,
        "label": "Return on Assets",
        "description": "Efficiency of asset utilisation",
    },
    # Debt service — higher coverage = safer
    "interest_coverage": {
        "thresholds": [-5.0, 0.0, 1.5, 3.0, 5.0],
        "direction": "higher_is_better",
        "weight": 0.10,
        "label": "Interest Coverage",
        "description": "Ability to service interest payments from EBIT",
    },
    # Cash flow — higher is safer
    "ocf_to_liabilities": {
        "thresholds": [-0.1, 0.0, 0.05, 0.1, 0.2],
        "direction": "higher_is_better",
        "weight": 0.05,
        "label": "Operating CF / Liabilities",
        "description": "Cash generation vs. total obligations",
    },
    "fcf_to_revenue": {
        "thresholds": [-0.2, -0.05, 0.0, 0.05, 0.1],
        "direction": "higher_is_better",
        "weight": 0.03,
        "label": "Free CF / Revenue",
        "description": "Free cash flow efficiency relative to revenue",
    },
}

TOTAL_WEIGHT = sum(v["weight"] for v in RATIO_CONFIG.values())


def _ratio_sub_score(value: float, config: dict) -> float:
    """
    Map a single ratio value to a 0–10 distress sub-score.
    Uses 5-level thresholds for granularity.
    """
    thresholds = config["thresholds"]
    direction  = config["direction"]

    if direction == "higher_is_better":
        if value >= thresholds[4]:   return 0.0
        elif value >= thresholds[3]: return 2.0
        elif value >= thresholds[2]: return 4.0
        elif value >= thresholds[1]: return 6.5
        elif value >= thresholds[0]: return 8.5
        else:                        return 10.0
    else:  # lower_is_better
        if value <= thresholds[0]:   return 0.0
        elif value <= thresholds[1]: return 2.0
        elif value <= thresholds[2]: return 4.0
        elif value <= thresholds[3]: return 6.5
        elif value <= thresholds[4]: return 8.5
        else:                        return 10.0


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-company risk scores (0–100) and categories.
    Adds columns: risk_score, risk_category, altman_z,
                  sub_score_<ratio>, weighted_<ratio>.
    """
    df = df.copy()

    weighted_scores = []
    for ratio, cfg in RATIO_CONFIG.items():
        if ratio not in df.columns:
            df[f"sub_{ratio}"] = 5.0
            df[f"w_{ratio}"]   = 5.0 * cfg["weight"]
            continue

        df[f"sub_{ratio}"] = df[ratio].apply(lambda v: _ratio_sub_score(v, cfg))
        df[f"w_{ratio}"]   = df[f"sub_{ratio}"] * cfg["weight"]
        weighted_scores.append(df[f"w_{ratio}"])

    raw_score = sum(weighted_scores) / TOTAL_WEIGHT  # 0–10
    df["risk_score"] = (raw_score * 10).round(1)     # scale to 0–100

    # Altman Z-score approximation (modified for non-US markets)
    # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    # X1 = Working capital / Total assets
    # X2 = Retained earnings proxy (net_profit / total_assets)
    # X3 = EBIT / Total assets
    # X4 = Equity / Total liabilities
    # X5 = Revenue / Total assets
    if all(c in df.columns for c in ["current_assets","current_liabilities",
                                      "total_assets","net_profit","ebit",
                                      "total_equity","total_liabilities","revenue"]):
        ta = df["total_assets"].replace(0, np.nan)
        tl = df["total_liabilities"].replace(0, np.nan)
        X1 = (df["current_assets"] - df["current_liabilities"]) / ta
        X2 = df["net_profit"] / ta
        X3 = df["ebit"] / ta
        X4 = df["total_equity"] / tl
        X5 = df["revenue"] / ta
        df["altman_z"] = (1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5).fillna(0).round(3)
    else:
        df["altman_z"] = np.nan

    # Risk category
    df["risk_category"] = pd.cut(
        df["risk_score"],
        bins=[-1, 33, 60, 101],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    return df


def get_top_risk_factors(row: pd.Series, top_n: int = 4) -> list:
    """
    Return the top_n ratio sub-scores (most distressed) for a company row.
    Returns list of dicts: {label, sub_score, value, description}.
    """
    factors = []
    for ratio, cfg in RATIO_CONFIG.items():
        sub_col = f"sub_{ratio}"
        if sub_col in row.index:
            factors.append({
                "ratio":       ratio,
                "label":       cfg["label"],
                "sub_score":   float(row[sub_col]),
                "value":       round(float(row.get(ratio, 0)), 4),
                "weight":      cfg["weight"],
                "description": cfg["description"],
            })
    factors.sort(key=lambda x: -x["sub_score"])
    return factors[:top_n]


def build_explanation(risk_score: float, risk_category: str,
                      top_factors: list, altman_z: float) -> str:
    """
    Generate a plain-language financial risk explanation.
    """
    lines = []

    # Overall verdict
    if risk_category == "High Risk":
        lines.append(
            f"This company shows significant signs of financial distress "
            f"with a risk score of {risk_score:.0f}/100."
        )
    elif risk_category == "Medium Risk":
        lines.append(
            f"This company exhibits moderate financial stress "
            f"(risk score {risk_score:.0f}/100), warranting close monitoring."
        )
    else:
        lines.append(
            f"This company appears financially stable "
            f"(risk score {risk_score:.0f}/100) with no immediate distress signals."
        )

    # Key drivers
    high_factors = [f for f in top_factors if f["sub_score"] >= 6.5]
    med_factors  = [f for f in top_factors if 3.5 <= f["sub_score"] < 6.5]

    if high_factors:
        names = " and ".join(f["label"] for f in high_factors[:2])
        lines.append(f"Primary risk drivers: {names} indicate elevated distress.")

    if med_factors:
        names = ", ".join(f["label"] for f in med_factors[:2])
        lines.append(f"Secondary concerns: {names} require attention.")

    # Altman Z context
    if not np.isnan(altman_z):
        if altman_z < 1.1:
            lines.append(
                f"The Altman Z-Score of {altman_z:.2f} places this company "
                "in the distress zone (Z < 1.1)."
            )
        elif altman_z < 2.6:
            lines.append(
                f"The Altman Z-Score of {altman_z:.2f} places this company "
                "in the grey zone (1.1 – 2.6)."
            )
        else:
            lines.append(
                f"The Altman Z-Score of {altman_z:.2f} indicates relative "
                "financial health (Z > 2.6)."
            )

    return " ".join(lines)


def run_isolation_forest(X_scaled: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """
    Detect statistical outliers in financial feature space.
    Returns boolean array: True = anomaly (high-risk outlier).
    """
    iso = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X_scaled)
    return preds == -1  # -1 means anomaly in sklearn convention
