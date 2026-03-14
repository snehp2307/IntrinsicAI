"""
xai.py  —  Legacy XAI helpers
==============================
Provides:
  explain()            — attributions, sensitivity, waterfall  (called by app.py)
  compute_global_importance()
  compute_local_contributions()
  waterfall_data()
  compute_counterfactuals()
  build_rich_explanation()
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance as sk_perm

from risk_engine import RATIO_CONFIG, _ratio_sub_score, TOTAL_WEIGHT
from preprocessing import FEATURE_COLS

FEAT_LIST = list(RATIO_CONFIG.keys())

_BASELINE = {
    "current_ratio": 1.80, "quick_ratio": 1.30,
    "debt_to_equity": 1.50, "debt_to_assets": 0.45,
    "profit_margin": 0.06, "return_on_assets": 0.03,
    "interest_coverage": 4.00, "ocf_to_liabilities": 0.08,
    "fcf_to_revenue": 0.04,
}


def _score(ratios: dict) -> float:
    total = sum(
        _ratio_sub_score(float(ratios.get(r, _BASELINE.get(r, 5.0))), cfg) * cfg["weight"]
        for r, cfg in RATIO_CONFIG.items()
    )
    return round((total / TOTAL_WEIGHT) * 10, 3)


# ── explain() — called directly by app.py ────────────────────────────────────

def explain(ratios: dict, risk_score: float, risk_category: str) -> dict:
    """
    Return attribution, sensitivity, and waterfall data for the predict page.
    This is the legacy entry-point; newer routes use xai_engine.run_all_xai().
    """
    attributions = _attributions(ratios, risk_score)
    sensitivity  = _sensitivity(ratios)
    wf           = _waterfall(attributions, risk_score)
    return {
        "attributions": attributions,
        "sensitivity":  sensitivity,
        "waterfall":    wf,
    }


def _attributions(ratios: dict, risk_score: float) -> list:
    """Simple marginal attribution: score with vs without each feature."""
    base = _score(_BASELINE)
    out  = []
    for feat, cfg in RATIO_CONFIG.items():
        val    = ratios.get(feat, _BASELINE.get(feat, 0.0))
        sub    = _ratio_sub_score(float(val), cfg)
        w      = cfg["weight"] / TOTAL_WEIGHT
        contrib= (sub - 5.0) * w * 10   # vs neutral sub-score of 5
        out.append({
            "feature":     feat,
            "label":       cfg["label"],
            "value":       round(float(val), 4),
            "sub_score":   round(sub, 2),
            "contribution":round(contrib, 3),
            "effect":      "raises_risk" if contrib > 0 else "lowers_risk",
        })
    out.sort(key=lambda x: -abs(x["contribution"]))
    return out


def _sensitivity(ratios: dict, steps: int = 20) -> list:
    """
    One-at-a-time sensitivity: vary each feature ±50% and record score impact.
    Returns list of {feature, label, range, scores}.
    """
    out = []
    for feat, cfg in RATIO_CONFIG.items():
        base_val = ratios.get(feat, _BASELINE.get(feat, 1.0))
        lo, hi   = base_val * 0.5, base_val * 1.5
        vals     = np.linspace(lo, hi, steps)
        scores   = []
        for v in vals:
            tmp = dict(ratios); tmp[feat] = float(v)
            scores.append(round(_score(tmp), 2))
        out.append({
            "feature": feat,
            "label":   cfg["label"],
            "values":  [round(float(v), 4) for v in vals],
            "scores":  scores,
        })
    return out


def _waterfall(attributions: list, final_score: float) -> dict:
    steps   = []
    running = 50.0   # neutral baseline
    for a in attributions[:8]:
        steps.append({
            "label": a["label"],
            "start": round(running, 2),
            "delta": round(a["contribution"], 2),
            "end":   round(running + a["contribution"], 2),
            "color": "#ef4444" if a["contribution"] > 0 else "#22c55e",
        })
        running += a["contribution"]
    return {"baseline": 50.0, "final_score": round(final_score, 2), "steps": steps}


# ── Global Feature Importance ─────────────────────────────────────────────────

def compute_global_importance(df: pd.DataFrame) -> list:
    feature_cols = [c for c in FEAT_LIST if c in df.columns]
    X = df[feature_cols].fillna(0).values.astype(float)
    y = df["risk_score"].values.astype(float)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
    model.fit(X, y)

    mdi  = model.feature_importances_
    perm = sk_perm(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    pm   = perm.importances_mean
    ps   = perm.importances_std

    mdi_n  = mdi / (mdi.sum() + 1e-12)
    pm_pos = np.clip(pm, 0, None)
    pm_n   = pm_pos / (pm_pos.sum() + 1e-12)
    comb   = 0.5 * mdi_n + 0.5 * pm_n

    results = []
    for i, feat in enumerate(feature_cols):
        cfg = RATIO_CONFIG.get(feat, {})
        results.append({
            "feature": feat, "label": cfg.get("label", feat),
            "description": cfg.get("description", ""),
            "direction": cfg.get("direction", ""), "weight": cfg.get("weight", 0),
            "mdi_importance":  round(float(mdi_n[i]) * 100, 2),
            "perm_importance": round(float(pm_n[i])  * 100, 2),
            "perm_std":        round(float(ps[i]), 4),
            "combined_score":  round(float(comb[i])  * 100, 2),
        })

    results.sort(key=lambda x: -x["combined_score"])
    for rank, r in enumerate(results):
        r["rank"] = rank + 1
    return results


# ── Local Contributions ───────────────────────────────────────────────────────

def compute_local_contributions(row: pd.Series, population_df: pd.DataFrame) -> list:
    results = []
    for feat, cfg in RATIO_CONFIG.items():
        sub_col = f"sub_{feat}"
        if sub_col not in row.index:
            continue
        sub_score = float(row[sub_col])
        mean_sub  = float(population_df[sub_col].mean()) if sub_col in population_df else 5.0
        weight    = cfg["weight"] / TOTAL_WEIGHT
        contrib   = (sub_score - mean_sub) * weight * 10

        results.append({
            "feature": feat, "label": cfg["label"],
            "description": cfg["description"], "direction": cfg["direction"],
            "value": round(float(row.get(feat, 0)), 4),
            "sub_score": round(sub_score, 2),
            "mean_sub_score": round(mean_sub, 2),
            "contribution": round(contrib, 3),
            "weight_pct": round(cfg["weight"] / TOTAL_WEIGHT * 100, 1),
            "direction_flag": "increase" if contrib > 0 else "decrease",
        })

    results.sort(key=lambda x: -abs(x["contribution"]))
    total_abs = sum(abs(r["contribution"]) for r in results) + 1e-9
    for r in results:
        r["pct_of_total"] = round(abs(r["contribution"]) / total_abs * 100, 1)
    return results


# ── Waterfall data ────────────────────────────────────────────────────────────

def waterfall_data(contributions: list, baseline: float, final_score: float) -> dict:
    steps   = []
    running = baseline
    for c in contributions[:8]:
        steps.append({
            "label": c["label"],
            "start": round(running, 2),
            "delta": round(c["contribution"], 2),
            "end":   round(running + c["contribution"], 2),
            "color": "#ef4444" if c["contribution"] > 0 else "#22c55e",
        })
        running += c["contribution"]
    return {"baseline": round(baseline, 2), "final_score": round(final_score, 2), "steps": steps}


# ── Counterfactuals ───────────────────────────────────────────────────────────

def compute_counterfactuals(row: pd.Series) -> list:
    results = []
    for feat, cfg in RATIO_CONFIG.items():
        sub_col = f"sub_{feat}"
        if sub_col not in row.index:
            continue
        sub_score = float(row[sub_col])
        if sub_score < 4.0:
            continue

        current_val = float(row.get(feat, 0))
        direction   = cfg["direction"]
        weight      = cfg["weight"] / TOTAL_WEIGHT

        target_val = None; target_sub = sub_score; target_label = ""

        if direction == "higher_is_better":
            for t in sorted(cfg["thresholds"]):
                if t > current_val:
                    ts = _ratio_sub_score(t + 0.01, cfg)
                    if (sub_score - ts) >= 3:
                        target_val, target_sub, target_label = t, ts, f"Increase to ≥ {t}"
                        break
            if target_val is None:
                target_val = cfg["thresholds"][-1]
                target_sub = _ratio_sub_score(target_val, cfg)
                target_label = f"Target ≥ {target_val}"
        else:
            for t in sorted(cfg["thresholds"], reverse=True):
                if t < current_val:
                    ts = _ratio_sub_score(t - 0.01, cfg)
                    if (sub_score - ts) >= 3:
                        target_val, target_sub, target_label = t, ts, f"Reduce to ≤ {t}"
                        break
            if target_val is None:
                target_val = cfg["thresholds"][0]
                target_sub = _ratio_sub_score(target_val, cfg)
                target_label = f"Target ≤ {target_val}"

        improvement = (sub_score - target_sub) * weight * 10
        results.append({
            "feature": feat, "label": cfg["label"],
            "current_value": round(current_val, 3),
            "current_sub_score": round(sub_score, 1),
            "target_value": round(target_val, 3) if target_val is not None else None,
            "target_sub_score": round(target_sub, 1),
            "target_label": target_label,
            "score_improvement": round(improvement, 1),
            "action": f"{'Increase' if direction=='higher_is_better' else 'Reduce'} {cfg['label']} from {current_val:.3f} toward {target_val:.3f}.",
            "priority": "Critical" if sub_score >= 8 else "High" if sub_score >= 6 else "Medium",
        })

    results.sort(key=lambda x: -x["score_improvement"])
    return results[:6]


# ── Rich NL Explanation ───────────────────────────────────────────────────────

def build_rich_explanation(company_name, risk_score, risk_category,
                           contributions, counterfactuals, altman_z,
                           global_importance) -> dict:
    urgency = {"High Risk": "Immediate intervention recommended.",
               "Medium Risk": "Close monitoring advised."}.get(risk_category, "Continue current discipline.")
    top3 = [c["label"] for c in contributions if c["contribution"] > 0][:3]
    summary = (
        f"{company_name} shows a risk score of {risk_score:.0f}/100 ({risk_category}). "
        + (f"Top drivers: {', '.join(top3)}. " if top3 else "")
        + urgency
    )

    factor_stories = []
    for c in contributions[:6]:
        factor_stories.append({
            "label": c["label"], "value": c["value"],
            "sub_score": c["sub_score"], "contribution": round(c["contribution"], 2),
            "impact_pts": round(abs(c["contribution"]), 1),
            "direction": c["direction"], "direction_flag": c["direction_flag"],
            "narrative": c["description"],
            "signal": "danger" if c["sub_score"] >= 6.5 else "warning" if c["sub_score"] >= 3.5 else "safe",
        })

    az = float(altman_z) if altman_z is not None and not (isinstance(altman_z, float) and np.isnan(altman_z)) else None
    if az is not None:
        zone = "distress zone (Z < 1.1)" if az < 1.1 else "grey zone (1.1–2.6)" if az < 2.6 else "safe zone (Z > 2.6)"
        altman_comment = f"Altman Z-Score of {az:.3f} is in the {zone}."
    else:
        altman_comment = "Altman Z-Score unavailable."

    global_comment = ""
    if global_importance:
        names = ", ".join(g["label"] for g in global_importance[:3])
        global_comment = f"Dataset-wide top drivers: {names}."

    return {
        "summary": summary, "factor_stories": factor_stories,
        "altman_comment": altman_comment, "global_comment": global_comment,
        "recommendations": counterfactuals,
        "methodology": "Weighted ratio scoring + GBM adaptive model + Kernel SHAP + LIME + DiCE.",
    }