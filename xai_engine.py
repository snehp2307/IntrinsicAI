"""
xai_engine.py
=============
Explainable AI Engine — SHAP, LIME, DiCE + Feature Importance

All methods gracefully fall back to built-in approximations
if the optional libraries are not installed.

Install for full functionality:
    pip install shap lime dice-ml
"""

import numpy as np
import json

# Optional imports with fallback flags
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import dice_ml
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False


# ── Feature Importance (always available) ─────────────────────────────────────

def feature_importance_explanation(model, feature_cols: list) -> dict:
    """Global feature importance from Random Forest Gini scores."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    pairs = sorted(
        zip(feature_cols, importances.tolist()),
        key=lambda x: -x[1]
    )
    total = sum(v for _, v in pairs) or 1.0

    return {
        "method": "Feature Importance",
        "available": True,
        "global_importance": [
            {
                "feature": f,
                "importance": round(v, 4),
                "pct": round(v / total * 100, 1),
            }
            for f, v in pairs
        ],
        "description": (
            "Global feature importance measures how much each financial ratio "
            "contributes to predictions across all companies, based on Gini impurity "
            "reduction in the Random Forest."
        ),
    }


def local_contribution(model, scaler, x_raw: np.ndarray, feature_cols: list,
                        feature_labels: dict) -> list:
    """Approximate local contribution per prediction."""
    x_sc = scaler.transform(x_raw.reshape(1, -1))[0]

    if hasattr(model, "feature_importances_"):
        glob_imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        glob_imp = np.abs(model.coef_[0])
    else:
        glob_imp = np.ones(len(feature_cols)) / len(feature_cols)

    contrib = glob_imp * np.abs(x_sc)
    total   = contrib.sum() or 1.0

    results = []
    for i, feat in enumerate(feature_cols):
        results.append({
            "feature":      feat,
            "label":        feature_labels.get(feat, feat),
            "raw_value":    round(float(x_raw[i]), 4),
            "contribution": round(float(contrib[i] / total * 100), 1),
            "scaled_value": round(float(x_sc[i]), 4),
            "is_risky":     feat in ["Attr16"] and float(x_raw[i]) > 0.6,
        })

    results.sort(key=lambda r: -r["contribution"])
    return results


# ── SHAP ──────────────────────────────────────────────────────────────────────

def shap_explanation(model, scaler, x_raw: np.ndarray,
                     X_train: np.ndarray, feature_cols: list,
                     feature_labels: dict) -> dict:
    """
    SHAP TreeExplainer for Random Forest.
    Falls back to approximation if shap not installed.
    """
    if not SHAP_AVAILABLE:
        return _shap_approximation(model, scaler, x_raw, feature_cols, feature_labels)

    try:
        x_sc = scaler.transform(x_raw.reshape(1, -1))

        # Use small background sample for speed
        bg_size = min(100, len(X_train))
        bg = shap.sample(X_train, bg_size)

        explainer    = shap.TreeExplainer(model, bg)
        shap_values  = explainer.shap_values(x_sc)

        # For binary classification, take class 1 (bankrupt) values
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        base_value = float(explainer.expected_value[1] if isinstance(
            explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value)

        features = []
        for i, feat in enumerate(feature_cols):
            features.append({
                "feature":    feat,
                "label":      feature_labels.get(feat, feat),
                "raw_value":  round(float(x_raw[i]), 4),
                "shap_value": round(float(sv[i]), 4),
                "impact":     "Increases Risk" if sv[i] > 0 else "Reduces Risk",
                "color":      "red" if sv[i] > 0 else "green",
            })

        features.sort(key=lambda r: -abs(r["shap_value"]))

        return {
            "method":      "SHAP (TreeExplainer)",
            "available":   True,
            "base_value":  round(base_value * 100, 2),
            "features":    features,
            "description": (
                "SHAP (SHapley Additive exPlanations) uses cooperative game theory "
                "to fairly attribute the prediction to each feature. Red bars push "
                "toward bankruptcy; green bars push toward health."
            ),
        }
    except Exception as e:
        return {**_shap_approximation(model, scaler, x_raw, feature_cols, feature_labels),
                "warning": f"SHAP calculation failed ({e}), using approximation."}


def _shap_approximation(model, scaler, x_raw, feature_cols, feature_labels):
    """Built-in SHAP approximation when library unavailable."""
    x_sc = scaler.transform(x_raw.reshape(1, -1))[0]

    if hasattr(model, "feature_importances_"):
        gi = model.feature_importances_
    else:
        gi = np.ones(len(feature_cols)) / len(feature_cols)

    # Signed contribution: positive = increases bankruptcy risk
    sv = gi * x_sc

    features = []
    for i, feat in enumerate(feature_cols):
        features.append({
            "feature":    feat,
            "label":      feature_labels.get(feat, feat),
            "raw_value":  round(float(x_raw[i]), 4),
            "shap_value": round(float(sv[i]), 4),
            "impact":     "Increases Risk" if sv[i] > 0 else "Reduces Risk",
            "color":      "red" if sv[i] > 0 else "green",
        })

    features.sort(key=lambda r: -abs(r["shap_value"]))
    prob = float(model.predict_proba(scaler.transform(x_raw.reshape(1,-1)))[0, 1])

    return {
        "method":     "SHAP (Approximation — install 'shap' for exact values)",
        "available":  not SHAP_AVAILABLE,
        "approximated": True,
        "base_value": round(prob * 100, 2),
        "features":   features,
        "description": (
            "Approximate SHAP values based on feature importance × scaled feature value. "
            "Install the 'shap' library for exact Shapley values."
        ),
    }


# ── LIME ──────────────────────────────────────────────────────────────────────

def lime_explanation(model, scaler, x_raw: np.ndarray,
                     X_train: np.ndarray, feature_cols: list,
                     feature_labels: dict) -> dict:
    """
    LIME local explanation.
    Falls back to perturbation-based approximation if not installed.
    """
    if not LIME_AVAILABLE:
        return _lime_approximation(model, scaler, x_raw, feature_cols, feature_labels)

    try:
        # LIME expects un-scaled data but scaled predictions
        def predict_fn(X):
            return model.predict_proba(scaler.transform(X))

        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_cols,
            class_names=["Healthy", "Bankrupt"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

        exp = explainer.explain_instance(
            x_raw,
            predict_fn,
            num_features=len(feature_cols),
            top_labels=1
        )

        lime_list = exp.as_list(label=1)
        features = []
        for condition, weight in lime_list:
            # Extract feature name from condition string
            feat_name = None
            for fc in feature_cols:
                if fc in condition:
                    feat_name = fc
                    break
            if feat_name is None:
                feat_name = condition[:10]

            features.append({
                "feature":   feat_name,
                "label":     feature_labels.get(feat_name, feat_name),
                "condition": condition,
                "weight":    round(float(weight), 4),
                "impact":    "Increases Risk" if weight > 0 else "Reduces Risk",
                "color":     "red" if weight > 0 else "green",
            })

        features.sort(key=lambda r: -abs(r["weight"]))

        return {
            "method":    "LIME (Local Interpretable Model-agnostic Explanations)",
            "available": True,
            "features":  features,
            "description": (
                "LIME perturbs the input slightly and fits a local linear model to "
                "explain this specific prediction. Positive weights push toward "
                "bankruptcy; negative weights suggest financial health."
            ),
        }
    except Exception as e:
        return {**_lime_approximation(model, scaler, x_raw, feature_cols, feature_labels),
                "warning": f"LIME failed ({e}), using approximation."}


def _lime_approximation(model, scaler, x_raw, feature_cols, feature_labels):
    """Perturbation-based LIME approximation."""
    n_perturb = 200
    np.random.seed(42)
    x_sc = scaler.transform(x_raw.reshape(1, -1))[0]
    base_prob = float(model.predict_proba(x_sc.reshape(1, -1))[0, 1])

    weights = []
    for i in range(len(feature_cols)):
        delta = np.zeros(len(feature_cols))
        delta[i] = x_sc[i] * 0.1 if x_sc[i] != 0 else 0.1
        perturbed = x_sc.copy()
        perturbed += delta
        new_prob = float(model.predict_proba(perturbed.reshape(1, -1))[0, 1])
        weights.append(new_prob - base_prob)

    features = []
    for i, feat in enumerate(feature_cols):
        features.append({
            "feature":   feat,
            "label":     feature_labels.get(feat, feat),
            "condition": f"{feat} = {x_raw[i]:.4f}",
            "weight":    round(float(weights[i]), 4),
            "impact":    "Increases Risk" if weights[i] > 0 else "Reduces Risk",
            "color":     "red" if weights[i] > 0 else "green",
        })

    features.sort(key=lambda r: -abs(r["weight"]))

    return {
        "method":       "LIME (Approximation — install 'lime' for exact values)",
        "available":    not LIME_AVAILABLE,
        "approximated": True,
        "features":     features,
        "description":  (
            "Approximate LIME via feature perturbation analysis. "
            "Install the 'lime' library for full Local Interpretable explanations."
        ),
    }


# ── DiCE — Counterfactual Explanations ───────────────────────────────────────

def dice_explanation(model, scaler, x_raw: np.ndarray,
                     X_train: np.ndarray, feature_cols: list,
                     feature_labels: dict, n_cfs: int = 3) -> dict:
    """
    DiCE Counterfactual Explanations.
    Answers: "What would need to change for a DIFFERENT outcome?"
    Falls back to gradient-based approximation if not installed.
    """
    if not DICE_AVAILABLE:
        return _dice_approximation(model, scaler, x_raw, feature_cols,
                                    feature_labels, n_cfs)

    try:
        import pandas as pd

        x_sc    = scaler.transform(x_raw.reshape(1, -1))[0]
        df_train = pd.DataFrame(X_train, columns=feature_cols)
        df_train["class"] = model.predict(X_train)
        df_test  = pd.DataFrame([x_sc], columns=feature_cols)

        data_dice = dice_ml.Data(
            dataframe=df_train,
            continuous_features=feature_cols,
            outcome_name="class"
        )
        model_dice = dice_ml.Model(model=model, backend="sklearn")
        exp        = dice_ml.Dice(data_dice, model_dice, method="random")

        current_pred = int(model.predict(x_sc.reshape(1, -1))[0])
        desired      = 1 - current_pred  # flip outcome

        cfs = exp.generate_counterfactuals(
            df_test, total_CFs=n_cfs, desired_class=desired
        )
        cf_df = cfs.cf_examples_list[0].final_cfs_df

        counterfactuals = []
        for _, row in cf_df.iterrows():
            changes = []
            for feat in feature_cols:
                orig = float(x_sc[feature_cols.index(feat)])
                new  = float(row[feat])
                if abs(new - orig) > 0.001:
                    orig_unscaled = float(x_raw[feature_cols.index(feat)])
                    # Inverse-transform single feature approx
                    changes.append({
                        "feature":   feat,
                        "label":     feature_labels.get(feat, feat),
                        "original":  round(orig_unscaled, 4),
                        "suggested": round(new, 4),
                        "change":    round(new - orig, 4),
                    })
            counterfactuals.append(changes)

        desired_str = "Healthy" if desired == 0 else "Bankrupt"
        return {
            "method":          "DiCE (Diverse Counterfactual Explanations)",
            "available":       True,
            "current_outcome": "Bankrupt" if current_pred == 1 else "Healthy",
            "desired_outcome": desired_str,
            "counterfactuals": counterfactuals,
            "description": (
                f"DiCE shows what minimal changes to financial ratios would flip the "
                f"prediction to '{desired_str}'. This helps identify actionable "
                f"steps to improve financial health."
            ),
        }
    except Exception as e:
        return {**_dice_approximation(model, scaler, x_raw, feature_cols,
                                       feature_labels, n_cfs),
                "warning": f"DiCE failed ({e}), using approximation."}


def _dice_approximation(model, scaler, x_raw, feature_cols, feature_labels, n_cfs=3):
    """
    Built-in counterfactual generation.
    Perturbs features toward their 'healthy' direction until prediction flips.
    """
    x_sc        = scaler.transform(x_raw.reshape(1, -1))[0]
    current_pred = int(model.predict(x_sc.reshape(1, -1))[0])
    current_prob = float(model.predict_proba(x_sc.reshape(1, -1))[0, 1])
    desired      = 1 - current_pred

    if hasattr(model, "feature_importances_"):
        gi = model.feature_importances_
    else:
        gi = np.ones(len(feature_cols)) / len(feature_cols)

    # Generate counterfactuals by moving top features toward desired direction
    counterfactuals = []
    for cf_num in range(n_cfs):
        x_cf = x_sc.copy()
        changes = []
        step_size = 0.05 * (1 + cf_num * 0.5)  # larger steps for more diverse CFs

        # Move features in order of importance
        for i in np.argsort(-gi):
            feat = feature_cols[i]
            orig = float(x_raw[i])
            # Move toward healthy range
            direction = -1 if current_pred == 1 else 1
            x_cf[i] += direction * step_size * (1 + cf_num * 0.3)

            new_pred = int(model.predict(x_cf.reshape(1, -1))[0])
            if new_pred == desired:
                # Record what changed
                for j, f in enumerate(feature_cols):
                    if abs(x_cf[j] - x_sc[j]) > 0.01:
                        orig_v = float(x_raw[j])
                        delta  = x_cf[j] - x_sc[j]
                        # Approximate inverse-transform
                        suggested = orig_v + delta * abs(orig_v) * 0.1
                        changes.append({
                            "feature":   f,
                            "label":     feature_labels.get(f, f),
                            "original":  round(orig_v, 4),
                            "suggested": round(suggested, 4),
                            "change":    round(delta, 4),
                            "direction": "↑ Increase" if delta > 0 else "↓ Decrease",
                        })
                break

        if changes:
            counterfactuals.append(changes)

    desired_str = "Healthy" if desired == 0 else "Bankrupt"
    return {
        "method":          "DiCE (Approximation — install 'dice-ml' for exact values)",
        "available":       not DICE_AVAILABLE,
        "approximated":    True,
        "current_outcome": "Bankrupt" if current_pred == 1 else "Healthy",
        "desired_outcome": desired_str,
        "counterfactuals": counterfactuals,
        "description": (
            f"Counterfactual analysis shows what changes to financial ratios would "
            f"shift the prediction to '{desired_str}'. "
            f"Install 'dice-ml' for diverse, theoretically grounded counterfactuals."
        ),
    }


# ── Run All XAI ───────────────────────────────────────────────────────────────

def run_all_xai(model, scaler, x_raw: np.ndarray, X_train: np.ndarray,
                feature_cols: list, feature_labels: dict) -> dict:
    """
    Run all available XAI methods.
    """
    results = {}

    results["feature_importance"] = feature_importance_explanation(
        model, feature_cols
    )
    results["local_contribution"] = local_contribution(
        model, scaler, x_raw, feature_cols, feature_labels
    )

    results["shap"] = shap_explanation(
        model, scaler, x_raw, X_train, feature_cols, feature_labels
    )

    results["lime"] = lime_explanation(
        model, scaler, x_raw, X_train, feature_cols, feature_labels
    )

    results["dice"] = dice_explanation(
        model, scaler, x_raw, X_train, feature_cols, feature_labels
    )

    return results


def get_xai_status() -> dict:
    return {
        "shap":    {"available": SHAP_AVAILABLE,  "install": "pip install shap"},
        "lime":    {"available": LIME_AVAILABLE,  "install": "pip install lime"},
        "dice":    {"available": DICE_AVAILABLE,  "install": "pip install dice-ml"},
    }