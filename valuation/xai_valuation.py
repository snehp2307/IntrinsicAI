"""
xai_valuation.py
================
Explainable AI layer for valuation outputs.

Provides:
  - driver_summary()        : Human-readable key driver breakdown
  - sensitivity_narrative() : Natural language sensitivity impact
  - generate_explanation()  : Full NL explanation paragraph
  - wacc_sensitivity_table(): How IV changes as WACC shifts
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dcf_model import DCFResult, ValuationInput


# ── Driver labels ─────────────────────────────────────────────────────────────

_DRIVER_LABELS = {
    "revenue_growth_rate":  "Revenue Growth",
    "operating_margin":     "Operating Margin",
    "wacc":                 "Discount Rate (WACC)",
    "terminal_growth_rate": "Terminal Growth Rate",
    "reinvestment_rate":    "Reinvestment Rate",
}


def driver_summary(result: "DCFResult") -> list[dict]:
    """
    Return sorted list of valuation drivers with weights and labels.
    Example: [{"driver": "Revenue Growth", "weight": 42.0}, ...]
    """
    weights = result.driver_weights or {}
    rows = []
    for key, pct in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        rows.append({
            "key":    key,
            "driver": _DRIVER_LABELS.get(key, key.replace("_", " ").title()),
            "weight": round(pct, 1),
            "value":  round(getattr(result.inputs, key, 0) * 100, 2),
            "unit":   "%",
        })
    return rows


def sensitivity_narrative(result: "DCFResult") -> list[dict]:
    """
    For each key driver, describe the impact of a +1% change on IV.
    Returns list of narrative strings.
    """
    from .dcf_model import ValuationInput, _run_dcf_core

    base_iv = result.intrinsic_value_per_share
    inp = result.inputs
    narratives = []

    perturbations = {
        "wacc":                 (+0.01,  "increases by 1%"),
        "revenue_growth_rate":  (-0.02,  "drops by 2%"),
        "terminal_growth_rate": (-0.01,  "drops by 1%"),
        "operating_margin":     (-0.02,  "falls by 2%"),
    }

    for param, (delta, description) in perturbations.items():
        try:
            d2 = {**inp.__dict__, param: getattr(inp, param) + delta}
            d2.pop("base_fcf_override", None)
            new_iv = _run_dcf_core(ValuationInput(**d2)).intrinsic_value_per_share
            change = new_iv - base_iv
            direction = "drops" if change < 0 else "rises"
            label = _DRIVER_LABELS.get(param, param)
            narratives.append({
                "param":       param,
                "label":       label,
                "description": description,
                "base_iv":     round(base_iv, 2),
                "new_iv":      round(new_iv, 2),
                "change":      round(change, 2),
                "narrative": (
                    f"If {label} {description}, intrinsic value "
                    f"{direction} from ₹{base_iv:,.2f} → ₹{new_iv:,.2f} "
                    f"(₹{abs(change):,.2f} {direction})."
                ),
            })
        except Exception:
            continue

    return narratives


def generate_explanation(result: "DCFResult") -> str:
    """
    Produce a concise natural-language paragraph explaining the valuation.
    """
    inp   = result.inputs
    label = result.valuation_label
    iv    = result.intrinsic_value_per_share
    price = inp.current_price
    mos   = result.margin_of_safety * 100

    # Top 2 drivers
    drivers = driver_summary(result)
    top2 = [d["driver"] for d in drivers[:2]] if len(drivers) >= 2 else ["growth", "margin"]

    if label == "Undervalued":
        stance = (
            f"This stock appears undervalued. At a market price of ₹{price:,.2f}, "
            f"our DCF model estimates intrinsic value at ₹{iv:,.2f}, "
            f"implying a {abs(mos):.1f}% margin of safety. "
            f"The valuation is primarily driven by {top2[0]} and {top2[1]}, "
            f"both of which exceed what the current price implies."
        )
    elif label == "Overvalued":
        stance = (
            f"This stock appears overvalued. The market price of ₹{price:,.2f} "
            f"exceeds our DCF-derived intrinsic value of ₹{iv:,.2f} "
            f"by {abs(mos):.1f}%. The market is pricing in growth assumptions "
            f"that appear aggressive given the {top2[0]} and {top2[1]} trends."
        )
    else:
        stance = (
            f"This stock appears fairly valued. The market price of ₹{price:,.2f} "
            f"is broadly in line with our intrinsic value estimate of ₹{iv:,.2f}. "
            f"Key drivers — {top2[0]} and {top2[1]} — are consistent with "
            f"current market expectations."
        )

    wacc_line = (
        f" The discount rate (WACC) used is {inp.wacc*100:.1f}%, "
        f"with a terminal growth assumption of {inp.terminal_growth_rate*100:.1f}%."
    )

    if result.warnings:
        warn_line = f" Note: {result.warnings[0]}"
    else:
        warn_line = ""

    return stance + wacc_line + warn_line


def wacc_sensitivity_table(result: "DCFResult",
                            wacc_range: float = 0.04,
                            steps: int = 9) -> list[dict]:
    """
    Return a table showing how intrinsic value changes across a WACC range.
    wacc_range: ± range around base WACC (default ±4%)
    steps     : number of steps (default 9 → -4% to +4% in 1% increments)
    """
    from .dcf_model import ValuationInput, _run_dcf_core

    inp      = result.inputs
    base_iv  = result.intrinsic_value_per_share
    base_w   = inp.wacc
    step_sz  = (2 * wacc_range) / (steps - 1)

    table = []
    for i in range(steps):
        w = round(base_w - wacc_range + i * step_sz, 4)
        if w <= inp.terminal_growth_rate:
            continue
        try:
            d2 = {**inp.__dict__, "wacc": w}
            d2.pop("base_fcf_override", None)
            iv = _run_dcf_core(ValuationInput(**d2)).intrinsic_value_per_share
        except Exception:
            iv = 0.0
        table.append({
            "wacc":              round(w * 100, 2),
            "intrinsic_value":   round(iv, 2),
            "vs_base":           round(iv - base_iv, 2),
            "is_base":           abs(w - base_w) < 1e-5,
        })
    return table


def full_xai_payload(result: "DCFResult") -> dict:
    """
    Aggregate all XAI outputs into a single dict for the API response.
    """
    return {
        "explanation":         generate_explanation(result),
        "driver_summary":      driver_summary(result),
        "sensitivity":         sensitivity_narrative(result),
        "wacc_sensitivity":    wacc_sensitivity_table(result),
        "valuation_label":     result.valuation_label,
        "margin_of_safety":    round(result.margin_of_safety * 100, 2),
        "intrinsic_value":     round(result.intrinsic_value_per_share, 2),
        "current_price":       round(result.inputs.current_price, 2),
        "warnings":            result.warnings,
    }