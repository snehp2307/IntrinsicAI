"""
reverse_dcf.py
==============
Solve for the growth rate implied by the current market price.
Uses bisection search over [-30%, +150%] growth space.
"""
from .dcf_model import ValuationInput, run_dcf


def _iv_at_growth(inp: ValuationInput, g: float) -> float:
    d = {**inp.__dict__, "revenue_growth_rate": g}
    d.pop("base_fcf_override", None)
    try:
        return run_dcf(ValuationInput(**d)).intrinsic_value_per_share
    except Exception:
        return 0.0


def implied_growth_rate(inp: ValuationInput, price: float = None) -> dict:
    target = price or inp.current_price
    if target <= 0:
        return {"error": "Price must be positive."}

    lo, hi = -0.30, 1.50
    for _ in range(120):
        mid   = (lo + hi) / 2
        f_mid = _iv_at_growth(inp, mid) - target
        if abs(f_mid) < 0.5 or (hi - lo) < 1e-6:
            break
        f_lo = _iv_at_growth(inp, lo) - target
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo = mid

    implied_g = (lo + hi) / 2
    user_g    = inp.revenue_growth_rate
    diff      = implied_g - user_g

    if implied_g > user_g + 0.05:
        note = (f"Market prices in {implied_g:.1%} growth — {diff:.1%} above your estimate. "
                "Stock may be overvalued unless higher growth is achievable.")
    elif implied_g < user_g - 0.05:
        note = (f"Market prices in only {implied_g:.1%} growth — {abs(diff):.1%} below your estimate. "
                "Stock may be undervalued relative to your assumptions.")
    else:
        note = (f"Market-implied growth ({implied_g:.1%}) closely matches your estimate ({user_g:.1%}). "
                "Stock appears fairly priced.")

    table = []
    for g_pct in range(-10, 55, 5):
        g  = g_pct / 100
        iv = round(_iv_at_growth(inp, g), 2)
        mos = round((iv - target) / iv * 100, 1) if iv > 0 else None
        table.append({"growth_rate": g_pct, "intrinsic_value": iv, "margin_of_safety": mos})

    return {
        "implied_growth_rate":    round(implied_g * 100, 2),
        "user_growth_assumption": round(user_g * 100, 2),
        "growth_premium":         round(diff * 100, 2),
        "current_price":          round(target, 2),
        "iv_at_implied":          round(_iv_at_growth(inp, implied_g), 2),
        "interpretation":         note,
        "sensitivity_table":      table,
    }