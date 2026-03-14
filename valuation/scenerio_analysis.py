"""
scenario_analysis.py  —  Bear / Base / Bull DCF scenarios
"""
from .dcf_model import ValuationInput, run_dcf, dcf_to_dict

SCENARIOS = [
    dict(name="bear", label="Bear Case", color="#ef4444",
         g_delta=-0.06, margin_delta=-0.04, wacc_delta=+0.02,
         ri_delta=+0.05, tg_delta=-0.01, prob=0.25),
    dict(name="base", label="Base Case", color="#3b82f6",
         g_delta=0, margin_delta=0, wacc_delta=0,
         ri_delta=0, tg_delta=0, prob=0.50),
    dict(name="bull", label="Bull Case", color="#22c55e",
         g_delta=+0.06, margin_delta=+0.04, wacc_delta=-0.01,
         ri_delta=-0.05, tg_delta=+0.01, prob=0.25),
]


def run_scenarios(inp: ValuationInput) -> dict:
    results = {}
    ivs     = {}

    for sc in SCENARIOS:
        sc_inp = ValuationInput(
            net_income=             inp.net_income,
            depreciation=           inp.depreciation,
            amortization=           inp.amortization,
            capex=                  inp.capex,
            working_capital_change= inp.working_capital_change,
            revenue_growth_rate=    max(inp.revenue_growth_rate + sc["g_delta"],  0.0),
            operating_margin=       max(inp.operating_margin    + sc["margin_delta"], 0.0),
            tax_rate=               inp.tax_rate,
            reinvestment_rate=      max(min(inp.reinvestment_rate + sc["ri_delta"], 0.99), 0.0),
            wacc=                   max(inp.wacc + sc["wacc_delta"], 0.01),
            terminal_growth_rate=   max(inp.terminal_growth_rate + sc["tg_delta"], 0.0),
            forecast_years=         inp.forecast_years,
            current_price=          inp.current_price,
            shares_outstanding=     inp.shares_outstanding,
            net_debt=               inp.net_debt,
            base_fcf_override=      inp.base_fcf_override,
        )
        r = run_dcf(sc_inp)
        d = dcf_to_dict(r)
        results[sc["name"]] = {
            **d,
            "scenario_name": sc["label"],
            "color":         sc["color"],
            "probability":   sc["prob"],
            "assumptions": {
                "revenue_growth_rate":  round((inp.revenue_growth_rate + sc["g_delta"])*100, 1),
                "wacc":                 round((inp.wacc + sc["wacc_delta"])*100, 1),
                "operating_margin":     round((inp.operating_margin + sc["margin_delta"])*100, 1),
                "terminal_growth_rate": round((inp.terminal_growth_rate + sc["tg_delta"])*100, 1),
            },
        }
        ivs[sc["name"]] = r.intrinsic_value_per_share

    weighted_iv  = sum(ivs[sc["name"]] * sc["prob"] for sc in SCENARIOS)
    bear_iv      = ivs.get("bear", weighted_iv)
    base_iv      = ivs.get("base", weighted_iv)
    bull_iv      = ivs.get("bull", weighted_iv)
    price        = inp.current_price
    spread_pct   = round((bull_iv - bear_iv) / max(base_iv, 1) * 100, 1)
    mos_wtd      = round((weighted_iv - price) / weighted_iv * 100, 1) if weighted_iv > 0 and price > 0 else None
    uncertainty  = ("HIGH" if spread_pct > 80 else "MODERATE" if spread_pct > 40 else "LOW")
    commentary   = (
        f"Probability-weighted IV: ₹{weighted_iv:,.2f}. "
        f"Range: ₹{bear_iv:,.2f} (Bear) → ₹{bull_iv:,.2f} (Bull) — "
        f"{spread_pct:.0f}% spread. Uncertainty: {uncertainty}."
    )
    return {
        "scenarios":                results,
        "weighted_intrinsic_value": round(weighted_iv, 2),
        "current_price":            price,
        "margin_of_safety_weighted": mos_wtd,
        "commentary":               commentary,
        "summary": {
            "bear_iv":    round(bear_iv, 2),
            "base_iv":    round(base_iv, 2),
            "bull_iv":    round(bull_iv, 2),
            "spread":     round(bull_iv - bear_iv, 2),
            "spread_pct": spread_pct,
        },
    }