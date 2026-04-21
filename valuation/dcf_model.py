"""
dcf_model.py
============
Warren Buffett Owner Earnings + Aswath Damodaran DCF Engine

References:
  Buffett (1986) Berkshire Hathaway Annual Letter
  Damodaran (2012) Investment Valuation, 3rd ed. Chapter 12
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ValuationInput:
    # Owner Earnings components (₹ Crores)
    net_income:             float = 0.0
    depreciation:           float = 0.0
    amortization:           float = 0.0
    capex:                  float = 0.0
    working_capital_change: float = 0.0

    # Growth & profitability
    revenue_growth_rate:    float = 0.10   # decimal
    operating_margin:       float = 0.15
    tax_rate:               float = 0.25
    reinvestment_rate:      float = 0.30

    # Discount
    wacc:                   float = 0.10
    terminal_growth_rate:   float = 0.05

    # Forecast
    forecast_years:         int   = 10

    # Market
    current_price:          float = 0.0
    shares_outstanding:     float = 1.0   # millions
    net_debt:               float = 0.0

    base_fcf_override:      Optional[float] = None


@dataclass
class YearProjection:
    year:            int
    growth_rate:     float
    fcf:             float
    discount_factor: float
    pv_fcf:          float


@dataclass
class DCFResult:
    inputs:                   ValuationInput
    owner_earnings:           float
    projections:              List[YearProjection]
    pv_fcfs:                  float
    terminal_value:           float
    pv_terminal_value:        float
    enterprise_value:         float
    equity_value:             float
    intrinsic_value_per_share:float
    margin_of_safety:         float
    valuation_label:          str
    driver_weights:           dict
    warnings:                 List[str] = field(default_factory=list)


def compute_owner_earnings(inp: ValuationInput) -> float:
    return (inp.net_income + inp.depreciation + inp.amortization
            - inp.capex - inp.working_capital_change)


def _declining_growth(base: float, year: int, terminal: float, n: int) -> float:
    if n <= 1:
        return terminal
    t = (year - 1) / (n - 1)
    return base * (1 - t) + terminal * t


def _run_dcf_core(inp: ValuationInput) -> DCFResult:
    """Core DCF calculation WITHOUT computing driver weights (avoids recursion)."""
    warnings = []

    if inp.wacc <= inp.terminal_growth_rate:
        warnings.append("WACC must exceed terminal growth rate. Adjusting.")
        inp = ValuationInput(**{**inp.__dict__, "terminal_growth_rate": inp.wacc - 0.005})

    owner_earnings = compute_owner_earnings(inp)
    base_fcf = inp.base_fcf_override if inp.base_fcf_override is not None else owner_earnings
    if base_fcf <= 0:
        warnings.append("Negative base FCF — DCF reliability is limited.")

    projections: List[YearProjection] = []
    pv_fcfs = 0.0
    fcf = base_fcf

    for yr in range(1, inp.forecast_years + 1):
        g   = _declining_growth(inp.revenue_growth_rate, yr,
                                 inp.terminal_growth_rate, inp.forecast_years)
        fcf = fcf * (1 + g)
        fcf_free = fcf * (1 - inp.reinvestment_rate)
        df  = 1 / ((1 + inp.wacc) ** yr)
        pv  = fcf_free * df
        projections.append(YearProjection(
            year=yr, growth_rate=round(g,6),
            fcf=round(fcf_free,4), discount_factor=round(df,6), pv_fcf=round(pv,4)
        ))
        pv_fcfs += pv

    last_fcf = projections[-1].fcf
    terminal_value = last_fcf * (1 + inp.terminal_growth_rate) / (
        inp.wacc - inp.terminal_growth_rate)
    pv_terminal = terminal_value / ((1 + inp.wacc) ** inp.forecast_years)

    enterprise_value = pv_fcfs + pv_terminal
    equity_value     = enterprise_value - inp.net_debt
    iv_per_share     = equity_value / inp.shares_outstanding if inp.shares_outstanding > 0 else 0.0

    price = inp.current_price
    if iv_per_share > 0 and price > 0:
        mos = (iv_per_share - price) / iv_per_share
    elif iv_per_share > 0:
        mos = 1.0
    else:
        mos = -1.0

    label = ("Undervalued" if mos > 0.15 else
             "Fairly Valued" if mos > -0.15 else "Overvalued")

    return DCFResult(
        inputs=inp, owner_earnings=round(owner_earnings,4),
        projections=projections, pv_fcfs=round(pv_fcfs,4),
        terminal_value=round(terminal_value,4), pv_terminal_value=round(pv_terminal,4),
        enterprise_value=round(enterprise_value,4), equity_value=round(equity_value,4),
        intrinsic_value_per_share=round(iv_per_share,4),
        margin_of_safety=round(mos,4), valuation_label=label,
        driver_weights={}, warnings=warnings,
    )


def run_dcf(inp: ValuationInput) -> DCFResult:
    """Full DCF with driver weight analysis."""
    result = _run_dcf_core(inp)
    driver_weights = _compute_driver_weights(inp, result.intrinsic_value_per_share)
    return DCFResult(
        inputs=result.inputs, owner_earnings=result.owner_earnings,
        projections=result.projections, pv_fcfs=result.pv_fcfs,
        terminal_value=result.terminal_value, pv_terminal_value=result.pv_terminal_value,
        enterprise_value=result.enterprise_value, equity_value=result.equity_value,
        intrinsic_value_per_share=result.intrinsic_value_per_share,
        margin_of_safety=result.margin_of_safety, valuation_label=result.valuation_label,
        driver_weights=driver_weights, warnings=result.warnings,
    )


def _compute_driver_weights(inp: ValuationInput, base_iv: float) -> dict:
    if base_iv == 0:
        return {k: 0.0 for k in ["revenue_growth_rate","operating_margin",
                                   "wacc","terminal_growth_rate","reinvestment_rate"]}
    perturbations = {
        "revenue_growth_rate":  0.02,
        "operating_margin":     0.02,
        "wacc":                 0.01,
        "terminal_growth_rate": 0.005,
        "reinvestment_rate":    0.05,
    }
    sensitivities = {}
    for param, delta in perturbations.items():
        d2 = {**inp.__dict__, param: getattr(inp, param) + delta}
        d2.pop("base_fcf_override", None)
        try:
            r2 = _run_dcf_core(ValuationInput(**d2))
            sensitivities[param] = abs(r2.intrinsic_value_per_share - base_iv)
        except Exception:
            sensitivities[param] = 0.0
    total = sum(sensitivities.values()) or 1.0
    return {k: round(v / total * 100, 1) for k, v in sensitivities.items()}


def dcf_to_dict(r: DCFResult) -> dict:
    return {
        "owner_earnings":            r.owner_earnings,
        "pv_fcfs":                   round(r.pv_fcfs, 2),
        "terminal_value":            round(r.terminal_value, 2),
        "pv_terminal_value":         round(r.pv_terminal_value, 2),
        "enterprise_value":          round(r.enterprise_value, 2),
        "equity_value":              round(r.equity_value, 2),
        "intrinsic_value_per_share": round(r.intrinsic_value_per_share, 2),
        "margin_of_safety":          round(r.margin_of_safety * 100, 2),
        "valuation_label":           r.valuation_label,
        "driver_weights":            r.driver_weights,
        "warnings":                  r.warnings,
        "projections": [{
            "year":        p.year,
            "growth_rate": round(p.growth_rate * 100, 2),
            "fcf":         round(p.fcf, 2),
            "pv_fcf":      round(p.pv_fcf, 2),
        } for p in r.projections],
    }