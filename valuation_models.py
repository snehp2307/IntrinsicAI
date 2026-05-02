"""
valuation.py
============
Industry-Grade Financial Valuation Models

Models implemented:
  1. Altman Z-Score         — Classic bankruptcy prediction (1968)
  2. Ohlson O-Score         — Logit-based bankruptcy probability (1980)
  3. Beneish M-Score        — Earnings manipulation detection (1999)
  4. Piotroski F-Score      — Financial strength scoring (2000)
  5. DCF Analysis           — Discounted Cash Flow intrinsic value
  6. Liquidity Ratios       — Current, Quick, Cash ratio analysis
  7. Debt Coverage          — Interest coverage & debt service ratios
"""

import math


# ── 1. Altman Z-Score ─────────────────────────────────────────────────────────

def altman_z_score(working_capital, total_assets, retained_earnings,
                    ebit, market_cap, total_liabilities, sales) -> dict:
    """
    Altman Z-Score for public companies.
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    Interpretation:
      Z > 2.99  → Safe Zone
      1.81–2.99 → Grey Zone
      Z < 1.81  → Distress Zone
    """
    if total_assets == 0:
        return {"error": "Total assets cannot be zero"}

    x1 = working_capital  / total_assets       # Liquidity
    x2 = retained_earnings / total_assets      # Profitability
    x3 = ebit / total_assets                   # Operating efficiency
    x4 = market_cap / max(total_liabilities, 0.001)  # Leverage
    x5 = sales / total_assets                  # Asset turnover

    z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

    if z > 2.99:
        zone = "Safe Zone"
        risk = "Low"
        color = "green"
        interpretation = "Company is financially healthy. Bankruptcy is unlikely in the near term."
    elif z > 1.81:
        zone = "Grey Zone"
        risk = "Moderate"
        color = "amber"
        interpretation = "Company shows signs of financial stress. Monitor closely."
    else:
        zone = "Distress Zone"
        risk = "High"
        color = "red"
        interpretation = "Company is in financial distress. High bankruptcy probability."

    return {
        "model": "Altman Z-Score",
        "score": round(z, 4),
        "zone": zone,
        "risk": risk,
        "color": color,
        "interpretation": interpretation,
        "components": {
            "X1 (Working Capital/Total Assets)":       round(x1, 4),
            "X2 (Retained Earnings/Total Assets)":     round(x2, 4),
            "X3 (EBIT/Total Assets)":                  round(x3, 4),
            "X4 (Market Cap/Total Liabilities)":       round(x4, 4),
            "X5 (Sales/Total Assets)":                 round(x5, 4),
        },
        "thresholds": {"safe": 2.99, "distress": 1.81},
        "weights": {"X1": 1.2, "X2": 1.4, "X3": 3.3, "X4": 0.6, "X5": 1.0},
    }


# ── 2. Ohlson O-Score ─────────────────────────────────────────────────────────

def ohlson_o_score(total_assets, total_liabilities, working_capital,
                    current_liabilities, current_assets, net_income,
                    funds_from_operations, gdp_deflated_assets=None) -> dict:
    """
    Ohlson O-Score — logistic regression bankruptcy prediction.
    Uses 9 financial ratios. Returns probability of bankruptcy.
    P(bankruptcy) = 1 / (1 + e^(-O))
    """
    if total_assets <= 0:
        return {"error": "Total assets must be positive"}

    # SIZE: log(Total Assets / GNP Price Level Index) — use log(TA) as proxy
    size = math.log(max(total_assets, 1))

    # Leverage
    tlta = total_liabilities / total_assets  # Total Liabilities / Total Assets

    # Working Capital / Total Assets
    wcta = working_capital / total_assets

    # Current Liabilities / Current Assets
    clca = current_liabilities / max(current_assets, 0.001)

    # Funds from Operations / Total Liabilities
    ffotl = funds_from_operations / max(total_liabilities, 0.001)

    # Net Income / Total Assets
    nita = net_income / total_assets

    # Change in Net Income (approximated as sign)
    intwo = 1 if net_income < 0 else 0  # 1 if net income negative

    # Negative equity indicator
    oeneg = 1 if total_liabilities > total_assets else 0

    # Change in net income indicator (simplified)
    chin = net_income / max(abs(net_income) + 0.001, 0.001)

    # Ohlson O-Score formula
    o = (
        -1.32
        - 0.407 * size
        + 6.03  * tlta
        - 1.43  * wcta
        + 0.076 * clca
        - 1.72  * oeneg
        - 2.37  * nita
        - 1.83  * ffotl
        + 0.285 * intwo
        - 0.521 * chin
    )

    prob = 1 / (1 + math.exp(-o))

    if prob < 0.20:
        risk = "Low"
        color = "green"
        interpretation = "Low probability of bankruptcy within the next year."
    elif prob < 0.50:
        risk = "Moderate"
        color = "amber"
        interpretation = "Moderate bankruptcy risk. Financial performance needs monitoring."
    else:
        risk = "High"
        color = "red"
        interpretation = "High probability of financial failure. Immediate action required."

    return {
        "model": "Ohlson O-Score",
        "score": round(o, 4),
        "probability": round(prob * 100, 2),
        "risk": risk,
        "color": color,
        "interpretation": interpretation,
        "components": {
            "SIZE (log Total Assets)":              round(size, 4),
            "TLTA (Total Liab/Total Assets)":       round(tlta, 4),
            "WCTA (Working Capital/Total Assets)":  round(wcta, 4),
            "CLCA (Current Liab/Current Assets)":   round(clca, 4),
            "FFOTL (Funds from Ops/Total Liab)":    round(ffotl, 4),
            "NITA (Net Income/Total Assets)":       round(nita, 4),
            "INTWO (Negative NI indicator)":        intwo,
            "OENEG (Negative equity indicator)":    oeneg,
            "CHIN (Change in NI)":                  round(chin, 4),
        },
    }


# ── 3. Beneish M-Score ────────────────────────────────────────────────────────

def beneish_m_score(
    receivables_t, receivables_t1, sales_t, sales_t1,
    gross_profit_t, gross_profit_t1,
    assets_t, assets_t1,
    ppe_t, ppe_t1,
    total_accruals_t, total_accruals_t1,
    sg_expense_t, sg_expense_t1,
    long_term_debt_t, long_term_debt_t1,
    current_assets_t, current_assets_t1,
    current_liabilities_t, current_liabilities_t1,
    net_income_t, cash_from_ops_t
) -> dict:
    """
    Beneish M-Score — earnings manipulation detection.
    M < -2.22 → Unlikely manipulator
    M > -1.78 → Likely manipulator
    """
    eps = 0.001  # prevent division by zero

    # DSRI: Days Sales in Receivables Index
    dsri = (receivables_t / max(sales_t, eps)) / max(receivables_t1 / max(sales_t1, eps), eps)

    # GMI: Gross Margin Index
    gmi = ((sales_t1 - gross_profit_t1) / max(sales_t1, eps)) / max(
          (sales_t - gross_profit_t) / max(sales_t, eps), eps)

    # AQI: Asset Quality Index
    aqi = (1 - (current_assets_t + ppe_t) / max(assets_t, eps)) / max(
          1 - (current_assets_t1 + ppe_t1) / max(assets_t1, eps), eps)

    # SGI: Sales Growth Index
    sgi = sales_t / max(sales_t1, eps)

    # DEPI: Depreciation Index
    depi = (ppe_t1 / max(ppe_t1 + gross_profit_t1, eps)) / max(
           (ppe_t / max(ppe_t + gross_profit_t, eps)), eps)

    # SGAI: SG&A Expense Index
    sgai = (sg_expense_t / max(sales_t, eps)) / max(sg_expense_t1 / max(sales_t1, eps), eps)

    # LVGI: Leverage Index
    lvgi = ((long_term_debt_t + current_liabilities_t) / max(assets_t, eps)) / max(
           (long_term_debt_t1 + current_liabilities_t1) / max(assets_t1, eps), eps)

    # TATA: Total Accruals to Total Assets
    tata = (net_income_t - cash_from_ops_t) / max(assets_t, eps)

    # M-Score
    m = (-4.84
         + 0.920  * dsri
         + 0.528  * gmi
         + 0.404  * aqi
         + 0.892  * sgi
         + 0.115  * depi
         - 0.172  * sgai
         + 4.679  * tata
         - 0.327  * lvgi)

    if m < -2.22:
        verdict = "Unlikely Manipulator"
        risk = "Low"
        color = "green"
        interpretation = "Financial statements appear reliable. Low manipulation risk."
    elif m < -1.78:
        verdict = "Grey Zone"
        risk = "Moderate"
        color = "amber"
        interpretation = "Some signs of potential earnings manipulation. Investigate further."
    else:
        verdict = "Likely Manipulator"
        risk = "High"
        color = "red"
        interpretation = "High likelihood of earnings manipulation. Financial statements may be unreliable."

    return {
        "model": "Beneish M-Score",
        "score": round(m, 4),
        "verdict": verdict,
        "risk": risk,
        "color": color,
        "interpretation": interpretation,
        "components": {
            "DSRI (Days Sales Receivables Index)": round(dsri, 4),
            "GMI (Gross Margin Index)":            round(gmi, 4),
            "AQI (Asset Quality Index)":           round(aqi, 4),
            "SGI (Sales Growth Index)":            round(sgi, 4),
            "DEPI (Depreciation Index)":           round(depi, 4),
            "SGAI (SG&A Expense Index)":           round(sgai, 4),
            "LVGI (Leverage Index)":               round(lvgi, 4),
            "TATA (Total Accruals/Total Assets)":  round(tata, 4),
        },
        "thresholds": {"manipulator": -1.78, "safe": -2.22},
    }


# ── 4. Piotroski F-Score ──────────────────────────────────────────────────────

def piotroski_f_score(
    net_income, total_assets, operating_cash_flow,
    roa_prev, long_term_debt, long_term_debt_prev,
    current_ratio, current_ratio_prev,
    shares_outstanding, shares_outstanding_prev,
    gross_margin, gross_margin_prev,
    asset_turnover, asset_turnover_prev
) -> dict:
    """
    Piotroski F-Score — 9-point financial strength scoring.
    Score 0–3: Weak  |  4–6: Neutral  |  7–9: Strong
    """
    scores = {}

    # PROFITABILITY (4 signals)
    roa = net_income / max(total_assets, 0.001)
    scores["ROA Positive"]         = 1 if roa > 0 else 0
    scores["Operating CF Positive"]= 1 if operating_cash_flow > 0 else 0
    scores["ROA Improving"]        = 1 if roa > roa_prev else 0
    scores["Accrual Quality"]      = 1 if operating_cash_flow / max(total_assets, 0.001) > roa else 0

    # LEVERAGE & LIQUIDITY (3 signals)
    debt_ratio      = long_term_debt / max(total_assets, 0.001)
    debt_ratio_prev = long_term_debt_prev / max(total_assets, 0.001)
    scores["Leverage Decreasing"]    = 1 if debt_ratio < debt_ratio_prev else 0
    scores["Liquidity Improving"]    = 1 if current_ratio > current_ratio_prev else 0
    scores["No Share Dilution"]      = 1 if shares_outstanding <= shares_outstanding_prev else 0

    # OPERATING EFFICIENCY (2 signals)
    scores["Gross Margin Improving"] = 1 if gross_margin > gross_margin_prev else 0
    scores["Asset Turnover Improving"]= 1 if asset_turnover > asset_turnover_prev else 0

    f_score = sum(scores.values())

    if f_score >= 7:
        verdict = "Strong"
        risk = "Low"
        color = "green"
        interpretation = "Company has strong fundamentals. Good investment or low bankruptcy risk."
    elif f_score >= 4:
        verdict = "Neutral"
        risk = "Moderate"
        color = "amber"
        interpretation = "Mixed financial signals. Monitor key metrics for improvement or deterioration."
    else:
        verdict = "Weak"
        risk = "High"
        color = "red"
        interpretation = "Weak fundamentals across multiple dimensions. High financial distress risk."

    return {
        "model": "Piotroski F-Score",
        "score": f_score,
        "out_of": 9,
        "verdict": verdict,
        "risk": risk,
        "color": color,
        "interpretation": interpretation,
        "components": {k: ("✓ Pass" if v else "✗ Fail") for k, v in scores.items()},
        "category_scores": {
            "Profitability (4 pts)": sum([
                scores["ROA Positive"], scores["Operating CF Positive"],
                scores["ROA Improving"], scores["Accrual Quality"]
            ]),
            "Leverage & Liquidity (3 pts)": sum([
                scores["Leverage Decreasing"], scores["Liquidity Improving"],
                scores["No Share Dilution"]
            ]),
            "Operating Efficiency (2 pts)": sum([
                scores["Gross Margin Improving"], scores["Asset Turnover Improving"]
            ]),
        }
    }


# ── 5. DCF Analysis ───────────────────────────────────────────────────────────

def dcf_analysis(
    free_cash_flow: float,
    growth_rate_5yr: float,
    terminal_growth_rate: float,
    wacc: float,
    net_debt: float,
    shares_outstanding: float,
    current_price: float = None
) -> dict:
    """
    Discounted Cash Flow (DCF) valuation.
    Projects FCF for 5 years, then terminal value.
    Returns intrinsic value per share and margin of safety.
    """
    if wacc <= terminal_growth_rate:
        terminal_growth_rate = wacc - 0.01

    # Project FCFs for 5 years
    projected_fcf = []
    pv_fcfs = []
    fcf = free_cash_flow

    for year in range(1, 6):
        fcf = fcf * (1 + growth_rate_5yr)
        pv  = fcf / ((1 + wacc) ** year)
        projected_fcf.append(round(fcf, 2))
        pv_fcfs.append(round(pv, 2))

    # Terminal value (Gordon Growth Model)
    terminal_fcf = projected_fcf[-1] * (1 + terminal_growth_rate)
    terminal_value = terminal_fcf / (wacc - terminal_growth_rate)
    pv_terminal = terminal_value / ((1 + wacc) ** 5)

    # Enterprise Value
    enterprise_value = sum(pv_fcfs) + pv_terminal

    # Equity Value
    equity_value = enterprise_value - net_debt
    intrinsic_value_per_share = equity_value / max(shares_outstanding, 1)

    result = {
        "model": "DCF Analysis",
        "enterprise_value": round(enterprise_value, 2),
        "equity_value": round(equity_value, 2),
        "intrinsic_value_per_share": round(intrinsic_value_per_share, 2),
        "terminal_value": round(pv_terminal, 2),
        "pv_of_fcfs": round(sum(pv_fcfs), 2),
        "projected_fcf": projected_fcf,
        "pv_fcfs": pv_fcfs,
        "assumptions": {
            "5yr Growth Rate": f"{growth_rate_5yr*100:.1f}%",
            "Terminal Growth Rate": f"{terminal_growth_rate*100:.1f}%",
            "WACC": f"{wacc*100:.1f}%",
        },
    }

    if current_price and current_price > 0:
        margin_of_safety = (intrinsic_value_per_share - current_price) / current_price * 100
        result["current_price"] = current_price
        result["margin_of_safety"] = round(margin_of_safety, 2)

        if margin_of_safety > 20:
            result["valuation"] = "Undervalued"
            result["color"] = "green"
            result["interpretation"] = f"Stock appears undervalued by {margin_of_safety:.1f}%. Potential buy opportunity."
        elif margin_of_safety > -10:
            result["valuation"] = "Fairly Valued"
            result["color"] = "amber"
            result["interpretation"] = "Stock is trading near intrinsic value."
        else:
            result["valuation"] = "Overvalued"
            result["color"] = "red"
            result["interpretation"] = f"Stock appears overvalued by {abs(margin_of_safety):.1f}%."
    else:
        result["color"] = "blue"
        result["interpretation"] = f"Estimated intrinsic value: {intrinsic_value_per_share:.2f} per share."

    return result


# ── 6. Liquidity & Coverage Ratios ───────────────────────────────────────────

def liquidity_analysis(
    current_assets, current_liabilities,
    inventory, cash, total_debt,
    ebit, interest_expense,
    operating_cash_flow
) -> dict:
    """
    Comprehensive liquidity and debt coverage analysis.
    """
    eps = 0.001

    current_ratio = current_assets / max(current_liabilities, eps)
    quick_ratio   = (current_assets - inventory) / max(current_liabilities, eps)
    cash_ratio    = cash / max(current_liabilities, eps)
    interest_cov  = ebit / max(interest_expense, eps)
    debt_service  = operating_cash_flow / max(total_debt, eps)

    def rate_ratio(name, val, thresholds, higher_better=True):
        lo, hi = thresholds
        if higher_better:
            if val >= hi: return "green", "Healthy"
            if val >= lo: return "amber", "Marginal"
            return "red", "Weak"
        else:
            if val <= lo: return "green", "Healthy"
            if val <= hi: return "amber", "Marginal"
            return "red", "Weak"

    ratios = [
        {
            "name": "Current Ratio",
            "value": round(current_ratio, 2),
            "formula": "Current Assets / Current Liabilities",
            "healthy_range": "1.5 – 3.0",
            **dict(zip(["color", "status"],
                   rate_ratio("cr", current_ratio, (1.0, 1.5)))),
            "interpretation": "Measures ability to pay short-term obligations.",
        },
        {
            "name": "Quick Ratio",
            "value": round(quick_ratio, 2),
            "formula": "(Current Assets - Inventory) / Current Liabilities",
            "healthy_range": "1.0 – 2.0",
            **dict(zip(["color", "status"],
                   rate_ratio("qr", quick_ratio, (0.8, 1.0)))),
            "interpretation": "Stricter liquidity — excludes inventory.",
        },
        {
            "name": "Cash Ratio",
            "value": round(cash_ratio, 2),
            "formula": "Cash / Current Liabilities",
            "healthy_range": "0.2 – 0.5",
            **dict(zip(["color", "status"],
                   rate_ratio("cashr", cash_ratio, (0.1, 0.2)))),
            "interpretation": "Most conservative liquidity measure.",
        },
        {
            "name": "Interest Coverage",
            "value": round(interest_cov, 2),
            "formula": "EBIT / Interest Expense",
            "healthy_range": "3.0+",
            **dict(zip(["color", "status"],
                   rate_ratio("ic", interest_cov, (1.5, 3.0)))),
            "interpretation": "Ability to cover interest payments from earnings.",
        },
        {
            "name": "Debt Service Coverage",
            "value": round(debt_service, 2),
            "formula": "Operating Cash Flow / Total Debt",
            "healthy_range": "0.25+",
            **dict(zip(["color", "status"],
                   rate_ratio("dsc", debt_service, (0.1, 0.25)))),
            "interpretation": "Cash flow available to service total debt.",
        },
    ]

    overall_reds   = sum(1 for r in ratios if r["color"] == "red")
    overall_greens = sum(1 for r in ratios if r["color"] == "green")

    if overall_reds >= 3:
        overall = "High Risk"
        overall_color = "red"
    elif overall_reds >= 1:
        overall = "Moderate Risk"
        overall_color = "amber"
    else:
        overall = "Healthy"
        overall_color = "green"

    return {
        "model": "Liquidity & Coverage Analysis",
        "overall": overall,
        "overall_color": overall_color,
        "ratios": ratios,
        "risk": overall,
        "color": overall_color,
        "interpretation": f"{overall_greens}/5 ratios are healthy. {overall_reds}/5 show weakness.",
    }


# ── Run All Available Models ──────────────────────────────────────────────────


# ── 7. Gordon Growth Model (Dividend Discount Model) ─────────────────────────

def gordon_growth_model(
    earnings_per_share: float,
    dividend_payout_ratio: float,
    required_return: float,
    growth_rate: float,
    current_stock_price: float = None,
) -> dict:
    """
    Gordon Growth Model (Dividend Discount Model).
    V = D1 / (r - g)  where D1 = projected next-year dividend.

    Assumptions:
      - Constant perpetual growth (sustainable for stable mature firms).
      - r > g (required return > growth rate).
    """
    result = {"model": "Gordon Growth Model (DDM)"}

    if required_return <= growth_rate:
        return {
            **result,
            "error": "Required return must exceed growth rate.",
            "color": "grey",
            "risk": "N/A",
        }

    dividend_current = earnings_per_share * dividend_payout_ratio
    d1               = dividend_current * (1 + growth_rate)
    intrinsic_value  = d1 / (required_return - growth_rate)

    result.update({
        "dividend_current":  round(dividend_current, 4),
        "d1":                round(d1, 4),
        "intrinsic_value":   round(intrinsic_value, 4),
        "required_return":   round(required_return * 100, 2),
        "growth_rate":       round(growth_rate * 100, 2),
    })

    if current_stock_price:
        margin = (intrinsic_value - current_stock_price) / current_stock_price * 100
        result["current_price"]      = round(current_stock_price, 4)
        result["margin_of_safety"]   = round(margin, 1)
        if margin > 15:
            result["valuation"]       = "Undervalued"
            result["color"]           = "green"
            result["interpretation"]  = f"Stock is {margin:.1f}% below intrinsic DDM value. Potentially attractive."
        elif margin > -10:
            result["valuation"]       = "Fairly Valued"
            result["color"]           = "amber"
            result["interpretation"]  = "Stock trades close to DDM intrinsic value."
        else:
            result["valuation"]       = "Overvalued"
            result["color"]           = "red"
            result["interpretation"]  = f"Stock trades {abs(margin):.1f}% above DDM intrinsic value."
    else:
        result["color"]          = "blue"
        result["valuation"]      = "Computed"
        result["interpretation"] = f"DDM intrinsic value: {intrinsic_value:.4f} per share (normalised)."

    result["risk"] = result.get("valuation", "N/A")
    return result


# ── 8. EV/EBITDA Multiple Analysis ───────────────────────────────────────────

def ev_ebitda_analysis(
    total_debt: float,
    cash: float,
    equity_market_value: float,
    ebitda: float,
    industry_median_ev_ebitda: float = 10.0,
) -> dict:
    """
    Enterprise Value / EBITDA multiple.
    EV = Market Cap + Total Debt − Cash
    Low EV/EBITDA → potentially undervalued; High → expensive.
    """
    result = {"model": "EV/EBITDA Multiple Analysis"}

    if ebitda <= 0:
        return {
            **result,
            "error": "EBITDA must be positive.",
            "color": "red",
            "risk": "High",
            "interpretation": "Negative EBITDA indicates operating losses — company cannot be valued on this metric.",
        }

    ev         = equity_market_value + total_debt - cash
    ev_ebitda  = ev / ebitda

    result.update({
        "enterprise_value": round(ev, 4),
        "ebitda":           round(ebitda, 4),
        "ev_ebitda":        round(ev_ebitda, 2),
        "industry_median":  round(industry_median_ev_ebitda, 2),
        "premium_discount": round((ev_ebitda / industry_median_ev_ebitda - 1) * 100, 1),
    })

    if ev_ebitda < industry_median_ev_ebitda * 0.8:
        result.update({
            "valuation":       "Discount to Peers",
            "color":           "green",
            "risk":            "Low",
            "interpretation":  (f"EV/EBITDA of {ev_ebitda:.1f}x is {result['premium_discount']:.0f}% "
                                f"below the industry median of {industry_median_ev_ebitda:.1f}x. "
                                "Company may be undervalued relative to peers."),
        })
    elif ev_ebitda < industry_median_ev_ebitda * 1.2:
        result.update({
            "valuation":      "In-Line with Peers",
            "color":          "amber",
            "risk":           "Moderate",
            "interpretation": (f"EV/EBITDA of {ev_ebitda:.1f}x is roughly in line with "
                               f"the industry median ({industry_median_ev_ebitda:.1f}x)."),
        })
    else:
        result.update({
            "valuation":      "Premium to Peers",
            "color":          "red",
            "risk":           "High",
            "interpretation": (f"EV/EBITDA of {ev_ebitda:.1f}x is {result['premium_discount']:.0f}% "
                               f"above the industry median. Company is priced at a significant premium."),
        })

    result["components"] = {
        "EV (Enterprise Value)":  round(ev, 4),
        "EBITDA":                 round(ebitda, 4),
        "EV/EBITDA Multiple":     round(ev_ebitda, 2),
        "Industry Median":        industry_median_ev_ebitda,
    }
    return result


# ── 9. DuPont Analysis ────────────────────────────────────────────────────────

def dupont_analysis(
    net_income: float,
    sales: float,
    total_assets: float,
    equity: float,
) -> dict:
    """
    3-Factor DuPont Decomposition of Return on Equity.
    ROE = Net Profit Margin × Asset Turnover × Equity Multiplier

    Identifies whether profitability issues stem from:
    - Operations (low margin)
    - Asset efficiency (low turnover)
    - Leverage (low/high multiplier)
    """
    eps     = 0.0001
    npm     = net_income / max(sales, eps)          # Net Profit Margin
    at      = sales / max(total_assets, eps)        # Asset Turnover
    em      = total_assets / max(equity, eps)       # Equity Multiplier
    roe     = npm * at * em                         # Return on Equity

    def rate(val, lo, hi):
        if val >= hi: return "green", "Strong"
        if val >= lo: return "amber", "Moderate"
        return "red", "Weak"

    npm_color, npm_status = rate(npm,  0.05, 0.15)
    at_color,  at_status  = rate(at,   0.5,  1.0)
    em_color,  em_status  = rate(em,   1.0,  3.0) if em <= 5 else ("red", "High Leverage")
    roe_color, roe_status = rate(roe,  0.08, 0.15)

    reds = sum(1 for c in [npm_color, at_color, em_color] if c == "red")

    return {
        "model": "DuPont Analysis",
        "roe":   round(roe * 100, 2),
        "risk":  "High" if reds >= 2 else "Moderate" if reds == 1 else "Low",
        "color": "red" if reds >= 2 else "amber" if reds == 1 else "green",
        "interpretation": (
            f"ROE of {roe*100:.1f}% "
            f"{'is strong' if roe >= 0.15 else 'is moderate' if roe >= 0.08 else 'is weak'}. "
            f"Net Margin: {npm*100:.1f}% ({npm_status}), "
            f"Asset Turnover: {at:.2f}x ({at_status}), "
            f"Equity Multiplier: {em:.2f}x ({em_status})."
        ),
        "components": {
            "Net Profit Margin":    {"value": round(npm * 100, 2), "unit": "%",  "color": npm_color, "status": npm_status},
            "Asset Turnover":       {"value": round(at, 3),        "unit": "x",  "color": at_color,  "status": at_status},
            "Equity Multiplier":    {"value": round(em, 3),        "unit": "x",  "color": em_color,  "status": em_status},
            "Return on Equity":     {"value": round(roe * 100, 2), "unit": "%",  "color": roe_color, "status": roe_status},
        },
    }


# ── 10. Merton Distance-to-Default (simplified) ───────────────────────────────

def merton_distance_to_default(
    asset_value: float,
    debt_face_value: float,
    asset_volatility: float,
    risk_free_rate: float,
    time_horizon: float = 1.0,
) -> dict:
    """
    Simplified Merton Distance-to-Default.
    DD = (ln(V/F) + (r - 0.5σ²)T) / (σ√T)

    Higher DD → further from default → safer.
    DD < 0 → technically insolvent.
    """
    import math
    result = {"model": "Merton Distance-to-Default"}

    if asset_value <= 0 or debt_face_value <= 0 or asset_volatility <= 0:
        return {
            **result,
            "error": "Asset value, debt, and volatility must be positive.",
            "color": "grey",
            "risk": "N/A",
        }

    try:
        numerator   = (math.log(asset_value / debt_face_value)
                       + (risk_free_rate - 0.5 * asset_volatility**2) * time_horizon)
        denominator = asset_volatility * math.sqrt(time_horizon)
        dd          = numerator / denominator

        # Approximate default probability: N(-DD) ~ Φ(-DD)
        def norm_cdf(x):
            import math
            t = 1 / (1 + 0.2316419 * abs(x))
            d = 0.3989423 * math.exp(-x * x / 2)
            p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
            return p if x < 0 else 1 - p

        default_prob = norm_cdf(-dd) * 100

        if dd > 3.0:
            zone, risk, color = "Safe", "Low", "green"
            interp = f"Distance-to-Default of {dd:.2f}σ. Very low default probability ({default_prob:.2f}%)."
        elif dd > 1.0:
            zone, risk, color = "Caution", "Moderate", "amber"
            interp = f"Distance-to-Default of {dd:.2f}σ. Moderate financial stress. Default prob: {default_prob:.2f}%."
        else:
            zone, risk, color = "Distress", "High", "red"
            interp = f"Distance-to-Default of {dd:.2f}σ. High default risk. Default probability: {default_prob:.2f}%."

        return {
            **result,
            "distance_to_default": round(dd, 4),
            "default_probability": round(default_prob, 2),
            "zone":    zone,
            "risk":    risk,
            "color":   color,
            "interpretation": interp,
            "components": {
                "Asset Value / Debt":    round(asset_value / debt_face_value, 4),
                "Asset Volatility (σ)":  round(asset_volatility * 100, 2),
                "Risk-Free Rate":        round(risk_free_rate * 100, 2),
                "Time Horizon (years)":  time_horizon,
                "Distance-to-Default":  round(dd, 4),
                "Default Probability":   round(default_prob, 2),
            },
        }
    except Exception as e:
        return {**result, "error": str(e), "color": "grey", "risk": "N/A"}


# ── 11. Graham Number ─────────────────────────────────────────────────────────

def graham_number(
    earnings_per_share: float,
    book_value_per_share: float,
    current_price: float = None,
) -> dict:
    """
    Benjamin Graham's intrinsic value formula.
    Graham Number = √(22.5 × EPS × BVPS)

    Represents the maximum price a defensive investor should pay.
    """
    result = {"model": "Graham Number"}

    if earnings_per_share <= 0:
        return {**result, "error": "EPS must be positive for Graham Number.",
                "color": "grey", "risk": "N/A"}
    if book_value_per_share <= 0:
        return {**result, "error": "Book Value per share must be positive.",
                "color": "grey", "risk": "N/A"}

    import math
    gn = math.sqrt(22.5 * earnings_per_share * book_value_per_share)

    result.update({
        "graham_number": round(gn, 2),
        "eps": round(earnings_per_share, 2),
        "bvps": round(book_value_per_share, 2),
        "formula": "√(22.5 × EPS × BVPS)",
    })

    if current_price and current_price > 0:
        margin = (gn - current_price) / current_price * 100
        result["current_price"] = round(current_price, 2)
        result["margin_of_safety"] = round(margin, 1)
        if margin > 20:
            result.update({"valuation": "Undervalued", "color": "green", "risk": "Low",
                           "interpretation": f"CMP ₹{current_price:.0f} is {margin:.0f}% below Graham Number ₹{gn:.0f}. Strong margin of safety."})
        elif margin > -10:
            result.update({"valuation": "Fairly Valued", "color": "amber", "risk": "Moderate",
                           "interpretation": f"CMP ₹{current_price:.0f} is near Graham Number ₹{gn:.0f}. Fair pricing."})
        else:
            result.update({"valuation": "Overvalued", "color": "red", "risk": "High",
                           "interpretation": f"CMP ₹{current_price:.0f} is {abs(margin):.0f}% above Graham Number ₹{gn:.0f}. No margin of safety."})
    else:
        result.update({"color": "blue", "valuation": "Computed", "risk": "N/A",
                       "interpretation": f"Graham Number: ₹{gn:.2f} per share."})

    result["components"] = {
        "EPS": round(earnings_per_share, 2),
        "BVPS": round(book_value_per_share, 2),
        "Graham Multiplier": 22.5,
        "Graham Number": round(gn, 2),
    }
    return result


# ── 12. P/E Relative Valuation ────────────────────────────────────────────────

def pe_relative_valuation(
    earnings_per_share: float,
    current_price: float,
    sector_pe: float = 20.0,
    market_pe: float = 22.0,
) -> dict:
    """
    Price-to-Earnings relative valuation.
    Compares company P/E to sector and market averages.
    Fair Value = EPS × Sector P/E
    """
    result = {"model": "P/E Relative Valuation"}

    if earnings_per_share <= 0:
        return {**result, "error": "EPS must be positive for P/E valuation.",
                "color": "grey", "risk": "N/A"}
    if current_price <= 0:
        return {**result, "error": "Current price must be positive.",
                "color": "grey", "risk": "N/A"}

    company_pe = current_price / earnings_per_share
    fair_value = earnings_per_share * sector_pe
    pe_premium = (company_pe / sector_pe - 1) * 100
    margin = (fair_value - current_price) / current_price * 100

    result.update({
        "pe_ratio": round(company_pe, 2),
        "sector_pe": round(sector_pe, 2),
        "market_pe": round(market_pe, 2),
        "fair_value": round(fair_value, 2),
        "pe_premium_discount": round(pe_premium, 1),
        "margin_of_safety": round(margin, 1),
        "formula": "Fair Value = EPS × Sector P/E",
    })

    if company_pe < sector_pe * 0.75:
        result.update({"valuation": "Undervalued", "color": "green", "risk": "Low",
                       "interpretation": f"P/E of {company_pe:.1f}x is {abs(pe_premium):.0f}% below sector avg {sector_pe:.1f}x. Potentially undervalued."})
    elif company_pe < sector_pe * 1.25:
        result.update({"valuation": "Fairly Valued", "color": "amber", "risk": "Moderate",
                       "interpretation": f"P/E of {company_pe:.1f}x is roughly in line with sector avg {sector_pe:.1f}x."})
    else:
        result.update({"valuation": "Overvalued", "color": "red", "risk": "High",
                       "interpretation": f"P/E of {company_pe:.1f}x is {pe_premium:.0f}% above sector avg {sector_pe:.1f}x. Richly valued."})

    result["components"] = {
        "Company P/E": round(company_pe, 2),
        "Sector P/E": round(sector_pe, 2),
        "Market P/E": round(market_pe, 2),
        "EPS": round(earnings_per_share, 2),
        "Fair Value": round(fair_value, 2),
    }
    return result


# ── 13. Residual Income Model ─────────────────────────────────────────────────

def residual_income_model(
    book_value_per_share: float,
    roe: float,
    cost_of_equity: float,
    growth_rate: float = 0.03,
    current_price: float = None,
    forecast_years: int = 5,
) -> dict:
    """
    Residual Income Valuation (Edwards-Bell-Ohlson).
    IV = BVPS + Σ(Residual Income_t / (1+r)^t) + Terminal RI

    Residual Income = (ROE - Cost of Equity) × Book Value
    """
    result = {"model": "Residual Income Model"}

    if book_value_per_share <= 0:
        return {**result, "error": "Book value per share must be positive.",
                "color": "grey", "risk": "N/A"}
    if cost_of_equity <= growth_rate:
        growth_rate = cost_of_equity - 0.01

    spread = roe - cost_of_equity  # Economic profit spread
    bv = book_value_per_share
    pv_ri_total = 0.0

    for yr in range(1, forecast_years + 1):
        ri = spread * bv
        pv_ri = ri / ((1 + cost_of_equity) ** yr)
        pv_ri_total += pv_ri
        bv = bv * (1 + growth_rate)  # BV grows

    # Terminal residual income (perpetuity)
    terminal_ri = (spread * bv * (1 + growth_rate)) / (cost_of_equity - growth_rate)
    pv_terminal = terminal_ri / ((1 + cost_of_equity) ** forecast_years)

    intrinsic_value = book_value_per_share + pv_ri_total + pv_terminal

    result.update({
        "intrinsic_value": round(intrinsic_value, 2),
        "book_value": round(book_value_per_share, 2),
        "pv_residual_income": round(pv_ri_total, 2),
        "pv_terminal_ri": round(pv_terminal, 2),
        "roe": round(roe * 100, 2),
        "cost_of_equity": round(cost_of_equity * 100, 2),
        "economic_spread": round(spread * 100, 2),
        "formula": "IV = BVPS + Σ(RI_t/(1+r)^t) + Terminal RI",
    })

    if current_price and current_price > 0:
        margin = (intrinsic_value - current_price) / current_price * 100
        result["current_price"] = round(current_price, 2)
        result["margin_of_safety"] = round(margin, 1)
        if margin > 15:
            result.update({"valuation": "Undervalued", "color": "green", "risk": "Low",
                           "interpretation": f"RI intrinsic value ₹{intrinsic_value:.0f} vs CMP ₹{current_price:.0f}. {margin:.0f}% upside."})
        elif margin > -10:
            result.update({"valuation": "Fairly Valued", "color": "amber", "risk": "Moderate",
                           "interpretation": f"RI intrinsic value ₹{intrinsic_value:.0f} is close to CMP ₹{current_price:.0f}."})
        else:
            result.update({"valuation": "Overvalued", "color": "red", "risk": "High",
                           "interpretation": f"RI intrinsic value ₹{intrinsic_value:.0f} is {abs(margin):.0f}% below CMP ₹{current_price:.0f}."})
    else:
        result.update({"color": "blue", "valuation": "Computed", "risk": "N/A",
                       "interpretation": f"Residual Income intrinsic value: ₹{intrinsic_value:.2f}."})

    result["components"] = {
        "Book Value/Share": round(book_value_per_share, 2),
        "ROE": f"{roe*100:.1f}%",
        "Cost of Equity": f"{cost_of_equity*100:.1f}%",
        "Economic Spread": f"{spread*100:.1f}%",
        "PV Residual Income": round(pv_ri_total, 2),
        "PV Terminal RI": round(pv_terminal, 2),
    }
    return result


# ── Run All Available Models ──────────────────────────────────────────────────

def run_quick_valuation(inputs: dict) -> dict:
    """
    Run all available valuation models.
    """
    results = {}

    # Map our 6 features to balance sheet proxies (normalised to TA = 1.0)
    retained_earnings_ratio  = inputs.get("Attr6",  0.20)   # RE / TA
    debt_ratio               = inputs.get("Attr16", 0.45)   # TL / TA
    gross_profit_dep_liab    = inputs.get("Attr13", 0.35)   # GP+Dep / TL
    cash_flow_ind            = inputs.get("Attr5",  30.0)   # Cash flow indicator
    gross_profit_dep_sales   = inputs.get("Attr12", 0.10)   # GP+Dep / Sales
    profit_on_sales_ratio    = inputs.get("Attr15", 0.05)   # ProfitSales / TA

    total_assets       = 1.0
    total_liabilities  = max(debt_ratio, 0.001)
    equity             = max(1.0 - debt_ratio, 0.001)
    retained_earnings  = retained_earnings_ratio
    ebit               = profit_on_sales_ratio
    net_income         = ebit * 0.75
    sales              = profit_on_sales_ratio / max(gross_profit_dep_sales, 0.001)
    sales              = max(sales, 0.001)
    working_capital    = max(1.0 - debt_ratio - 0.3, 0.001)
    current_liab       = total_liabilities * 0.4
    current_assets     = working_capital + current_liab
    cash               = max(cash_flow_ind / 365, 0.001)
    ebitda             = ebit * 1.2
    gross_profit       = max(gross_profit_dep_sales * sales, 0.001)
    depreciation       = max(gross_profit_dep_sales * sales * 0.15, 0.001)
    market_cap         = max(equity * 1.5, 0.1)

    # ── Tier 0: Always available ──────────────────────────────────────────────
    try:
        results["altman"] = altman_z_score(
            working_capital=working_capital, total_assets=total_assets,
            retained_earnings=retained_earnings, ebit=ebit,
            market_cap=market_cap, total_liabilities=total_liabilities, sales=sales
        )
    except Exception as e:
        results["altman"] = {"error": str(e), "model": "Altman Z-Score"}



    # ── Tier 1: Starter+ ─────────────────────────────────────────────────────
    try:
        results["ohlson"] = ohlson_o_score(
            total_assets=total_assets, total_liabilities=total_liabilities,
            working_capital=working_capital, current_liabilities=current_liab,
            current_assets=current_assets, net_income=net_income,
            funds_from_operations=net_income * 1.1
        )
    except Exception as e:
        results["ohlson"] = {"error": str(e), "model": "Ohlson O-Score"}

    try:
        results["dcf"] = dcf_analysis(
            free_cash_flow=ebit * 0.9,
            growth_rate_5yr=0.05,
            terminal_growth_rate=0.025,
            wacc=0.10,
            net_debt=total_liabilities - cash,
            shares_outstanding=1.0,
            current_price=market_cap
        )
    except Exception as e:
        results["dcf"] = {"error": str(e), "model": "DCF Analysis"}



    # ── Tier 2: Professional+ ─────────────────────────────────────────────────
    try:
        results["piotroski"] = piotroski_f_score(
            net_income=net_income,
            total_assets=total_assets,
            operating_cash_flow=ebit * 0.9,
            roa_prev=ebit - 0.05,
            long_term_debt=total_liabilities * 0.5,
            long_term_debt_prev=total_liabilities * 0.5 + 0.02,
            current_ratio=current_assets/max(current_liab, 0.001),
            current_ratio_prev=current_assets/max(current_liab, 0.001) - 0.05,
            shares_outstanding=1.0,
            shares_outstanding_prev=1.0,
            gross_margin=gross_profit/max(sales, 0.001),
            gross_margin_prev=gross_profit/max(sales, 0.001) - 0.02,
            asset_turnover=sales/total_assets,
            asset_turnover_prev=sales/total_assets - 0.01
        )
    except Exception as e:
        results["piotroski"] = {"error": str(e), "model": "Piotroski F-Score"}

    try:
        results["dupont"] = dupont_analysis(
            net_income=net_income, sales=sales,
            total_assets=total_assets, equity=equity
        )
    except Exception as e:
        results["dupont"] = {"error": str(e), "model": "DuPont Analysis"}

    try:
        results["ev_ebitda"] = ev_ebitda_analysis(
            total_debt=total_liabilities, cash=cash,
            equity_market_value=market_cap, ebitda=max(ebitda, 0.001),
            industry_median_ev_ebitda=9.5
        )
    except Exception as e:
        results["ev_ebitda"] = {"error": str(e), "model": "EV/EBITDA"}

    try:
        results["liquidity"] = liquidity_analysis(
            current_assets=current_assets, current_liabilities=current_liab,
            inventory=current_assets * 0.3, cash=cash,
            total_debt=total_liabilities, ebit=ebit,
            interest_expense=total_liabilities * 0.05,
            operating_cash_flow=ebit * 0.9
        )
    except Exception as e:
        results["liquidity"] = {"error": str(e), "model": "Liquidity Analysis"}



    # ── Tier 3: Enterprise only ───────────────────────────────────────────────
    try:
        results["beneish"] = beneish_m_score(
            receivables_t=sales*0.1, receivables_t1=sales*0.1,
            sales_t=sales, sales_t1=sales,
            gross_profit_t=gross_profit, gross_profit_t1=gross_profit,
            assets_t=total_assets, assets_t1=total_assets,
            ppe_t=total_assets*0.5, ppe_t1=total_assets*0.5,
            total_accruals_t=ebit*0.1, total_accruals_t1=ebit*0.1,
            sg_expense_t=gross_profit*0.2, sg_expense_t1=gross_profit*0.2,
            long_term_debt_t=total_liabilities*0.5, long_term_debt_t1=total_liabilities*0.5,
            current_assets_t=current_assets, current_assets_t1=current_assets,
            current_liabilities_t=current_liab, current_liabilities_t1=current_liab,
            net_income_t=net_income, cash_from_ops_t=net_income*1.1
        )
    except Exception as e:
        results["beneish"] = {"error": str(e), "model": "Beneish M-Score"}

    try:
        results["gordon"] = gordon_growth_model(
            earnings_per_share=net_income,
            dividend_payout_ratio=0.4,
            required_return=0.10,
            growth_rate=0.03,
            current_stock_price=market_cap
        )
    except Exception as e:
        results["gordon"] = {"error": str(e), "model": "Gordon Growth Model"}

    try:
        results["merton"] = merton_distance_to_default(
            asset_value=total_assets,
            debt_face_value=total_liabilities,
            asset_volatility=0.25,
            risk_free_rate=0.04,
            time_horizon=1.0
        )
    except Exception as e:
        results["merton"] = {"error": str(e), "model": "Merton Distance-to-Default"}

    return results