"""
routes.py
=========
Flask Blueprint: /valuation/*

Endpoints:
  GET  /valuation/search_company?q=TCS
  GET  /valuation/fetch_market_data?symbol=TCS
  POST /valuation/run_dcf
  POST /valuation/reverse_dcf
  POST /valuation/scenario_analysis
  POST /valuation/ai_explanation
  GET  /valuation/dashboard
"""
from __future__ import annotations

import logging
import time
import functools
from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for, flash

log = logging.getLogger(__name__)

from .data_loader  import search_companies, get_financials
from .market_data  import get_market_data
from .dcf_model    import ValuationInput, run_dcf, dcf_to_dict
from .reverse_dcf  import implied_growth_rate
from .scenario_analysis import run_scenarios
from .xai_valuation import full_xai_payload
from .ai_engine     import generate_ai_explanation

valuation_bp = Blueprint("valuation", __name__,
                          url_prefix="/valuation",
                          template_folder="../templates",
                          static_folder="../static")


# ── Auth guard — protects ALL valuation routes ────────────────────────────────

@valuation_bp.before_request
def _require_login():
    """Redirect unauthenticated users to login for all valuation endpoints."""
    if "user_id" not in session:
        # API endpoints return 401 JSON; page routes redirect to login
        if request.path.startswith("/valuation/") and (
            request.is_json or request.content_type == "application/json"
            or request.method == "POST"
        ):
            return jsonify({"error": "Authentication required", "status": "unauthorized"}), 401
        flash("Please log in to access valuation tools.", "warning")
        return redirect(url_for("login", next=request.full_path))



# ── Helpers ───────────────────────────────────────────────────────────────────

def _fv(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _build_input(data: dict, market_price: float = 0.0) -> ValuationInput:
    """Map request JSON → ValuationInput dataclass."""
    return ValuationInput(
        net_income             = _fv(data, "net_income"),
        depreciation           = _fv(data, "depreciation"),
        amortization           = _fv(data, "amortization"),
        capex                  = _fv(data, "capex"),
        working_capital_change = _fv(data, "working_capital_change"),
        revenue_growth_rate    = _fv(data, "revenue_growth_rate",  0.10),
        operating_margin       = _fv(data, "operating_margin",     0.15),
        tax_rate               = _fv(data, "tax_rate",             0.25),
        reinvestment_rate      = _fv(data, "reinvestment_rate",    0.30),
        wacc                   = _fv(data, "wacc",                 0.10),
        terminal_growth_rate   = _fv(data, "terminal_growth_rate", 0.05),
        forecast_years         = int(_fv(data, "forecast_years",   10)),
        current_price          = market_price or _fv(data, "current_price"),
        shares_outstanding     = _fv(data, "shares_outstanding",   1.0),
        net_debt               = _fv(data, "net_debt"),
        data_source            = data.get("data_source", "unknown"),
        data_warnings          = data.get("data_warnings"),
    )


# ── GET /valuation/dashboard ──────────────────────────────────────────────────

@valuation_bp.route("/dashboard")
def dashboard():
    return render_template("valuation_dashboard.html")


# ── GET /valuation/search_company ─────────────────────────────────────────────

@valuation_bp.route("/search_company")
def search_company():
    q = request.args.get("q", "").strip()
    if len(q) < 1:
        return jsonify({"results": []})
    results = search_companies(q, limit=12)
    return jsonify({"results": results})


# ── GET /valuation/fetch_market_data ─────────────────────────────────────────

@valuation_bp.route("/fetch_market_data")
def fetch_market_data():
    symbol   = request.args.get("symbol", "").strip().upper()
    exchange = request.args.get("exchange", "NSE").upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    # Fetch financials from local dataset (detailed or directory stub)
    fin = get_financials(symbol)
    if not fin:
        return jsonify({
            "error": f"No financial data found for '{symbol}'. "
                     f"Try searching by NSE symbol (e.g. SBIN, RELIANCE, TCS)."
        }), 404

    # Fetch live price
    market = get_market_data(
        symbol,
        shares_outstanding=fin.get("shares_outstanding", 1.0),
        exchange=exchange,
        token=fin.get("token", ""),
    )

    return jsonify({
        "symbol":            fin["symbol"],
        "company_name":      fin["company_name"],
        "sector":            fin.get("sector", ""),
        "exchange":          fin.get("exchange", "NSE"),
        "ltp":               market["ltp"],
        "market_cap":        market["market_cap_millions"],
        "price_source":      market["source"],
        "shares_outstanding":fin["shares_outstanding"],
        "financials":        fin,
        "history":           fin.get("history", []),
        "is_stub":           fin.get("_stub", False),
    })


# ── POST /valuation/run_dcf ───────────────────────────────────────────────────

@valuation_bp.route("/run_dcf", methods=["POST"])
def run_dcf_route():
    data = request.get_json(force=True, silent=True) or {}

    symbol   = data.get("symbol", "").strip().upper()
    exchange = data.get("exchange", "NSE").upper()

    # Auto-fill from dataset if symbol provided
    fin = get_financials(symbol) if symbol else None

    # Merge dataset defaults with any user overrides
    merged = {**(fin or {}), **data}

    # Live price
    if symbol and fin:
        market = get_market_data(symbol,
                                  shares_outstanding=fin.get("shares_outstanding", 1.0),
                                  exchange=exchange,
                                  token=fin.get("token", ""))
        market_price = market["ltp"]
    else:
        market_price = _fv(data, "current_price")

    inp = _build_input(merged, market_price)

    try:
        t0 = time.time()
        result  = run_dcf(inp)
        payload = dcf_to_dict(result)
        payload["xai"] = full_xai_payload(result)
        if fin:
            payload["history"] = fin.get("history", [])
        log.info("[run_dcf] %s completed in %.2fs", symbol or "manual", time.time() - t0)
        return jsonify({"status": "ok", **payload})
    except Exception as e:
        log.error("[run_dcf] %s failed: %s", symbol or "manual", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ── POST /valuation/reverse_dcf ───────────────────────────────────────────────

@valuation_bp.route("/reverse_dcf", methods=["POST"])
def reverse_dcf_route():
    data   = request.get_json(force=True, silent=True) or {}
    symbol = data.get("symbol", "").strip().upper()
    fin    = get_financials(symbol) if symbol else None
    merged = {**(fin or {}), **data}

    if symbol and fin:
        market = get_market_data(symbol,
                                  shares_outstanding=fin.get("shares_outstanding", 1.0),
                                  token=fin.get("token", ""))
        market_price = market["ltp"]
    else:
        market_price = _fv(data, "current_price")

    inp = _build_input(merged, market_price)

    try:
        t0 = time.time()
        result = implied_growth_rate(inp, price=market_price)
        log.info("[reverse_dcf] %s completed in %.2fs", symbol or "manual", time.time() - t0)
        return jsonify({"status": "ok", **result})
    except Exception as e:
        log.error("[reverse_dcf] %s failed: %s", symbol or "manual", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ── POST /valuation/scenario_analysis ────────────────────────────────────────

@valuation_bp.route("/scenario_analysis", methods=["POST"])
def scenario_analysis_route():
    data   = request.get_json(force=True, silent=True) or {}
    symbol = data.get("symbol", "").strip().upper()
    fin    = get_financials(symbol) if symbol else None
    merged = {**(fin or {}), **data}

    if symbol and fin:
        market = get_market_data(symbol,
                                  shares_outstanding=fin.get("shares_outstanding", 1.0),
                                  token=fin.get("token", ""))
        market_price = market["ltp"]
    else:
        market_price = _fv(data, "current_price")

    inp = _build_input(merged, market_price)

    try:
        t0 = time.time()
        result = run_scenarios(inp)
        log.info("[scenario_analysis] %s completed in %.2fs", symbol or "manual", time.time() - t0)
        return jsonify({"status": "ok", **result})
    except Exception as e:
        log.error("[scenario_analysis] %s failed: %s", symbol or "manual", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ── POST /valuation/ai_explanation ────────────────────────────────────────────

@valuation_bp.route("/ai_explanation", methods=["POST"])
def ai_explanation_route():
    """
    Generate an AI-powered equity research explanation using Mistral.
    Accepts the same payload as run_dcf + the DCF result fields.
    """
    data = request.get_json(force=True, silent=True) or {}

    try:
        t0 = time.time()
        result = generate_ai_explanation(data)
        log.info("[ai_explanation] %s completed in %.2fs (source=%s, cached=%s)",
                 data.get("symbol", "manual"), time.time() - t0,
                 result.get("ai_source"), result.get("ai_cached"))
        return jsonify({"status": "ok", **result})
    except Exception as e:
        log.error("[ai_explanation] failed: %s", e)
        return jsonify({
            "status": "error",
            "ai_explanation": f"AI explanation unavailable. Error: {str(e)}",
            "ai_source": "error",
        }), 500


# ── POST /valuation/multi_model ───────────────────────────────────────────────

@valuation_bp.route("/multi_model", methods=["POST"])
def multi_model_route():
    """
    Run all available valuation models using real company financial data.
    Returns individual model results + composite verdict.
    """
    import math
    from valuation_models import (
        altman_z_score, ohlson_o_score, piotroski_f_score,
        dupont_analysis, ev_ebitda_analysis, gordon_growth_model,
        merton_distance_to_default, beneish_m_score, liquidity_analysis,
        graham_number, pe_relative_valuation, residual_income_model,
    )

    data = request.get_json(force=True, silent=True) or {}

    symbol   = data.get("symbol", "").strip().upper()
    exchange = data.get("exchange", "NSE").upper()

    # Auto-fill from dataset
    fin = get_financials(symbol) if symbol else None
    merged = {**(fin or {}), **data}

    # Live price
    if symbol and fin:
        market = get_market_data(symbol,
                                  shares_outstanding=fin.get("shares_outstanding", 1.0),
                                  exchange=exchange,
                                  token=fin.get("token", ""))
        market_price = market["ltp"]
        market_cap   = market.get("market_cap_millions", 0) * 1e6  # Convert to ₹
    else:
        market_price = _fv(data, "current_price")
        market_cap   = market_price * _fv(merged, "shares_outstanding", 1.0)

    # ── Extract financial metrics from real data ────────────────────────────
    net_income       = _fv(merged, "net_income", 0)
    depreciation     = _fv(merged, "depreciation", 0)
    capex            = _fv(merged, "capex", 0)
    revenue          = _fv(merged, "revenue", 0)
    if revenue <= 0:
        revenue = net_income / max(_fv(merged, "operating_margin", 0.15), 0.01)
    ebit             = _fv(merged, "ebit", 0)
    if ebit <= 0:
        ebit = net_income / 0.75
    ebitda           = ebit + depreciation
    net_debt         = _fv(merged, "net_debt", 0)
    shares           = _fv(merged, "shares_outstanding", 1.0)

    # Real balance sheet data
    total_assets      = _fv(merged, "total_assets", 0)
    total_liabilities = _fv(merged, "total_liabilities", 0)
    equity            = _fv(merged, "equity", 0)
    current_assets    = _fv(merged, "current_assets", 0)
    current_liab      = _fv(merged, "current_liabilities", 0)
    cash              = _fv(merged, "cash", 0)

    # Real previous-year data (from yfinance multi-year extraction)
    revenue_prev        = _fv(merged, "revenue_prev", 0)
    net_income_prev     = _fv(merged, "net_income_prev", 0)
    total_assets_prev   = _fv(merged, "total_assets_prev", 0)
    total_liab_prev     = _fv(merged, "total_liabilities_prev", 0)
    equity_prev         = _fv(merged, "equity_prev", 0)
    current_assets_prev = _fv(merged, "current_assets_prev", 0)
    current_liab_prev   = _fv(merged, "current_liabilities_prev", 0)

    # Real detailed line items
    retained_earnings   = _fv(merged, "retained_earnings", 0)
    inventory           = _fv(merged, "inventory", 0)
    receivables         = _fv(merged, "receivables", 0)
    receivables_prev    = _fv(merged, "receivables_prev", 0)
    ppe                 = _fv(merged, "ppe", 0)
    ppe_prev            = _fv(merged, "ppe_prev", 0)
    gross_profit        = _fv(merged, "gross_profit", 0)
    gross_profit_prev   = _fv(merged, "gross_profit_prev", 0)
    sga_expense         = _fv(merged, "sga_expense", 0)
    sga_expense_prev    = _fv(merged, "sga_expense_prev", 0)
    interest_expense    = _fv(merged, "interest_expense", 0)
    long_term_debt      = _fv(merged, "long_term_debt", 0)
    long_term_debt_prev = _fv(merged, "long_term_debt_prev", 0)

    # Dividend and valuation data
    dividend_payout     = _fv(merged, "payout_ratio", 0)
    trailing_pe         = _fv(merged, "trailing_pe", 0)
    beta                = _fv(merged, "beta", 1.0)

    # Derived values
    fcf              = _fv(merged, "free_cash_flow", 0) or (net_income + depreciation - capex)
    operating_cf     = _fv(merged, "operating_cash_flow", 0) or fcf
    growth           = _fv(merged, "revenue_growth_rate", 0.10)
    margin           = _fv(merged, "operating_margin", 0.15)
    wacc             = _fv(merged, "wacc", 0.10)
    terminal_g       = _fv(merged, "terminal_growth_rate", 0.05)
    data_source      = merged.get("data_source", "unknown")

    # Flag: do we have real previous-year data?
    has_prev_year = total_assets_prev > 0 and revenue_prev > 0

    # Fallback estimates only when real data is zero
    if total_assets <= 0:
        total_assets = max(revenue * 0.8, 1.0)
    if total_liabilities <= 0:
        total_liabilities = max(net_debt, total_assets * 0.4)
    if equity <= 0:
        equity = max(total_assets - total_liabilities, 0.001)
    if current_assets <= 0:
        current_liab = total_liabilities * 0.4
        current_assets = max(total_assets * 0.2, 0.001) + current_liab
    elif current_liab <= 0:
        current_liab = total_liabilities * 0.4
    if cash <= 0:
        cash = max(total_assets * 0.05, 0.001)
    if retained_earnings <= 0:
        retained_earnings = net_income * 3  # Last resort approximation
    if interest_expense <= 0:
        interest_expense = max(total_liabilities * 0.06, 0.001)

    working_capital = current_assets - current_liab

    # EPS and BVPS
    eps  = net_income / max(shares, 1) if shares > 0 else 0
    bvps = equity / max(shares, 1) if shares > 0 else 0
    roe  = net_income / max(equity, 0.001)

    # Sector P/E medians (industry-specific)
    sector = merged.get("sector", "")
    SECTOR_PE = {
        "Financial Services": 15, "Technology": 30, "Consumer Cyclical": 25,
        "Healthcare": 28, "Industrials": 20, "Energy": 12,
        "Consumer Defensive": 22, "Basic Materials": 14, "Utilities": 16,
        "Communication Services": 18, "Real Estate": 20,
    }
    sector_pe = SECTOR_PE.get(sector, 20.0)

    results = {}
    t0 = time.time()

    # ── 1. Altman Z-Score ─────────────────────────────────────────────────
    try:
        results["altman"] = altman_z_score(
            working_capital=working_capital, total_assets=total_assets,
            retained_earnings=retained_earnings, ebit=ebit,
            market_cap=market_cap or equity * 1.5,
            total_liabilities=total_liabilities, sales=revenue,
        )
    except Exception as e:
        results["altman"] = {"error": str(e), "model": "Altman Z-Score"}

    # ── 2. Ohlson O-Score ─────────────────────────────────────────────────
    try:
        results["ohlson"] = ohlson_o_score(
            total_assets=total_assets, total_liabilities=total_liabilities,
            working_capital=working_capital, current_liabilities=current_liab,
            current_assets=current_assets, net_income=net_income,
            funds_from_operations=operating_cf,
        )
    except Exception as e:
        results["ohlson"] = {"error": str(e), "model": "Ohlson O-Score"}

    # ── 3. Piotroski F-Score (using REAL prev-year data) ──────────────────
    try:
        if has_prev_year:
            roa_prev = net_income_prev / max(total_assets_prev, 0.001)
            cr_prev  = current_assets_prev / max(current_liab_prev, 0.001)
            gm       = gross_profit / max(revenue, 0.001) if gross_profit > 0 else margin
            gm_prev  = gross_profit_prev / max(revenue_prev, 0.001) if gross_profit_prev > 0 else gm * 0.97
            at_prev  = revenue_prev / max(total_assets_prev, 0.001)
            ltd_prev = long_term_debt_prev if long_term_debt_prev > 0 else total_liab_prev * 0.6
        else:
            # No previous year — cannot compute Piotroski accurately
            results["piotroski"] = {"error": "Previous year financial data unavailable for comparative analysis.", "model": "Piotroski F-Score"}
            raise StopIteration  # skip to next

        results["piotroski"] = piotroski_f_score(
            net_income=net_income, total_assets=total_assets,
            operating_cash_flow=operating_cf,
            roa_prev=roa_prev,
            long_term_debt=long_term_debt if long_term_debt > 0 else total_liabilities * 0.6,
            long_term_debt_prev=ltd_prev,
            current_ratio=current_assets / max(current_liab, 0.001),
            current_ratio_prev=cr_prev,
            shares_outstanding=shares,
            shares_outstanding_prev=shares,  # From yfinance — same fiscal year
            gross_margin=gm,
            gross_margin_prev=gm_prev,
            asset_turnover=revenue / max(total_assets, 0.001),
            asset_turnover_prev=at_prev,
        )
    except StopIteration:
        pass
    except Exception as e:
        results["piotroski"] = {"error": str(e), "model": "Piotroski F-Score"}

    # ── 4. DuPont Analysis ────────────────────────────────────────────────
    try:
        results["dupont"] = dupont_analysis(
            net_income=net_income, sales=revenue,
            total_assets=total_assets, equity=equity,
        )
    except Exception as e:
        results["dupont"] = {"error": str(e), "model": "DuPont Analysis"}

    # ── 5. EV/EBITDA (using real EV/EBITDA from yfinance if available) ────
    try:
        # Sector EV/EBITDA medians
        SECTOR_EV = {
            "Financial Services": 12, "Technology": 20, "Consumer Cyclical": 14,
            "Healthcare": 16, "Industrials": 11, "Energy": 8,
            "Consumer Defensive": 13, "Basic Materials": 9, "Utilities": 10,
            "Communication Services": 12, "Real Estate": 15,
        }
        industry_ev = SECTOR_EV.get(sector, 12.0)

        results["ev_ebitda"] = ev_ebitda_analysis(
            total_debt=total_liabilities, cash=cash,
            equity_market_value=market_cap or equity * 1.5,
            ebitda=max(ebitda, 0.001),
            industry_median_ev_ebitda=industry_ev,
        )
    except Exception as e:
        results["ev_ebitda"] = {"error": str(e), "model": "EV/EBITDA"}

    # ── 6. Gordon Growth (DDM) — using REAL dividend data ─────────────────
    try:
        actual_payout = dividend_payout if dividend_payout > 0 else 0.30
        results["gordon"] = gordon_growth_model(
            earnings_per_share=eps,
            dividend_payout_ratio=actual_payout,
            required_return=wacc,
            growth_rate=min(terminal_g, wacc - 0.01),
            current_stock_price=market_price,
        )
    except Exception as e:
        results["gordon"] = {"error": str(e), "model": "Gordon Growth Model"}

    # ── 7. Merton Distance-to-Default — using real beta for volatility ────
    try:
        # Estimate asset volatility from beta (levered → unlevered)
        debt_equity = total_liabilities / max(equity, 0.001)
        asset_vol = max(beta * 0.15 / (1 + debt_equity * 0.65), 0.08)  # Unlevered vol proxy

        results["merton"] = merton_distance_to_default(
            asset_value=total_assets,
            debt_face_value=max(total_liabilities, 0.001),
            asset_volatility=asset_vol,
            risk_free_rate=0.07,  # Indian 10Y govt bond
            time_horizon=1.0,
        )
    except Exception as e:
        results["merton"] = {"error": str(e), "model": "Merton DD"}

    # ── 8. Beneish M-Score (using REAL prev-year data) ────────────────────
    try:
        if has_prev_year and gross_profit > 0 and revenue_prev > 0:
            # Use real receivables, or estimate from revenue
            recv_t  = receivables if receivables > 0 else revenue * 0.12
            recv_t1 = receivables_prev if receivables_prev > 0 else revenue_prev * 0.12
            gp_t    = gross_profit
            gp_t1   = gross_profit_prev if gross_profit_prev > 0 else gp_t * 0.95
            sga_t   = sga_expense if sga_expense > 0 else gp_t * 0.20
            sga_t1  = sga_expense_prev if sga_expense_prev > 0 else sga_t * 0.97
            ppe_t   = ppe if ppe > 0 else total_assets * 0.40
            ppe_t1  = ppe_prev if ppe_prev > 0 else total_assets_prev * 0.40
            accruals_t  = net_income - operating_cf
            accruals_t1 = net_income_prev - (operating_cf * 0.95) if net_income_prev else accruals_t * 0.95

            results["beneish"] = beneish_m_score(
                receivables_t=recv_t, receivables_t1=recv_t1,
                sales_t=revenue, sales_t1=revenue_prev,
                gross_profit_t=gp_t, gross_profit_t1=gp_t1,
                assets_t=total_assets, assets_t1=total_assets_prev,
                ppe_t=ppe_t, ppe_t1=ppe_t1,
                total_accruals_t=accruals_t, total_accruals_t1=accruals_t1,
                sg_expense_t=sga_t, sg_expense_t1=sga_t1,
                long_term_debt_t=long_term_debt if long_term_debt > 0 else total_liabilities * 0.55,
                long_term_debt_t1=long_term_debt_prev if long_term_debt_prev > 0 else total_liab_prev * 0.55,
                current_assets_t=current_assets, current_assets_t1=current_assets_prev,
                current_liabilities_t=current_liab, current_liabilities_t1=current_liab_prev,
                net_income_t=net_income, cash_from_ops_t=operating_cf,
            )
        else:
            results["beneish"] = {"error": "Previous year financial data unavailable for Beneish M-Score analysis.", "model": "Beneish M-Score"}
    except Exception as e:
        results["beneish"] = {"error": str(e), "model": "Beneish M-Score"}

    # ── 9. Liquidity Analysis — using REAL inventory & interest ───────────
    try:
        inv = inventory if inventory > 0 else current_assets * 0.25
        int_exp = interest_expense if interest_expense > 0 else max(total_liabilities * 0.06, 0.001)

        results["liquidity"] = liquidity_analysis(
            current_assets=current_assets, current_liabilities=current_liab,
            inventory=inv, cash=cash,
            total_debt=total_liabilities, ebit=ebit,
            interest_expense=int_exp,
            operating_cash_flow=operating_cf,
        )
    except Exception as e:
        results["liquidity"] = {"error": str(e), "model": "Liquidity Analysis"}

    # ── 10. Graham Number ─────────────────────────────────────────────────
    try:
        if eps > 0 and bvps > 0:
            results["graham"] = graham_number(
                earnings_per_share=eps,
                book_value_per_share=bvps,
                current_price=market_price,
            )
        else:
            results["graham"] = {"error": "Positive EPS and Book Value required.", "model": "Graham Number"}
    except Exception as e:
        results["graham"] = {"error": str(e), "model": "Graham Number"}

    # ── 11. P/E Relative Valuation ────────────────────────────────────────
    try:
        if eps > 0 and market_price > 0:
            results["pe_relative"] = pe_relative_valuation(
                earnings_per_share=eps,
                current_price=market_price,
                sector_pe=sector_pe,
                market_pe=22.0,  # Nifty 50 historical avg
            )
        else:
            results["pe_relative"] = {"error": "Positive EPS and market price required.", "model": "P/E Relative Valuation"}
    except Exception as e:
        results["pe_relative"] = {"error": str(e), "model": "P/E Relative Valuation"}

    # ── 12. Residual Income Model ─────────────────────────────────────────
    try:
        if bvps > 0 and roe != 0:
            results["residual_income"] = residual_income_model(
                book_value_per_share=bvps,
                roe=roe,
                cost_of_equity=wacc,
                growth_rate=terminal_g,
                current_price=market_price,
            )
        else:
            results["residual_income"] = {"error": "Positive book value and ROE required.", "model": "Residual Income Model"}
    except Exception as e:
        results["residual_income"] = {"error": str(e), "model": "Residual Income Model"}

    # ── Composite Verdict ─────────────────────────────────────────────────────
    verdicts = []
    signals  = {"bullish": 0, "neutral": 0, "bearish": 0, "total": 0}

    def _classify(model_key, result_dict):
        if "error" in result_dict:
            return
        signals["total"] += 1
        risk  = result_dict.get("risk", "").lower()
        color = result_dict.get("color", "")
        val   = result_dict.get("valuation", "").lower()
        zone  = result_dict.get("zone", "").lower()
        model = result_dict.get("model", model_key)

        if color == "green" or risk == "low" or "under" in val or "safe" in zone or "strong" in val.lower() if val else False:
            signals["bullish"] += 1
            verdicts.append({"model": model, "signal": "Bullish", "color": "green",
                             "summary": result_dict.get("interpretation", "")[:120]})
        elif color == "red" or risk == "high" or "over" in val or "distress" in zone:
            signals["bearish"] += 1
            verdicts.append({"model": model, "signal": "Bearish", "color": "red",
                             "summary": result_dict.get("interpretation", "")[:120]})
        else:
            signals["neutral"] += 1
            verdicts.append({"model": model, "signal": "Neutral", "color": "amber",
                             "summary": result_dict.get("interpretation", "")[:120]})

    for key, val in results.items():
        _classify(key, val)

    # Composite recommendation
    total = max(signals["total"], 1)
    bull_pct = signals["bullish"] / total * 100
    bear_pct = signals["bearish"] / total * 100

    if bull_pct >= 60:
        composite = "Buy"
        composite_color = "green"
        composite_desc = f"{signals['bullish']}/{total} models signal positive. Strong multi-factor consensus."
    elif bear_pct >= 50:
        composite = "Avoid"
        composite_color = "red"
        composite_desc = f"{signals['bearish']}/{total} models signal negative. Significant downside risk."
    elif bull_pct >= 40:
        composite = "Watchlist"
        composite_color = "blue"
        composite_desc = f"Mixed signals ({signals['bullish']} bullish, {signals['neutral']} neutral, {signals['bearish']} bearish). Monitor for clarity."
    else:
        composite = "Hold"
        composite_color = "amber"
        composite_desc = f"Balanced signals. No strong conviction in either direction."

    elapsed = round((time.time() - t0) * 1000)
    log.info("[multi_model] %s — %d models in %dms, verdict=%s", symbol or "manual", total, elapsed, composite)

    return jsonify({
        "status": "ok",
        "models": results,
        "composite": {
            "verdict":     composite,
            "color":       composite_color,
            "description": composite_desc,
            "signals":     signals,
            "details":     verdicts,
        },
        "latency_ms": elapsed,
    })