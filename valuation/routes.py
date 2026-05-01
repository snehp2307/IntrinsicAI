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
  GET  /valuation/dashboard
"""
from __future__ import annotations

import logging
import time
from flask import Blueprint, request, jsonify, render_template, session

log = logging.getLogger(__name__)

from .data_loader  import search_companies, get_financials
from .market_data  import get_market_data
from .dcf_model    import ValuationInput, run_dcf, dcf_to_dict
from .reverse_dcf  import implied_growth_rate
from .scenario_analysis import run_scenarios
from .xai_valuation import full_xai_payload

valuation_bp = Blueprint("valuation", __name__,
                          url_prefix="/valuation",
                          template_folder="../templates",
                          static_folder="../static")


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