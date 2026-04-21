"""
data_loader.py
==============
Loads company financial data from local datasets.

Two data sources:
  1. data/nifty500_companies.json — 300+ companies for search/autocomplete
  2. Built-in sample financials — 5 companies with detailed DCF data

When a company is found in the directory but has no financials,
the route will still work using yfinance prices + default DCF assumptions.
"""
from __future__ import annotations

import json
import os
import logging
import statistics
from typing import Optional

log = logging.getLogger(__name__)

_DATA_PATH       = os.path.join(os.path.dirname(__file__), "..", "data", "valuation_data.json")
_NIFTY500_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "nifty500_companies.json")
_CACHE: Optional[dict] = None
_DIRECTORY_CACHE: Optional[list] = None


# ── Company Directory (300+ companies for search) ─────────────────────────────

def _load_directory() -> list:
    """Load the NIFTY 500 company directory for search/autocomplete."""
    global _DIRECTORY_CACHE
    if _DIRECTORY_CACHE is not None:
        return _DIRECTORY_CACHE

    if os.path.exists(_NIFTY500_PATH):
        try:
            with open(_NIFTY500_PATH, "r", encoding="utf-8") as f:
                _DIRECTORY_CACHE = json.load(f)
            log.info("Loaded company directory: %d companies", len(_DIRECTORY_CACHE))
            return _DIRECTORY_CACHE
        except Exception as e:
            log.error("Failed to load company directory: %s", e)

    _DIRECTORY_CACHE = []
    return _DIRECTORY_CACHE


# ── Financial Dataset (detailed DCF data) ─────────────────────────────────────

def _load_dataset() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if os.path.exists(_DATA_PATH):
        try:
            with open(_DATA_PATH, "r") as f:
                _CACHE = json.load(f)
            log.info("Loaded valuation dataset: %d companies", len(_CACHE))
            return _CACHE
        except Exception as e:
            log.error("Failed to load valuation dataset: %s", e)
    # Return built-in sample dataset as fallback
    _CACHE = _sample_dataset()
    return _CACHE


def reload_dataset():
    """Force reload from disk (useful after updates)."""
    global _CACHE, _DIRECTORY_CACHE
    _CACHE = None
    _DIRECTORY_CACHE = None
    return _load_dataset()


# ── Company search (uses directory + financials) ─────────────────────────────

def search_companies(query: str, limit: int = 12) -> list[dict]:
    """
    Search by symbol or company name (case-insensitive substring match).
    Searches the NIFTY 500 directory first, then the financials dataset.
    Returns list of {symbol, company_name, sector, industry, exchange}.
    """
    q = query.strip().upper()
    if not q:
        return []

    results = []
    seen = set()

    # Search NIFTY 500 directory first (300+ companies)
    directory = _load_directory()
    for entry in directory:
        sym  = entry.get("symbol", "").upper()
        name = entry.get("company_name", "").upper()
        if q in sym or q in name:
            if sym not in seen:
                seen.add(sym)
                results.append({
                    "symbol":       sym,
                    "company_name": entry.get("company_name", sym),
                    "sector":       entry.get("sector", ""),
                    "industry":     entry.get("industry", ""),
                    "exchange":     entry.get("exchange", "NSE"),
                    "token":        entry.get("token", ""),
                })
            if len(results) >= limit:
                return results

    # Also search the financials dataset (may have entries not in directory)
    dataset = _load_dataset()
    for sym, data in dataset.items():
        if sym in seen:
            continue
        name = data.get("company_name", "").upper()
        if q in sym or q in name:
            seen.add(sym)
            results.append({
                "symbol":       sym,
                "company_name": data.get("company_name", sym),
                "sector":       data.get("sector", ""),
                "industry":     data.get("industry", ""),
                "exchange":     data.get("exchange", "NSE"),
                "token":        data.get("token", ""),
            })
            if len(results) >= limit:
                break

    return results


def get_company_info(symbol: str) -> Optional[dict]:
    """Return full company entry or None."""
    dataset = _load_dataset()
    return dataset.get(symbol.upper())


# ── Financial aggregation ─────────────────────────────────────────────────────

def get_financials(symbol: str) -> Optional[dict]:
    """
    Return processed financial inputs suitable for DCF.
    Averages the most recent 3–5 years for stability.
    """
    data = get_company_info(symbol)
    if not data:
        return None

    fin = data.get("financials", [])
    if not fin:
        return None

    # Use most recent 5 years (or whatever is available)
    recent = fin[:5]

    def avg(field: str, fallback: float = 0.0) -> float:
        vals = [r.get(field, fallback) for r in recent if r.get(field) is not None]
        return round(statistics.mean(vals), 2) if vals else fallback

    def growth_rate(field: str) -> float:
        """CAGR from oldest to newest in available data."""
        vals = [r.get(field) for r in fin if r.get(field)]
        if len(vals) < 2:
            return 0.08
        try:
            n    = len(vals) - 1
            cagr = (vals[0] / vals[-1]) ** (1 / n) - 1
            return round(max(min(cagr, 0.50), -0.20), 4)
        except Exception:
            return 0.08

    # Working capital change (latest year)
    wc_latest = recent[0].get("working_capital", 0)
    wc_prev   = recent[0].get("working_capital_prev") or (recent[1].get("working_capital", 0) if len(recent) > 1 else 0)
    wc_change = wc_latest - wc_prev

    return {
        "symbol":               symbol.upper(),
        "company_name":         data.get("company_name", symbol),
        "sector":               data.get("sector", ""),
        "shares_outstanding":   data.get("shares_outstanding", 1.0),
        "exchange":             data.get("exchange", "NSE"),
        "token":                data.get("token", ""),

        # Owner Earnings components (₹ Crores, averaged)
        "net_income":             avg("net_income"),
        "depreciation":           avg("depreciation"),
        "amortization":           avg("amortization"),
        "capex":                  avg("capex"),
        "working_capital_change": round(wc_change, 2),

        # DCF drivers
        "revenue_growth_rate":    growth_rate("revenue"),
        "operating_margin":       round(avg("ebit") / max(avg("revenue"), 1), 4),
        "free_cash_flow":         avg("free_cash_flow"),
        "net_debt":               round(avg("debt") - avg("cash"), 2),

        # Raw history for charts
        "history": [
            {
                "year":         r.get("year"),
                "revenue":      r.get("revenue", 0),
                "net_income":   r.get("net_income", 0),
                "free_cash_flow": r.get("free_cash_flow", 0),
                "ebit":         r.get("ebit", 0),
            }
            for r in fin
        ],
    }


# ── Built-in sample dataset ───────────────────────────────────────────────────

def _sample_dataset() -> dict:
    """
    Small built-in sample dataset covering 5 well-known NSE stocks.
    Replace or extend with data/valuation_data.json for production.
    """
    return {
        "TCS": {
            "company_name": "Tata Consultancy Services Ltd",
            "symbol": "TCS", "exchange": "NSE", "token": "11536",
            "sector": "Information Technology", "industry": "IT Services",
            "shares_outstanding": 364.6,
            "financials": [
                {"year":2024,"revenue":240893,"net_income":46099,"ebit":57200,
                 "free_cash_flow":40200,"depreciation":9800,"amortization":800,
                 "capex":5200,"working_capital":42000,"working_capital_prev":38000,"debt":0,"cash":9547},
                {"year":2023,"revenue":225458,"net_income":42303,"ebit":53100,
                 "free_cash_flow":37800,"depreciation":9100,"amortization":750,
                 "capex":4900,"working_capital":38000,"working_capital_prev":33000,"debt":0,"cash":8900},
                {"year":2022,"revenue":191754,"net_income":38327,"ebit":48600,
                 "free_cash_flow":33900,"depreciation":8400,"amortization":700,
                 "capex":4300,"working_capital":33000,"working_capital_prev":29000,"debt":0,"cash":8100},
                {"year":2021,"revenue":164177,"net_income":33388,"ebit":41700,
                 "free_cash_flow":29200,"depreciation":7600,"amortization":650,
                 "capex":3900,"working_capital":29000,"working_capital_prev":26000,"debt":0,"cash":7300},
                {"year":2020,"revenue":156949,"net_income":32340,"ebit":38500,
                 "free_cash_flow":27100,"depreciation":7100,"amortization":600,
                 "capex":3600,"working_capital":26000,"working_capital_prev":23000,"debt":0,"cash":6800},
            ]
        },
        "INFY": {
            "company_name": "Infosys Ltd",
            "symbol": "INFY", "exchange": "NSE", "token": "1594",
            "sector": "Information Technology", "industry": "IT Services",
            "shares_outstanding": 416.0,
            "financials": [
                {"year":2024,"revenue":153670,"net_income":26248,"ebit":32100,
                 "free_cash_flow":22900,"depreciation":5800,"amortization":600,
                 "capex":3200,"working_capital":18000,"working_capital_prev":15000,"debt":2000,"cash":12000},
                {"year":2023,"revenue":146767,"net_income":24095,"ebit":29800,
                 "free_cash_flow":21000,"depreciation":5400,"amortization":550,
                 "capex":3000,"working_capital":15000,"working_capital_prev":13000,"debt":2500,"cash":11000},
                {"year":2022,"revenue":121641,"net_income":22110,"ebit":27500,
                 "free_cash_flow":19400,"depreciation":4900,"amortization":500,
                 "capex":2700,"working_capital":13000,"working_capital_prev":12000,"debt":3000,"cash":10000},
                {"year":2021,"revenue":100472,"net_income":19351,"ebit":23800,
                 "free_cash_flow":16800,"depreciation":4300,"amortization":450,
                 "capex":2400,"working_capital":12000,"working_capital_prev":11000,"debt":3500,"cash":9500},
                {"year":2020,"revenue":90791,"net_income":16639,"ebit":20900,
                 "free_cash_flow":14600,"depreciation":3900,"amortization":400,
                 "capex":2100,"working_capital":11000,"working_capital_prev":10000,"debt":4000,"cash":9000},
            ]
        },
        "RELIANCE": {
            "company_name": "Reliance Industries Ltd",
            "symbol": "RELIANCE", "exchange": "NSE", "token": "2885",
            "sector": "Energy & Retail", "industry": "Conglomerate",
            "shares_outstanding": 677.0,
            "financials": [
                {"year":2024,"revenue":899000,"net_income":69621,"ebit":90200,
                 "free_cash_flow":48000,"depreciation":35000,"amortization":5000,
                 "capex":92000,"working_capital":120000,"working_capital_prev":100000,"debt":332000,"cash":160000},
                {"year":2023,"revenue":868765,"net_income":73670,"ebit":95400,
                 "free_cash_flow":50000,"depreciation":32000,"amortization":4500,
                 "capex":88000,"working_capital":100000,"working_capital_prev":85000,"debt":320000,"cash":140000},
                {"year":2022,"revenue":721634,"net_income":60705,"ebit":79300,
                 "free_cash_flow":41000,"depreciation":28000,"amortization":4000,
                 "capex":75000,"working_capital":85000,"working_capital_prev":72000,"debt":298000,"cash":120000},
                {"year":2021,"revenue":486326,"net_income":53739,"ebit":68900,
                 "free_cash_flow":35000,"depreciation":24000,"amortization":3500,
                 "capex":63000,"working_capital":72000,"working_capital_prev":62000,"debt":280000,"cash":110000},
                {"year":2020,"revenue":611645,"net_income":39880,"ebit":55600,
                 "free_cash_flow":28000,"depreciation":20000,"amortization":3000,
                 "capex":55000,"working_capital":62000,"working_capital_prev":53000,"debt":261000,"cash":98000},
            ]
        },
        "HDFCBANK": {
            "company_name": "HDFC Bank Ltd",
            "symbol": "HDFCBANK", "exchange": "NSE", "token": "1333",
            "sector": "Banking", "industry": "Private Sector Bank",
            "shares_outstanding": 756.0,
            "financials": [
                {"year":2024,"revenue":220000,"net_income":64600,"ebit":80000,
                 "free_cash_flow":55000,"depreciation":6200,"amortization":800,
                 "capex":5500,"working_capital":180000,"working_capital_prev":160000,"debt":1800000,"cash":210000},
                {"year":2023,"revenue":183000,"net_income":50000,"ebit":63000,
                 "free_cash_flow":43000,"depreciation":5600,"amortization":700,
                 "capex":4900,"working_capital":160000,"working_capital_prev":140000,"debt":1600000,"cash":190000},
                {"year":2022,"revenue":150000,"net_income":36961,"ebit":47500,
                 "free_cash_flow":32000,"depreciation":4800,"amortization":600,
                 "capex":4200,"working_capital":140000,"working_capital_prev":122000,"debt":1400000,"cash":170000},
                {"year":2021,"revenue":130000,"net_income":31116,"ebit":40000,
                 "free_cash_flow":27000,"depreciation":4100,"amortization":500,
                 "capex":3600,"working_capital":122000,"working_capital_prev":108000,"debt":1200000,"cash":155000},
                {"year":2020,"revenue":113000,"net_income":26257,"ebit":34000,
                 "free_cash_flow":23000,"depreciation":3500,"amortization":450,
                 "capex":3100,"working_capital":108000,"working_capital_prev":95000,"debt":1100000,"cash":140000},
            ]
        },
        "WIPRO": {
            "company_name": "Wipro Ltd",
            "symbol": "WIPRO", "exchange": "NSE", "token": "3787",
            "sector": "Information Technology", "industry": "IT Services",
            "shares_outstanding": 520.0,
            "financials": [
                {"year":2024,"revenue":89765,"net_income":11008,"ebit":14200,
                 "free_cash_flow":10500,"depreciation":4100,"amortization":500,
                 "capex":2800,"working_capital":25000,"working_capital_prev":22000,"debt":15000,"cash":19000},
                {"year":2023,"revenue":90488,"net_income":11368,"ebit":14900,
                 "free_cash_flow":10900,"depreciation":3800,"amortization":460,
                 "capex":2600,"working_capital":22000,"working_capital_prev":20000,"debt":18000,"cash":17000},
                {"year":2022,"revenue":79312,"net_income":12238,"ebit":15600,
                 "free_cash_flow":11400,"depreciation":3400,"amortization":420,
                 "capex":2300,"working_capital":20000,"working_capital_prev":18000,"debt":20000,"cash":15000},
                {"year":2021,"revenue":61943,"net_income":10795,"ebit":13800,
                 "free_cash_flow":10100,"depreciation":3000,"amortization":380,
                 "capex":2000,"working_capital":18000,"working_capital_prev":16000,"debt":22000,"cash":13000},
                {"year":2020,"revenue":57628,"net_income":9736,"ebit":12400,
                 "free_cash_flow":9100,"depreciation":2700,"amortization":340,
                 "capex":1800,"working_capital":16000,"working_capital_prev":14000,"debt":24000,"cash":11000},
            ]
        },
    }