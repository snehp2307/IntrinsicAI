"""
market_data.py  (formerly angel_api.py)
=======================================
Stock market data provider using yfinance as the primary source.

Supports:
  - NSE stocks (auto-appends .NS suffix)
  - BSE stocks (auto-appends .BO suffix)
  - Global stocks (pass raw ticker)

Features:
  - LTP price cache with configurable TTL (default 5 minutes)
  - Automatic NSE → BSE → raw symbol fallback
  - Mock prices for development/offline use
  - Historical OHLCV data

Environment variables:
  SKIP_YFINANCE=1     Skip yfinance entirely, use mock prices (for offline dev)
  PRICE_CACHE_TTL=300  Cache TTL in seconds (default 300 = 5 min)
"""
from __future__ import annotations

import os
import logging
import time
import hashlib
from typing import Optional

log = logging.getLogger(__name__)

# ── yfinance import (graceful fallback) ───────────────────────────────────────
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    log.warning("yfinance not installed (pip install yfinance). Using mock prices.")


# ── Price cache ───────────────────────────────────────────────────────────────
_PRICE_CACHE: dict = {}
_PRICE_CACHE_TTL = int(os.environ.get("PRICE_CACHE_TTL", "300"))  # 5 minutes

_HIST_CACHE: dict = {}
_HIST_CACHE_TTL  = 600  # 10 minutes for historical data


# ── Exchange suffix mapping ───────────────────────────────────────────────────

_EXCHANGE_SUFFIX = {
    "NSE": ".NS",
    "BSE": ".BO",
}


def _yf_symbol(symbol: str, exchange: str = "NSE") -> str:
    """Convert a bare symbol to a yfinance-compatible ticker."""
    sym = symbol.upper().strip()
    # If already has a suffix (e.g., AAPL, MSFT, TCS.NS), return as-is
    if "." in sym:
        return sym
    suffix = _EXCHANGE_SUFFIX.get(exchange.upper(), "")
    return f"{sym}{suffix}"


# ── Core price fetcher ────────────────────────────────────────────────────────

def _fetch_yf_price(yf_symbol: str) -> Optional[dict]:
    """
    Fetch price data from yfinance for a single symbol.
    Has a hard 10-second timeout to prevent hangs.
    Returns dict with price fields or None on failure.
    """
    if not _YF_AVAILABLE:
        return None

    import math
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    def _do_fetch():
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="5d")
        return hist

    try:
        # Hard 10-second timeout on yfinance call
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_fetch)
            try:
                hist = future.result(timeout=10)
            except FuturesTimeout:
                log.warning("yfinance timeout (10s) for %s", yf_symbol)
                future.cancel()
                return None

        if hist is None or hist.empty:
            return None

        # Drop rows with NaN Close
        hist = hist.dropna(subset=["Close"])
        if hist.empty:
            return None

        row = hist.iloc[-1]
        ltp = float(row["Close"])

        if math.isnan(ltp) or math.isinf(ltp) or ltp <= 0:
            return None

        ltp = round(ltp, 2)

        def safe_float(val, fallback=0.0):
            try:
                v = float(val)
                return round(v, 2) if not (math.isnan(v) or math.isinf(v)) else fallback
            except (TypeError, ValueError):
                return fallback

        vol = int(safe_float(row.get("Volume", 0)))

        return {
            "ltp":    ltp,
            "open":   safe_float(row["Open"], ltp),
            "high":   safe_float(row["High"], ltp),
            "low":    safe_float(row["Low"], ltp),
            "close":  ltp,
            "volume": vol,
        }
    except Exception as e:
        log.debug("yfinance fetch failed for %s: %s", yf_symbol, e)
        return None


def _fetch_with_fallback(symbol: str, exchange: str = "NSE") -> tuple[dict, str]:
    """
    Try fetching price with automatic fallback chain:
      1. NSE (.NS) or specified exchange
      2. BSE (.BO) if NSE fails
      3. Raw symbol (for global stocks)

    Returns: (price_dict, source_label)
    """
    sym = symbol.upper().strip()

    # Attempt 1: Primary exchange
    primary = _yf_symbol(sym, exchange)
    data = _fetch_yf_price(primary)
    if data:
        return data, f"yfinance ({exchange.upper()})"

    # Attempt 2: Alternate Indian exchange
    if exchange.upper() == "NSE":
        alt = f"{sym}.BO"
        data = _fetch_yf_price(alt)
        if data:
            return data, "yfinance (BSE fallback)"
    elif exchange.upper() == "BSE":
        alt = f"{sym}.NS"
        data = _fetch_yf_price(alt)
        if data:
            return data, "yfinance (NSE fallback)"

    # Attempt 3: Raw symbol (for global stocks like AAPL, MSFT)
    if "." not in sym:
        data = _fetch_yf_price(sym)
        if data:
            return data, "yfinance (global)"

    return {}, ""


# ── Public API: get_ltp ───────────────────────────────────────────────────────

def get_ltp(symbol: str, exchange: str = "NSE", token: str = "") -> dict:
    """
    Fetch Last Traded Price for a symbol.
    Results are cached for PRICE_CACHE_TTL seconds.

    Args:
        symbol:   Stock symbol (e.g., TCS, RELIANCE, INFY)
        exchange: NSE or BSE (default: NSE)
        token:    Ignored (kept for backward compatibility)

    Returns:
        {
          "symbol":   "TCS",
          "exchange": "NSE",
          "ltp":      3850.25,
          "open":     3820.00,
          "high":     3870.00,
          "low":      3810.00,
          "close":    3845.00,
          "volume":   1234567,
          "source":   "yfinance (NSE)" | "mock"
        }
    """
    cache_key = f"{symbol.upper()}:{exchange.upper()}"
    now = time.time()

    # ── Check cache ───────────────────────────────────────────────────────────
    cached = _PRICE_CACHE.get(cache_key)
    if cached and (now - cached.get("_ts", 0)) < _PRICE_CACHE_TTL:
        log.debug("Price cache hit for %s (%.0fs old)", cache_key, now - cached["_ts"])
        return {k: v for k, v in cached.items() if k != "_ts"}

    t0 = time.time()

    # ── Skip yfinance if env says so ──────────────────────────────────────────
    skip_yf = os.environ.get("SKIP_YFINANCE", "").strip().lower()
    if skip_yf in ("1", "true", "yes"):
        result = _mock_price(symbol, exchange)
        _PRICE_CACHE[cache_key] = {**result, "_ts": now}
        log.info("Mock LTP for %s: ₹%.2f (skip_yfinance=true)", symbol, result["ltp"])
        return result

    # ── Fetch from yfinance with fallback ─────────────────────────────────────
    price_data, source = _fetch_with_fallback(symbol, exchange)

    if price_data:
        result = {
            "symbol":   symbol.upper(),
            "exchange": exchange.upper(),
            "ltp":      price_data["ltp"],
            "open":     price_data["open"],
            "high":     price_data["high"],
            "low":      price_data["low"],
            "close":    price_data["close"],
            "volume":   price_data["volume"],
            "source":   source,
        }
        _PRICE_CACHE[cache_key] = {**result, "_ts": time.time()}
        log.info("LTP for %s: ₹%.2f via %s (%.2fs)",
                 symbol, result["ltp"], source, time.time() - t0)
        return result

    # ── Final fallback: mock ──────────────────────────────────────────────────
    result = _mock_price(symbol, exchange)
    _PRICE_CACHE[cache_key] = {**result, "_ts": time.time()}
    log.warning("All sources failed for %s, using mock: ₹%.2f (%.2fs)",
                symbol, result["ltp"], time.time() - t0)
    return result


# ── Mock price generator ──────────────────────────────────────────────────────

def _mock_price(symbol: str, exchange: str = "NSE") -> dict:
    """Return deterministic mock prices for development/testing."""
    seed  = int(hashlib.md5(symbol.upper().encode()).hexdigest()[:8], 16)
    price = round(500 + (seed % 4000), 2)
    return {
        "symbol":   symbol.upper(),
        "exchange": exchange.upper(),
        "ltp":      price,
        "open":     round(price * 0.99, 2),
        "high":     round(price * 1.02, 2),
        "low":      round(price * 0.97, 2),
        "close":    round(price * 1.00, 2),
        "volume":   seed % 5_000_000,
        "source":   "mock",
    }


# ── Historical OHLC ──────────────────────────────────────────────────────────

def get_historical_prices(symbol: str, exchange: str = "NSE",
                           token: str = "", days: int = 365) -> list[dict]:
    """
    Return list of daily OHLCV dicts for the past `days` days.
    Results are cached for 10 minutes.
    """
    cache_key = f"hist:{symbol.upper()}:{exchange.upper()}:{days}"
    now = time.time()

    # Check cache
    cached = _HIST_CACHE.get(cache_key)
    if cached and (now - cached.get("_ts", 0)) < _HIST_CACHE_TTL:
        return cached["data"]

    skip_yf = os.environ.get("SKIP_YFINANCE", "").strip().lower()
    if skip_yf in ("1", "true", "yes") or not _YF_AVAILABLE:
        return []

    try:
        yf_sym = _yf_symbol(symbol, exchange)
        hist = yf.Ticker(yf_sym).history(period=f"{days}d")
        if hist is not None and not hist.empty:
            records = []
            for dt, row in hist.iterrows():
                records.append({
                    "date":   str(dt.date()),
                    "open":   round(float(row["Open"]), 2),
                    "high":   round(float(row["High"]), 2),
                    "low":    round(float(row["Low"]), 2),
                    "close":  round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                })
            _HIST_CACHE[cache_key] = {"data": records, "_ts": time.time()}
            return records
    except Exception as e:
        log.warning("Historical fetch failed for %s: %s", symbol, e)

    return []


# ── yfinance Financial Data Extraction ────────────────────────────────────

_FIN_CACHE: dict = {}
_FIN_CACHE_TTL = 3600  # 1 hour for financial data

def fetch_yf_financials(symbol: str, exchange: str = "NSE") -> Optional[dict]:
    """
    Extract real company financials from yfinance:
      - Income statement (revenue, net income, EBIT, depreciation)
      - Balance sheet (assets, liabilities, cash, debt, equity, shares)
      - Cash flow (operating CF, capex, FCF)
      - Historical growth rates (CAGR)

    Returns dict suitable for DCF inputs, or None on failure.
    Cached for 1 hour.
    """
    cache_key = f"fin:{symbol.upper()}:{exchange.upper()}"
    now = time.time()

    cached = _FIN_CACHE.get(cache_key)
    if cached and (now - cached.get("_ts", 0)) < _FIN_CACHE_TTL:
        log.debug("Financial cache hit for %s", symbol)
        result = {k: v for k, v in cached.items() if k != "_ts"}
        return result

    if not _YF_AVAILABLE:
        return None

    skip_yf = os.environ.get("SKIP_YFINANCE", "").strip().lower()
    if skip_yf in ("1", "true", "yes"):
        return None

    import math
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTE

    def _do_fetch():
        yf_sym = _yf_symbol(symbol, exchange)
        ticker = yf.Ticker(yf_sym)

        # Fetch financial statements
        inc = ticker.income_stmt        # columns = fiscal years
        bs  = ticker.balance_sheet
        cf  = ticker.cashflow
        info = ticker.info or {}

        return inc, bs, cf, info, yf_sym

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_fetch)
            try:
                inc, bs, cf, info, yf_sym = future.result(timeout=15)
            except FTE:
                log.warning("yfinance financial fetch timeout (15s) for %s", symbol)
                return None

        if inc is None or inc.empty:
            log.debug("No income statement data for %s", symbol)
            return None

        # ── Helper to safely extract a value from a DataFrame ─────────
        def _get(df, labels, col_idx=0, default=0.0):
            """Try multiple label names (yfinance labels vary by company)."""
            if df is None or df.empty:
                return default
            for label in labels:
                if label in df.index:
                    try:
                        val = df.iloc[df.index.get_loc(label), col_idx]
                        if val is not None and not (isinstance(val, float) and math.isnan(val)):
                            return float(val)
                    except Exception:
                        continue
            return default

        def _get_all_years(df, labels):
            """Get values across all available years for CAGR."""
            if df is None or df.empty:
                return []
            for label in labels:
                if label in df.index:
                    try:
                        vals = []
                        for i in range(len(df.columns)):
                            v = df.iloc[df.index.get_loc(label), i]
                            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                                vals.append(float(v))
                        return vals
                    except Exception:
                        continue
            return []

        # ── Extract from Income Statement (₹ in actual units from yfinance) ──
        # yfinance returns values in the company's reporting currency (₹ for Indian stocks)
        # Values are typically in raw units, need to convert to crores for Indian stocks
        scale = 1e7  # 1 Cr = 10 million = 1e7

        revenue     = _get(inc, ["Total Revenue", "Revenue", "Operating Revenue"]) / scale
        net_income  = _get(inc, ["Net Income", "Net Income Common Stockholders",
                                  "Net Income From Continuing Operations"]) / scale
        ebit        = _get(inc, ["EBIT", "Operating Income", "Operating Profit"]) / scale
        depreciation = _get(inc, ["Depreciation And Amortization", "Depreciation",
                                   "Reconciled Depreciation"]) / scale
        tax_expense = _get(inc, ["Tax Provision", "Income Tax Expense", "Tax Expense"]) / scale

        # ── Extract from Balance Sheet ────────────────────────────────
        total_assets   = _get(bs, ["Total Assets"]) / scale
        total_liab     = _get(bs, ["Total Liabilities Net Minority Interest",
                                    "Total Liab", "Total Non Current Liabilities Net Minority Interest"]) / scale
        cash           = _get(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
                                    "Cash Financial"]) / scale
        total_debt     = _get(bs, ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"]) / scale
        equity         = _get(bs, ["Total Equity Gross Minority Interest", "Stockholders Equity",
                                    "Common Stock Equity"]) / scale
        current_assets = _get(bs, ["Current Assets"]) / scale
        current_liab   = _get(bs, ["Current Liabilities"]) / scale
        working_capital = current_assets - current_liab

        # Previous year working capital for WC change
        wc_prev = 0.0
        if bs is not None and len(bs.columns) > 1:
            ca_prev = _get(bs, ["Current Assets"], col_idx=1) / scale
            cl_prev = _get(bs, ["Current Liabilities"], col_idx=1) / scale
            wc_prev = ca_prev - cl_prev

        wc_change = working_capital - wc_prev

        # ── Extract from Cash Flow Statement ──────────────────────────
        operating_cf = _get(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
                                  "Total Cash From Operating Activities"]) / scale
        capex        = abs(_get(cf, ["Capital Expenditure", "Purchase Of Property Plant And Equipment",
                                      "Capital Expenditures"])) / scale
        fcf_reported = _get(cf, ["Free Cash Flow"]) / scale

        # ── Shares outstanding ────────────────────────────────────────
        shares_outstanding = info.get("sharesOutstanding", 0)
        if shares_outstanding:
            shares_outstanding = shares_outstanding / 1e6  # Convert to millions
        else:
            shares_outstanding = _get(bs, ["Share Issued", "Ordinary Shares Number"]) / 1e6
        if shares_outstanding <= 0:
            shares_outstanding = 1.0

        # ── Calculate historical growth (CAGR) ────────────────────────
        rev_history = _get_all_years(inc, ["Total Revenue", "Revenue", "Operating Revenue"])
        ni_history  = _get_all_years(inc, ["Net Income", "Net Income Common Stockholders"])

        def _cagr(vals):
            """CAGR from most recent (idx 0) to oldest. Clamp to [-20%, +50%]."""
            if len(vals) < 2:
                return None
            newest, oldest = vals[0], vals[-1]
            if oldest <= 0 or newest <= 0:
                return None
            n = len(vals) - 1
            try:
                rate = (newest / oldest) ** (1 / n) - 1
                return round(max(min(rate, 0.50), -0.20), 4)
            except Exception:
                return None

        revenue_cagr    = _cagr(rev_history)
        net_income_cagr = _cagr(ni_history)

        # ── Operating margin ──────────────────────────────────────────
        op_margin = None
        if revenue > 0 and ebit > 0:
            op_margin = round(ebit / revenue, 4)
        elif revenue > 0 and net_income > 0:
            op_margin = round(net_income / revenue * 1.2, 4)  # rough pre-tax proxy

        # ── Tax rate from actual data ─────────────────────────────────
        pretax_income = _get(inc, ["Pretax Income", "Income Before Tax"]) / scale
        actual_tax_rate = None
        if pretax_income > 0 and tax_expense > 0:
            actual_tax_rate = round(tax_expense / pretax_income, 4)

        # ── Net debt ──────────────────────────────────────────────────
        net_debt = total_debt - cash

        # ── Validate — reject if all zeros (yfinance returned empty structure) ──
        if revenue <= 0 and net_income <= 0 and total_assets <= 0:
            log.info("yfinance returned empty financials for %s", symbol)
            return None

        # ── Build historical entries for chart ────────────────────────
        history = []
        if inc is not None:
            for i, col in enumerate(inc.columns[:5]):
                try:
                    yr = col.year if hasattr(col, 'year') else str(col)[:4]
                    history.append({
                        "year":          int(yr) if isinstance(yr, (int, float)) else yr,
                        "revenue":       round(_get(inc, ["Total Revenue", "Revenue"], i) / scale, 2),
                        "net_income":    round(_get(inc, ["Net Income", "Net Income Common Stockholders"], i) / scale, 2),
                        "ebit":          round(_get(inc, ["EBIT", "Operating Income"], i) / scale, 2),
                        "free_cash_flow": round(_get(cf, ["Free Cash Flow"], i) / scale, 2) if cf is not None and not cf.empty else 0,
                    })
                except Exception:
                    continue

        result = {
            "symbol":               symbol.upper(),
            "exchange":             exchange.upper(),
            "data_source":          "yfinance_live",

            # Income statement
            "net_income":           round(net_income, 2),
            "depreciation":         round(depreciation, 2) if depreciation > 0 else round(revenue * 0.04, 2),
            "amortization":         round(depreciation * 0.1, 2),
            "revenue":              round(revenue, 2),
            "ebit":                 round(ebit, 2),

            # Cash flow
            "capex":                round(capex, 2) if capex > 0 else round(revenue * 0.05, 2),
            "operating_cash_flow":  round(operating_cf, 2),
            "free_cash_flow":       round(fcf_reported if fcf_reported else (operating_cf - capex), 2),
            "working_capital_change": round(wc_change, 2),

            # Balance sheet
            "total_assets":         round(total_assets, 2),
            "total_liabilities":    round(total_liab, 2),
            "cash":                 round(cash, 2),
            "total_debt":           round(total_debt, 2),
            "equity":               round(equity, 2),
            "current_assets":       round(current_assets, 2),
            "current_liabilities":  round(current_liab, 2),
            "net_debt":             round(net_debt, 2),

            # Computed
            "shares_outstanding":   round(shares_outstanding, 2),
            "revenue_growth_rate":  revenue_cagr if revenue_cagr is not None else 0.08,
            "operating_margin":     op_margin if op_margin is not None else 0.12,
            "tax_rate":             actual_tax_rate if actual_tax_rate is not None else 0.25,

            # Growth flags
            "revenue_cagr_available":    revenue_cagr is not None,
            "operating_margin_computed": op_margin is not None,
            "tax_rate_computed":         actual_tax_rate is not None,

            # ── Previous year data (for Piotroski, Beneish, etc.) ─────
            "revenue_prev":             round(_get(inc, ["Total Revenue", "Revenue"], col_idx=1) / scale, 2) if inc is not None and len(inc.columns) > 1 else 0,
            "net_income_prev":          round(_get(inc, ["Net Income", "Net Income Common Stockholders"], col_idx=1) / scale, 2) if inc is not None and len(inc.columns) > 1 else 0,
            "total_assets_prev":        round(_get(bs, ["Total Assets"], col_idx=1) / scale, 2) if bs is not None and len(bs.columns) > 1 else 0,
            "total_liabilities_prev":   round(_get(bs, ["Total Liabilities Net Minority Interest", "Total Liab"], col_idx=1) / scale, 2) if bs is not None and len(bs.columns) > 1 else 0,
            "equity_prev":              round(_get(bs, ["Total Equity Gross Minority Interest", "Stockholders Equity"], col_idx=1) / scale, 2) if bs is not None and len(bs.columns) > 1 else 0,
            "current_assets_prev":      round(_get(bs, ["Current Assets"], col_idx=1) / scale, 2) if bs is not None and len(bs.columns) > 1 else 0,
            "current_liabilities_prev": round(_get(bs, ["Current Liabilities"], col_idx=1) / scale, 2) if bs is not None and len(bs.columns) > 1 else 0,

            # ── Detailed line items for multi-model ───────────────────
            "retained_earnings":        round(_get(bs, ["Retained Earnings", "Retained Profit"]) / scale, 2),
            "inventory":                round(_get(bs, ["Inventory", "Net Inventory"]) / scale, 2),
            "receivables":              round(_get(bs, ["Accounts Receivable", "Net Receivables", "Receivables"]) / scale, 2),
            "receivables_prev":         round(_get(bs, ["Accounts Receivable", "Net Receivables", "Receivables"], col_idx=1) / scale, 2) if bs is not None and len(bs.columns) > 1 else 0,
            "ppe":                      round(_get(bs, ["Net PPE", "Property Plant And Equipment", "Gross PPE"]) / scale, 2),
            "ppe_prev":                 round(_get(bs, ["Net PPE", "Property Plant And Equipment", "Gross PPE"], col_idx=1) / scale, 2) if bs is not None and len(bs.columns) > 1 else 0,
            "gross_profit":             round(_get(inc, ["Gross Profit"]) / scale, 2),
            "gross_profit_prev":        round(_get(inc, ["Gross Profit"], col_idx=1) / scale, 2) if inc is not None and len(inc.columns) > 1 else 0,
            "sga_expense":              round(_get(inc, ["Selling General And Administration", "General And Administrative Expense"]) / scale, 2),
            "sga_expense_prev":         round(_get(inc, ["Selling General And Administration", "General And Administrative Expense"], col_idx=1) / scale, 2) if inc is not None and len(inc.columns) > 1 else 0,
            "interest_expense":         round(abs(_get(inc, ["Interest Expense", "Interest Expense Non Operating", "Net Interest Income"])) / scale, 2),
            "long_term_debt":           round(_get(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]) / scale, 2),
            "long_term_debt_prev":      round(_get(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], col_idx=1) / scale, 2) if bs is not None and len(bs.columns) > 1 else 0,

            # ── Dividend data ─────────────────────────────────────────
            "dividend_per_share":       info.get("dividendRate", 0) or 0,
            "dividend_yield":           info.get("dividendYield", 0) or 0,
            "payout_ratio":             info.get("payoutRatio", 0) or 0,

            # ── Valuation multiples from yfinance ─────────────────────
            "trailing_pe":              info.get("trailingPE", 0) or 0,
            "forward_pe":               info.get("forwardPE", 0) or 0,
            "price_to_book":            info.get("priceToBook", 0) or 0,
            "ev_to_ebitda":             info.get("enterpriseToEbitda", 0) or 0,
            "beta":                     info.get("beta", 1.0) or 1.0,

            # History
            "history":              history,

            # Company info from yfinance
            "company_name":         info.get("longName", info.get("shortName", symbol)),
            "sector":               info.get("sector", ""),
            "industry":             info.get("industry", ""),
        }

        _FIN_CACHE[cache_key] = {**result, "_ts": time.time()}
        log.info("yfinance financials for %s: Revenue=%.0f Cr, NI=%.0f Cr, FCF=%.0f Cr, Shares=%.1fM",
                 symbol, revenue, net_income, result["free_cash_flow"], shares_outstanding)
        return result

    except Exception as e:
        log.warning("yfinance financial extraction failed for %s: %s", symbol, e)
        return None


# ── Market cap estimate ───────────────────────────────────────────────────────

def get_market_data(symbol: str, shares_outstanding: float = 1.0,
                    exchange: str = "NSE", token: str = "") -> dict:
    """
    Return LTP + derived market cap.
    shares_outstanding: in millions
    """
    price_data = get_ltp(symbol, exchange, token)
    ltp        = price_data["ltp"]
    market_cap = round(ltp * shares_outstanding, 2)  # ₹ millions

    return {
        **price_data,
        "shares_outstanding": shares_outstanding,
        "market_cap_millions": market_cap,
    }


# ── Cache management ─────────────────────────────────────────────────────────

def clear_price_cache():
    """Clear all cached prices (useful for testing)."""
    _PRICE_CACHE.clear()
    _HIST_CACHE.clear()
    log.info("Price cache cleared.")
