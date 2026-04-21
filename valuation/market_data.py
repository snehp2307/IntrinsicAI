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
