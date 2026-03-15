"""
angel_api.py
============
Angel One Smart API integration for live market data.

Fetches:
  - Current stock price (LTP)
  - OHLC + volume
  - Market cap (price × shares from dataset)

Angel One Smart API docs:
  https://smartapi.angelbroking.com/docs

Setup: Set environment variables:
  ANGEL_API_KEY
  ANGEL_CLIENT_ID
  ANGEL_PASSWORD
  ANGEL_TOTP_SECRET   (optional — for TOTP-based login)
"""
from __future__ import annotations

import os
import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# ── Optional SmartAPI import (graceful fallback to mock) ──────────────────────
try:
    from SmartApi import SmartConnect
    import pyotp
    _SMARTAPI_AVAILABLE = True
except ImportError:
    _SMARTAPI_AVAILABLE = False
    log.warning("SmartApi package not installed. Using mock price data.")


# ── Token cache (avoid re-login on every request) ────────────────────────────
_SESSION_CACHE: dict = {}
_SESSION_TTL   = 3600  # seconds


def _get_smart_api() -> Optional["SmartConnect"]:
    """Login and return an authenticated SmartConnect instance (cached)."""
    if not _SMARTAPI_AVAILABLE:
        return None

    api_key   = os.environ.get("ANGEL_API_KEY", "")
    client_id = os.environ.get("ANGEL_CLIENT_ID", "")
    password  = os.environ.get("ANGEL_PASSWORD", "")
    totp_key  = os.environ.get("ANGEL_TOTP_SECRET", "")

    if not all([api_key, client_id, password]):
        log.warning("Angel One credentials not configured.")
        return None

    now = time.time()
    cached = _SESSION_CACHE.get("obj")
    if cached and (now - _SESSION_CACHE.get("ts", 0)) < _SESSION_TTL:
        return cached

    try:
        obj = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_key).now() if totp_key else ""
        data = obj.generateSession(client_id, password, totp)
        if data.get("status"):
            _SESSION_CACHE.update({"obj": obj, "ts": now})
            log.info("Angel One session established.")
            return obj
        log.error("Angel One login failed: %s", data.get("message"))
    except Exception as e:
        log.error("Angel One connection error: %s", e)
    return None


# ── Exchange / segment mapping ────────────────────────────────────────────────

_EXCHANGE_MAP = {
    "NSE": "NSE",
    "BSE": "BSE",
    "NFO": "NFO",
}


def _resolve_exchange(symbol: str, exchange: str = "NSE") -> str:
    return _EXCHANGE_MAP.get(exchange.upper(), "NSE")


# ── Live price fetch ──────────────────────────────────────────────────────────

def get_ltp(symbol: str, exchange: str = "NSE", token: str = "") -> dict:
    """
    Fetch Last Traded Price for a symbol.

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
          "source":   "angel_one" | "mock"
        }
    """
    obj = _get_smart_api()

    if obj and token:
        try:
            exch = _resolve_exchange(symbol, exchange)
            resp = obj.ltpData(exch, symbol.upper(), token)
            if resp and resp.get("status"):
                d = resp.get("data", {})
                return {
                    "symbol":   symbol.upper(),
                    "exchange": exch,
                    "ltp":      float(d.get("ltp", 0)),
                    "open":     float(d.get("open", 0)),
                    "high":     float(d.get("high", 0)),
                    "low":      float(d.get("low", 0)),
                    "close":    float(d.get("close", 0)),
                    "volume":   int(d.get("tradedVolume", 0)),
                    "source":   "angel_one",
                }
        except Exception as e:
            log.warning("Angel LTP fetch failed for %s: %s", symbol, e)

    # ── Fallback: yfinance ────────────────────────────────────────────────────
    return _yfinance_fallback(symbol, exchange)


def _yfinance_fallback(symbol: str, exchange: str = "NSE") -> dict:
    """Try yfinance as a secondary data source."""
    try:
        import yfinance as yf
        suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
        ticker = yf.Ticker(f"{symbol.upper()}{suffix}")
        info   = ticker.fast_info
        hist   = ticker.history(period="1d")
        ltp    = float(info.get("lastPrice") or info.get("regularMarketPrice") or 0)
        vol    = int(info.get("threeMonthAverageVolume") or 0)

        if hist is not None and not hist.empty:
            row = hist.iloc[-1]
            return {
                "symbol":   symbol.upper(),
                "exchange": exchange.upper(),
                "ltp":      round(ltp or float(row["Close"]), 2),
                "open":     round(float(row["Open"]), 2),
                "high":     round(float(row["High"]), 2),
                "low":      round(float(row["Low"]), 2),
                "close":    round(float(row["Close"]), 2),
                "volume":   vol,
                "source":   "yfinance",
            }
    except Exception as e:
        log.warning("yfinance fallback failed for %s: %s", symbol, e)

    # ── Final fallback: mock ──────────────────────────────────────────────────
    return _mock_price(symbol, exchange)


def _mock_price(symbol: str, exchange: str = "NSE") -> dict:
    """Return deterministic mock prices for development/testing."""
    import hashlib
    seed  = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
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


# ── Historical OHLC ───────────────────────────────────────────────────────────

def get_historical_prices(symbol: str, exchange: str = "NSE",
                           token: str = "", days: int = 365) -> list[dict]:
    """
    Return list of daily OHLCV dicts for the past `days` days.
    Falls back to yfinance if Angel One is unavailable.
    """
    try:
        import yfinance as yf
        suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
        hist   = yf.Ticker(f"{symbol.upper()}{suffix}").history(period=f"{days}d")
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