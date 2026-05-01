"""
ai_engine.py
============
Mistral-powered AI explanation engine for Intrinsic AI.

Generates institutional-grade equity research analysis using:
  - mistral-small-latest model
  - Structured valuation prompt (Buffett + Damodaran style)
  - In-memory LRU cache to avoid repeated API calls
  - Graceful fallback to rule-based explanation on failure

Environment:
  MISTRAL_API_KEY  — Required for AI explanations
"""
from __future__ import annotations

import os
import logging
import time
import hashlib
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)

# ── Mistral client (lazy init) ───────────────────────────────────────────────

_client = None


def _get_client():
    """Lazily initialise the Mistral client from env."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        log.warning("MISTRAL_API_KEY not set — AI explanations disabled.")
        return None

    try:
        # mistralai v2.x — class is in client subpackage
        try:
            from mistralai.client import Mistral
        except ImportError:
            # mistralai v1.x — class is at package root
            from mistralai import Mistral
        _client = Mistral(api_key=api_key)
        log.info("Mistral client initialised (model: mistral-small-latest)")
        return _client
    except ImportError:
        log.warning("mistralai package not installed — AI explanations disabled.")
        return None
    except Exception as e:
        log.error("Failed to initialise Mistral client: %s", e)
        return None


# ── Prompt builder ───────────────────────────────────────────────────────────

def _build_prompt(data: dict) -> str:
    """Build the equity research prompt from valuation data."""
    company      = data.get("company_name", data.get("symbol", "Unknown"))
    symbol       = data.get("symbol", "")
    sector       = data.get("sector", "")
    cmp          = data.get("current_price", 0)
    iv           = data.get("intrinsic_value", 0)
    mos          = data.get("margin_of_safety", 0)
    label        = data.get("valuation_label", "")
    wacc         = data.get("wacc", 0)
    growth       = data.get("revenue_growth_rate", 0)
    terminal     = data.get("terminal_growth_rate", 0)
    margin       = data.get("operating_margin", 0)
    fcf          = data.get("free_cash_flow", 0)
    net_debt     = data.get("net_debt", 0)

    # Sensitivity summary
    sensitivity_lines = ""
    for s in data.get("sensitivity", []):
        sensitivity_lines += f"  - If {s.get('label','')} {s.get('description','')}: IV moves ₹{s.get('change', 0):+,.2f}\n"

    # Warnings
    warnings = data.get("warnings", [])
    warn_text = "\n".join(f"  - {w}" for w in warnings) if warnings else "  None"

    return f"""You are an elite equity research analyst combining Warren Buffett's owner earnings philosophy, Aswath Damodaran's DCF valuation discipline, and institutional sell-side research rigor.

Analyze this Indian company valuation and produce a professional investment analysis:

═══════════════════════════════════════
COMPANY: {company} ({symbol})
SECTOR:  {sector}
═══════════════════════════════════════

VALUATION SNAPSHOT:
  Current Market Price (CMP): ₹{cmp:,.2f}
  Intrinsic Value (DCF):      ₹{iv:,.2f}
  Margin of Safety:           {mos:+.1f}%
  Verdict:                    {label}

DCF ASSUMPTIONS:
  WACC (Discount Rate):       {wacc:.1f}%
  Revenue Growth Rate:        {growth:.1f}%
  Terminal Growth Rate:       {terminal:.1f}%
  Operating Margin:           {margin:.1f}%
  Free Cash Flow (₹ Cr):     ₹{fcf:,.0f}
  Net Debt (₹ Cr):           ₹{net_debt:,.0f}

SENSITIVITY ANALYSIS:
{sensitivity_lines if sensitivity_lines else "  Not available"}

MODEL WARNINGS:
{warn_text}

═══════════════════════════════════════

Generate a detailed equity research note covering:

1. **Valuation Verdict** — Is this stock overvalued, undervalued, or fairly valued? By how much?
2. **Market Mispricing** — Why might the market be mispricing this stock? What is the market embedding vs. your DCF?
3. **Key Assumptions Scrutiny** — Are the DCF assumptions reasonable? Which ones are most aggressive?
4. **Risk Factors** — What are the 3-4 biggest risks that could invalidate this valuation?
5. **Long-Term Investor View** — What would Buffett or Damodaran say about this at current prices?
6. **Actionable Summary** — One clear, confident paragraph for a long-term value investor.

FORMATTING RULES:
- Use clear section headers with bold text
- Use ₹ for all Indian currency values
- Be specific with numbers — don't be vague
- Write 250-400 words total
- Write in professional but accessible language
- Do NOT use markdown code blocks or backticks
"""


# ── Cache key builder ────────────────────────────────────────────────────────

def _cache_key(data: dict) -> str:
    """Create a cache key from the core valuation inputs."""
    parts = (
        str(data.get("symbol", "")),
        str(round(data.get("current_price", 0), 1)),
        str(round(data.get("intrinsic_value", 0), 1)),
        str(round(data.get("wacc", 0), 2)),
        str(round(data.get("revenue_growth_rate", 0), 2)),
    )
    return hashlib.md5("|".join(parts).encode()).hexdigest()


# ── In-memory cache ──────────────────────────────────────────────────────────

_AI_CACHE: dict[str, dict] = {}
_MAX_CACHE = 200  # Keep at most 200 cached explanations


# ── Main API ─────────────────────────────────────────────────────────────────

def generate_ai_explanation(valuation_data: dict) -> dict:
    """
    Generate an AI-powered equity research explanation using Mistral.

    Args:
        valuation_data: dict with keys like company_name, symbol, current_price,
                        intrinsic_value, margin_of_safety, wacc, etc.

    Returns:
        {
          "ai_explanation": "...",      # The LLM-generated analysis
          "ai_model": "mistral-small-latest",
          "ai_source": "mistral" | "fallback",
          "ai_cached": True/False,
          "ai_latency_ms": 1234,
        }
    """
    key = _cache_key(valuation_data)

    # ── Check cache ──────────────────────────────────────────────────────────
    if key in _AI_CACHE:
        log.debug("AI cache hit for %s", valuation_data.get("symbol", "?"))
        cached = _AI_CACHE[key].copy()
        cached["ai_cached"] = True
        return cached

    # ── Try Mistral ──────────────────────────────────────────────────────────
    client = _get_client()
    if client is None:
        return _fallback_response(valuation_data, reason="API client not available")

    prompt = _build_prompt(valuation_data)
    t0 = time.time()

    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTE

        def _call():
            return client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior equity research analyst at a top-tier "
                            "investment bank. Provide institutional-grade analysis."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=800,
            )

        # Hard 15-second timeout
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call)
            try:
                response = future.result(timeout=15)
            except FTE:
                log.warning("Mistral API timed out (15s) for %s",
                            valuation_data.get("symbol", "?"))
                return _fallback_response(valuation_data, reason="API timeout")

        content = response.choices[0].message.content
        latency = round((time.time() - t0) * 1000)

        # Token usage logging
        usage = getattr(response, "usage", None)
        if usage:
            log.info("[Mistral] %s — %d prompt + %d completion tokens, %dms",
                     valuation_data.get("symbol", "?"),
                     getattr(usage, "prompt_tokens", 0),
                     getattr(usage, "completion_tokens", 0),
                     latency)

        result = {
            "ai_explanation": content,
            "ai_model":       "mistral-small-latest",
            "ai_source":      "mistral",
            "ai_cached":      False,
            "ai_latency_ms":  latency,
        }

        # Store in cache
        if len(_AI_CACHE) >= _MAX_CACHE:
            # Evict oldest entry
            oldest_key = next(iter(_AI_CACHE))
            del _AI_CACHE[oldest_key]
        _AI_CACHE[key] = result.copy()

        return result

    except Exception as e:
        log.error("Mistral API call failed: %s", e)
        return _fallback_response(valuation_data, reason=str(e))


# ── Fallback ─────────────────────────────────────────────────────────────────

def _fallback_response(valuation_data: dict, reason: str = "") -> dict:
    """Return a structured response using the existing rule-based explanation."""
    log.info("Using fallback explanation (reason: %s)", reason)

    company = valuation_data.get("company_name", "this company")
    cmp     = valuation_data.get("current_price", 0)
    iv      = valuation_data.get("intrinsic_value", 0)
    mos     = valuation_data.get("margin_of_safety", 0)
    label   = valuation_data.get("valuation_label", "Unknown")
    wacc    = valuation_data.get("wacc", 0)

    explanation = (
        f"AI explanation unavailable ({reason}). "
        f"Showing quantitative valuation summary instead.\n\n"
        f"Based on our DCF model, {company} appears {label.lower()}. "
        f"At a market price of ₹{cmp:,.2f} versus an intrinsic value of "
        f"₹{iv:,.2f}, the margin of safety is {mos:+.1f}%. "
        f"The discount rate (WACC) used is {wacc:.1f}%."
    )

    return {
        "ai_explanation": explanation,
        "ai_model":       "fallback",
        "ai_source":      "fallback",
        "ai_cached":      False,
        "ai_latency_ms":  0,
    }


# ── Cache management ─────────────────────────────────────────────────────────

def clear_ai_cache():
    """Clear the AI explanation cache."""
    _AI_CACHE.clear()
    log.info("AI explanation cache cleared.")
