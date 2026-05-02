"""
valuation/
==========
Modular DCF valuation engine.
"""
from .dcf_model        import ValuationInput, DCFResult, run_dcf, _run_dcf_core, dcf_to_dict
from .reverse_dcf      import implied_growth_rate
from .scenario_analysis import run_scenarios
from .xai_valuation    import full_xai_payload
from .data_loader      import search_companies, get_financials, get_company_info
from .market_data      import get_ltp, get_market_data, fetch_yf_financials

__all__ = [
    "ValuationInput", "DCFResult", "run_dcf", "_run_dcf_core", "dcf_to_dict",
    "implied_growth_rate", "run_scenarios", "full_xai_payload",
    "search_companies", "get_financials", "get_company_info",
    "get_ltp", "get_market_data", "fetch_yf_financials",
]