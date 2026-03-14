"""
user_store.py
=============
Handles user-submitted company data:
  1. Persist every prediction to  data/user_companies.csv
  2. Expose saved entries via get_all_submissions()
  3. Merge user data with the 5000-company base dataset
  4. AdaptiveModel: GBM trained on merged data, used for blended scoring
     final_score = 0.60 * rule_based_score + 0.40 * ml_score
"""
import os, logging, threading
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from preprocessing import FEATURE_COLS

log = logging.getLogger(__name__)

USER_CSV = os.path.join(os.path.dirname(__file__), "data", "user_companies.csv")

_SAVE_COLS = [
    "submitted_at", "company_name",
    "current_assets","current_liabilities","inventory",
    "total_liabilities","total_assets","total_debt","total_equity",
    "net_profit","revenue","ebit","interest_expense",
    "operating_cash_flow","free_cash_flow",
] + FEATURE_COLS + ["risk_score","risk_category","altman_z"]

_write_lock = threading.Lock()


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _load_csv() -> pd.DataFrame:
    if not os.path.exists(USER_CSV):
        return pd.DataFrame(columns=_SAVE_COLS)
    try:
        df = pd.read_csv(USER_CSV)
        for c in _SAVE_COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df
    except Exception as e:
        log.warning("Could not read user CSV: %s", e)
        return pd.DataFrame(columns=_SAVE_COLS)


def save_submission(raw_inputs: dict, scored_series: pd.Series) -> None:
    """Append one submission row to the CSV store (thread-safe)."""
    from datetime import datetime, timezone
    record = {"submitted_at": datetime.now(timezone.utc).isoformat()}
    for col in _SAVE_COLS[1:]:                          # skip submitted_at
        if col in raw_inputs:
            record[col] = raw_inputs[col]
        elif col in scored_series.index:
            v = scored_series[col]
            record[col] = None if (isinstance(v, float) and np.isnan(v)) else v
        else:
            record[col] = None
    with _write_lock:
        df = _load_csv()
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(USER_CSV, index=False)
    log.info("Saved user entry: %s  score=%.1f",
             record.get("company_name","?"), record.get("risk_score", 0))


def get_all_submissions() -> list:
    """Return saved submissions as list-of-dicts, newest first."""
    df = _load_csv()
    if df.empty:
        return []
    df = df.sort_values("submitted_at", ascending=False)
    out = []
    for _, row in df.iterrows():
        def sv(k):
            v = row.get(k)
            if v is None: return None
            try:
                f = float(v)
                return None if np.isnan(f) else round(f, 4)
            except Exception:
                return str(v)
        out.append({
            "submitted_at":  str(row.get("submitted_at","")),
            "company_name":  str(row.get("company_name","")),
            "risk_score":    sv("risk_score"),
            "risk_category": str(row.get("risk_category","")),
            "altman_z":      sv("altman_z"),
            "current_ratio": sv("current_ratio"),
            "debt_to_equity":sv("debt_to_equity"),
            "profit_margin": sv("profit_margin"),
            "interest_coverage": sv("interest_coverage"),
        })
    return out


# ── Adaptive model ────────────────────────────────────────────────────────────

class AdaptiveModel:
    """
    GradientBoostingRegressor trained on base_df + user submissions.
    User rows are up-weighted 3× to amplify their influence.
    predict_one() returns None until first fit() completes.
    """
    def __init__(self):
        self._model   = None
        self._scaler  = StandardScaler()
        self._lock    = threading.Lock()
        self.trained  = False
        self.n_samples = 0
        self.n_user    = 0
        self.feature_importances: dict = {}

    def fit(self, base_df: pd.DataFrame) -> None:
        user_df  = _load_csv().dropna(subset=["risk_score"])
        self.n_user = len(user_df)

        frames = [base_df]
        if not user_df.empty:
            fc = [c for c in FEATURE_COLS if c in user_df.columns]
            for c in FEATURE_COLS:
                if c not in user_df.columns:
                    user_df[c] = base_df[c].median() if c in base_df.columns else 0.0
                user_df[c] = user_df[c].fillna(base_df[c].median() if c in base_df.columns else 0.0)
            frames += [user_df] * 3          # weight 3×

        combined = pd.concat(frames, ignore_index=True)
        fc = [c for c in FEATURE_COLS if c in combined.columns]
        X  = combined[fc].fillna(0).values.astype(float)
        y  = combined["risk_score"].values.astype(float)

        with self._lock:
            X_s = self._scaler.fit_transform(X)
            mdl = GradientBoostingRegressor(
                n_estimators=300, max_depth=4,
                learning_rate=0.08, subsample=0.8, random_state=42)
            mdl.fit(X_s, y)
            self._model    = mdl
            self.trained   = True
            self.n_samples = len(X)
            imp = mdl.feature_importances_ / (mdl.feature_importances_.sum() + 1e-12)
            self.feature_importances = {fc[i]: round(float(imp[i])*100, 2) for i in range(len(fc))}
        log.info("AdaptiveModel fitted: %d samples (%d user).", self.n_samples, self.n_user)

    def predict_one(self, ratios: dict) -> float | None:
        with self._lock:
            if not self.trained or self._model is None:
                return None
            x  = np.array([[ratios.get(c, 0.0) for c in FEATURE_COLS]], dtype=float)
            xs = self._scaler.transform(x)
            return round(float(np.clip(self._model.predict(xs)[0], 0, 100)), 1)

    def status(self) -> dict:
        return {
            "trained":      self.trained,
            "n_samples":    self.n_samples,
            "n_user":       self.n_user,
            "importances":  self.feature_importances,
        }

    def retrain_async(self, base_df: pd.DataFrame) -> None:
        threading.Thread(target=self.fit, args=(base_df,), daemon=True).start()


adaptive_model = AdaptiveModel()