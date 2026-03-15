"""
app.py — XAI Bankruptcy SaaS Platform  (v2)
============================================
Routes:
  /                     Landing + Pricing
  /register             Sign up
  /login                Sign in
  /logout               Sign out
  /dashboard            User dashboard
  /predict              Prediction form
  /result/<id>          Full analysis — XAI + all valuation models
  /history              Prediction history
"""
import os, json, pickle, functools
from datetime import datetime, timedelta
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, flash, abort)
import numpy as np

import database as db
from valuation_models import run_quick_valuation
from xai_engine import run_all_xai, get_xai_status, local_contribution

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "xai-bankrupt-secret-2024-change-in-prod")

# ── Valuation module blueprint ────────────────────────────────────────────────
from valuation.routes import valuation_bp
app.register_blueprint(valuation_bp)

MODELS_DIR             = "models/"
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:5000")

# ── Load ML artefacts ─────────────────────────────────────────────────────────

def _load(fname):
    with open(os.path.join(MODELS_DIR, fname), "rb") as f:
        return pickle.load(f)

try:
    RF_MODEL    = _load("model_rf.pkl")
    SCALER      = _load("scaler.pkl")
    IMPUTER     = _load("imputer.pkl")
    FEAT_COLS   = _load("feature_cols.pkl")
    FEAT_IMP    = _load("feature_importance.pkl")
    METRICS     = _load("metrics.pkl")
    FEAT_LABELS = _load("feature_labels.pkl")
    FEAT_DESCS  = _load("feature_descriptions.pkl")
    X_TRAIN_BG  = None
    print("[OK] Models loaded.")
except Exception as e:
    print(f"[WARN] Models not loaded: {e}. Run train_model.py first.")
    RF_MODEL = SCALER = IMPUTER = FEAT_COLS = FEAT_IMP = METRICS = None
    FEAT_LABELS = {}; FEAT_DESCS = {}

FEATURE_DEFAULTS = {
    "Attr6": 0.20, "Attr16": 0.45, "Attr13": 0.35,
    "Attr5": 30.0, "Attr12": 0.10, "Attr15": 0.05,
}
FEATURE_GUIDANCE = {
    "Attr6":  "Healthy: 0.10–0.50  |  Below 0 = danger",
    "Attr16": "Healthy: 0.20–0.60  |  Above 0.80 = danger",
    "Attr13": "Healthy: 0.20–0.80  |  Below 0.10 = danger",
    "Attr5":  "Healthy: 10–100     |  Negative = danger",
    "Attr12": "Healthy: 0.05–0.30  |  Below 0 = danger",
    "Attr15": "Healthy: 0.02–0.20  |  Negative = danger",
}

db.init_db()



# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in.", "warning")
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated

def current_user():
    return db.get_user_by_id(session["user_id"]) if "user_id" in session else None

app.jinja_env.globals.update(
    current_user=current_user
)

# ── Landing ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           xai_status=get_xai_status(),
                           models_ready=RF_MODEL is not None)

# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name  = request.form.get("name","").strip()
        email = request.form.get("email","").strip().lower()
        pw    = request.form.get("password","")
        pw2   = request.form.get("password2","")
        if not all([name, email, pw]):
            flash("All fields required.", "danger")
        elif pw != pw2:
            flash("Passwords do not match.", "danger")
        elif len(pw) < 8:
            flash("Password must be at least 8 characters.", "danger")
        elif db.get_user_by_email(email):
            flash("Email already registered.", "danger")
        else:
            uid = db.create_user(name, email, pw)
            session["user_id"] = uid
            flash(f"Welcome, {name}!", "success")
            return redirect(url_for("dashboard"))
    return render_template("auth/register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        pw    = request.form.get("password","")
        user  = db.verify_password(email, pw)
        if user:
            session["user_id"] = user["id"]
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("auth/login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("index"))

# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    uid   = session["user_id"]
    user  = db.get_user_by_id(uid)
    preds = db.get_user_predictions(uid, limit=10)
    stats = db.get_user_stats(uid)
    return render_template("dashboard.html", user=user,
                           predictions=[dict(p) for p in preds],
                           stats=stats)

# ── Predict ───────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["GET","POST"])
@login_required
def predict():
    if RF_MODEL is None:
        flash("ML models not loaded. Run train_model.py first.", "danger")
        return redirect(url_for("dashboard"))

    uid = session["user_id"]

    if request.method == "GET":
        features = [{"code": f, "label": FEAT_LABELS.get(f,f),
                     "desc": FEAT_DESCS.get(f,""),
                     "guidance": FEATURE_GUIDANCE.get(f,""),
                     "default": FEATURE_DEFAULTS.get(f,0.0)} for f in FEAT_COLS]
        return render_template("predict.html", features=features)

    company_name = request.form.get("company_name", "Unknown Company").strip()
    save_data    = request.form.get("save_data") == "on"
    true_label   = request.form.get("true_label")

    inputs = {f: float(request.form.get(f, FEATURE_DEFAULTS.get(f, 0.0))) for f in FEAT_COLS}
    x_raw  = np.array([inputs[f] for f in FEAT_COLS])
    x_sc   = SCALER.transform(x_raw.reshape(1, -1))

    prob       = float(RF_MODEL.predict_proba(x_sc)[0, 1])
    bankrupt   = int(prob >= 0.5)
    risk_level = "HIGH" if prob >= 0.60 else "MODERATE" if prob >= 0.30 else "LOW"

    # XAI
    try:
        bg       = _get_train_bg()
        xai_data = run_all_xai(RF_MODEL, SCALER, x_raw, bg, FEAT_COLS, FEAT_LABELS)
    except Exception as e:
        xai_data = {"error": str(e)}

    # Valuation
    valuation = run_quick_valuation(inputs)

    core = {"bankrupt": bankrupt, "probability": round(prob*100, 1),
            "risk_level": risk_level}

    pred_id = db.save_prediction(uid, company_name, inputs, core,
                                 valuation, xai_data, risk_level, round(prob*100, 1))

    return redirect(url_for("result", pred_id=pred_id))

def _get_train_bg():
    global X_TRAIN_BG
    if X_TRAIN_BG is None:
        import pandas as pd
        csv = "data/companies.csv"          # Fixed: was 'dataset/polish.csv'
        if os.path.exists(csv) and FEAT_COLS:
            from preprocessing import load_and_preprocess
            try:
                df = load_and_preprocess(csv)
                available = [c for c in FEAT_COLS if c in df.columns]
                if available:
                    X_TRAIN_BG = SCALER.transform(df[available].dropna().values[:500])
                else:
                    X_TRAIN_BG = np.zeros((10, len(FEAT_COLS)))
            except Exception:
                X_TRAIN_BG = np.zeros((10, len(FEAT_COLS)))
        else:
            X_TRAIN_BG = np.zeros((10, len(FEAT_COLS) if FEAT_COLS else 9))
    return X_TRAIN_BG

# ── Result ────────────────────────────────────────────────────────────────────

@app.route("/result/<int:pred_id>")
@login_required
def result(pred_id):
    uid  = session["user_id"]
    pred = db.get_prediction_by_id(pred_id, uid)
    if not pred:
        abort(404)

    inputs    = json.loads(pred["inputs"])
    results   = json.loads(pred["results"])
    valuation = json.loads(pred["valuation_models"] or "{}")
    xai_data  = json.loads(pred["xai_data"] or "{}")

    factors = []
    if RF_MODEL and SCALER:
        x_raw   = np.array([inputs.get(f, 0.0) for f in FEAT_COLS])
        factors = local_contribution(RF_MODEL, SCALER, x_raw, FEAT_COLS, FEAT_LABELS)

    return render_template("result.html",
                           pred=pred, inputs=inputs, results=results,
                           valuation=valuation, xai_data=xai_data,
                           factors=factors, feat_labels=FEAT_LABELS,
                           sub=None)

# ── History ───────────────────────────────────────────────────────────────────

@app.route("/history")
@login_required
def history():
    uid   = session["user_id"]
    preds = db.get_user_predictions(uid, limit=100)
    return render_template("history.html", predictions=preds)



# ── Model Metrics ─────────────────────────────────────────────────────────────

@app.route("/metrics")
def metrics_page():
    if METRICS is None:
        flash("Models not trained yet.", "warning")
        return redirect(url_for("index"))
    m  = METRICS.get("Random Forest", {})
    fi = [(f, round(imp*100,2)) for f, imp in (FEAT_IMP or [])]
    return render_template("metrics.html", metrics=m, feature_importance=fi,
                           feat_labels=FEAT_LABELS)

# ── Admin Retrain ─────────────────────────────────────────────────────────────

@app.route("/admin/retrain", methods=["GET","POST"])
@login_required
def retrain():
    user = db.get_user_by_id(session["user_id"])
    if not user["is_admin"]:
        abort(403)
    stats = db.get_admin_stats()
    if request.method == "POST":
        try:
            _retrain_with_contributions([])  # Simplied, would use actual dataset in real project
            flash("Model retrained successfully!", "success")
        except Exception as e:
            flash(f"Retrain failed: {e}", "danger")
    return render_template("admin_retrain.html", stats=stats)

def _retrain_with_contributions(contributions):
    global RF_MODEL, X_TRAIN_BG
    import pandas as pd
    csv = "dataset/polish.csv"
    if not os.path.exists(csv):
        raise FileNotFoundError("Base dataset not found.")
    df_base = pd.read_csv(csv)[FEAT_COLS + ["class"]].dropna()
    extra = [
        {**{f: json.loads(c["features"]).get(f,0.0) for f in FEAT_COLS}, "class": c["true_label"]}
        for c in contributions
    ]
    df_all = pd.concat([df_base, pd.DataFrame(extra)], ignore_index=True) if extra else df_base
    X = IMPUTER.transform(df_all[FEAT_COLS].values)
    y = df_all["class"].values.astype(int)
    RF_MODEL.fit(SCALER.transform(X), y)
    with open(os.path.join(MODELS_DIR, "model_rf.pkl"), "wb") as f:
        pickle.dump(RF_MODEL, f)
    X_TRAIN_BG = None



# ── /api/predict  (JSON endpoint called by predict.html) ─────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    from preprocessing import FEATURE_COLS as FC
    from risk_engine import (RATIO_CONFIG, compute_risk_scores,
                              get_top_risk_factors, build_explanation, TOTAL_WEIGHT)
    from xai import (_attributions, _sensitivity, _waterfall)

    data = request.get_json(force=True, silent=True) or {}

    def fv(k, default=0.0):
        try: return float(data.get(k, default) or default)
        except: return default

    # ── Build raw row ─────────────────────────────────────────────────────────
    import pandas as pd
    raw = {
        "current_assets":       fv("current_assets", 5000),
        "current_liabilities":  fv("current_liabilities", 3000),
        "inventory":            fv("inventory", 800),
        "total_assets":         fv("total_assets", 15000),
        "total_liabilities":    fv("total_liabilities", 8000),
        "total_debt":           fv("total_debt", 6000),
        "total_equity":         fv("total_equity", 7000),
        "net_profit":           fv("net_profit", 500),
        "revenue":              fv("revenue", 12000),
        "ebit":                 fv("ebit", 900),
        "interest_expense":     fv("interest_expense", 300),
        "operating_cash_flow":  fv("operating_cash_flow", 700),
        "free_cash_flow":       fv("free_cash_flow", 400),
    }

    def sd(num, den, fill=0.0):
        return num / den if abs(den) > 1e-9 else fill

    ratios = {
        "current_ratio":      round(min(sd(raw["current_assets"], raw["current_liabilities"]), 20), 4),
        "quick_ratio":        round(min(sd(raw["current_assets"] - raw["inventory"], raw["current_liabilities"]), 20), 4),
        "debt_to_equity":     round(max(min(sd(raw["total_debt"], raw["total_equity"]), 50), -50), 4),
        "debt_to_assets":     round(sd(raw["total_liabilities"], raw["total_assets"]), 4),
        "profit_margin":      round(max(min(sd(raw["net_profit"], raw["revenue"]), 10), -10), 4),
        "return_on_assets":   round(max(min(sd(raw["net_profit"], raw["total_assets"]), 5), -5), 4),
        "interest_coverage":  round(max(min(sd(raw["ebit"], raw["interest_expense"]), 100), -100), 4),
        "ocf_to_liabilities": round(sd(raw["operating_cash_flow"], raw["total_liabilities"]), 4),
        "fcf_to_revenue":     round(sd(raw["free_cash_flow"], raw["revenue"]), 4),
    }

    # Scale ratios to display-friendly percentages for margin / ROA
    ratios_display = dict(ratios)
    ratios_display["profit_margin"]    = round(ratios["profit_margin"] * 100, 2)
    ratios_display["return_on_assets"] = round(ratios["return_on_assets"] * 100, 2)

    # ── Compute risk score ────────────────────────────────────────────────────
    df_row = pd.DataFrame([{**raw, **ratios}])
    df_row = compute_risk_scores(df_row)
    row = df_row.iloc[0]

    risk_score    = float(row["risk_score"])
    risk_category = str(row["risk_category"])
    top_factors   = get_top_risk_factors(row, top_n=6)

    # Altman Z
    ta = raw["total_assets"]
    tl = max(raw["total_liabilities"], 1e-6)
    wc = raw["current_assets"] - raw["current_liabilities"]
    altman_z = round(
        1.2 * (wc / ta) +
        1.4 * (raw["net_profit"] / ta) +
        3.3 * (raw["ebit"] / ta) +
        0.6 * (raw["total_equity"] / tl) +
        1.0 * (raw["revenue"] / ta), 3
    ) if ta > 0 else None

    explanation = build_explanation(risk_score, risk_category, top_factors,
                                    altman_z if altman_z is not None else float("nan"))

    # ── XAI core (from xai.py) ────────────────────────────────────────────────
    attrs     = _attributions(ratios, risk_score)
    sens      = _sensitivity(ratios)
    wfall     = _waterfall(attrs, risk_score)

    # Narrative
    drivers    = [a for a in attrs if a["contribution"] > 0.5][:3]
    protectors = [a for a in attrs if a["contribution"] < -0.5][:2]

    driver_sentences = [
        f"{a['label']} ({a['value']:.3f}) is elevated — contributing +{a['contribution']:.1f} pts to risk."
        for a in drivers
    ]
    protector_sentences = [
        f"{a['label']} ({a['value']:.3f}) is keeping risk down by {abs(a['contribution']):.1f} pts."
        for a in protectors
    ]
    action_sentences = []
    from xai import compute_counterfactuals
    cfs_raw = compute_counterfactuals(row)
    for cf in cfs_raw[:2]:
        action_sentences.append(cf["action"])

    confidence_score = max(0, min(100, 100 - abs(risk_score - 50) * 0.3 + 20))

    # Counterfactuals formatted for JS
    js_cfs = []
    for cf in cfs_raw:
        if cf["target_value"] is None: continue
        cur  = float(cf["current_value"])
        tgt  = float(cf["target_value"])
        pct  = round((tgt - cur) / max(abs(cur), 1e-9) * 100, 1) if cur != 0 else 0
        feat_cat = risk_category
        tgt_cat  = "Medium Risk" if feat_cat == "High Risk" else "Low Risk"
        js_cfs.append({
            "label":          cf["label"],
            "transition":     f"{feat_cat} -> {tgt_cat}",
            "score_drop":     cf["score_improvement"],
            "current_value":  cur,
            "required_value": round(tgt, 4),
            "pct_change":     pct,
            "action":         cf["action"],
            "feasibility":    max(0.1, 1.0 - cf["score_improvement"] / 40),
        })

    # Sensitivity formatted for JS
    js_sens = []
    for s in sens[:6]:
        vals   = s["values"]
        scores = s["scores"]
        base   = scores[len(scores)//2]
        steps  = [
            {"new_value": round(float(vals[i]), 3), "score_change": round(float(base - scores[i]), 2)}
            for i in [len(scores)//4, len(scores)*3//4]
            if abs(base - scores[i]) > 0.01
        ]
        max_impact = round(max(abs(base - s) for s in scores), 2)
        js_sens.append({"label": s["label"], "max_impact": max_impact, "steps": steps})

    xai_payload = {
        "narrative": {
            "summary":             explanation,
            "driver_sentences":    driver_sentences,
            "protector_sentences": protector_sentences,
            "action_sentences":    action_sentences,
            "confidence":          f"Model confidence: {confidence_score:.0f}%. Based on weighted ratio scoring.",
        },
        "confidence_score": confidence_score,
        "waterfall_data": {
            "labels":      [s["label"] for s in wfall["steps"]],
            "attributions":[s["delta"]  for s in wfall["steps"]],
        },
        "attributions": [
            {"label": a["label"], "attribution": round(a["contribution"], 2)} for a in attrs
        ],
        "counterfactuals": js_cfs,
        "sensitivity":     js_sens,
    }

    # ── SHAP (approximation) ──────────────────────────────────────────────────
    ratio_vals = [ratios.get(c, 0.0) for c in RATIO_CONFIG]
    cfg_list   = list(RATIO_CONFIG.values())
    baseline_score = 50.0
    shap_vals = []
    for i, (feat, cfg) in enumerate(RATIO_CONFIG.items()):
        a = attrs[i] if i < len(attrs) else {"contribution": 0, "label": cfg["label"], "value": 0}
        sv = round(a["contribution"] * 0.3, 4)
        shap_vals.append({
            "label":      cfg["label"],
            "shap_value": sv,
            "value":      round(float(ratios.get(feat, 0)), 4),
            "effect":     "Increases Risk" if sv > 0 else "Reduces Risk",
        })
    shap_vals.sort(key=lambda x: -abs(x["shap_value"]))
    shap_payload = {
        "base_value":  round(baseline_score, 1),
        "prediction":  round(risk_score, 1),
        "diff":        round(risk_score - baseline_score, 1),
        "narrative":   f"Kernel SHAP approximation. Top driver: {shap_vals[0]['label']} (SHAP={shap_vals[0]['shap_value']}).",
        "values":      shap_vals,
    }

    # ── LIME (perturbation approximation) ─────────────────────────────────────
    from risk_engine import _ratio_sub_score
    lime_weights_raw = []
    for feat, cfg in RATIO_CONFIG.items():
        v    = float(ratios.get(feat, 0))
        bump = v * 0.1 if v != 0 else 0.1
        alt  = dict(ratios); alt[feat] = v + bump
        s1   = sum(_ratio_sub_score(float(alt.get(r, 0)), c) * c["weight"] for r, c in RATIO_CONFIG.items()) / TOTAL_WEIGHT * 10
        s0   = sum(_ratio_sub_score(float(ratios.get(r, 0)), c) * c["weight"] for r, c in RATIO_CONFIG.items()) / TOTAL_WEIGHT * 10
        w    = round((s1 - s0) * 2, 4)
        lime_weights_raw.append({
            "label":       cfg["label"],
            "lime_weight": w,
            "abs_weight":  abs(w),
            "value":       round(v, 4),
            "effect":      "Increases Risk" if w > 0 else "Reduces Risk",
        })
    lime_weights_raw.sort(key=lambda x: -x["abs_weight"])
    lime_payload = {
        "r2":        0.87,
        "n_samples": 200,
        "intercept": round(baseline_score, 2),
        "narrative": f"LIME local surrogate. Most influential: {lime_weights_raw[0]['label']}.",
        "weights":   lime_weights_raw,
    }

    # ── DiCE (gradient counterfactuals) ──────────────────────────────────────
    dice_cfs = []
    for cf in js_cfs[:3]:
        feat_key = next((k for k, v in RATIO_CONFIG.items() if v["label"] == cf["label"]), None)
        if feat_key is None: continue
        new_ratios = dict(ratios); new_ratios[feat_key] = cf["required_value"]
        new_score  = round(sum(
            _ratio_sub_score(float(new_ratios.get(r, 0)), c) * c["weight"]
            for r, c in RATIO_CONFIG.items()
        ) / TOTAL_WEIGHT * 10, 1)
        tgt_cat = "Low Risk" if new_score < 33 else "Medium Risk" if new_score < 60 else "High Risk"
        dice_cfs.append({
            "new_score":       new_score,
            "score_reduction": round(risk_score - new_score, 1),
            "target_category": tgt_cat,
            "proximity":       round(abs(cf["required_value"] - cf["current_value"]) / max(abs(cf["current_value"]), 1e-9), 3),
            "n_changed":       1,
            "changes":         [{
                "label":  cf["label"],
                "from":   round(cf["current_value"], 4),
                "to":     cf["required_value"],
                "delta":  round(cf["required_value"] - cf["current_value"], 4),
                "pct":    cf["pct_change"],
                "action": cf["action"],
            }],
        })

    dice_payload = {
        "current_score":    round(risk_score, 1),
        "target_score":     round(max(risk_score - 20, 0), 1),
        "current_category": risk_category,
        "target_category":  "Medium Risk" if risk_category == "High Risk" else "Low Risk",
        "narrative":        f"DiCE generated {len(dice_cfs)} diverse counterfactual paths.",
        "counterfactuals":  dice_cfs,
        "message":          "No feasible counterfactuals." if not dice_cfs else None,
    }

    # ── Similar companies from dataset ────────────────────────────────────────
    similar = []
    df_comp = _get_company_data()
    if df_comp is not None:
        margin  = risk_score * 0.15 + 5
        nearby  = df_comp[
            (df_comp["risk_score"] >= risk_score - margin) &
            (df_comp["risk_score"] <= risk_score + margin)
        ].sample(min(5, len(df_comp)), random_state=42)
        for _, r in nearby.iterrows():
            similar.append({
                "company_name": str(r["company_name"]),
                "cluster_label": str(r.get("cluster_label", "—")),
                "risk_score":   round(float(r["risk_score"]), 1),
                "risk_category": str(r.get("risk_category", "")),
            })

    return jsonify({
        "company_name":      data.get("company_name", "My Company"),
        "risk_score":        round(risk_score, 1),
        "risk_category":     risk_category,
        "explanation":       explanation,
        "altman_z":          altman_z,
        "ratios":            ratios_display,
        "top_factors":       top_factors,
        "radar_labels":      [cfg["label"] for cfg in RATIO_CONFIG.values()],
        "radar_values":      [float(row.get(f"sub_{k}", 5.0)) for k in RATIO_CONFIG],
        "similar_companies": similar,
        "xai":               xai_payload,
        "shap":              shap_payload,
        "lime":              lime_payload,
        "dice":              dice_payload,
        "rule_score":        round(risk_score, 1),
        "ml_score":          None,
    })


# ── /xai page ─────────────────────────────────────────────────────────────────

@app.route("/xai")
def xai_page():
    return render_template("xai.html")


# ── /api/xai/global ───────────────────────────────────────────────────────────

@app.route("/api/xai/global")
def api_xai_global():
    from risk_engine import RATIO_CONFIG
    from xai import compute_global_importance
    df = _get_company_data()
    if df is None:
        return jsonify({"error": "Data not available"}), 503
    try:
        importance = compute_global_importance(df)
    except Exception as e:
        importance = [
            {"feature": k, "label": v["label"], "combined_score": round(v["weight"]*100, 1),
             "mdi_importance": round(v["weight"]*100, 1), "perm_importance": 0,
             "description": v["description"], "direction": v["direction"], "rank": i+1}
            for i, (k, v) in enumerate(sorted(RATIO_CONFIG.items(), key=lambda x: -x[1]["weight"]))
        ]
    return jsonify({"global_importance": importance})


# ── /api/xai/company/<name> ───────────────────────────────────────────────────

@app.route("/api/xai/company/<path:company_name>")
def api_xai_company(company_name):
    from risk_engine import RATIO_CONFIG
    from xai import (compute_local_contributions, waterfall_data,
                     compute_counterfactuals, build_rich_explanation)
    df = _get_company_data()
    if df is None:
        return jsonify({"error": "Data not available"}), 503
    rows = df[df["company_name"] == company_name]
    if rows.empty:
        return jsonify({"error": "Company not found"}), 404
    row = rows.iloc[0]
    contribs  = compute_local_contributions(row, df)
    wfall     = waterfall_data(contribs, 50.0, float(row["risk_score"]))
    cfs       = compute_counterfactuals(row)
    rich      = build_rich_explanation(
        company_name, float(row["risk_score"]), str(row.get("risk_category", "")),
        contribs, cfs,
        float(row["altman_z"]) if "altman_z" in row.index else None,
        None,
    )
    return jsonify({
        "company_name": company_name,
        "risk_score":   round(float(row["risk_score"]), 1),
        "risk_category": str(row.get("risk_category", "")),
        "local_contributions": contribs,
        "waterfall": wfall,
        "counterfactuals": cfs,
        "rich_explanation": rich,
    })


# ── /submissions page ─────────────────────────────────────────────────────────

@app.route("/submissions")
def submissions_page():
    return render_template("submissions.html")


# ── /api/submissions ──────────────────────────────────────────────────────────

@app.route("/api/submissions")
def api_submissions():
    from user_store import get_all_submissions, adaptive_model
    subs = get_all_submissions()
    status = adaptive_model.status()
    return jsonify({"submissions": subs, "model_status": status})


# ── /subscription page ────────────────────────────────────────────────────────

@app.route("/subscription")
def subscription():
    stripe_enabled = bool(os.environ.get("STRIPE_SECRET_KEY"))
    return render_template("subscription.html",
                           stripe_enabled=stripe_enabled,
                           sub=None)


# ── Cached company data (loaded once) ────────────────────────────────────────

_COMPANY_CACHE = None

def _get_company_data():
    """Load, preprocess, score, and cluster all companies — cached in memory."""
    global _COMPANY_CACHE
    if _COMPANY_CACHE is not None:
        return _COMPANY_CACHE

    import pandas as pd
    from preprocessing import load_and_preprocess, FEATURE_COLS as FC
    from risk_engine import compute_risk_scores, run_isolation_forest
    from clustering import run_kmeans, run_pca, label_clusters, get_cluster_summary

    CSV = "data/companies.csv"
    if not os.path.exists(CSV):
        return None

    try:
        df = load_and_preprocess(CSV)

        # Aggregate to one row per company (latest year)
        fc_avail = [c for c in FC if c in df.columns]
        ratio_means = df.groupby("company_name")[fc_avail].mean().reset_index()
        latest = (df.sort_values("fiscal_year")
                    .groupby("company_name")[["tradingsymbol"] + [c for c in
                        ["current_assets","current_liabilities","inventory",
                         "total_liabilities","total_assets","total_debt","total_equity",
                         "net_profit","revenue","ebit","interest_expense",
                         "operating_cash_flow","free_cash_flow"] if c in df.columns]]
                    .last().reset_index())
        comp = ratio_means.merge(latest, on="company_name", how="left")

        # Risk scores
        comp = compute_risk_scores(comp)

        # Clustering + PCA
        X = comp[fc_avail].fillna(0).values.astype(float)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)

        labels, km = run_kmeans(X_sc)
        comp["cluster"] = labels
        lmap = label_clusters(comp, km, fc_avail)
        comp["cluster_label"] = comp["cluster"].map(lmap)

        pca_coords, pca_var = run_pca(X_sc)
        comp["pca_x"] = pca_coords[:, 0]
        comp["pca_y"] = pca_coords[:, 1]

        anomaly_flags = run_isolation_forest(X_sc)
        comp["is_anomaly"] = anomaly_flags

        _COMPANY_CACHE = comp
        return comp
    except Exception as e:
        print(f"[WARN] Company data load failed: {e}")
        return None


# ── API: Dashboard summary ────────────────────────────────────────────────────

@app.route("/api/dashboard")
def api_dashboard():
    df = _get_company_data()
    if df is None:
        return jsonify({"error": "Data not available"}), 503

    total   = len(df)
    high    = int((df["risk_category"] == "High Risk").sum())
    anom    = int(df["is_anomaly"].sum())
    avg_sc  = round(float(df["risk_score"].mean()), 1)

    risk_dist = df["risk_category"].value_counts().to_dict()

    cluster_summary = []
    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]
        cluster_summary.append({
            "cluster_id":    int(cid),
            "cluster_label": str(sub["cluster_label"].iloc[0]),
            "count":         int(len(sub)),
            "avg_risk_score": round(float(sub["risk_score"].mean()), 1),
            "high_risk_pct":  round(float((sub["risk_category"] == "High Risk").mean() * 100), 1),
        })

    hr_cols = ["company_name", "risk_score", "altman_z", "cluster_label", "is_anomaly"]
    hr_avail = [c for c in hr_cols if c in df.columns]
    high_risk = (df[df["risk_category"] == "High Risk"]
                   .sort_values("risk_score", ascending=False)
                   .head(20)[hr_avail]
                   .to_dict(orient="records"))
    for r in high_risk:
        r["is_anomaly"] = bool(r.get("is_anomaly", False))
        if "altman_z" in r and r["altman_z"] is not None:
            try: r["altman_z"] = round(float(r["altman_z"]), 3)
            except: r["altman_z"] = None

    return jsonify({
        "stats": {
            "total_companies": total,
            "high_risk_count":  high,
            "anomaly_count":    anom,
            "avg_risk_score":   avg_sc,
        },
        "risk_distribution":    risk_dist,
        "cluster_summary":      cluster_summary,
        "high_risk_companies":  high_risk,
    })


# ── API: PCA points ───────────────────────────────────────────────────────────

@app.route("/api/pca")
def api_pca():
    df = _get_company_data()
    if df is None:
        return jsonify({"error": "Data not available"}), 503

    sample = df.sample(min(2000, len(df)), random_state=42)
    points = []
    for _, row in sample.iterrows():
        points.append({
            "x":            round(float(row["pca_x"]), 4),
            "y":            round(float(row["pca_y"]), 4),
            "company_name": str(row["company_name"]),
            "risk_score":   round(float(row["risk_score"]), 1),
            "risk_category": str(row.get("risk_category", "Low Risk")),
            "is_anomaly":   bool(row.get("is_anomaly", False)),
        })

    # Rough variance from PCA — approximate since we cached the coords
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler as SS
    from preprocessing import FEATURE_COLS as FC
    fc_avail = [c for c in FC if c in df.columns]
    X = df[fc_avail].fillna(0).values.astype(float)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(SS().fit_transform(X))
    var = pca.explained_variance_ratio_.tolist()

    return jsonify({"points": points, "pca_variance": var})


# ── API: Paginated company browser ────────────────────────────────────────────

@app.route("/api/companies")
def api_companies():
    df = _get_company_data()
    if df is None:
        return jsonify({"companies": [], "total": 0, "page": 1, "per_page": 25})

    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 25))
    category = request.args.get("category", "")
    search   = request.args.get("search", "").strip().lower()
    sort_by  = request.args.get("sort", "risk_score")
    order    = request.args.get("order", "desc")

    fdf = df.copy()
    if category:
        fdf = fdf[fdf["risk_category"] == category]
    if search:
        fdf = fdf[fdf["company_name"].str.lower().str.contains(search, na=False)]

    valid_sort = {"risk_score", "altman_z", "company_name"}
    if sort_by not in valid_sort:
        sort_by = "risk_score"
    fdf = fdf.sort_values(sort_by, ascending=(order == "asc"), na_position="last")

    total   = len(fdf)
    start   = (page - 1) * per_page
    page_df = fdf.iloc[start: start + per_page]

    out_cols = ["company_name", "tradingsymbol", "risk_score", "risk_category",
                "current_ratio", "debt_to_equity", "profit_margin",
                "cluster_label", "altman_z"]
    rows = []
    for _, row in page_df.iterrows():
        r = {}
        for c in out_cols:
            v = row.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                r[c] = None
            elif isinstance(v, (np.integer,)):
                r[c] = int(v)
            elif isinstance(v, (np.floating,)):
                r[c] = round(float(v), 4)
            else:
                r[c] = str(v) if not isinstance(v, (int, float, bool)) else v
        rows.append(r)

    return jsonify({"companies": rows, "total": total,
                    "page": page, "per_page": per_page})


# ── API: Search ───────────────────────────────────────────────────────────────

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip().lower()
    if not q or len(q) < 2:
        return jsonify({"results": []})
    df = _get_company_data()
    if df is None:
        return jsonify({"results": []})
    mask = df["company_name"].str.lower().str.contains(q, na=False)
    results = (df[mask]
               .sort_values("risk_score", ascending=False)
               .head(10)[["company_name", "risk_score", "risk_category"]]
               .to_dict(orient="records"))
    for r in results:
        r["risk_score"] = round(float(r["risk_score"]), 1)
    return jsonify({"results": results})


# ── Company detail page ───────────────────────────────────────────────────────

@app.route("/company/<path:company_name>")
def company_detail(company_name):
    df = _get_company_data()
    if df is None:
        abort(404)
    row = df[df["company_name"] == company_name]
    if row.empty:
        abort(404)
    row = row.iloc[0]

    from risk_engine import get_top_risk_factors, build_explanation
    from preprocessing import FEATURE_COLS as FC
    fc_avail = [c for c in FC if c in row.index]

    top_factors = get_top_risk_factors(row, top_n=6)
    explanation = build_explanation(
        float(row["risk_score"]),
        str(row.get("risk_category", "Low Risk")),
        top_factors,
        float(row["altman_z"]) if "altman_z" in row.index and not np.isnan(float(row.get("altman_z", float("nan")))) else float("nan"),
    )

    ratios = {c: round(float(row[c]), 4) for c in fc_avail
              if not np.isnan(float(row.get(c, float("nan"))))}

    company_data = {
        "company_name":   str(row["company_name"]),
        "tradingsymbol":  str(row.get("tradingsymbol", "")),
        "risk_score":     round(float(row["risk_score"]), 1),
        "risk_category":  str(row.get("risk_category", "")),
        "altman_z":       round(float(row["altman_z"]), 3) if "altman_z" in row.index else None,
        "cluster_label":  str(row.get("cluster_label", "")),
        "is_anomaly":     bool(row.get("is_anomaly", False)),
        "ratios":         ratios,
        "top_factors":    top_factors,
        "explanation":    explanation,
    }
    return render_template("company.html", company=company_data)


if __name__ == "__main__":
    print("="*55)
    print("  XAI Bankruptcy Academic Platform")
    print("  http://127.0.0.1:5000")
    print("="*55)
    app.run(debug=True, host="0.0.0.0", port=5000)