/**
 * valuation.js
 * ============
 * Front-end logic for the Intrinsic Value Analyser dashboard.
 * Talks to Flask routes:
 *   GET  /valuation/search_company
 *   GET  /valuation/fetch_market_data
 *   POST /valuation/run_dcf
 *   POST /valuation/reverse_dcf
 *   POST /valuation/scenario_analysis
 */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
let _currentSymbol   = "";
let _currentExchange = "NSE";
let _marketData      = null;
let _dcfResult       = null;
let _charts          = {};

// ── Chart.js global defaults ──────────────────────────────────────────────────
Chart.defaults.color           = "#8892a4";
Chart.defaults.borderColor     = "#1e2d3d";
Chart.defaults.font.family     = "'IBM Plex Mono', monospace";
Chart.defaults.plugins.legend.labels.boxWidth = 12;

// ── Autocomplete ──────────────────────────────────────────────────────────────
const symbolInput    = document.getElementById("symbolInput");
const autocompleteList = document.getElementById("autocompleteList");
let _acIndex = -1;
let _acTimer;

symbolInput.addEventListener("input", () => {
  clearTimeout(_acTimer);
  const q = symbolInput.value.trim();
  if (q.length < 1) { hideAC(); return; }
  _acTimer = setTimeout(() => fetchAC(q), 200);
});

symbolInput.addEventListener("keydown", e => {
  const items = autocompleteList.querySelectorAll("li");
  if (e.key === "ArrowDown") { _acIndex = Math.min(_acIndex + 1, items.length - 1); highlightAC(items); e.preventDefault(); }
  else if (e.key === "ArrowUp") { _acIndex = Math.max(_acIndex - 1, 0); highlightAC(items); e.preventDefault(); }
  else if (e.key === "Enter") {
    if (_acIndex >= 0 && items[_acIndex]) items[_acIndex].click();
    else runFullAnalysis();
  }
  else if (e.key === "Escape") hideAC();
});

document.addEventListener("click", e => {
  if (!symbolInput.contains(e.target)) hideAC();
});

async function fetchAC(q) {
  try {
    const r = await fetch(`/valuation/search_company?q=${encodeURIComponent(q)}`);
    const d = await r.json();
    renderAC(d.results || []);
  } catch { hideAC(); }
}

function renderAC(results) {
  if (!results.length) { hideAC(); return; }
  _acIndex = -1;
  autocompleteList.innerHTML = results.map(c => `
    <li data-sym="${c.symbol}" data-exch="${c.exchange || 'NSE'}">
      <span class="ac-sym">${c.symbol}</span>
      <span class="ac-name">${c.company_name}</span>
      ${c.sector ? `<span class="ac-sector" style="margin-left:auto;font-size:.7rem;opacity:.5">${c.sector}</span>` : ''}
    </li>`).join("");
  autocompleteList.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => {
      symbolInput.value  = li.dataset.sym;
      _currentSymbol     = li.dataset.sym;
      _currentExchange   = li.dataset.exch || "NSE";
      document.getElementById("exchangeSelect").value = _currentExchange;
      hideAC();
      runFullAnalysis();
    });
  });
  autocompleteList.classList.remove("d-none");
}

function highlightAC(items) {
  items.forEach((li, i) => li.classList.toggle("active", i === _acIndex));
}
function hideAC() {
  autocompleteList.classList.add("d-none");
  _acIndex = -1;
}

// ── Main analysis runner ──────────────────────────────────────────────────────
async function runFullAnalysis(useOverrides = false) {
  const symbol   = symbolInput.value.trim().toUpperCase();
  const exchange = document.getElementById("exchangeSelect").value;
  const wacc     = parseFloat(document.getElementById("waccInput").value) / 100 || 0.10;

  if (!symbol) { showError("Please enter a company symbol or name."); return; }
  _currentSymbol   = symbol;
  _currentExchange = exchange;

  showLoading(true);
  hideError();
  document.getElementById("resultsSection").classList.add("d-none");

  try {
    // 1. Fetch market + financial data
    const mktResp = await fetch(
      `/valuation/fetch_market_data?symbol=${encodeURIComponent(symbol)}&exchange=${exchange}`
    );
    const mkt = await mktResp.json();
    if (mkt.error) { showError(mkt.error); return; }
    _marketData = mkt;

    updateCompanyBar(mkt);

    // Show notice if using default assumptions (no detailed financials)
    if (mkt.is_stub) {
      const bar = document.getElementById("companyBar");
      const existing = bar.querySelector(".stub-notice");
      if (!existing) {
        bar.insertAdjacentHTML("beforeend",
          `<div class="stub-notice" style="width:100%;margin-top:6px;padding:6px 10px;border-radius:6px;background:rgba(59,130,246,.12);color:#93c5fd;font-size:.78rem;">
            <i class="fas fa-info-circle me-1"></i>
            Using sector-default assumptions — override in the Assumptions panel below for accurate valuation.
          </div>`);
      }
    }

    // Build DCF payload (merge dataset defaults with user overrides if any)
    const payload = buildPayload(mkt, wacc, useOverrides);

    // 2. Run DCF in parallel with scenarios and reverse DCF
    const [dcfResp, scResp, rdResp] = await Promise.all([
      postJSON("/valuation/run_dcf",          payload),
      postJSON("/valuation/scenario_analysis", payload),
      postJSON("/valuation/reverse_dcf",       payload),
    ]);

    if (dcfResp.status !== "ok") { showError(dcfResp.message || "DCF failed."); return; }

    _dcfResult = dcfResp;

    renderResults(dcfResp, scResp, rdResp);
    populateAssumptions(payload);

  } catch (err) {
    showError("Network error: " + err.message);
  } finally {
    showLoading(false);
  }
}

function buildPayload(mkt, wacc, useOverrides) {
  const fin    = mkt.financials || {};
  const base   = {
    symbol:               _currentSymbol,
    exchange:             _currentExchange,
    net_income:           fin.net_income           || 0,
    depreciation:         fin.depreciation          || 0,
    amortization:         fin.amortization          || 0,
    capex:                fin.capex                 || 0,
    working_capital_change: fin.working_capital_change || 0,
    revenue_growth_rate:  fin.revenue_growth_rate   || 0.10,
    operating_margin:     fin.operating_margin      || 0.15,
    tax_rate:             0.25,
    reinvestment_rate:    0.30,
    wacc:                 wacc,
    terminal_growth_rate: Math.max(wacc - 0.05, 0.03),
    forecast_years:       10,
    current_price:        mkt.ltp                   || 0,
    shares_outstanding:   mkt.shares_outstanding    || 1.0,
    net_debt:             fin.net_debt              || 0,
  };

  if (useOverrides) {
    const overrideFields = ["revenue_growth_rate","operating_margin","wacc",
                             "terminal_growth_rate","reinvestment_rate","tax_rate"];
    overrideFields.forEach(f => {
      const el = document.getElementById("ov_" + f);
      if (el) base[f] = parseFloat(el.value) / 100 || base[f];
    });
  }
  return base;
}

// ── Render all results ────────────────────────────────────────────────────────
function renderResults(dcf, sc, rd) {
  renderKPIs(dcf);
  renderFCFChart(dcf);
  renderHistoryChart(_marketData);
  renderDriversChart(dcf);
  renderXAI(dcf);
  renderScenarios(sc);
  renderReverseDCF(rd);
  renderWACCTable(dcf);
  document.getElementById("resultsSection").classList.remove("d-none");

  // Fire AI explanation request asynchronously (doesn't block results)
  fetchAIExplanation(dcf);
}

// ── KPI Cards ─────────────────────────────────────────────────────────────────
function renderKPIs(dcf) {
  const iv    = dcf.intrinsic_value_per_share;
  const price = dcf.inputs?.current_price || _marketData?.ltp || 0;
  const mos   = dcf.margin_of_safety;
  const label = dcf.valuation_label;

  setText("kpiIV",    "₹" + fmt(iv));
  setText("kpiPrice", "₹" + fmt(price));
  setText("kpiMoS",   mos > 0 ? "+" + fmt(mos) + "%" : fmt(mos) + "%");

  const vEl = document.getElementById("kpiVerdict");
  vEl.textContent = label;
  vEl.className   = "val-kpi-value " + verdictClass(label);

  const mosEl = document.getElementById("kpiMoS");
  mosEl.className = "val-kpi-value " + (mos > 0 ? "verdict-undervalued" : mos < -5 ? "verdict-overvalued" : "verdict-fair");
}

// ── FCF Chart ─────────────────────────────────────────────────────────────────
function renderFCFChart(dcf) {
  const proj = dcf.projections || [];
  destroyChart("fcfChart");
  _charts.fcfChart = new Chart(document.getElementById("fcfChart"), {
    type: "bar",
    data: {
      labels: proj.map(p => `Y${p.year}`),
      datasets: [
        {
          label: "FCF (₹ Cr)",
          data:  proj.map(p => p.fcf),
          backgroundColor: "rgba(59,130,246,.55)",
          borderColor: "#3b82f6",
          borderWidth: 1,
          borderRadius: 4,
        },
        {
          label: "PV of FCF (₹ Cr)",
          data:  proj.map(p => p.pv_fcf),
          backgroundColor: "rgba(99,102,241,.35)",
          borderColor: "#6366f1",
          borderWidth: 1,
          borderRadius: 4,
        }
      ]
    },
    options: { ...baseChartOpts(), plugins: { ...baseChartOpts().plugins } }
  });
}

// ── History Chart ─────────────────────────────────────────────────────────────
function renderHistoryChart(mkt) {
  const hist = (mkt?.history || []).slice().reverse();
  if (!hist.length) return;
  destroyChart("historyChart");
  _charts.historyChart = new Chart(document.getElementById("historyChart"), {
    type: "line",
    data: {
      labels: hist.map(h => h.year),
      datasets: [
        {
          label: "Revenue (₹ Cr)",
          data: hist.map(h => h.revenue),
          borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,.1)",
          tension: .35, fill: true, pointRadius: 3,
        },
        {
          label: "Net Income (₹ Cr)",
          data: hist.map(h => h.net_income),
          borderColor: "#34d399", backgroundColor: "rgba(52,211,153,.08)",
          tension: .35, fill: true, pointRadius: 3,
        }
      ]
    },
    options: baseChartOpts()
  });
}

// ── Drivers doughnut ─────────────────────────────────────────────────────────
function renderDriversChart(dcf) {
  const drvs = dcf.xai?.driver_summary || [];
  if (!drvs.length) return;
  destroyChart("driversChart");
  _charts.driversChart = new Chart(document.getElementById("driversChart"), {
    type: "doughnut",
    data: {
      labels: drvs.map(d => `${d.driver} (${d.value}%)`),
      datasets: [{
        data: drvs.map(d => d.weight),
        backgroundColor: ["#3b82f6","#6366f1","#8b5cf6","#f59e0b","#10b981"],
        borderWidth: 2,
        borderColor: "#0e1523",
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "right", labels: { font: { size: 11 } } } },
      cutout: "62%",
    }
  });
}

// ── XAI panel (rule-based + sensitivity) ─────────────────────────────────────
function renderXAI(dcf) {
  const xai = dcf.xai || {};
  setText("xaiExplanation", xai.explanation || "No explanation available.");

  const sensitivityEl = document.getElementById("sensitivityNarrative");
  const items = xai.sensitivity || [];
  sensitivityEl.innerHTML = items.map(s => `
    <div class="val-sensitivity-item">
      <strong>${s.label}</strong> ${s.description} →
      <span class="${s.change < 0 ? 'arrow-down' : 'arrow-up'}">
        ${s.change < 0 ? '▼' : '▲'} ₹${Math.abs(s.change).toLocaleString('en-IN', {maximumFractionDigits:2})}
      </span>
      <span class="text-muted"> (₹${fmt(s.base_iv)} → ₹${fmt(s.new_iv)})</span>
    </div>`).join("");

  // Warnings
  const warns = dcf.warnings || [];
  if (warns.length) {
    sensitivityEl.insertAdjacentHTML("afterend",
      warns.map(w => `<div class="val-warning"><i class="fas fa-exclamation-triangle me-1"></i>${w}</div>`).join(""));
  }
}

// ── Mistral AI Explanation (async) ───────────────────────────────────────────
async function fetchAIExplanation(dcf) {
  const aiPanel = document.getElementById("aiExplanationPanel");
  if (!aiPanel) return;

  const contentEl = document.getElementById("aiExplanationContent");
  const metaEl    = document.getElementById("aiExplanationMeta");

  // Show loading state
  aiPanel.classList.remove("d-none");
  contentEl.innerHTML = `
    <div class="text-center py-3">
      <div class="spinner-border spinner-border-sm text-info" role="status"></div>
      <span class="ms-2 text-muted small">Generating institutional-grade valuation analysis...</span>
    </div>`;
  metaEl.innerHTML = "";

  // Build payload from DCF result + market data
  const xai = dcf.xai || {};
  const payload = {
    company_name:         _marketData?.company_name || dcf.inputs?.symbol || "Unknown",
    symbol:               _currentSymbol || "",
    sector:               _marketData?.sector || "",
    current_price:        xai.current_price || dcf.inputs?.current_price || _marketData?.ltp || 0,
    intrinsic_value:      xai.intrinsic_value || dcf.intrinsic_value_per_share || 0,
    margin_of_safety:     xai.margin_of_safety || (dcf.margin_of_safety * 100) || 0,
    valuation_label:      xai.valuation_label || dcf.valuation_label || "",
    wacc:                 ((dcf.inputs?.wacc || 0.10) * 100),
    revenue_growth_rate:  ((dcf.inputs?.revenue_growth_rate || 0.10) * 100),
    terminal_growth_rate: ((dcf.inputs?.terminal_growth_rate || 0.05) * 100),
    operating_margin:     ((dcf.inputs?.operating_margin || 0.15) * 100),
    free_cash_flow:       dcf.inputs?.net_income || 0,
    net_debt:             dcf.inputs?.net_debt || 0,
    sensitivity:          xai.sensitivity || [],
    warnings:             dcf.warnings || [],
  };

  try {
    const resp = await postJSON("/valuation/ai_explanation", payload);

    if (resp.status === "ok" && resp.ai_explanation) {
      // Format the explanation: convert newlines to HTML
      const formatted = resp.ai_explanation
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');

      contentEl.innerHTML = `<div class="ai-analysis-text">${formatted}</div>`;

      // Meta info
      const source = resp.ai_source === "mistral" ? "Mistral Small" : "Quantitative Fallback";
      const cached = resp.ai_cached ? " · Cached" : "";
      const latency = resp.ai_latency_ms ? ` · ${resp.ai_latency_ms}ms` : "";
      metaEl.innerHTML = `
        <span class="badge bg-dark text-info" style="font-size:.7rem;font-weight:400">
          <i class="fas fa-robot me-1"></i>${source}${cached}${latency}
        </span>`;
    } else {
      contentEl.innerHTML = `<p class="text-muted small">AI explanation unavailable. Showing quantitative valuation summary instead.</p>`;
    }
  } catch (err) {
    contentEl.innerHTML = `<p class="text-muted small">AI explanation unavailable. Showing quantitative valuation summary instead.</p>`;
  }
}

// ── Scenarios ─────────────────────────────────────────────────────────────────
function renderScenarios(sc) {
  if (!sc || sc.status === "error") return;
  const s = sc.scenarios || {};
  const labels = ["Bear", "Base", "Bull"];
  const colors = ["#ef4444","#3b82f6","#22c55e"];
  const ivs    = [s.bear?.intrinsic_value_per_share, s.base?.intrinsic_value_per_share, s.bull?.intrinsic_value_per_share];
  const price  = sc.current_price || 0;

  destroyChart("scenarioChart");
  _charts.scenarioChart = new Chart(document.getElementById("scenarioChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Intrinsic Value (₹)",
          data: ivs,
          backgroundColor: colors.map(c => c + "99"),
          borderColor: colors,
          borderWidth: 2,
          borderRadius: 6,
        },
        {
          label: "Market Price (₹)",
          data: [price, price, price],
          type: "line",
          borderColor: "#fbbf24",
          borderWidth: 2,
          borderDash: [5,5],
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: baseChartOpts()
  });

  const sum = sc.summary || {};
  document.getElementById("scenarioSummary").innerHTML = `
    <div class="scenario-meta">
      <span class="scenario-pill pill-bear">Bear ₹${fmt(sum.bear_iv)}</span>
      <span class="scenario-pill pill-base">Base ₹${fmt(sum.base_iv)}</span>
      <span class="scenario-pill pill-bull">Bull ₹${fmt(sum.bull_iv)}</span>
    </div>
    <p class="mt-2 text-muted" style="font-size:.8rem">${sc.commentary || ""}</p>
  `;
}

// ── Reverse DCF ───────────────────────────────────────────────────────────────
function renderReverseDCF(rd) {
  if (!rd || rd.status === "error") return;

  document.getElementById("reverseDCFPanel").innerHTML = `
    <div class="val-rdcf-row">
      <span class="val-rdcf-label">Market-Implied Growth</span>
      <span class="val-rdcf-value text-warning">${rd.implied_growth_rate}%</span>
    </div>
    <div class="val-rdcf-row">
      <span class="val-rdcf-label">Your Growth Estimate</span>
      <span class="val-rdcf-value text-info">${rd.user_growth_assumption}%</span>
    </div>
    <div class="val-rdcf-row">
      <span class="val-rdcf-label">Growth Premium</span>
      <span class="val-rdcf-value ${rd.growth_premium > 0 ? 'text-danger' : 'text-success'}">
        ${rd.growth_premium > 0 ? '+' : ''}${rd.growth_premium}%
      </span>
    </div>
    <p class="mt-2 text-muted" style="font-size:.78rem">${rd.interpretation || ""}</p>
  `;

  const tbl = rd.sensitivity_table || [];
  if (!tbl.length) return;
  destroyChart("reverseChart");
  _charts.reverseChart = new Chart(document.getElementById("reverseChart"), {
    type: "line",
    data: {
      labels: tbl.map(r => r.growth_rate + "%"),
      datasets: [{
        label: "Intrinsic Value (₹)",
        data:  tbl.map(r => r.intrinsic_value),
        borderColor: "#6366f1",
        backgroundColor: "rgba(99,102,241,.08)",
        tension: .35,
        fill: true,
        pointRadius: 2,
      }]
    },
    options: baseChartOpts()
  });
}

// ── WACC Sensitivity Table ────────────────────────────────────────────────────
function renderWACCTable(dcf) {
  const rows = dcf.xai?.wacc_sensitivity || [];
  if (!rows.length) return;

  const head = document.getElementById("waccTableHead");
  const body = document.getElementById("waccTableRow");

  head.innerHTML = `<th class="text-start">WACC (%)</th>` +
    rows.map(r => `<th class="${r.is_base ? 'wacc-base' : ''}">${r.wacc}%${r.is_base ? " ★" : ""}</th>`).join("");

  body.innerHTML = `<td class="text-start text-muted">IV / Share</td>` +
    rows.map(r => {
      const cls = r.is_base ? "wacc-base" : (r.vs_base > 0 ? "wacc-good" : "wacc-bad");
      return `<td class="${cls}">₹${fmt(r.intrinsic_value)}</td>`;
    }).join("");
}

// ── Assumption fields ─────────────────────────────────────────────────────────
function populateAssumptions(payload) {
  const fields = [
    { key:"revenue_growth_rate",  label:"Revenue Growth (%)" },
    { key:"operating_margin",     label:"Operating Margin (%)" },
    { key:"wacc",                 label:"WACC (%)" },
    { key:"terminal_growth_rate", label:"Terminal Growth (%)" },
    { key:"reinvestment_rate",    label:"Reinvestment Rate (%)" },
    { key:"tax_rate",             label:"Tax Rate (%)" },
  ];
  const container = document.getElementById("assumptionFields");
  container.innerHTML = fields.map(f => `
    <div class="col-6 col-md-4 col-lg-2">
      <label class="form-label val-assump-label">${f.label}</label>
      <input id="ov_${f.key}" type="number" step="0.1"
             value="${((payload[f.key] || 0) * 100).toFixed(1)}"
             class="form-control form-control-sm val-input">
    </div>`).join("");
}

// ── Company bar ───────────────────────────────────────────────────────────────
function updateCompanyBar(mkt) {
  setText("companyName",   mkt.company_name || mkt.symbol);
  setText("companySector", mkt.sector       || "");
  setText("companyLTP",    "₹" + fmt(mkt.ltp));
  setText("priceSource",   mkt.price_source || "");
  document.getElementById("companyBar").classList.remove("d-none");
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmt(n) {
  if (n == null || isNaN(n)) return "—";
  return Number(n).toLocaleString("en-IN", { maximumFractionDigits: 2, minimumFractionDigits: 2 });
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function verdictClass(label) {
  if (!label) return "";
  const l = label.toLowerCase();
  if (l.includes("under")) return "verdict-undervalued";
  if (l.includes("over"))  return "verdict-overvalued";
  return "verdict-fair";
}

function destroyChart(id) {
  if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
}

function baseChartOpts() {
  return {
    responsive: true,
    interaction: { intersect: false, mode: "index" },
    plugins: {
      legend: { labels: { font: { size: 11 } } },
      tooltip: {
        backgroundColor: "#0e1523",
        borderColor: "#1e2d3d",
        borderWidth: 1,
        titleColor: "#e2e8f0",
        bodyColor:  "#8892a4",
        callbacks: {
          label: ctx => ` ${ctx.dataset.label}: ₹${fmt(ctx.raw)}`
        }
      }
    },
    scales: {
      x: { grid: { color: "#1e2d3d" }, ticks: { color: "#8892a4", font: { size: 10 } } },
      y: { grid: { color: "#1e2d3d" }, ticks: { color: "#8892a4", font: { size: 10 },
           callback: v => "₹" + (v >= 1e5 ? (v/1e5).toFixed(1) + "L" : v >= 1e3 ? (v/1e3).toFixed(0) + "K" : v) }}
    }
  };
}

async function postJSON(url, payload) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return resp.json();
}

function showLoading(on) {
  document.getElementById("loadingBar").classList.toggle("d-none", !on);
  document.getElementById("analyseBtn").disabled = on;
}

function showError(msg) {
  const el = document.getElementById("errorAlert");
  el.textContent = msg;
  el.classList.remove("d-none");
  showLoading(false);
}

function hideError() {
  document.getElementById("errorAlert").classList.add("d-none");
}

// ── Mode switching (Search / Manual) ──────────────────────────────────────────
function switchMode(mode) {
  const searchMode  = document.getElementById("searchMode");
  const manualMode  = document.getElementById("manualMode");
  const tabSearch   = document.getElementById("modeSearch");
  const tabManual   = document.getElementById("modeManual");

  if (mode === "manual") {
    searchMode.classList.add("d-none");
    manualMode.classList.remove("d-none");
    tabSearch.classList.remove("active");
    tabManual.classList.add("active");
  } else {
    searchMode.classList.remove("d-none");
    manualMode.classList.add("d-none");
    tabSearch.classList.add("active");
    tabManual.classList.remove("active");
  }
  // Hide previous results & errors
  hideError();
  document.getElementById("resultsSection").classList.add("d-none");
  document.getElementById("companyBar").classList.add("d-none");
}

// ── Manual analysis runner ────────────────────────────────────────────────────
async function runManualAnalysis() {
  const fv = (id, def = 0) => {
    const el = document.getElementById(id);
    const v = parseFloat(el ? el.value : def);
    return isNaN(v) ? def : v;
  };

  const companyName = (document.getElementById("manualCompanyName")?.value || "Custom Company").trim();
  if (!companyName) { showError("Please enter a company name."); return; }

  const netIncome  = fv("manualNetIncome", 5000);
  const depr       = fv("manualDepr", 1000);
  const amort      = fv("manualAmort", 100);
  const capex      = fv("manualCapex", 2000);
  const wcChange   = fv("manualWCChange", 500);
  const netDebt    = fv("manualNetDebt", 0);
  const price      = fv("manualPrice", 0);
  const shares     = fv("manualShares", 1);
  const wacc       = fv("manualWacc", 10) / 100;
  const growth     = fv("manualGrowth", 10) / 100;
  const margin     = fv("manualMargin", 15) / 100;
  const taxRate    = fv("manualTax", 25) / 100;
  const reinvest   = fv("manualReinvest", 30) / 100;
  const terminal   = fv("manualTerminal", 5) / 100;
  const years      = Math.max(1, Math.min(30, Math.round(fv("manualYears", 10))));

  if (shares <= 0) { showError("Shares outstanding must be greater than zero."); return; }
  if (wacc <= 0 || wacc > 1) { showError("WACC must be between 0.1% and 100%."); return; }
  if (wacc <= terminal) { showError("WACC must be greater than terminal growth rate."); return; }

  showLoading(true);
  hideError();
  document.getElementById("resultsSection").classList.add("d-none");

  const payload = {
    symbol:                "",
    net_income:            netIncome,
    depreciation:          depr,
    amortization:          amort,
    capex:                 capex,
    working_capital_change: wcChange,
    net_debt:              netDebt,
    current_price:         price,
    shares_outstanding:    shares,
    wacc:                  wacc,
    revenue_growth_rate:   growth,
    operating_margin:      margin,
    tax_rate:              taxRate,
    reinvestment_rate:     reinvest,
    terminal_growth_rate:  terminal,
    forecast_years:        years,
  };

  // Show company bar with manual data
  _marketData = {
    company_name: companyName,
    symbol: "MANUAL",
    sector: "Manual Input",
    ltp: price,
    price_source: "User Input",
    shares_outstanding: shares,
    history: [],
  };
  updateCompanyBar(_marketData);

  try {
    const [dcfResp, scResp, rdResp] = await Promise.all([
      postJSON("/valuation/run_dcf",          payload),
      postJSON("/valuation/scenario_analysis", payload),
      postJSON("/valuation/reverse_dcf",       payload),
    ]);

    if (dcfResp.status !== "ok") { showError(dcfResp.message || "DCF valuation failed."); return; }

    _dcfResult = dcfResp;
    renderResults(dcfResp, scResp, rdResp);
    populateAssumptions(payload);
  } catch (err) {
    showError("Network error: " + err.message);
  } finally {
    showLoading(false);
  }
}