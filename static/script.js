/**
 * script.js — FinRisk Intelligence shared utilities
 */

/**
 * Format a number with commas and optional decimal places.
 * @param {number|null} val
 * @param {number} decimals
 * @returns {string}
 */
function fmt(val, decimals = 0) {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return Number(val).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/**
 * Clamp a number between lo and hi.
 */
function clamp(val, lo, hi) {
  return Math.min(hi, Math.max(lo, val));
}

/**
 * Debounce a function call.
 */
function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

/**
 * Global Chart.js defaults for the terminal aesthetic.
 */
if (typeof Chart !== "undefined") {
  Chart.defaults.color = "#8892a4";
  Chart.defaults.font.family = "IBM Plex Mono";
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.tooltip.backgroundColor = "#0f1524";
  Chart.defaults.plugins.tooltip.borderColor = "#2a3550";
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleColor = "#e2e8f0";
  Chart.defaults.plugins.tooltip.bodyColor = "#8892a4";
  Chart.defaults.plugins.tooltip.padding = 10;
}
