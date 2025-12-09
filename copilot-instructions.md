<!-- Copilot / AI agent instructions for this repo -->
# Project overview

This repository is a single-file Streamlit application: `mutual_fund_dashboard.py`. The app ingests user CSVs (or local CSVs in `BASE_FOLDER`), fetches NAVs from the public `mfapi.in` API, computes metrics (current value, gain/loss, XIRR) and renders interactive charts via Plotly.

# Quick run

- **Run locally:** `streamlit run mutual_fund_dashboard.py` (PowerShell: use `streamlit run mutual_fund_dashboard.py`).
- **Python deps:** `streamlit`, `pandas`, `plotly`, `requests`. Example install: `pip install streamlit pandas plotly requests`.

# Architecture & dataflow (what to know)

- Single-entry UI: everything is implemented in `mutual_fund_dashboard.py` (no separate backend/service). Treat this file as the canonical source of truth.
- Input sources:
  - Streamlit file uploader (multiple CSVs) for portfolio builds.
  - Local CSV files loaded automatically when `BASE_FOLDER` + `<Fund Name>.csv` exists (path is currently hard-coded to a Windows path in `BASE_FOLDER`).
  - External NAV API: `https://api.mfapi.in/mf/{fund_code}` (used for current and historical NAVs).
- Processing flow: raw CSV bytes -> `detect_delimiter()` -> `load_and_clean_csv_bytes()` -> normalized DataFrame with columns titled `Date`, `Units`, `NAV`, `Amount` -> compute per-fund totals and XIRR via `xirr()` -> display via Plotly (`px.line`, `go.Figure`, `px.pie`).

# Project-specific conventions and patterns

- Column normalization: the app canonicalizes column names to Title-case keys (`Date`, `Units`, `NAV`, `Amount`) inside `load_and_clean_csv_bytes()` — code and downstream logic assume those exact names.
- Date parsing: uses `pd.to_datetime(..., dayfirst=True)`; expect day-first date formats in CSVs (DD-MM-YYYY common).
- Numeric cleaning: currency symbols (`₹`) and comma thousand separators are stripped before numeric conversion.
- Delimiter sniffing: `detect_delimiter()` prefers `;` when present; code expects CSVs with either `,` or `;` delimiters.
- Fund mapping: `mutual_funds` dict maps human-friendly fund names to MFAPI codes. Matching uploaded filenames to this dict uses substring case-insensitive checks.

# Key functions to reference (examples)

- `detect_delimiter(sample_bytes: bytes) -> str` — returns delimiter used when reading CSV bytes.
- `load_and_clean_csv_bytes(raw_bytes: bytes) -> pd.DataFrame` — canonical CSV parsing and cleaning; responsible for column name normalization and numeric conversions.
- `xirr(cashflows, dates, guess=0.1)` — Newton–Raphson implementation for annualized XIRR; cashflows use negative amounts for outflows and a final positive terminal value.

# Debugging and common edits

- To change where local CSVs are read, edit `BASE_FOLDER` (currently a Windows absolute path). Prefer making this configurable via an env var or Streamlit text input for portability.
- If NAVs appear missing, check the `mutual_funds` mapping keys and the substring matching logic used when inferring fund codes from filenames.
- When adding a new fund mapping, add an entry to the `mutual_funds` dict at the top of the file.
- Keep UI-friendly messages (use `st.error`, `st.warning`, `st.info`) rather than raising exceptions — the app uses try/except and Streamlit messages to preserve interactive flow.

# Tests & verification

- No automated tests present. Quick local verification steps:
  - Install dependencies.
  - Run `streamlit run mutual_fund_dashboard.py` and exercise the UI: upload a sample CSV, or place `<Fund Name>.csv` in `BASE_FOLDER` and select that fund.
  - Inspect the console for exceptions and Streamlit UI for expected metrics (Total Invested, Current Value, XIRR).

# Integration points & external dependencies

- Public API: `mfapi.in` (HTTP GET) — avoid aggressive polling; API calls include simple timeouts in places (consider adding retries/backoff for production).
- Local filesystem: `BASE_FOLDER` for automatic CSV discovery — path must be writable/readable by the environment running Streamlit.

# What the AI should not change without confirmation

- Do not change the user-visible column canonicalization (`Date`, `Units`, `NAV`, `Amount`) because many computations rely on them.
- Avoid removing or overhauling the `xirr()` algorithm; instead, if numerical stability is a concern, propose tests or optional alternative implementations.

---
If any section is unclear or you want additional examples (sample CSV shape, suggested env var for `BASE_FOLDER`, or an alternate XIRR implementation with tests), tell me which area to expand.
