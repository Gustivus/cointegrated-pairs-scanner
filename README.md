# Cointegration Scanner 2.1

Scan the S&P 500 for statistically significant, *trade-ready* cointegrated pairs in one pass.  
Pulls constituents, downloads prices, runs Engle–Granger with BH-FDR correction, then **refines** survivors using spread stats and mean-reversion speed. Produces a full scan and a ranked, trade-ready list.

---

##  Features

- **Data acquisition:** S&P 500 members (Wikipedia) + Adjusted Close via `yfinance`
- **Data handling:** log prices, intersection alignment, minimum sample filters
- **Candidate prefilter:** absolute correlation with optional **top-K neighbors** per ticker
- **Engle–Granger** cointegration tests (`statsmodels`) with **Benjamini–Hochberg (FDR)**
- **Refinement** of survivors:
  - OLS hedge ratio (β) on log prices (with constant)
  - Spread volatility bounds (**auto-tuned** from survivor distribution)
  - ADF stationarity on residual spread
  - AR(1) half-life filter for mean-reversion speed
- **Adaptive relaxation loop** to reach at least `MIN_PAIRS` while respecting safety floors/ceilings
- Robust downloads (retries, timeouts) and deterministic seed
- Outputs:
  - `pairs_scan_all.csv` — raw cointegration results with p/q values
  - `pairs_trade_ready.csv` — final, ranked list with diagnostics

---

##  Requirements

```bash
pip install pandas numpy yfinance statsmodels lxml requests
# Optional (progress bar)
pip install tqdm
```

> **Python:** 3.9+ recommended.

---

##  Quick Start

```bash
python3 cointegration_scanner_2.1.py
```

**Outputs (in the working directory):**
- `pairs_scan_all.csv` — all tested pairs with `p_value`, `q_value`, and `n`
- `pairs_trade_ready.csv` — filtered pairs with `corr`, `beta`, `spread_vol`, `adf_p`, `phi`, `half_life`

**Example console log:**
```
Fetching S&P 500 tickers…
Universe size: 500
Downloading Adj Close (yfinance)…
Prefiltering candidates by absolute correlation…
Candidate pairs: 3812
Running Engle–Granger cointegration tests…
Tested pairs: 3650
Applying Benjamini–Hochberg FDR…
Saved scan results to pairs_scan_all.csv
Survivors after FDR: 142
Adaptive spread-vol bounds -> [0.0065, 0.1420]
[refine step 0] thresholds: corr>=0.80, sv∈[0.0065,0.1420], adf_p<0.05, hl∈[5,120]  -> ok=18  diag=…
[refine step 1] thresholds: … -> ok=22  diag=…
Saved trade-ready pairs to pairs_trade_ready.csv (rows: 22)
```

---

##  Configuration

Edit the **Config** block near the top of `cointegration_scanner_2.1.py` to fit your needs.

| Setting | Default | Description |
|---|---:|---|
| `START_DATE` / `END_DATE` | `"2024-01-01"` / `None` | Data window (`None` = today). |
| `MAX_TICKERS` | `500` | Cap S&P universe (≤503). |
| `MIN_DAYS` | `180` | Min valid observations per ticker to retain it. |
| `MIN_OVERLAP` | `160` | Minimum overlapping bars per pair (test/refine). |
| `CORR_MIN` | `0.70` | **Prefilter** absolute correlation threshold on log prices. |
| `TOP_K_PER_TICKER` | `25` | Keep only top-K most-correlated neighbors per symbol (set `None` to disable). |
| `EG_TREND` | `"c"` | Engle–Granger trend term: `'c'`, `'ct'`, `'ctt'`, `'nc'`. |
| `EG_MAX_LAG` | `None` | Let statsmodels choose (`autolag="aic"`). |
| `Q_ALPHA` | `0.05` | BH-FDR significance; survivors have `q_value < Q_ALPHA`. |
| `CORR_FILTER` | `0.80` | **Refiner** correlation threshold (stricter than prefilter). |
| `SPREAD_VOL_MIN/MAX` | placeholders | Ignored initially; replaced by **adaptive** bounds from survivors. |
| `ADF_ALPHA` | `0.05` | ADF p-value cutoff on residual spread. |
| `HL_MIN / HL_MAX` | `5 / 120` | AR(1) half-life bounds (bars). |
| `MIN_PAIRS` | `20` | Target number of trade-ready pairs. |
| `MAX_RELAX_STEPS` | `8` | Max adaptive relaxation iterations. |
| Safety rails | see code | Floors/ceilings for corr/ADF/vol/half-life during relaxation. |
| `OUT_ALL` / `OUT_READY` | filenames | Output CSV names. |
| `TIMEOUT` / `RETRIES` | `20 / 3` | HTTP timeout and download retries. |
| `RANDOM_SEED` | `1337` | Deterministic ordering/repro. |

**Tuning tips**
- **Too strict (few/no pairs):** extend `START_DATE`; raise `TOP_K_PER_TICKER`; lower `CORR_MIN`; increase `MAX_RELAX_STEPS`.
- **Too loose (many pairs):** shrink date window; raise `CORR_FILTER`; lower `ADF_CEIL` or `SV_MAX_CEIL`.

---

##  Methodology

1. **Universe & Data**
   - Scrape S&P 500 tickers from Wikipedia (normalize dots→dashes for Yahoo, e.g., `BRK.B` → `BRK-B`).
   - Download **Adjusted Close** via `yfinance`; drop tickers with `< MIN_DAYS` valid points.
   - Work in **log prices**; align per-pair on intersected dates.

2. **Candidate Prefilter**
   - Compute absolute Pearson correlation on log prices.
   - Keep pairs with `|corr| ≥ CORR_MIN`.
   - Optionally restrict to **top-K neighbors** per ticker to reduce pair count.

3. **Engle–Granger + FDR**
   - Run `statsmodels.tsa.stattools.coint` (trend=`EG_TREND`, `autolag="aic"`).
   - Apply **Benjamini–Hochberg** to p-values → `q_value`; keep `q_value < Q_ALPHA`.

4. **Refinement (Trade-Ready Filters)**
   - OLS regression `y = α + β x` on log prices → **hedge ratio β**.
   - Residual spread `s = y − β x`; compute **std(s)**.
   - **Adaptive spread-vol bounds:** derive from survivors’ spread std (≈ 5–95th pct), then clamp to safety rails.
   - **ADF** stationarity test on `s` (require `adf_p < ADF_ALPHA`).
   - **AR(1) half-life:** estimate on `s`; accept `HL_MIN ≤ hl ≤ HL_MAX`.

5. **Adaptive Relaxation**
   - If results `< MIN_PAIRS`, gently relax thresholds (bounded by floors/ceilings) and re-run the refiner up to `MAX_RELAX_STEPS`. Keep “best so far.”

6. **Ranking**
   - Composite score prioritizes: **low `q_value`**, **high `|corr|`**, **lower `spread_vol`**, and **half-life near the mid-range**.

---

##  Output Files & Schemas

### `pairs_scan_all.csv`
| Column | Meaning |
|---|---|
| `Pair` | `TICKER1-TICKER2` |
| `Ticker1`, `Ticker2` | Pair constituents |
| `n` | Overlapping observations used |
| `p_value` | Engle–Granger p-value |
| `q_value` | BH-adjusted p-value (FDR) |
| `rank` | Order by ascending `p_value` |

### `pairs_trade_ready.csv`
| Column | Meaning |
|---|---|
| `Pair`, `Ticker1`, `Ticker2`, `n` | As above |
| `p_value`, `q_value` | From EG + BH |
| `corr` | Absolute Pearson corr on log prices (refiner sample) |
| `beta` | OLS hedge ratio |
| `spread_vol` | Std of residual spread |
| `adf_p` | ADF p-value on spread |
| `phi` | AR(1) coefficient on spread |
| `half_life` | AR(1) half-life (bars) |

> The console prints per-step diagnostics (`tested`, `fail_overlap`, `fail_corr`, `fail_sv`, `fail_adf`, `fail_hl`, `ok`) to reveal which filter is binding.

---

##  Troubleshooting

- **“No symbols with sufficient data”**  
  Extend the window (`START_DATE` earlier), reduce `MIN_DAYS`, or lower `MAX_TICKERS`.

- **“No testable pairs found”**  
  Lower `CORR_MIN` or set `TOP_K_PER_TICKER=None`. Ensure `MIN_OVERLAP` fits your window.

- **Few/zero trade-ready pairs**  
  Increase `MAX_RELAX_STEPS`, slightly lower `CORR_FILTER`, widen `HL_MIN/HL_MAX`, or allow higher `ADF_ALPHA` (bounded by `ADF_CEIL`). Check the step diagnostics.

- **HTTP/Download errors**  
  Re-run (script retries automatically). If persistent, reduce `MAX_TICKERS` or increase `TIMEOUT`.

---

##  Notes & Extensions

- **Custom universe:** Replace `get_sp500_symbols()` with a static list/CSV (sectors, ETFs, ADRs).
- **Different frequency:** Resample to weekly/monthly before log transform for slower regimes.
- **Alternate tests:** Consider Johansen, Hurst screens, or Kalman filter hedge ratio.
- **Execution hooks:** After writing `pairs_trade_ready.csv`, trigger your signal engine/report.

---

## ⚠️ Disclaimer

This tool is for research and education. Cointegration and statistical signals do not guarantee profits.  
Backtest with realistic frictions and risk controls before trading live.

---

##  MIT License

Copyright (c) 2025 Conor Maguire

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
