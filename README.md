# Cointegration Scanner 2.1

Scan the S&P 500 for statistically significant, *trade-ready* cointegrated pairs in one pass.

This script:
- Pulls S&P 500 constituents
- Downloads prices
- Runs Engle–Granger tests with Benjamini–Hochberg FDR control
- Refines survivors into a final trade-ready list

---

## 📌 Features

- **Data acquisition** from Wikipedia + Yahoo Finance (`yfinance`)
- **Candidate prefilter** by absolute correlation (with optional top-K neighbors per ticker)
- **Engle–Granger** cointegration tests
- **Multiple testing control** via Benjamini–Hochberg (FDR)
- **Refinement** using:
  - OLS hedge ratio β
  - Spread volatility bounds (auto-tuned)
  - ADF stationarity test
  - AR(1) half-life filter
- **Adaptive relaxation loop** to reach `MIN_PAIRS` while staying within safety bounds
- Outputs:
  - `pairs_scan_all.csv` — raw cointegration results
  - `pairs_trade_ready.csv` — final ranked list

---

## ⚙️ Requirements

```bash
pip install pandas numpy yfinance statsmodels lxml requests
# Optional for progress bar
pip install tqdm
🚀 Quick Start
bash
Copy
Edit
python3 cointegration_scanner_2.1.py
Outputs:

pairs_scan_all.csv — all tested pairs with p/q-values

pairs_trade_ready.csv — trade-ready pairs with diagnostics

Example console log:

yaml
Copy
Edit
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
[refine step 0] thresholds: corr>=0.80, sv∈[0.0065,0.1420], adf_p<0.05, hl∈[5,120]  -> ok=18
[refine step 1] thresholds: … -> ok=22
Saved trade-ready pairs to pairs_trade_ready.csv (rows: 22)
🔧 Configuration
Edit the Config section in the script.

Setting	Default	Description
START_DATE / END_DATE	"2024-01-01" / None	Date range (None = today)
MAX_TICKERS	500	Max S&P tickers to use
MIN_DAYS	180	Min valid obs per ticker
MIN_OVERLAP	160	Min overlap per pair
CORR_MIN	0.70	Prefilter abs-corr threshold
TOP_K_PER_TICKER	25	Limit neighbors per ticker (None = all)
Q_ALPHA	0.05	FDR threshold
CORR_FILTER	0.80	Stricter corr in refiner
ADF_ALPHA	0.05	ADF p-value cutoff
HL_MIN / HL_MAX	5 / 120	Half-life bounds
MIN_PAIRS	20	Target trade-ready pairs
MAX_RELAX_STEPS	8	Max relax iterations

## 📈 Methodology
Data Fetch

Get S&P 500 list from Wikipedia

Download Adjusted Close prices via yfinance

Filter for min observations and log-transform

Prefilter

Compute abs-corr between all tickers

Keep pairs ≥ CORR_MIN

Optionally keep only TOP_K_PER_TICKER neighbors

Cointegration

Run Engle–Granger tests

Apply Benjamini–Hochberg FDR correction

Refinement

Estimate hedge ratio β

Calculate spread volatility & filter by adaptive bounds

ADF test on spread

AR(1) half-life filter

Adaptive Relaxation

Relax thresholds within safety bounds until MIN_PAIRS met

Ranking

Rank by q-value, correlation, spread vol, and half-life proximity to mid-range

📄 Output Files
pairs_scan_all.csv
Column	Meaning
Pair	TICKER1-TICKER2
Ticker1, Ticker2	Symbols
n	Overlapping obs
p_value	EG p-value
q_value	FDR-adjusted p-value
rank	Order by p-value

pairs_trade_ready.csv
Column	Meaning
Pair	TICKER1-TICKER2
corr	Abs-corr in refiner sample
beta	Hedge ratio
spread_vol	Spread std
adf_p	ADF p-value
phi	AR(1) coefficient
half_life	AR(1) half-life

💡 Tips
Too few pairs?
Extend START_DATE, lower CORR_MIN, increase MAX_RELAX_STEPS.

Too many pairs?
Tighten CORR_FILTER/ADF_ALPHA or shrink the date range.

Stable results:
Keep RANDOM_SEED fixed.

⚠️ Disclaimer
This script is for research and educational purposes.
Statistical cointegration ≠ profitable trading. Backtest with realistic costs and risk controls before trading.
