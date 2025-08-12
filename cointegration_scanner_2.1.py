#!/usr/bin/env python3
"""
cointegration_scanner_2.1.py — Scan + Refine to trade-ready pairs in one shot.


What it does:
- Pull S&P 500 members (Wikipedia)
- Download Adjusted Close with yfinance
- Use log prices + date intersection alignment
- Prefilter candidate pairs by absolute correlation (optional top-K neighbors per ticker)
- Run Engle–Granger cointegration tests on candidates
- Apply Benjamini–Hochberg (FDR) to p-values
- Refine survivors:
    * OLS hedge ratio beta on log prices (with constant)
    * Spread volatility bounds
    * ADF stationarity on residual spread
    * AR(1) half-life filter (mean-reversion speed)
- Adaptive relaxation loop to reach at least MIN_PAIRS outputs (within safety bounds)
- Output:
    pairs_scan_all.csv (raw coint tests + q-values)
    pairs_trade_ready.csv (final, trade-ready list)

Dependencies:
    pip install pandas numpy yfinance statsmodels lxml requests
    (optional) pip install tqdm
"""


import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

# =======================
# Config (edit as needed)
# =======================
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Data window
START_DATE = "2024-01-01"
END_DATE   = None  # None = today

# Universe
MAX_TICKERS = 500          # <= 503
MIN_DAYS    = 180          # drop symbols with fewer valid observations
MIN_OVERLAP = 160          # minimum overlapping bars per pair

# Prefilter (reduces pair count before Engle–Granger)
CORR_MIN = 0.70            # absolute corr on log prices to keep a pair
TOP_K_PER_TICKER = 25      # top-K most correlated neighbors per symbol; set None to disable

# Engle–Granger params
EG_TREND = "c"             # 'c','ct','ctt','nc'
EG_MAX_LAG = None          # let statsmodels choose

# Multiple testing control (Benjamini–Hochberg FDR)
Q_ALPHA = 0.05

# Refining thresholds (initial; spread-vol bounds will be adapted from data)
CORR_FILTER = 0.80         # stricter corr filter at refiner stage
SPREAD_VOL_MIN = 0.5       # initial placeholder; will be reset adaptively
SPREAD_VOL_MAX = 50.0      # initial placeholder; will be reset adaptively
ADF_ALPHA = 0.05           # residual stationarity p-value threshold
HL_MIN = 5                 # min AR(1) half-life (bars)
HL_MAX = 120               # max half-life (bars)

# Adaptive relaxation targets/bounds
MIN_PAIRS = 20             # target number of trade-ready pairs
MAX_RELAX_STEPS = 8        # cap relaxation steps

# safety floors/ceilings during relaxation
CORR_FLOOR   = 0.65
ADF_CEIL     = 0.20
SV_MIN_FLOOR = 0.001
SV_MAX_CEIL  = 1.0
HL_MIN_FLOOR = 2
HL_MAX_CEIL  = 252

# Output files
OUT_ALL  = "pairs_scan_all.csv"        # all tested pairs + p/q
OUT_READY = "pairs_trade_ready.csv"    # final filtered, trade-ready

# Misc
TIMEOUT = 20
RETRIES = 3
RANDOM_SEED = 1337
# =======================


def get_sp500_symbols(url=SP500_URL):
    """Scrape S&P 500 tickers from Wikipedia and normalize for yfinance."""
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    # pandas warns about literal HTML; fine here.
    tables = pd.read_html(resp.text)
    df = tables[0].copy()
    syms = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return syms


def safe_download(tickers, start, end, retries=RETRIES):
    """Download Adj Close for a list of tickers, with retries, return wide DataFrame."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            last_exc = None
            break
        except Exception as e:
            last_exc = e
            time.sleep(1.5 * attempt)

    if last_exc is not None:
        raise last_exc

    # Build a wide Adj Close dataframe
    if isinstance(df.columns, pd.MultiIndex):
        out = {}
        for sym in df.columns.levels[0]:
            try:
                ser = df[sym]["Adj Close"]
                if not ser.dropna().empty:
                    out[sym] = ser
            except Exception:
                continue
        data = pd.DataFrame(out)
    else:
        data = df.rename(columns={"Adj Close": tickers[0]})[["Adj Close"]]

    data = data.dropna(axis=1, how="all")
    return data


def benjamini_hochberg(pvals: pd.Series, alpha=Q_ALPHA) -> pd.Series:
    """Return BH-adjusted p-values (q-values)."""
    p = pvals.values
    n = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    qvals = np.empty_like(q, dtype=float)
    qvals[order] = q_sorted
    return pd.Series(qvals, index=pvals.index)


def prefilter_candidates(log_prices: pd.DataFrame,
                         corr_min=CORR_MIN,
                         top_k=TOP_K_PER_TICKER,
                         min_days=MIN_DAYS):
    """Reduce pair count via abs-correlation and optional top-K per symbol."""
    good = log_prices.count() >= min_days
    lp = log_prices.loc[:, good[good].index]
    if lp.shape[1] == 0:
        raise ValueError("No tickers with sufficient data.")

    corr = lp.corr().abs()
    pairs = set()
    cols = corr.columns.tolist()

    if top_k is not None:
        for sym in cols:
            top = (
                corr[sym]
                .drop(labels=[sym], errors="ignore")
                .sort_values(ascending=False)
                .head(top_k)
            )
            for other, c in top.items():
                if c >= corr_min:
                    a, b = sorted([sym, other])
                    pairs.add((a, b))
    else:
        iu = np.triu_indices_from(corr.values, k=1)
        mask = corr.values[iu] >= corr_min
        for (i, j), flag in zip(zip(iu[0], iu[1]), mask):
            if flag:
                a, b = cols[i], cols[j]
                pairs.add((a, b))

    return lp, sorted(list(pairs))


def ar1_half_life(series: pd.Series):
    """Estimate AR(1) half-life of mean reversion for a stationary series."""
    s = series.dropna().values
    if len(s) < 20:
        return None, None
    y = s[1:]
    x = s[:-1]
    X = sm.add_constant(x)
    try:
        model = sm.OLS(y, X).fit()
        phi = model.params[1]
    except Exception:
        return None, None
    if not np.isfinite(phi):
        return None, None
    if phi <= 0 or phi >= 1:
        return phi, None
    hl = -np.log(2) / np.log(phi)
    return phi, hl


def refine_once(survivors_df, logp_df,
                corr_min, sv_min, sv_max, adf_alpha, hl_min, hl_max, min_overlap):
    """Run the refiner with given thresholds; return (ready_df, diag_counts)."""
    rows = []
    diag = {"tested": 0, "fail_corr": 0, "fail_sv": 0, "fail_adf": 0, "fail_hl": 0, "fail_overlap": 0, "ok": 0}
    for _, row in survivors_df.iterrows():
        a = row["Ticker1"]; b = row["Ticker2"]
        s1 = logp_df[a].dropna(); s2 = logp_df[b].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < min_overlap:
            diag["fail_overlap"] += 1
            continue
        s1 = s1.loc[idx]; s2 = s2.loc[idx]
        diag["tested"] += 1

        # correlation
        corr = float(s1.corr(s2))
        if not np.isfinite(corr) or abs(corr) < corr_min:
            diag["fail_corr"] += 1
            continue

        # OLS: y ~ const + beta * x
        try:
            X = sm.add_constant(s2.values)
            beta = float(sm.OLS(s1.values, X).fit().params[1])
        except Exception:
            diag["fail_corr"] += 1
            continue

        # Spread and its volatility
        spread = s1 - beta * s2
        sv = float(spread.std())
        if not np.isfinite(sv) or sv < sv_min or sv > sv_max:
            diag["fail_sv"] += 1
            continue

        # ADF stationarity on residual
        try:
            _, adf_p, *_ = adfuller(spread.values, autolag="AIC")
        except Exception:
            diag["fail_adf"] += 1
            continue
        if not np.isfinite(adf_p) or adf_p >= adf_alpha:
            diag["fail_adf"] += 1
            continue

        # AR(1) half-life
        phi, hl = ar1_half_life(spread)
        if hl is None or not np.isfinite(hl) or hl < hl_min or hl > hl_max:
            diag["fail_hl"] += 1
            continue

        rows.append({
            "Pair": f"{a}-{b}", "Ticker1": a, "Ticker2": b,
            "n": int(len(idx)),
            "p_value": float(row["p_value"]), "q_value": float(row["q_value"]),
            "corr": float(corr), "beta": float(beta),
            "spread_vol": float(sv), "adf_p": float(adf_p),
            "phi": float(phi) if phi is not None else np.nan,
            "half_life": float(hl)
        })
        diag["ok"] += 1

    ready = pd.DataFrame(rows)
    return ready, diag


def main():
    np.random.seed(RANDOM_SEED)

    print("Fetching S&P 500 tickers…")
    syms = get_sp500_symbols()
    if MAX_TICKERS is not None:
        syms = syms[:MAX_TICKERS]
    print(f"Universe size: {len(syms)}")

    print("Downloading Adj Close (yfinance)…")
    data = safe_download(syms, START_DATE, END_DATE)

    # Keep columns with enough observations
    data = data.loc[:, data.count() >= MIN_DAYS]
    if data.shape[1] == 0:
        raise ValueError("No symbols with sufficient data in the chosen window.")

    # Log prices
    logp = np.log(data)

    # Prefilter pairs
    print("Prefiltering candidates by absolute correlation…")
    logp_pf, candidates = prefilter_candidates(
        logp, corr_min=CORR_MIN, top_k=TOP_K_PER_TICKER, min_days=MIN_DAYS
    )
    print(f"Candidate pairs: {len(candidates)}")

    # Engle–Granger on candidates
    results = []
    print("Running Engle–Granger cointegration tests…")
    for a, b in tqdm(candidates, ncols=80):
        s1 = logp_pf[a].dropna()
        s2 = logp_pf[b].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < MIN_OVERLAP:
            continue
        s1, s2 = s1.loc[idx], s2.loc[idx]
        try:
            score, p, _ = coint(s1.values, s2.values, trend=EG_TREND, maxlag=EG_MAX_LAG, autolag="aic")
            results.append({"Pair": f"{a}-{b}", "Ticker1": a, "Ticker2": b, "p_value": float(p), "n": int(len(idx))})
        except Exception:
            continue

    if not results:
        print("No testable pairs found.")
        return

    res = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    print(f"Tested pairs: {len(res)}")

    # BH FDR
    print("Applying Benjamini–Hochberg FDR…")
    res["q_value"] = benjamini_hochberg(res["p_value"], alpha=Q_ALPHA)
    res["rank"] = np.arange(1, len(res) + 1)

    # Save raw scan
    res.to_csv(OUT_ALL, index=False)
    print(f"Saved scan results to {OUT_ALL}")

    # Survivors for refiner
    survivors = res[(res["q_value"] < Q_ALPHA) & (res["n"] >= MIN_OVERLAP)].copy()
    print(f"Survivors after FDR: {len(survivors)}")

    # ---------- Adaptive spread-vol bounds (from survivors) ----------
    # Compute provisional spread std for each survivor to set sensible initial bounds.
    sv_measures = []
    for _, row in survivors.iterrows():
        a, b = row["Ticker1"], row["Ticker2"]
        s1 = logp[a].dropna(); s2 = logp[b].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < MIN_OVERLAP:
            continue
        s1, s2 = s1.loc[idx], s2.loc[idx]
        try:
            beta = float(sm.OLS(s1.values, sm.add_constant(s2.values)).fit().params[1])
            spread = s1 - beta * s2
            sv = float(spread.std())
            if np.isfinite(sv):
                sv_measures.append(sv)
        except Exception:
            pass

    if sv_measures:
        q05, q95 = np.quantile(sv_measures, [0.05, 0.95])
        # Clamp to safety bounds and prefer slightly generous window
        adaptive_sv_min = max(SV_MIN_FLOOR, min(q05, 0.05))
        adaptive_sv_max = min(SV_MAX_CEIL,  max(q95, 0.15))
        print(f"Adaptive spread-vol bounds -> [{adaptive_sv_min:.4f}, {adaptive_sv_max:.4f}]")
    else:
        # Fallback if we couldn't compute anything
        adaptive_sv_min, adaptive_sv_max = 0.005, 0.5
        print(f"Adaptive spread-vol bounds (fallback) -> [{adaptive_sv_min:.4f}, {adaptive_sv_max:.4f}]")

    # Seed thresholds for relaxation
    corr_min = CORR_FILTER
    sv_min, sv_max = adaptive_sv_min, adaptive_sv_max
    adf_alpha = ADF_ALPHA
    hl_min, hl_max = HL_MIN, HL_MAX

    # ---------- Adaptive relaxation loop ----------
    best = None
    for step in range(MAX_RELAX_STEPS + 1):
        ready, diag = refine_once(survivors, logp,
                                  corr_min, sv_min, sv_max, adf_alpha, hl_min, hl_max, MIN_OVERLAP)
        print(f"[refine step {step}] thresholds: corr>={corr_min:.2f}, "
              f"sv∈[{sv_min:.4f},{sv_max:.4f}], adf_p<{adf_alpha:.2f}, "
              f"hl∈[{hl_min},{hl_max}]  -> ok={len(ready)}  diag={diag}")

        if best is None or len(ready) > len(best):
            best = ready.copy()

        if len(ready) >= MIN_PAIRS:
            final = ready
            break

        # relax for next step (gentle, bounded)
        corr_min = max(CORR_FLOOR, corr_min - 0.02)
        sv_min   = max(SV_MIN_FLOOR, sv_min * 0.9)
        sv_max   = min(SV_MAX_CEIL,  sv_max * 1.2)
        adf_alpha = min(ADF_CEIL, adf_alpha + 0.02)
        hl_min   = max(HL_MIN_FLOOR, int(hl_min * 0.8))
        hl_max   = min(HL_MAX_CEIL,  int(hl_max * 1.25))
    else:
        final = best if best is not None else pd.DataFrame()

    if final is None or final.empty:
        print("No pairs passed after relaxation. Consider extending the date window or lowering CORR_MIN/TOP_K.")
        final = pd.DataFrame(columns=[
            "Pair","Ticker1","Ticker2","n","p_value","q_value","corr","beta",
            "spread_vol","adf_p","phi","half_life"
        ])
    else:
        # Rank: prioritize low q_value, high corr, mid-range half-life, lower spread_vol
        final["rank_q"] = final["q_value"].rank(method="min")
        final["rank_corr"] = (-final["corr"]).rank(method="min")
        final["rank_spreadvol"] = final["spread_vol"].rank(method="min")
        mid_hl = (HL_MIN + HL_MAX) / 2.0
        final["hl_dist"] = (final["half_life"] - mid_hl).abs()
        final["rank_hl"] = final["hl_dist"].rank(method="min")
        final["score"] = final["rank_q"] + final["rank_corr"] + final["rank_spreadvol"] + final["rank_hl"]
        final = final.sort_values(["score", "q_value", "corr"], ascending=[True, True, False]).reset_index(drop=True)

    # Final columns and save
    cols = ["Pair","Ticker1","Ticker2","n","p_value","q_value","corr","beta",
            "spread_vol","adf_p","phi","half_life"]
    final = final[cols]
    final.to_csv(OUT_READY, index=False)
    print(f"Saved trade-ready pairs to {OUT_READY} (rows: {len(final)})")
    if not final.empty:
        print(final.head(20).to_string(index=False))


if __name__ == "__main__":
    main()