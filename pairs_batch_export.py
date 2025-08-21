#!/usr/bin/env python3
"""
pairs_batch_export.py  —  TrendSpider-ready pair Z-score feed

Outputs
-------
1) TS-ready (minimal): out/pairs_ts/<PAIR>_<interval>.csv   with columns: Time,Z
2) (Optional) Diagnostics: out/pairs/<PAIR>_<interval>.csv  with: Y_Close,X_Close,Y_adj,X_adj,Beta,Spread,ZScore,ZClip

Key features
------------
- Only downloads tickers present in your pairs CSV (no whole-index pulls)
- Intraday is chunked and earliest start is clamped to Yahoo limits (prevents "not available" spam)
- Rolling OLS beta & Z-score are computed with no lookahead
- Warm-up rows are trimmed (no NaNs/Inf in Z)
- Tail-only export keeps files small and fast to ingest in TrendSpider

Usage examples
--------------
# Daily bars (classic stat-arb)
python pairs_batch_export.py --pairs_csv pairs_trade_ready.csv --out ./out_daily \
  --interval 1d --start 2020-01-01 --use_log yes --beta_lb 120 --z_win 60 \
  --trim_warmup yes --clip_z 5 --tail 3000 --write_diagnostics yes

# Hourly bars (stitched + clamped intraday)
python pairs_batch_export.py --pairs_csv pairs_trade_ready.csv --out ./out_hourly \
  --interval 1h --start 2024-01-01 --use_log yes --beta_lb 48 --z_win 24 \
  --trim_warmup yes --clip_z 5 --tail 5000 --write_diagnostics yes
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("Please: pip install yfinance pandas numpy")
    sys.exit(1)

# --------------------------- Config ---------------------------

DEFAULT_CHUNK_DAYS = {
    "1m": 6, "2m": 30, "5m": 30, "15m": 30, "30m": 30,
    "60m": 55, "90m": 55, "1h": 55,
}

INTRADAY_MAX_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "90m": 730, "1h": 730,
}

def is_intraday(interval: str) -> bool:
    k = interval.strip().lower()
    return k.endswith("m") or k.endswith("h")

def safe_symbol(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_", "."))

def parse_date(s: str | None):
    if s is None:
        return None
    return pd.to_datetime(s)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_naive_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure tz-naive DateTimeIndex, deduped, sorted."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df = df.copy()
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

# --------------------------- Download (stitched + clamped) ---------------------------

def download_bars_stitched(
    ticker: str,
    interval: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    auto_adjust: bool = True,
    actions: bool = False,
    backoff_sec: float = 1.0,
    chunk_days: int | None = None,
    probe_recent_days: int = 3,
) -> pd.DataFrame:
    """
    Download bars; chunk intraday; clamp earliest start to Yahoo's limits; skip symbols with no recent data.
    """
    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")

    # normalize end
    if end is None:
        end = now_utc
    else:
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    # normalize start
    if start is None:
        start = end - pd.Timedelta(days=60 if is_intraday(interval) else 365)
    else:
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")

    # Clamp intraday start to Yahoo window
    if is_intraday(interval):
        key = interval.lower()
        if key == "1h":
            key = "60m"
        max_days = INTRADAY_MAX_DAYS.get(key, 60)
        min_start_allowed = end - pd.Timedelta(days=max_days)
        if start < min_start_allowed:
            print(f"[{ticker}] clamp {start.date()} → {min_start_allowed.date()} for {interval} (Yahoo ~{max_days}d)")
            start = min_start_allowed

        # quick probe: skip if no recent bars (saves time)
        try:
            probe_start = end - pd.Timedelta(days=probe_recent_days)
            probe = yf.Ticker(ticker).history(
                start=probe_start, end=end, interval=interval,
                actions=False, auto_adjust=auto_adjust
            )
            probe = to_naive_sorted(probe)
            if probe.empty:
                print(f"[{ticker}] no {interval} data in last {probe_recent_days}d — skipping.")
                return pd.DataFrame()
        except Exception as e:
            print(f"[{ticker}] probe error for {interval}: {e} — skipping.")
            return pd.DataFrame()

    # Daily+ can be single call
    if not is_intraday(interval):
        df = yf.Ticker(ticker).history(
            start=start, end=end, interval=interval,
            actions=actions, auto_adjust=auto_adjust
        )
        return to_naive_sorted(df)

    # Intraday: chunk + stitch
    cd = chunk_days or DEFAULT_CHUNK_DAYS.get(interval.lower(), 30)
    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=cd), end)
        for attempt in range(3):
            try:
                dfp = yf.Ticker(ticker).history(
                    start=cur, end=nxt, interval=interval,
                    actions=actions, auto_adjust=auto_adjust
                )
                dfp = to_naive_sorted(dfp)
                if not dfp.empty:
                    chunks.append(dfp)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[{ticker}] ERROR {cur.date()}..{nxt.date()} @ {interval}: {e}")
                time.sleep(backoff_sec * (attempt + 1))
        cur = nxt
        time.sleep(0.3)  # polite pause
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, axis=0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

# --------------------------- Pair math ---------------------------

def rolling_ols_beta(y: pd.Series, x: pd.Series, lookback: int, min_obs: int = 20) -> pd.Series:
    """ Rolling OLS slope of y ~ beta * x (no lookahead). """
    beta = pd.Series(index=y.index, dtype=float)
    for i in range(len(y)):
        j0 = max(0, i - lookback + 1)
        yi = y.iloc[j0:i+1].dropna()
        xi = x.iloc[j0:i+1].dropna()
        join = yi.to_frame("y").join(xi.to_frame("x"), how="inner")
        if len(join) < max(5, min_obs):
            beta.iloc[i] = np.nan
            continue
        xx = join["x"].values
        yy = join["y"].values
        n  = len(xx)
        xSum = float(xx.sum()); ySum = float(yy.sum())
        x2   = float((xx*xx).sum()); xy = float((xx*yy).sum())
        den  = n * x2 - xSum * xSum
        beta.iloc[i] = np.nan if den == 0 else (n * xy - xSum * ySum) / den
    return beta

def rolling_zscore(s: pd.Series, win: int) -> pd.Series:
    m  = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std(ddof=1)
    z  = (s - m) / sd
    return z.replace([np.inf, -np.inf], np.nan)

# --------------------------- IO helpers ---------------------------

def read_pairs_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # Accept common schemas
    if "ticker1" in cols and "ticker2" in cols:
        a, b = cols["ticker1"], cols["ticker2"]
    elif "symbol1" in cols and "symbol2" in cols:
        a, b = cols["symbol1"], cols["symbol2"]
    elif "y" in cols and "x" in cols:
        a, b = cols["y"], cols["x"]
    else:
        raise ValueError("pairs_csv must have columns like (Ticker1,Ticker2) or (Symbol1,Symbol2) or (y,x).")
    out = df[[a, b]].rename(columns={a: "Y", b: "X"})
    out["Y"] = out["Y"].astype(str).str.strip()
    out["X"] = out["X"].astype(str).str.strip()
    return out

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Export TS-ready Z-score feeds (Time,Z) for cointegrated pairs.")
    ap.add_argument("--pairs_csv", required=True, help="CSV with (Ticker1,Ticker2) or (Symbol1,Symbol2) or (y,x)")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--interval", default="1d", help="yfinance interval (e.g., 1d, 1h, 60m, 5m)")
    ap.add_argument("--start", default=None, help="Start YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="End YYYY-MM-DD (optional)")
    ap.add_argument("--chunk_days", type=int, default=None, help="Override intraday chunk size (days)")
    ap.add_argument("--beta_lb", type=int, default=120, help="Rolling OLS lookback for beta")
    ap.add_argument("--z_win", type=int, default=60, help="Rolling z-score window")
    ap.add_argument("--use_log", choices=["yes","no"], default="yes", help="Log prices before beta/z calc")
    ap.add_argument("--adj_close", choices=["auto","raw"], default="auto", help="Use auto_adjust (auto) or raw (raw)")
    # TS-optimized toggles
    ap.add_argument("--trim_warmup", choices=["yes","no"], default="yes", help="Drop rows before beta & z exist")
    ap.add_argument("--clip_z", type=float, default=5.0, help="Clip ZScore to +/- clip_z as ZClip (diagnostics)")
    ap.add_argument("--tail", type=int, default=3000, help="Keep only the last N rows per pair after trim")
    ap.add_argument("--write_diagnostics", choices=["yes","no"], default="yes", help="Also write compact diagnostics CSVs")
    args = ap.parse_args()

    interval = args.interval.strip().lower()
    start    = parse_date(args.start)
    end      = parse_date(args.end)

    out_root     = os.path.abspath(args.out)
    out_tickers  = os.path.join(out_root, "tickers")
    out_pairs    = os.path.join(out_root, "pairs")
    out_pairs_ts = os.path.join(out_root, "pairs_ts")
    ensure_dir(out_root); ensure_dir(out_tickers); ensure_dir(out_pairs_ts)
    if args.write_diagnostics == "yes":
        ensure_dir(out_pairs)

    pairs = read_pairs_csv(args.pairs_csv)
    tickers = sorted(set(pairs["Y"]).union(set(pairs["X"])))
    print(f"Found {len(pairs)} pairs; {len(tickers)} unique tickers.\n")

    cache: dict[str, pd.DataFrame] = {}

    # -------------------- per-ticker downloads --------------------
    for tk in tickers:
        print(f"[DL] {tk} @ {interval} ...")
        df = download_bars_stitched(
            tk, interval, start, end,
            auto_adjust=(args.adj_close == "auto"),
            actions=False,
            chunk_days=args.chunk_days
        )
        if df.empty or "Close" not in df.columns:
            print(f"  -> no data written for {tk}.\n")
            continue

        out_df = df[["Open","High","Low","Close","Volume"]].copy()
        out_path = os.path.join(out_tickers, f"{safe_symbol(tk)}_{interval}.csv")
        out_df.to_csv(out_path, index_label="Time")
        cache[tk] = out_df
        print(f"  -> {len(out_df):,} bars -> {out_path}\n")

    # -------------------- per-pair build --------------------
    manifest = []
    for _, row in pairs.iterrows():
        ysym, xsym = row["Y"], row["X"]
        ydf = cache.get(ysym); xdf = cache.get(xsym)
        if ydf is None or xdf is None:
            print(f"[PAIR] Skip {ysym}-{xsym}: missing data.\n")
            continue

        join = ydf[["Close"]].rename(columns={"Close": f"{ysym}_Close"}).join(
               xdf[["Close"]].rename(columns={"Close": f"{xsym}_Close"}), how="inner")
        if join.empty:
            print(f"[PAIR] Skip {ysym}-{xsym}: no overlap.\n")
            continue

        # log or linear
        if args.use_log == "yes":
            y = np.log(join[f"{ysym}_Close"])
            x = np.log(join[f"{xsym}_Close"])
        else:
            y = join[f"{ysym}_Close"].astype(float)
            x = join[f"{xsym}_Close"].astype(float)

        beta   = rolling_ols_beta(y, x, lookback=args.beta_lb, min_obs=min(30, args.beta_lb//3 if args.beta_lb>=9 else 3))
        spread = y - beta * x
        z      = rolling_zscore(spread, win=args.z_win)

        # Build compact frame for diagnostics (optional) and for TS-ready export
        out_pair = pd.DataFrame({
            "Y_Close": join[f"{ysym}_Close"],
            "X_Close": join[f"{xsym}_Close"],
            "Y_adj": y, "X_adj": x,
            "Beta": beta, "Spread": spread, "ZScore": z
        }, index=join.index)

        # Trim warm-up & clean Z
        if args.trim_warmup == "yes":
            valid = out_pair["Beta"].notna() & out_pair["ZScore"].notna()
            out_pair = out_pair.loc[valid].copy()
            if out_pair.empty:
                print(f"[PAIR] {ysym}-{xsym}: no valid rows after warm-up; skipping.\n")
                continue

        out_pair["ZScore"] = out_pair["ZScore"].replace([np.inf, -np.inf], np.nan)
        out_pair = out_pair.dropna(subset=["ZScore"])
        out_pair = out_pair[~out_pair.index.duplicated(keep="last")].sort_index()

        # Tail trim
        tail_n = int(args.tail) if args.tail and args.tail > 0 else None
        if tail_n and len(out_pair) > tail_n:
            out_pair = out_pair.iloc[-tail_n:].copy()

        # ---------------- TS-READY export (Time,Z) ----------------
        out_pair.index = pd.to_datetime(out_pair.index)
        out_pair.index.name = "Time"
        ts_min = out_pair[["ZScore"]].rename(columns={"ZScore": "Z"}).reset_index()[["Time", "Z"]]
        ts_min.dropna(subset=["Z"], inplace=True)
        ts_min["Z"] = pd.to_numeric(ts_min["Z"], errors="coerce")
        ts_min.dropna(subset=["Z"], inplace=True)
        ts_min["Z"] = ts_min["Z"].round(6)

        pair_name = f"{safe_symbol(ysym)}-{safe_symbol(xsym)}_{interval}.csv"
        out_path_ts = os.path.join(out_pairs_ts, pair_name)
        ts_min.to_csv(out_path_ts, index=False)

        # ---------------- Diagnostics export (optional) ----------
        if args.write_diagnostics == "yes":
            z_clip_val = float(args.clip_z)
            out_pair["ZClip"] = out_pair["ZScore"].clip(-z_clip_val, z_clip_val)
            diag = out_pair.round(6)[["Y_Close","X_Close","Y_adj","X_adj","Beta","Spread","ZScore","ZClip"]]
            out_path_full = os.path.join(out_pairs, pair_name)
            diag.to_csv(out_path_full, index_label="Time")
            print(f"[PAIR] {ysym}-{xsym}: {len(ts_min):,} rows -> TS {out_path_ts} | Full {out_path_full}\n")
            file_full_rel = os.path.relpath(out_path_full, start=out_root)
        else:
            print(f"[PAIR] {ysym}-{xsym}: {len(ts_min):,} rows -> TS {out_path_ts}\n")
            file_full_rel = None

        manifest.append({
            "pair": f"{ysym}-{xsym}",
            "interval": interval,
            "file_ts": os.path.relpath(out_path_ts, start=out_root),
            "file_full": file_full_rel,
            "y": ysym, "x": xsym,
            "use_log": args.use_log == "yes",
            "beta_lb": args.beta_lb,
            "z_win": args.z_win,
            "tail": tail_n
        })

    # Manifest
    manifest_path = os.path.join(out_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "interval": interval,
            "start": args.start,
            "end": args.end,
            "chunk_days": args.chunk_days or DEFAULT_CHUNK_DAYS.get(interval, None),
            "intraday": is_intraday(interval),
            "pairs_count": len(manifest),
            "pairs": manifest
        }, f, indent=2)

    print(f"Done. Manifest → {manifest_path}")
    print("Tip: In TrendSpider, ingest out/pairs_ts/<PAIR>_<interval>.csv and use column 'Z' (two columns only).")

if __name__ == "__main__":
    main()
