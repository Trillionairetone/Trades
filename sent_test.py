"""
sent_test.py - Alpha Vantage USD news-sentiment backtest on M15 (standalone).

Railway start command:  python sent_test.py
Needs variables: OANDA_TOKEN (or OANDA_API_KEY) and ALPHAVANTAGE_API_KEY.
Backtest only. This file never places orders.

Idea: score recent USD news (relevance-weighted Alpha Vantage sentiment).
When the score crosses a threshold, trade EUR_USD / GBP_USD / USD_JPY in the
USD's direction ("f" = follow) or against it ("x" = fade).
Search period picks settings; the holdout period checks them on unseen data.
Costs: real bid/ask candles, so every trade pays the spread.
"""
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


def log(*a):
    print(*a, flush=True)


OANDA_TOKEN = (os.getenv("OANDA_TOKEN") or os.getenv("OANDA_API_KEY") or "").strip()
AV_KEY = (os.getenv("ALPHAVANTAGE_API_KEY") or "").strip()
OANDA_URL = os.getenv("OANDA_URL", "https://api-fxpractice.oanda.com").rstrip("/")
START = os.getenv("SENT_START", "2022-03-01")
SPLIT = os.getenv("SENT_SPLIT", "2024-07-01")          # search before, holdout after
MONTHS_PER_CALL = int(os.getenv("SENT_MONTHS_PER_CALL", "2"))
MAX_CALLS = int(os.getenv("SENT_MAX_CALLS", "24"))      # free tier is small per day
TIME_SHIFT_H = float(os.getenv("SENT_TIME_SHIFT_H", "5"))  # delay news by 5h: no peeking
MIN_ARTICLES = int(os.getenv("SENT_MIN_ARTICLES", "3"))
HOLD_BARS = 96                                           # time exit after 24h
PAIRS = {"EUR_USD": -1, "GBP_USD": -1, "USD_JPY": 1}     # +1 when USD is the base

WINDOWS_H = [6, 24]
THRESHOLDS = [0.15, 0.25, 0.35]
STOPS_ATR = [2.0, 4.0]
TARGETS_R = [1.0, 2.0]


def add_months(d, m):
    y, mo = divmod(d.month - 1 + m, 12)
    return d.replace(year=d.year + y, month=mo + 1, day=1)


def av_sentiment(start):
    rows, t0, calls = [], start, 0
    now = datetime.now(timezone.utc)
    while t0 < now and calls < MAX_CALLS:
        t1 = min(add_months(t0, MONTHS_PER_CALL), now)
        params = {"function": "NEWS_SENTIMENT", "tickers": "FOREX:USD",
                  "time_from": t0.strftime("%Y%m%dT%H%M"), "time_to": t1.strftime("%Y%m%dT%H%M"),
                  "limit": 1000, "sort": "RELEVANCE", "apikey": AV_KEY}
        calls += 1
        try:
            j = requests.get("https://www.alphavantage.co/query", params=params, timeout=60).json()
        except (requests.RequestException, ValueError) as e:
            j = {"Information": str(e)}
        if "feed" not in j:
            msg = str(j.get("Information") or j.get("Note") or j.get("Error Message") or j)[:200]
            log(f"ALPHA VANTAGE | {t0:%Y-%m-%d} | no data: {msg}")
            if "limit" in msg.lower():
                break
        else:
            n = 0
            for item in j["feed"]:
                try:
                    ts = datetime.strptime(item["time_published"][:15], "%Y%m%dT%H%M%S")
                except (KeyError, ValueError):
                    continue
                for s in item.get("ticker_sentiment", []):
                    if s.get("ticker") == "FOREX:USD":
                        rows.append((ts, float(s["relevance_score"]), float(s["ticker_sentiment_score"])))
                        n += 1
            log(f"ALPHA VANTAGE | {t0:%Y-%m-%d} to {t1:%Y-%m-%d} | {n} USD articles")
        t0 = t1
        time.sleep(13)
    log(f"ALPHA VANTAGE | {calls} calls used")
    return rows


def oanda_candles(pair, start):
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {OANDA_TOKEN}", "Accept-Datetime-Format": "UNIX"})
    t, end, rows, err = start.timestamp(), time.time(), [], ""
    while t < end:
        for attempt in range(4):
            try:
                r = s.get(f"{OANDA_URL}/v3/instruments/{pair}/candles", timeout=30,
                          params={"granularity": "M15", "price": "BA", "from": f"{t:.0f}", "count": 5000})
                if r.status_code == 200:
                    break
                err = f"HTTP {r.status_code} {r.text[:120]}"
            except requests.RequestException as e:
                err = str(e)
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"{pair} candles failed: {err}")
        cs = r.json().get("candles", [])
        if not cs:
            break
        for c in cs:
            if c.get("complete"):
                b, a = c["bid"], c["ask"]
                rows.append((float(c["time"]), *(float(b[k]) for k in "ohlc"), *(float(a[k]) for k in "ohlc")))
        last = float(cs[-1]["time"])
        if last <= t:
            break
        t = last + 1
    df = pd.DataFrame(rows, columns=["t", "bo", "bh", "bl", "bc", "ao", "ah", "al", "ac"]).drop_duplicates("t")
    df["time"] = pd.to_datetime(df["t"], unit="s")
    return df.set_index("time").drop(columns="t").sort_index()


def prep(df):
    P = {k: df[k].to_numpy() for k in df.columns}
    h, l, c = (df.bh + df.ah) / 2, (df.bl + df.al) / 2, (df.bc + df.ac) / 2
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    P["atr"] = tr.rolling(14).mean().to_numpy()
    P["time"] = df.index.to_numpy()
    return P


def score_series(P, art_t, cw, cr, window_h):
    bt = P["time"]
    hi = np.searchsorted(art_t, bt, side="right")
    lo = np.searchsorted(art_t, bt - np.timedelta64(int(window_h * 3600), "s"), side="right")
    n, wsum = hi - lo, cr[hi] - cr[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = (cw[hi] - cw[lo]) / wsum
    sc[(n < MIN_ARTICLES) | (wsum <= 0)] = 0.0
    return np.nan_to_num(sc)


def sim(P, i, d, risk, tgt):
    e = i + 1
    if e >= len(P["ao"]):
        return None
    end = min(e + HOLD_BARS, len(P["ao"]))
    if d > 0:
        entry = P["ao"][e]; stop = entry - risk; target = entry + tgt * risk
        hs, ht = P["bl"][e:end] <= stop, P["bh"][e:end] >= target
    else:
        entry = P["bo"][e]; stop = entry + risk; target = entry - tgt * risk
        hs, ht = P["ah"][e:end] >= stop, P["al"][e:end] <= target
    big = 10 ** 9
    ks = int(np.argmax(hs)) if hs.any() else big
    kt = int(np.argmax(ht)) if ht.any() else big
    if ks == big and kt == big:
        k, ex = end - 1 - e, (P["bc"][end - 1] if d > 0 else P["ac"][end - 1])
    elif ks <= kt:                      # stop checked first: the cautious choice
        k, ex = ks, stop
    else:
        k, ex = kt, target
    return (ex - entry) / risk * d, e + k, (P["ao"][e] - P["bo"][e]) / risk


def stats(rs):
    rs = np.asarray(rs, float)
    if len(rs) == 0:
        return 0, 0.0, 0.0, 0.0
    pos, neg = rs[rs > 0].sum(), -rs[rs < 0].sum()
    return len(rs), (rs > 0).mean() * 100, rs.mean(), (pos / neg if neg > 0 else 99.0)


def main():
    if not OANDA_TOKEN or not AV_KEY:
        log("Missing OANDA_TOKEN / OANDA_API_KEY or ALPHAVANTAGE_API_KEY. Add them in Railway variables.")
        sys.exit(1)
    start = datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    split = np.datetime64(SPLIT)
    mid = np.datetime64(start.replace(tzinfo=None)) + (split - np.datetime64(start.replace(tzinfo=None))) // 2
    log(f"SENT | USD news sentiment | M15 | search {START} to {SPLIT} | holdout {SPLIT} to today | "
        f"pairs {list(PAIRS)} | news delayed {TIME_SHIFT_H:g}h")

    feed = av_sentiment(start)
    if len(feed) < 200:
        log(f"Only {len(feed)} USD articles came back - not enough to test. Nothing else to do.")
        return
    art = pd.DataFrame(feed, columns=["ts", "rel", "sc"]).sort_values("ts")
    art_t = (art["ts"] + pd.Timedelta(hours=TIME_SHIFT_H)).to_numpy().astype("datetime64[ns]")
    cw = np.concatenate([[0.0], np.cumsum((art.rel * art.sc).to_numpy())])
    cr = np.concatenate([[0.0], np.cumsum(art.rel.to_numpy())])
    days = max(1, (art.ts.max() - art.ts.min()).days)
    log(f"News | {len(art)} USD articles | {len(art) / days:.1f} per day | "
        f"sentiment avg {art.sc.mean():+.3f}, typical spread {art.sc.std():.3f}")

    data = {}
    for pair in PAIRS:
        P = prep(oanda_candles(pair, start))
        P["time"] = P["time"].astype("datetime64[ns]")
        data[pair] = P
        log(f"{pair} | done | {len(P['ao'])} candles")

    results = []
    for w in WINDOWS_H:
        scores = {p: score_series(data[p], art_t, cw, cr, w) for p in PAIRS}
        for th in THRESHOLDS:
            for follow in (True, False):
                for sm in STOPS_ATR:
                    for tg in TARGETS_R:
                        trades = []          # (entry_time, R, spread share)
                        for pair, sign in PAIRS.items():
                            P, sc = data[pair], scores[pair]
                            prev = np.concatenate([[0.0], sc[:-1]])
                            sig = np.where((sc >= th) & (prev < th), 1, np.where((sc <= -th) & (prev > -th), -1, 0))
                            free = 0
                            for i in np.flatnonzero(sig):
                                if i < free or np.isnan(P["atr"][i]) or P["atr"][i] <= 0:
                                    continue
                                d = sig[i] * sign * (1 if follow else -1)
                                out = sim(P, i, d, sm * P["atr"][i], tg)
                                if out is None:
                                    continue
                                r, exit_i, spr = out
                                free = exit_i + 1
                                trades.append((P["time"][i + 1], r, spr))
                        label = f"{'f' if follow else 'x'} {w}h t{th:.2f} s{sm:g} {tg:g}R"
                        results.append((label, trades))

    log("")
    log("===== USD NEWS SENTIMENT | M15 | results in R =====")
    log("f = follow USD news, x = fade it | Nh = news window | t = score threshold | s = stop in ATR")
    log("SEARCH (picks settings)             HOLDOUT (unseen)")
    log("label                 n  win  avgR   PF   1st   2nd |    n  avgR   PF")
    rows = []
    for label, tr in results:
        s_tr = [t for t in tr if t[0] < split]
        h_tr = [t for t in tr if t[0] >= split]
        n, win, avg, pf = stats([t[1] for t in s_tr])
        a1 = stats([t[1] for t in s_tr if t[0] < mid])[2]
        a2 = stats([t[1] for t in s_tr if t[0] >= mid])[2]
        hn, _, havg, hpf = stats([t[1] for t in h_tr])
        spr = np.mean([t[2] for t in tr]) * 100 if tr else 0.0
        rows.append((avg, label, n, win, pf, a1, a2, hn, havg, hpf, spr))
    rows.sort(reverse=True)
    for avg, label, n, win, pf, a1, a2, hn, havg, hpf, spr in rows[:12]:
        log(f"{label:<19}{n:>5} {win:>3.0f}% {avg:+.2f} {pf:>4.2f} {a1:+.2f} {a2:+.2f} |"
            f" {hn:>4} {havg:+.2f} {hpf:>4.2f}  spread {spr:.0f}% of risk")
    log(f"(top 12 of {len(rows)} by search avgR)")
    passes = [r for r in rows if r[0] > 0 and r[5] > 0 and r[6] > 0 and r[4] >= 1.2 and r[2] >= 300]
    log(f"SEARCH passes: {len(passes)} -> " + (", ".join(p[1] for p in passes) if passes else "none"))
    for p in passes:
        ok = p[8] > 0 and p[9] >= 1.2
        log(f"  {p[1]} holdout: n {p[7]}, avgR {p[8]:+.2f}, PF {p[9]:.2f} -> {'HOLDS' if ok else 'fails'}")
    log("Pass: avgR > 0 in BOTH search halves, PF >= 1.2, 300+ trades, then positive with PF >= 1.2 in HOLDOUT.")
    log("Then it goes to the demo account first. Backtest finished.")


if __name__ == "__main__":
    main()
