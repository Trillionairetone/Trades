"""
UT+ bot for OANDA  (bot.py)

Signal : UT Bot style ATR trailing stop flip on M15, H1 or H4 (TIMEFRAME variable, default M15)
Filters: ADX (no ranges), Bollinger squeeze, tick volume, higher-timeframe trend (EMA50/EMA200),
         daily support/resistance, spread vs risk
Risk   : stop = 1.5 x ATR (STOP_MODE=atr) or the UT trailing-stop distance (STOP_MODE=ut), with a minimum size
Modes  : live (default)  |  backtest  (UT baseline vs UT+ on OANDA history)
         BACKTEST_STRATEGY=srbb backtests your original bot instead (daily S/R + Bollinger bounce)
         BACKTEST_STRATEGY=flux backtests the two Flux Charts scripts (ICT Power Of Three, Volumized Order Blocks)
         BACKTEST_STRATEGY=po3s backtests the session Power Of Three (Asian range -> London fake move -> fade)
         BACKTEST_STRATEGY=po3c backtests Power Of Three + LuxAlgo Reversal Signals + Flux Consolidation Zones
         BACKTEST_STRATEGY=po3e backtests Power Of Three + LuxAlgo Echo Forecast
         BACKTEST_STRATEGY=lbo  backtests the London breakout (trade WITH the Asian-range break) vs the fade
         BACKTEST_STRATEGY=po3x tunes Power Of Three + Echo exits (stop beyond the fake move, target the range)
         BACKTEST_STRATEGY=smc  backtests YOUR strategy: H4+H1+M15 trend aligned, break of structure,
                                entry on the pullback into the fair value gap / retest, structural stop
         BACKTEST_STRATEGY=smce tests 3 entries for your strategy: limit pullback vs market at the break vs confirmation candle
        BACKTEST_STRATEGY=po3v2 tests Power Of Three + trend filter + confirmation (break of structure, market entry)
        BACKTEST_STRATEGY=po3w wide test of PO3 + Echo (12 pairs, full history, + confirmation variants)
        BACKTEST_STRATEGY=tsm  daily trend-following (time-series momentum, 3/6/12-month lookbacks, 12 pairs)
        BACKTEST_STRATEGY=scan M15 signal scanner (10 timing/level signals, search 2014-2020, holdout 2020-today)
        BACKTEST_STRATEGY=combo backtests your strategy + Power Of Three together (fake move against the
                                H4/H1 trend, then your break of structure back with it)
         BACKTEST_END_DAYS=N   ends the test window N days ago (to test an untouched past period)
Safety : live entries are skipped near high-impact news (free Forex Factory feed), in the late New York
         hours, or when one currency is already over-exposed; trades are sized by % risk (RISK_PCT)
Live   : EXIT_MODE=fixed (stop + 1:3 target, default) or EXIT_MODE=flip (stop only, the bot closes the
         trade itself when UT Bot signals the other way; only trades the bot opened are managed)
Exits  : backtest compares "flip" (exit when UT Bot signals the other way) with "sr" (same, plus a take-profit
         zone just before the next daily support/resistance level)

Run:      python bot.py
Backtest: set Railway variable MODE=backtest (optionally BACKTEST_DAYS=180 and
          BACKTEST_KEYS=1,2,3 and BACKTEST_ADX=0,15,20,25 to compare settings) and redeploy,
          or run: python bot.py backtest 180
          Test another timeframe with TIMEFRAME=H1 (or H4), e.g. BACKTEST_DAYS=365.
"""
import logging
import gc
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("bot")


# ----------------------------------------------------------------------------
# CONFIG (every value can be overridden with a Railway variable of the same name)
# ----------------------------------------------------------------------------
def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(float(os.getenv(name, default)))


PAIRS = [
    p.strip().upper()
    for p in os.getenv(
        "PAIRS",
        "EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CAD,USD_CHF,NZD_USD,EUR_GBP,EUR_JPY",
    ).split(",")
    if p.strip()
]

UNITS = _i("UNITS", 1000)                    # fixed size per trade (1000 = 0.01 lot)
MAX_OPEN_TRADES = _i("MAX_OPEN_TRADES", 20)  # one trade per pair, this caps the total

UT_KEY = _f("UT_KEY", 1.0)                   # UT Bot "key value" (sensitivity)
UT_ATR = _i("UT_ATR", 10)                    # UT Bot ATR period

ATR_PERIOD = _i("ATR_PERIOD", 14)            # ATR used for stop distance
SL_ATR_MULT = _f("SL_ATR_MULT", 1.5)         # stop = ATR x this
MIN_STOP_PIPS = _f("MIN_STOP_PIPS", 8)       # never use a stop tighter than this
RR = _f("RR", 3.0)                           # take profit = RR x risk

ADX_MIN = _f("ADX_MIN", 20)                  # below this = ranging, no trade
SQUEEZE_PCT = _f("SQUEEZE_PCT", 0.20)        # Bollinger width in bottom 20% of last 100 bars = squeeze
VOL_MIN_RATIO = _f("VOL_MIN_RATIO", 0.5)     # signal candle volume vs 20-bar average
SR_BUFFER_R = _f("SR_BUFFER_R", 1.0)         # block if opposing daily level is within this many R
MAX_SPREAD_RISK = _f("MAX_SPREAD_RISK", 0.25)  # skip if spread > this share of risk

MAX_BARS_HELD = _i("MAX_BARS_HELD", 96)      # backtest only: close fixed-target trades after 24h
FLIP_MAX_BARS = _i("FLIP_MAX_BARS", {"M15": 2880, "M30": 1440, "H1": 960, "H4": 240}.get(os.getenv("TIMEFRAME", "M15").upper(), 960))  # backtest only: ~30-40 days
SR_TP_BUFFER_ATR = _f("SR_TP_BUFFER_ATR", 0.25)  # "sr" exit: take profit this many ATR before the level (the zone edge)
SR_MIN_R = _f("SR_MIN_R", 1.0)               # "sr" exit: ignore levels closer than this many R
SR_MAX_R = _f("SR_MAX_R", 6.0)               # "sr" exit: levels farther than this are ignored (trade just rides to the flip)
STALE_SECONDS = _i("STALE_SECONDS", 300)     # ignore signals older than this (live)


TIMEFRAME = os.getenv("TIMEFRAME", "M15").upper()
TF_MIN = {"M15": 15, "M30": 30, "H1": 60, "H4": 240}            # candle length in minutes
HTF_OF = {"M15": "H1", "M30": "H4", "H1": "H4", "H4": "D"}        # trend-filter timeframe
HTF_MIN = {"H1": 60, "H4": 240, "D": 1440}
SIG_WARMUP = {"M15": 5, "M30": 8, "H1": 15, "H4": 40}          # backtest: extra days of history for indicators
HTF_WARMUP = {"M15": 25, "M30": 70, "H1": 70, "H4": 320}        # backtest: extra days for the trend EMAs
if TIMEFRAME not in TF_MIN:
    raise RuntimeError(f"TIMEFRAME must be one of {list(TF_MIN)}, got {TIMEFRAME!r}")
HTF = HTF_OF[TIMEFRAME]
EXIT_MODE = os.getenv("EXIT_MODE", "fixed").lower()  # live exits: "fixed" = stop + RR target, "flip" = stop + close on UT flip
if EXIT_MODE not in ("fixed", "flip"):
    raise RuntimeError(f"EXIT_MODE must be 'fixed' or 'flip', got {EXIT_MODE!r}")
STOP_MODE = os.getenv("STOP_MODE", "atr").lower()    # "atr": stop = SL_ATR_MULT x ATR(14); "ut": stop = the UT trailing-stop distance (UT_KEY x ATR(UT_ATR))
if STOP_MODE not in ("atr", "ut"):
    raise RuntimeError(f"STOP_MODE must be 'atr' or 'ut', got {STOP_MODE!r}")
MANAGED_TAG = "utflip"
RISK_PCT = _f("RISK_PCT", 0.5)               # % of balance risked per trade (0 = fixed UNITS instead)
MAX_UNITS = _i("MAX_UNITS", 50000)           # hard cap on trade size
MAX_PER_CURRENCY = _i("MAX_PER_CURRENCY", 2) # max net same-direction trades per currency
NEWS_FILTER = os.getenv("NEWS_FILTER", "1") == "1"
NEWS_BLOCK_MIN = _i("NEWS_BLOCK_MIN", 30)    # no new trades this many minutes either side of high-impact news
NEWS_FAIL_MODE = os.getenv("NEWS_FAIL_MODE", "allow").lower()  # if the feed is down: "allow" trading or "block" it
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NY_BLOCK = os.getenv("NY_BLOCK", "16:30-18:30")  # New York time window with no new trades ("" to turn off)                               # marks trades opened by this bot, so old/manual trades are never touched


def pip(pair):
    return 0.01 if "JPY" in pair else 0.0001


def fmt_price(pair, p):
    return f"{p:.3f}" if "JPY" in pair else f"{p:.5f}"


# ----------------------------------------------------------------------------
# OANDA API
# ----------------------------------------------------------------------------
def _first_env(names):
    for n in names:
        v = os.getenv(n)
        if v:
            return v.strip()
    return None


class Oanda:
    def __init__(self):
        token = _first_env(["OANDA_API_KEY", "OANDA_API_TOKEN", "OANDA_TOKEN", "OANDA_ACCESS_TOKEN"])
        account = _first_env(["OANDA_ACCOUNT_ID", "OANDA_ACCOUNT"])
        if not token or not account:
            raise RuntimeError(
                "Missing OANDA token or account id. Set OANDA_API_KEY and OANDA_ACCOUNT_ID in Railway variables."
            )
        env = os.getenv("OANDA_ENV", "practice").lower()
        self.base = (
            "https://api-fxtrade.oanda.com" if env == "live" else "https://api-fxpractice.oanda.com"
        )
        self.account = account
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})

    def get(self, path, params=None):
        last = None
        for attempt in range(4):
            try:
                r = self.s.get(self.base + path, params=params, timeout=30)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (401, 429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}"
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text[:300]}")
            except requests.RequestException as e:
                last = str(e)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"GET {path} failed after retries: {last}")

    def post(self, path, body):
        # no retries on purpose, so an order can never be sent twice
        r = self.s.post(self.base + path, json=body, timeout=30)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"raw": r.text[:300]}

    def put(self, path, body):
        # no retries on purpose; a failed close is simply tried again on the next scan
        r = self.s.put(self.base + path, json=body, timeout=30)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"raw": r.text[:300]}


def parse_candles(raw):
    rows = []
    for c in raw:
        if not c.get("complete", False):
            continue
        rows.append(
            {
                "time": c["time"],
                "volume": c["volume"],
                "o": float(c["mid"]["o"]), "h": float(c["mid"]["h"]),
                "l": float(c["mid"]["l"]), "c": float(c["mid"]["c"]),
                "bo": float(c["bid"]["o"]), "bh": float(c["bid"]["h"]),
                "bl": float(c["bid"]["l"]), "bc": float(c["bid"]["c"]),
                "ao": float(c["ask"]["o"]), "ah": float(c["ask"]["h"]),
                "al": float(c["ask"]["l"]), "ac": float(c["ask"]["c"]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def fetch_recent(api, pair, gran, count):
    data = api.get(
        f"/v3/instruments/{pair}/candles",
        {"granularity": gran, "price": "BAM", "count": count},
    )
    return parse_candles(data.get("candles", []))


def fetch_range(api, pair, gran, start):
    """Download candles from start to now in chunks of 5000. Each chunk is turned into a compact table right
    away (the raw JSON for years of M15 data is huge and used to run the server out of memory)."""
    parts = []
    cur = start
    while True:
        data = api.get(
            f"/v3/instruments/{pair}/candles",
            {
                "granularity": gran,
                "price": "BAM",
                "from": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "count": 5000,
            },
        )
        raw = data.get("candles", [])
        if not raw:
            break
        last_time = raw[-1]["time"]
        got = len(raw)
        df = parse_candles(raw)
        del raw, data
        if not df.empty:
            parts.append(df)
        if got < 5000:
            break
        cur = pd.Timestamp(last_time).to_pydatetime() + timedelta(seconds=1)
        time.sleep(0.2)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def get_prices(api):
    data = api.get(f"/v3/accounts/{api.account}/pricing",
                   {"instruments": ",".join(PAIRS), "includeHomeConversions": "true"})
    for hc in data.get("homeConversions", []):
        try:
            _HOME_CONV[hc["currency"]] = float(hc["accountLoss"])
        except (KeyError, ValueError):
            pass
    out = {}
    for p in data.get("prices", []):
        if not p.get("tradeable", True):
            continue
        if p.get("bids") and p.get("asks"):
            out[p["instrument"]] = (float(p["bids"][0]["price"]), float(p["asks"][0]["price"]))
    return out


# ----------------------------------------------------------------------------
# INDICATORS
# ----------------------------------------------------------------------------
def calc_atr(df, n):
    pc = df["c"].shift(1)
    tr = pd.concat(
        [df["h"] - df["l"], (df["h"] - pc).abs(), (df["l"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()  # Wilder smoothing (same as TradingView RMA)


def calc_adx(df, n=14):
    up = df["h"].diff()
    dn = -df["l"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = calc_atr(df, n)
    pdi = 100 * plus.ewm(alpha=1.0 / n, adjust=False).mean() / atr
    mdi = 100 * minus.ewm(alpha=1.0 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False).mean()


def ut_bot(df, key, period):
    """UT Bot Alerts: ATR trailing stop, buy/sell when price crosses it."""
    src = df["c"].to_numpy()
    atr = calc_atr(df, period).to_numpy()
    n = len(src)
    stop = np.zeros(n)
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)
    for i in range(1, n):
        loss = key * atr[i]
        prev = stop[i - 1]
        if src[i] > prev and src[i - 1] > prev:
            stop[i] = max(prev, src[i] - loss)
        elif src[i] < prev and src[i - 1] < prev:
            stop[i] = min(prev, src[i] + loss)
        elif src[i] > prev:
            stop[i] = src[i] - loss
        else:
            stop[i] = src[i] + loss
        buy[i] = src[i] > stop[i] and src[i - 1] <= prev
        sell[i] = src[i] < stop[i] and src[i - 1] >= prev
    return stop, buy, sell


def prepare(m15, key=None):
    d = m15.copy()
    d["ut_stop"], d["ut_buy"], d["ut_sell"] = ut_bot(d, UT_KEY if key is None else key, UT_ATR)
    d["atr"] = calc_atr(d, ATR_PERIOD)
    d["ut_loss"] = (UT_KEY if key is None else key) * calc_atr(d, UT_ATR)  # distance of the UT trailing stop from price
    d["adx"] = calc_adx(d, 14)
    mid = d["c"].rolling(20).mean()
    sd = d["c"].rolling(20).std(ddof=0)
    d["bw"] = (4 * sd) / mid
    d["bw_cut"] = d["bw"].rolling(100).quantile(SQUEEZE_PCT)
    d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()
    d["close_time"] = d["time"] + pd.Timedelta(minutes=TF_MIN[TIMEFRAME])
    return d


def add_htf_trend(d, htf_df):
    """Attach the higher-timeframe trend (1 up, -1 down, 0 unknown) using only COMPLETED candles."""
    h = htf_df.copy().reset_index(drop=True)
    h["ema50"] = h["c"].ewm(span=50, adjust=False).mean()
    h["ema200"] = h["c"].ewm(span=200, adjust=False).mean()
    h["htf"] = np.where(h["ema50"] > h["ema200"], 1, -1)
    h.loc[h.index < 200, "htf"] = 0  # not enough history yet
    h["avail"] = h["time"] + pd.Timedelta(minutes=HTF_MIN[HTF])
    m = pd.merge_asof(
        d.sort_values("close_time"),
        h[["avail", "htf"]].sort_values("avail"),
        left_on="close_time",
        right_on="avail",
        direction="backward",
    )
    m["htf"] = m["htf"].fillna(0).astype(int)
    return m.reset_index(drop=True)


_SR_CACHE = {}


def sr_levels(daily, as_of, lookback=90, cache=False):
    """Confirmed daily swing highs (resistance) and lows (support) known at time `as_of`.
    cache=True is only for backtests, where the daily frame is long-lived."""
    if daily is None or daily.empty:
        return [], []
    avail = daily["time"] + pd.Timedelta(days=1)
    cnt = int((avail <= as_of).sum())
    ck = (id(daily), cnt, lookback)
    if cache and ck in _SR_CACHE:
        return _SR_CACHE[ck]
    d = daily.iloc[:cnt].tail(lookback).reset_index(drop=True)
    h = d["h"].to_numpy()
    l = d["l"].to_numpy()
    res, sup = [], []
    for i in range(2, len(d) - 2):
        if h[i] >= h[i - 2:i + 3].max():
            res.append(h[i])
        if l[i] <= l[i - 2:i + 3].min():
            sup.append(l[i])
    if cache:
        _SR_CACHE[ck] = (res, sup)
    return res, sup


def calc_risk(pair, atr, ut_loss=None, stop_mode=None):
    mode = stop_mode or STOP_MODE
    base = ut_loss if (mode == "ut" and ut_loss is not None) else SL_ATR_MULT * atr
    return max(base, MIN_STOP_PIPS * pip(pair))


def sr_blocked(direction, price, risk, res, sup):
    if direction == 1:
        above = [x for x in res if x > price]
        return bool(above) and (min(above) - price) < SR_BUFFER_R * risk
    below = [x for x in sup if x < price]
    return bool(below) and (price - max(below)) < SR_BUFFER_R * risk


def filter_reason(row, direction, risk, price, res, sup):
    """Return the name of the first filter that blocks the trade, or None."""
    for k in ("adx", "bw", "bw_cut", "vol_ratio", "atr"):
        if pd.isna(row[k]):
            return "warmup"
    if row["adx"] < ADX_MIN:
        return "adx (ranging)"
    if row["bw"] < row["bw_cut"]:
        return "squeeze"
    if row["vol_ratio"] < VOL_MIN_RATIO:
        return "volume"
    if row["htf"] != direction:
        return "h1_trend"
    if sr_blocked(direction, price, risk, res, sup):
        return "sr_level"
    return None


def trend_label(row):
    if pd.isna(row["adx"]) or row["adx"] < ADX_MIN or row["htf"] == 0:
        return "Sideways"
    return "Bullish" if row["htf"] == 1 else "Bearish"


# ----------------------------------------------------------------------------
# BACKTEST
# ----------------------------------------------------------------------------
def simulate(d, pair, daily, start_ts, use_filters, exit_mode="fixed", stop_mode=None):
    """exit_mode "fixed": stop + 1:RR target.  "flip": stop + exit when UT Bot signals the other way."""
    times = d["time"].tolist()
    bo, bh, bl, bc = (d[c].to_numpy() for c in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[c].to_numpy() for c in ("ao", "ah", "al", "ac"))
    atr = d["atr"].to_numpy()
    utl = d["ut_loss"].to_numpy()
    ubuy = d["ut_buy"].to_numpy()
    usell = d["ut_sell"].to_numpy()
    n = len(d)
    trades, skipped = [], {}
    busy = -1
    for i in range(n - 1):
        if times[i] < start_ts or i <= busy:
            continue
        direction = 1 if ubuy[i] else (-1 if usell[i] else 0)
        if direction == 0 or np.isnan(atr[i]):
            continue
        j = i + 1  # enter at the open of the next candle
        entry = ao[j] if direction == 1 else bo[j]
        risk = calc_risk(pair, atr[i], utl[i], stop_mode)
        reason = None
        if use_filters:
            reason = filter_reason(d.iloc[i], direction, risk, entry, [], [])
            if reason is None:  # only look up S/R when the cheaper filters pass
                res, sup = sr_levels(daily, d["close_time"].iloc[i], cache=True)
                if sr_blocked(direction, entry, risk, res, sup):
                    reason = "sr_level"
        if reason is None and (ao[j] - bo[j]) / risk > MAX_SPREAD_RISK:
            reason = "spread"
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        sl = entry - direction * risk
        r = None
        flipped = False
        k = j
        if exit_mode in ("flip", "sr"):
            end = min(j + FLIP_MAX_BARS, n - 1)
            tp = None
            if exit_mode == "sr":  # take profit at the edge of the next daily S/R zone
                res_l, sup_l = sr_levels(daily, d["close_time"].iloc[i], cache=True)
                if direction == 1:
                    cands = [x for x in res_l if x > entry]
                    lvl = min(cands) if cands else None
                else:
                    cands = [x for x in sup_l if x < entry]
                    lvl = max(cands) if cands else None
                if lvl is not None:
                    tp_c = lvl - direction * SR_TP_BUFFER_ATR * atr[i]
                    reward = direction * (tp_c - entry)
                    if SR_MIN_R * risk <= reward <= SR_MAX_R * risk:
                        tp = tp_c
            for k in range(j, end + 1):
                if direction == 1:  # long exits at the bid
                    if bl[k] <= sl:
                        r = -1.0
                        break
                    if tp is not None and bh[k] >= tp:
                        r = (tp - entry) / risk
                        break
                    if usell[k] and k + 1 <= n - 1:
                        r = (bo[k + 1] - entry) / risk
                        flipped = True
                        break
                else:  # short exits at the ask
                    if ah[k] >= sl:
                        r = -1.0
                        break
                    if tp is not None and al[k] <= tp:
                        r = (entry - tp) / risk
                        break
                    if ubuy[k] and k + 1 <= n - 1:
                        r = (entry - ao[k + 1]) / risk
                        flipped = True
                        break
        else:
            tp = entry + direction * RR * risk
            end = min(j + MAX_BARS_HELD, n - 1)
            for k in range(j, end + 1):
                if direction == 1:
                    if bl[k] <= sl:
                        r = -1.0
                        break
                    if bh[k] >= tp:
                        r = RR
                        break
                else:
                    if ah[k] >= sl:
                        r = -1.0
                        break
                    if al[k] <= tp:
                        r = RR
                        break
        if r is None:
            exit_p = bc[k] if direction == 1 else ac[k]
            r = direction * (exit_p - entry) / risk
        # a flip exit happens on the signal candle, so that same signal may open the reverse trade
        busy = k - 1 if flipped else k
        exit_t = times[k + 1] if flipped else times[k]
        trades.append({"pair": pair, "entry": times[j], "exit": exit_t, "dir": direction, "r": float(r)})
    return trades, skipped


def stats(trades):
    if not trades:
        return {"n": 0, "win": 0.0, "avg": 0.0, "total": 0.0, "pf": 0.0, "dd": 0.0}
    ts = sorted(trades, key=lambda t: t["exit"])
    r = np.array([t["r"] for t in ts])
    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {
        "n": len(r),
        "win": 100.0 * (r > 0).mean(),
        "avg": float(r.mean()),
        "total": float(r.sum()),
        "pf": float(min(gp / gl, 99.0)) if gl > 0 else 99.0,
        "dd": dd,
    }


def stat_line(label, trades):
    s = stats(trades)
    return (
        f"{label:<18}{s['n']:>6}{s['win']:>8.1f}%{s['avg']:>8.2f}"
        f"{s['total']:>9.1f}{s['pf']:>7.2f}{s['dd']:>9.1f}"
    )


def run_backtest(api, days):
    global ADX_MIN
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ts = pd.Timestamp(start)
    keys = [float(x) for x in os.getenv("BACKTEST_KEYS", "").split(",") if x.strip()] or [UT_KEY]
    adxs = [float(x) for x in os.getenv("BACKTEST_ADX", "").split(",") if x.strip()] or [ADX_MIN]
    stops = [x.strip().lower() for x in os.getenv("BACKTEST_STOPS", "").split(",") if x.strip()] or [STOP_MODE]
    for s in stops:
        if s not in ("atr", "ut"):
            raise RuntimeError(f"BACKTEST_STOPS values must be atr or ut, got {s!r}")
    show_sr = os.getenv("BACKTEST_SR", "0") == "1"      # the S/R take-profit exit is hidden unless BACKTEST_SR=1
    detail = os.getenv("BACKTEST_DETAIL", "1") != "0"   # BACKTEST_DETAIL=0 prints only the short summary table
    modes = ("flip", "sr") if show_sr else ("flip",)
    single = len(keys) == 1 and len(adxs) == 1 and len(stops) == 1
    orig_adx = ADX_MIN
    log.info(
        f"BACKTEST | {TIMEFRAME} | last {days} days | {len(PAIRS)} pairs | stop {stops} | UT_KEY {keys} | "
        f"ADX_MIN {adxs} | working through the pairs..."
    )
    acc, skip_acc, per_pair = {}, {}, {}
    for pair in PAIRS:
        try:
            sig = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=SIG_WARMUP[TIMEFRAME]))
            htf = fetch_range(api, pair, HTF, start - timedelta(days=HTF_WARMUP[TIMEFRAME]))
            dly = htf if HTF == "D" else fetch_range(api, pair, "D", start - timedelta(days=120))
            if len(sig) < 300 or len(htf) < 250:
                log.info(f"{pair} | not enough history, skipped")
                continue
            for key in keys:
                d = add_htf_trend(prepare(sig, key), htf)
                for stop in stops:
                    for mode in modes:  # plain UT Bot, no filters (does not depend on ADX)
                        tr, _ = simulate(d, pair, dly, start_ts, False, mode, stop)
                        acc.setdefault((stop, key, None, mode), []).extend(tr)
                    for adx in adxs:
                        ADX_MIN = adx
                        for mode in modes:
                            tr, sk = simulate(d, pair, dly, start_ts, True, mode, stop)
                            acc.setdefault((stop, key, adx, mode), []).extend(tr)
                            if mode == "flip":
                                sa = skip_acc.setdefault((stop, key, adx), {})
                                for k2, v in sk.items():
                                    sa[k2] = sa.get(k2, 0) + v
                                per_pair.setdefault(pair, {})[(stop, key, adx)] = stats(tr)
                        ADX_MIN = orig_adx
            log.info(f"{pair} | done | {len(sig)} {TIMEFRAME} candles")
        except Exception as e:
            ADX_MIN = orig_adx
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    def halves(trades):
        ordered = sorted(trades, key=lambda t: t["entry"])
        h = len(ordered) // 2
        return ordered[:h], ordered[h:]

    if detail:
        for stop in stops:
            for key in keys:
                log.info("")
                log.info(
                    f"===== BACKTEST | {TIMEFRAME} | stop {stop} | UT_KEY {key:g} | last {days} days | "
                    f"results in R (multiples of risk) ====="
                )
                log.info(f"{'':<18}{'trades':>6}{'win%':>9}{'avgR':>8}{'totalR':>9}{'PF':>7}{'maxDD':>9}")
                for mode in modes:
                    log.info(stat_line(f"{mode} UT", acc.get((stop, key, None, mode), [])))
                for adx in adxs:
                    tag = f"ADX{adx:g}"
                    for mode in modes:
                        tr = acc.get((stop, key, adx, mode), [])
                        first, second = halves(tr)
                        log.info(stat_line(f"{tag} {mode}", tr))
                        log.info(stat_line(f"{tag} {mode} 1st", first))
                        log.info(stat_line(f"{tag} {mode} 2nd", second))
                    top = sorted(skip_acc.get((stop, key, adx), {}).items(), key=lambda x: -x[1])[:4]
                    log.info("  blocked: " + (", ".join(f"{k2} {v}" for k2, v in top) or "none"))

    log.info("")
    log.info("===== SUMMARY | flip exit | avgR = average R per trade =====")
    log.info(f"{'stop|key|adx':<11}{'n':>5}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    for stop in stops:
        for key in keys:
            for adx in [None] + adxs:
                tr = acc.get((stop, key, adx, "flip"), [])
                s = stats(tr)
                first, second = halves(tr)
                label = f"{stop}|{key:g}|" + ("none" if adx is None else f"{adx:g}")
                log.info(
                    f"{label:<11}{s['n']:>5}{s['avg']:>6.2f}{s['pf']:>5.2f}"
                    f"{stats(first)['avg']:>6.2f}{stats(second)['avg']:>6.2f}"
                )
    log.info("none = plain UT Bot, no filters. 1st/2nd = older/newer half of the period.")

    if single:
        log.info("")
        log.info(f"{'pair':<10}{'flip UT+ R':>13}{'flip UT+ n':>12}")
        for pair, row in per_pair.items():
            r = row.get((stops[0], keys[0], adxs[0]))
            if r:
                log.info(f"{pair:<10}{r['total']:>13.1f}{r['n']:>12}")

    log.info("")
    log.info("Need: avgR above 0 in BOTH halves, PF above 1.2, a few hundred trades, and neighbouring settings that look similar.")
    log.info("Backtests ignore slippage, swap fees and news gaps. Treat them as a sanity check, not a promise.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: the original bot (daily S/R + Bollinger bounce, candle + volume, 1:3)
# Same rules as the old bot.py, so its results can be judged like everything else.
# ----------------------------------------------------------------------------
OLD_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD", "EUR_JPY", "GBP_JPY"]
SB_ZONE_ATR = _f("SB_ZONE_ATR", 1.0)            # "near" a daily level = within this many ATR
SB_SL_PAD_ATR = _f("SB_SL_PAD_ATR", 0.3)        # stop = 2-candle extreme +/- this many ATR
SB_MAX_RISK_ATR = _f("SB_MAX_RISK_ATR", 2.5)    # skip if the stop is wider than this many ATR
SB_MAX_SPREAD_FRAC = _f("SB_MAX_SPREAD_FRAC", 0.2)  # skip if spread > this share of the risk
SB_VOL_FACTOR = _f("SB_VOL_FACTOR", 0.8)        # candle volume must beat this share of the 20-candle average
SB_RR = _f("SB_RR", 3.0)                        # take profit = this many R
SB_MAX_BARS = _i("SB_MAX_BARS", 2880)           # backtest only: close a trade still open after this many candles


def prepare_srbb(sig, dly):
    d = sig.copy()
    o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["volume"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    a = tr.rolling(14).mean()                    # old bot: simple 14-candle ATR
    d["sa"] = a
    e50 = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    e50_prev = e50.shift(8)
    bull = (c > e50) & (e50 > e200) & (e50 > e50_prev)
    bear = (c < e50) & (e50 < e200) & (e50 < e50_prev)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    lower, upper = mid - 2 * sd, mid + 2 * sd
    vol_ok = v > SB_VOL_FACTOR * v.shift(1).rolling(20).mean()
    po, pcl, ph, pl = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    body = (c - o).abs()
    rng = (h - l).where(h - l > 0, 1e-9)
    low_wick = np.minimum(o, c) - l
    up_wick = h - np.maximum(o, c)
    bull_c = ((c > o) & (pcl < po) & (c >= po) & (o <= pcl)) | ((c > o) & (low_wick >= 2 * body) & (body / rng > 0.1))
    bear_c = ((c < o) & (pcl > po) & (c <= po) & (o >= pcl)) | ((c < o) & (up_wick >= 2 * body) & (body / rng > 0.1))

    # daily levels known when each candle closes: previous day's high/low and the 10-day extremes
    dd = dly[["time", "h", "l"]].copy().reset_index(drop=True)
    dd["s1"], dd["r1"] = dd["l"], dd["h"]
    dd["s10"], dd["r10"] = dd["l"].rolling(10).min(), dd["h"].rolling(10).max()
    dd["avail"] = dd["time"] + pd.Timedelta(days=1)
    d["close_time"] = d["time"] + pd.Timedelta(minutes=TF_MIN[TIMEFRAME])
    m = pd.merge_asof(d.sort_values("close_time"), dd[["avail", "s1", "s10", "r1", "r10"]].sort_values("avail"),
                      left_on="close_time", right_on="avail", direction="backward")
    m.index = d.index
    zone = SB_ZONE_ATR * a
    near_sup = ((l - m["s1"]).abs() <= zone) | (l < m["s1"]) | ((l - m["s10"]).abs() <= zone) | (l < m["s10"])
    near_res = ((h - m["r1"]).abs() <= zone) | (h > m["r1"]) | ((h - m["r10"]).abs() <= zone) | (h > m["r10"])
    touched_lo = np.minimum(l, pl) <= lower + 0.2 * a
    touched_hi = np.maximum(h, ph) >= upper - 0.2 * a
    level_b = (near_sup | touched_lo) & (c > lower)
    level_s = (near_res | touched_hi) & (c < upper)
    d["sb_buy"] = (bull & vol_ok & bull_c & level_b).fillna(False)
    d["sb_sell"] = (bear & vol_ok & bear_c & level_s).fillna(False)
    d["sb_sl_buy"] = np.minimum(l, pl) - SB_SL_PAD_ATR * a
    d["sb_sl_sell"] = np.maximum(h, ph) + SB_SL_PAD_ATR * a
    return d


def simulate_srbb(d, pair, start_ts):
    times = d["time"].tolist()
    bo, bh, bl, bc = (d[k].to_numpy() for k in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[k].to_numpy() for k in ("ao", "ah", "al", "ac"))
    a = d["sa"].to_numpy()
    buy, sell = d["sb_buy"].to_numpy(dtype=bool), d["sb_sell"].to_numpy(dtype=bool)
    slb, sls = d["sb_sl_buy"].to_numpy(), d["sb_sl_sell"].to_numpy()
    n = len(d)
    trades, skipped = [], {}
    busy = -1
    for i in range(n - 1):
        if times[i] < start_ts or i <= busy:
            continue
        direction = 1 if buy[i] else (-1 if sell[i] else 0)
        if direction == 0 or np.isnan(a[i]):
            continue
        j = i + 1
        entry = ao[j] if direction == 1 else bo[j]
        sl = slb[i] if direction == 1 else sls[i]
        risk = direction * (entry - sl)
        spread = ao[j] - bo[j]
        reason = None
        if risk <= 0:
            reason = "price beyond stop"
        elif risk > SB_MAX_RISK_ATR * a[i]:
            reason = "stop too wide"
        elif spread > SB_MAX_SPREAD_FRAC * risk:
            reason = "spread too wide"
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        tp = entry + direction * SB_RR * risk
        end = min(j + SB_MAX_BARS, n - 1)
        r = None
        k = j
        for k in range(j, end + 1):
            if direction == 1:
                if bl[k] <= sl:
                    r = -1.0
                    break
                if bh[k] >= tp:
                    r = SB_RR
                    break
            else:
                if ah[k] >= sl:
                    r = -1.0
                    break
                if al[k] <= tp:
                    r = SB_RR
                    break
        if r is None:
            exit_p = bc[k] if direction == 1 else ac[k]
            r = direction * (exit_p - entry) / risk
        busy = k
        trades.append({"pair": pair, "entry": times[j], "exit": times[k], "dir": direction, "r": float(r),
                       "risk_pips": risk / pip(pair), "spread_frac": spread / risk})
    return trades, skipped


def run_backtest_srbb(api, days):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ts = pd.Timestamp(start)
    pairs = PAIRS if os.getenv("PAIRS") else OLD_PAIRS
    log.info(f"BACKTEST | original bot (S/R + Bollinger bounce, 1:{SB_RR:g}) | {TIMEFRAME} | last {days} days | {len(pairs)} pairs")
    all_tr, skips, rows = [], {}, []
    for pair in pairs:
        try:
            sig = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=SIG_WARMUP[TIMEFRAME]))
            dly = fetch_range(api, pair, "D", start - timedelta(days=40))
            if len(sig) < 300 or len(dly) < 15:
                log.info(f"{pair} | not enough history, skipped")
                continue
            tr, sk = simulate_srbb(prepare_srbb(sig, dly), pair, start_ts)
            all_tr += tr
            for k2, v in sk.items():
                skips[k2] = skips.get(k2, 0) + v
            rows.append((pair, stats(tr)))
            log.info(f"{pair} | done | {len(tr)} trades")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    ordered = sorted(all_tr, key=lambda t: t["entry"])
    half = len(ordered) // 2
    log.info("")
    log.info(f"===== ORIGINAL BOT | {TIMEFRAME} | last {days} days | results in R =====")
    log.info(f"{'':<18}{'trades':>6}{'win%':>9}{'avgR':>8}{'totalR':>9}{'PF':>7}{'maxDD':>9}")
    log.info(stat_line("all trades", all_tr))
    log.info(stat_line("older half", ordered[:half]))
    log.info(stat_line("newer half", ordered[half:]))
    log.info("Skipped at entry: " + (", ".join(f"{k2} {v}" for k2, v in sorted(skips.items(), key=lambda x: -x[1])) or "none"))
    if all_tr:
        avg_pips = float(np.mean([t["risk_pips"] for t in all_tr]))
        avg_spread = float(np.mean([t["spread_frac"] for t in all_tr]))
        log.info(f"Average stop {avg_pips:.1f} pips | spread paid = {avg_spread * 100:.0f}% of the risk on average")
    log.info("")
    log.info(f"{'pair':<9}{'n':>5}{'win%':>7}{'totalR':>8}")
    for pair, s in rows:
        log.info(f"{pair:<9}{s['n']:>5}{s['win']:>6.1f}%{s['total']:>8.1f}")
    log.info("")
    log.info(f"Break-even win rate at 1:{SB_RR:g} is {100.0 / (1.0 + SB_RR):.0f}%. Need avgR above 0 in BOTH halves and PF above 1.2.")
    log.info("Not modelled: the old 5-trade cap and per-currency limit (this tests the signal itself), slippage, swap fees.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: Flux Charts scripts, ported from their Pine Script
#   PO3: "ICT Power Of Three | Flux Charts" with its default settings
#   OB : "Volumized Order Blocks | Flux Charts" zones, traded on the first re-entry into a zone
# ----------------------------------------------------------------------------
PO3_PRESETS = {  # algorithm mode -> (accumulation length, max accumulation size in ATR, min manipulation size in ATR)
    "small": (40, 5.0, 0.6),   # "Small Manipulation" (the script's default, and what your chart uses)
    "short": (11, 2.0, 1.0),   # "Short Accumulation"
    "big": (11, 5.0, 1.0),     # "Big Manipulation"
}
PO3_MODE = os.getenv("PO3_MODE", "small").lower()
PO3_SL_ATR = _f("PO3_SL_ATR", 5.0)   # risk "Normal" = 5 x ATR(50)
PO3_RR = _f("PO3_RR", 0.86)          # the script's fixed reward:risk
OB_SWING_LEN = _i("OB_SWING_LEN", 10)
OB_ZONES = _i("OB_ZONES", 3)          # zone count "Low" = the 3 newest zones of each type
OB_SL_PAD_ATR = _f("OB_SL_PAD_ATR", 0.1)
OB_RRS = [float(x) for x in os.getenv("OB_RRS", "1,2,3").split(",") if x.strip()]


def simulate_po3(d, pair, start_ts, tr1=None, tr4=None):
    acc_len, acc_mult, manip_mult = PO3_PRESETS.get(PO3_MODE, PO3_PRESETS["small"])
    t = d["time"].tolist()
    h, l = d["h"].to_numpy(), d["l"].to_numpy()
    bh, bl, bc = d["bh"].to_numpy(), d["bl"].to_numpy(), d["bc"].to_numpy()
    ao, ah, al, ac = d["ao"].to_numpy(), d["ah"].to_numpy(), d["al"].to_numpy(), d["ac"].to_numpy()
    bo = d["bo"].to_numpy()
    atr = calc_atr(d, 50).to_numpy()
    hh = d["h"].rolling(acc_len).max().to_numpy()
    ll = d["l"].rolling(acc_len).min().to_numpy()
    n = len(d)
    trades = []
    state, i = 0, 0
    acc_top = acc_bot = None
    acc_end = 0
    while i < n - 1:
        a = atr[i]
        if np.isnan(hh[i]) or np.isnan(a):
            i += 1
            continue
        if state == 0 and hh[i] - ll[i] <= acc_mult * a:          # accumulation found
            state, acc_end = 1, i
            acc_top, acc_bot = hh[i] + 0.5 * a, ll[i] - 0.5 * a
        if state == 1:
            if h[i] > acc_top or l[i] < acc_bot:                   # accumulation ends
                state = 2
            else:
                acc_end = i
        direction = 0
        if state == 2 and i > acc_end:                              # manipulation -> trade against it
            if h[i] > acc_top + manip_mult * a:
                direction = -1
            elif l[i] < acc_bot - manip_mult * a:
                direction = 1
        if direction != 0 and tr1 is not None and not (tr1[i] == direction and tr4[i] == direction):
            state = 0                       # trend filter: the fake move did not go against the H4+H1 trend
            direction = 0
        if direction == 0:
            i += 1
            continue
        j = i + 1
        entry = ao[j] if direction == 1 else bo[j]
        risk = PO3_SL_ATR * a
        sl = entry - direction * risk
        tp = entry + direction * PO3_RR * risk
        end = min(j + SB_MAX_BARS, n - 1)
        r, k = None, j
        for k in range(j, end + 1):
            if direction == 1:
                if bl[k] <= sl:
                    r = -1.0
                    break
                if bh[k] >= tp:
                    r = PO3_RR
                    break
            else:
                if ah[k] >= sl:
                    r = -1.0
                    break
                if al[k] <= tp:
                    r = PO3_RR
                    break
        if r is None:
            exit_p = bc[k] if direction == 1 else ac[k]
            r = direction * (exit_p - entry) / risk
        if t[j] >= start_ts:
            trades.append({"pair": pair, "entry": t[j], "exit": t[k], "dir": direction, "r": float(r),
                           "risk_pips": risk / pip(pair), "spread_frac": (ao[j] - bo[j]) / risk})
        state = 0
        i = k + 1   # the script only looks for a new setup after the trade has closed
    return trades


def simulate_ob(d, pair, start_ts):
    t = d["time"].tolist()
    h, l, c = d["h"].to_numpy(), d["l"].to_numpy(), d["c"].to_numpy()
    bo, bh, bl, bc = (d[k].to_numpy() for k in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[k].to_numpy() for k in ("ao", "ah", "al", "ac"))
    atr = calc_atr(d, 10).to_numpy()
    L = OB_SWING_LEN
    upper = d["h"].rolling(L).max().to_numpy()
    lower = d["l"].rolling(L).min().to_numpy()
    n = len(d)
    swing_type = 0
    top = btm = None            # [bar index, price, crossed]
    bull, bear = [], []         # newest first, like the script
    pos = {rr: None for rr in OB_RRS}
    trades = {rr: [] for rr in OB_RRS}
    skipped = {rr: 0 for rr in OB_RRS}

    def close_trade(rr, p, i, r):
        if p["record"]:
            trades[rr].append({"pair": pair, "entry": p["t"], "exit": t[i], "dir": p["dir"], "r": float(r),
                               "risk_pips": p["risk"] / pip(pair), "spread_frac": p["spread"] / p["risk"]})

    for i in range(n):
        # 1) manage open trades (one per pair per R:R setting)
        exited = set()
        for rr in OB_RRS:
            p = pos[rr]
            if p is None or i <= p["fill"]:
                continue
            r = None
            if p["dir"] == 1:
                if bl[i] <= p["sl"]:
                    r = -1.0
                elif bh[i] >= p["tp"]:
                    r = rr
            else:
                if ah[i] >= p["sl"]:
                    r = -1.0
                elif al[i] <= p["tp"]:
                    r = rr
            if r is None and i - p["fill"] >= SB_MAX_BARS:
                exit_p = bc[i] if p["dir"] == 1 else ac[i]
                r = p["dir"] * (exit_p - p["entry"]) / p["risk"]
            if r is not None:
                close_trade(rr, p, i, r)
                pos[rr] = None
                exited.add(rr)

        # 2) limit entry on the first re-entry into a live zone (zones known before this candle)
        if i > 0 and not np.isnan(atr[i - 1]):
            spread = ao[i] - bo[i]
            for rr in OB_RRS:
                if pos[rr] is not None or rr in exited:
                    continue
                done = False
                for side, zones in ((1, bull), (-1, bear)):
                    for z in zones[:OB_ZONES]:
                        if z["breaker"] or rr in z["traded"] or z["created"] >= i:
                            continue
                        if side == 1 and al[i] <= z["top"]:
                            entry = min(z["top"], ao[i])
                            sl = z["bottom"] - OB_SL_PAD_ATR * atr[i - 1]
                        elif side == -1 and bh[i] >= z["bottom"]:
                            entry = max(z["bottom"], bo[i])
                            sl = z["top"] + OB_SL_PAD_ATR * atr[i - 1]
                        else:
                            continue
                        z["traded"].add(rr)
                        risk = side * (entry - sl)
                        if risk <= 0 or spread > MAX_SPREAD_RISK * risk:
                            skipped[rr] += 1
                            continue
                        p = {"dir": side, "entry": entry, "sl": sl, "tp": entry + side * rr * risk, "risk": risk,
                             "fill": i, "t": t[i], "spread": spread, "record": t[i] >= start_ts}
                        stopped = (bl[i] <= sl) if side == 1 else (ah[i] >= sl)
                        if stopped:   # stopped out on the same candle it filled (assume the worst)
                            close_trade(rr, p, i, -1.0)
                        else:
                            pos[rr] = p
                        done = True
                        break
                    if done:
                        break

        # 3) the script's zone upkeep: invalidate on a wick through the zone, drop once fully reclaimed
        for z in list(bull):
            if not z["breaker"]:
                if l[i] < z["bottom"]:
                    z["breaker"] = True
            elif h[i] > z["top"]:
                bull.remove(z)
        for z in list(bear):
            if not z["breaker"]:
                if h[i] > z["top"]:
                    z["breaker"] = True
            elif l[i] < z["bottom"]:
                bear.remove(z)

        # 4) swings (confirmed OB_SWING_LEN candles later) and new zones on a close through a swing
        if i >= L:
            if h[i - L] > upper[i]:
                st = 0
            elif l[i - L] < lower[i]:
                st = 1
            else:
                st = swing_type
            if st == 0 and swing_type != 0:
                top = [i - L, h[i - L], False]
            if st == 1 and swing_type != 1:
                btm = [i - L, l[i - L], False]
            swing_type = st
        if top is not None and c[i] > top[1] and not top[2]:
            top[2] = True
            box_btm, box_top = l[i - 1], h[i - 1]
            for k in range(1, i - top[0]):
                if l[i - k] <= box_btm:
                    box_btm, box_top = l[i - k], h[i - k]
            if box_top - box_btm <= atr[i] * 3.5:
                bull.insert(0, {"top": box_top, "bottom": box_btm, "breaker": False, "traded": set(), "created": i})
                del bull[30:]
        if btm is not None and c[i] < btm[1] and not btm[2]:
            btm[2] = True
            box_btm, box_top = l[i - 1], h[i - 1]
            for k in range(1, i - btm[0]):
                if h[i - k] >= box_top:
                    box_top, box_btm = h[i - k], l[i - k]
            if box_top - box_btm <= atr[i] * 3.5:
                bear.insert(0, {"top": box_top, "bottom": box_btm, "breaker": False, "traded": set(), "created": i})
                del bear[30:]

    for rr in OB_RRS:   # anything still open at the end is closed at the last price
        p = pos[rr]
        if p is not None:
            exit_p = bc[n - 1] if p["dir"] == 1 else ac[n - 1]
            close_trade(rr, p, n - 1, p["dir"] * (exit_p - p["entry"]) / p["risk"])
    return trades, skipped


def run_backtest_flux(api, days):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ts = pd.Timestamp(start)
    log.info(f"BACKTEST | Flux Charts: Power Of Three ({PO3_MODE}) + Order Blocks (re-entry, R:R {OB_RRS}) | "
             f"{TIMEFRAME} | last {days} days | {len(PAIRS)} pairs")
    po3_all, ob_all, ob_skip = [], {rr: [] for rr in OB_RRS}, {rr: 0 for rr in OB_RRS}
    for pair in PAIRS:
        try:
            sig = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=10))
            if len(sig) < 300:
                log.info(f"{pair} | not enough history, skipped")
                continue
            p3 = simulate_po3(sig, pair, start_ts)
            obt, obs = simulate_ob(sig, pair, start_ts)
            po3_all += p3
            for rr in OB_RRS:
                ob_all[rr] += obt[rr]
                ob_skip[rr] += obs[rr]
            log.info(f"{pair} | done | PO3 {len(p3)} trades | OB {len(obt[OB_RRS[0]])} trades")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    def line(label, tr):
        s = stats(tr)
        ordered = sorted(tr, key=lambda x: x["entry"])
        h2 = len(ordered) // 2
        a1, a2 = stats(ordered[:h2])["avg"], stats(ordered[h2:])["avg"]
        return f"{label:<7}{s['n']:>5}{s['win']:>5.0f}%{s['avg']:>6.2f}{s['pf']:>5.2f}{a1:>6.2f}{a2:>6.2f}"

    log.info("")
    log.info(f"===== FLUX CHARTS | {TIMEFRAME} | last {days} days | results in R =====")
    log.info(f"{'':<7}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    log.info(line("PO3", po3_all))
    for rr in OB_RRS:
        log.info(line(f"OB 1:{rr:g}", ob_all[rr]))
    log.info("1st/2nd = average R in the older/newer half of the period.")
    if po3_all:
        log.info(f"PO3: average stop {np.mean([x['risk_pips'] for x in po3_all]):.1f} pips, spread "
                 f"{np.mean([x['spread_frac'] for x in po3_all]) * 100:.0f}% of risk. "
                 f"It targets {PO3_RR:g}R per 1R risked, so it needs {100 / (1 + PO3_RR):.0f}%+ wins to break even.")
    first = ob_all[OB_RRS[0]]
    if first:
        log.info(f"OB: average stop {np.mean([x['risk_pips'] for x in first]):.1f} pips, spread "
                 f"{np.mean([x['spread_frac'] for x in first]) * 100:.0f}% of risk. "
                 f"Skipped (spread over {MAX_SPREAD_RISK * 100:.0f}% of risk): {ob_skip[OB_RRS[0]]}.")
    log.info("Need: avgR above 0 in BOTH halves, PF above 1.2, a few hundred trades.")
    log.info("Not modelled: slippage, swap fees, news gaps.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: session Power Of Three (rules fixed before testing)
#   accumulation = the Asian range, 00:00-07:00 London time
#   manipulation = price pushes beyond that range (+0.5 ATR, then +0.6 ATR more, as in the Flux script)
#                  between 07:00 and 10:00 London time (the London open)
#   entry        = fade it at the next candle; stop 5 x ATR(50), target 0.86R (unchanged from the Flux script)
#   variant "+PD"= the fake move must also sweep yesterday's high (for a sell) or low (for a buy)
#   costs        = only the tightest-spread pairs, and no entries outside the London open
# ----------------------------------------------------------------------------
PO3S_PAIRS = [p.strip().upper() for p in os.getenv("PO3S_PAIRS", "EUR_USD,USD_JPY,GBP_USD").split(",") if p.strip()]
PO3S_ASIA_END = 7 * 60      # minutes after midnight, London time
PO3S_WINDOW_END = 10 * 60


def _fixed_exit(direction, j, entry, sl, tp, rr, risk, bh, bl, bc, ah, al, ac, n):
    end = min(j + SB_MAX_BARS, n - 1)
    k = j
    for k in range(j, end + 1):
        if direction == 1:
            if bl[k] <= sl:
                return -1.0, k
            if bh[k] >= tp:
                return rr, k
        else:
            if ah[k] >= sl:
                return -1.0, k
            if al[k] <= tp:
                return rr, k
    exit_p = bc[k] if direction == 1 else ac[k]
    return direction * (exit_p - entry) / risk, k


def simulate_po3_session(d, dly, pair, start_ts, need_sweep, follow=False):
    t = d["time"].tolist()
    h, l = d["h"].to_numpy(), d["l"].to_numpy()
    bo, bh, bl, bc = (d[k].to_numpy() for k in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[k].to_numpy() for k in ("ao", "ah", "al", "ac"))
    atr = calc_atr(d, 50).to_numpy()
    lt = d["time"].dt.tz_convert("Europe/London")
    lday = lt.dt.date.to_numpy()
    lmin = (lt.dt.hour * 60 + lt.dt.minute).to_numpy()
    n = len(d)
    pdh = pdl = None
    if need_sweep:  # yesterday's high/low, known when each candle closes
        dd = dly[["time", "h", "l"]].copy().reset_index(drop=True)
        dd["avail"] = dd["time"] + pd.Timedelta(days=1)
        ct = (d["time"] + pd.Timedelta(minutes=TF_MIN[TIMEFRAME])).to_frame("ct")
        m = pd.merge_asof(ct.sort_values("ct"), dd[["avail", "h", "l"]].sort_values("avail"),
                          left_on="ct", right_on="avail", direction="backward")
        pdh, pdl = m["h"].to_numpy(), m["l"].to_numpy()
    _, acc_mult, manip_mult = PO3_PRESETS["small"]
    trades = []
    busy_until = -1
    cur_day = None
    asia_hi = asia_lo = None
    asia_n = 0
    done_today = False
    acc_top = acc_bot = None
    for i in range(n - 1):
        if lday[i] != cur_day:
            cur_day, asia_hi, asia_lo, asia_n, done_today, acc_top = lday[i], -np.inf, np.inf, 0, False, None
        if lmin[i] < PO3S_ASIA_END:
            asia_hi, asia_lo, asia_n = max(asia_hi, h[i]), min(asia_lo, l[i]), asia_n + 1
            continue
        if lmin[i] >= PO3S_WINDOW_END or done_today or asia_n < 20 or np.isnan(atr[i]):
            continue
        if acc_top is None:  # set the accumulation box once, at the London open
            a0 = atr[i - 1]
            if np.isnan(a0) or asia_hi - asia_lo > acc_mult * a0:
                done_today = True
                continue
            acc_top, acc_bot = asia_hi + 0.5 * a0, asia_lo - 0.5 * a0
        if i <= busy_until:
            continue
        direction = 0
        if h[i] > acc_top + manip_mult * atr[i] and (pdh is None or h[i] > pdh[i]):
            direction = -1
        elif l[i] < acc_bot - manip_mult * atr[i] and (pdl is None or l[i] < pdl[i]):
            direction = 1
        if direction == 0:
            continue
        if follow:              # London breakout: go WITH the break instead of fading it
            direction = -direction
        done_today = True
        j = i + 1
        entry = ao[j] if direction == 1 else bo[j]
        risk = PO3_SL_ATR * atr[i]
        sl, tp = entry - direction * risk, entry + direction * PO3_RR * risk
        r, k = _fixed_exit(direction, j, entry, sl, tp, PO3_RR, risk, bh, bl, bc, ah, al, ac, n)
        busy_until = k
        if t[j] >= start_ts:
            trades.append({"pair": pair, "entry": t[j], "exit": t[k], "dir": direction, "r": float(r),
                           "risk_pips": risk / pip(pair), "spread_frac": (ao[j] - bo[j]) / risk})
    return trades


def _summary_line(label, tr):
    s = stats(tr)
    ordered = sorted(tr, key=lambda x: x["entry"])
    h2 = len(ordered) // 2
    a1, a2 = stats(ordered[:h2])["avg"], stats(ordered[h2:])["avg"]
    return f"{label:<8}{s['n']:>5}{s['win']:>5.0f}%{s['avg']:>6.2f}{s['pf']:>5.2f}{a1:>6.2f}{a2:>6.2f}"


def run_backtest_po3s(api, days):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ts = pd.Timestamp(start)
    log.info(f"BACKTEST | session Power Of Three | {TIMEFRAME} | last {days} days | pairs {PO3S_PAIRS}")
    ref, sess, sweep = [], [], []
    per_pair = []
    for pair in PO3S_PAIRS:
        try:
            sig = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=10))
            dly = fetch_range(api, pair, "D", start - timedelta(days=20))
            if len(sig) < 300:
                log.info(f"{pair} | not enough history, skipped")
                continue
            r0 = simulate_po3(sig, pair, start_ts)
            r1 = simulate_po3_session(sig, dly, pair, start_ts, need_sweep=False)
            r2 = simulate_po3_session(sig, dly, pair, start_ts, need_sweep=True)
            ref += r0
            sess += r1
            sweep += r2
            per_pair.append((pair, stats(r1), stats(r2)))
            log.info(f"{pair} | done | session {len(r1)} trades | +PD {len(r2)} trades")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    log.info("")
    log.info(f"===== SESSION PO3 | {TIMEFRAME} | last {days} days | results in R =====")
    log.info(f"{'':<8}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    log.info(_summary_line("Flux ref", ref))
    log.info(_summary_line("Session", sess))
    log.info(_summary_line("Sess+PD", sweep))
    log.info("Flux ref = the original script on the same pairs, any hour (for comparison only).")
    log.info("1st/2nd = average R in the older/newer half of the period.")
    log.info("")
    log.info(f"{'pair':<9}{'Sess n':>7}{'Sess R':>8}{'+PD n':>7}{'+PD R':>8}")
    for pair, s1, s2 in per_pair:
        log.info(f"{pair:<9}{s1['n']:>7}{s1['total']:>8.1f}{s2['n']:>7}{s2['total']:>8.1f}")
    if sess:
        log.info(f"Session: average stop {np.mean([x['risk_pips'] for x in sess]):.1f} pips, spread "
                 f"{np.mean([x['spread_frac'] for x in sess]) * 100:.0f}% of risk.")
    log.info(f"Targets {PO3_RR:g}R per 1R risked, so it needs {100 / (1 + PO3_RR):.0f}%+ wins to break even.")
    log.info("Need: avgR above 0 in BOTH halves, PF above 1.2, a few hundred trades.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: Power Of Three combined with two more scripts (rules fixed before testing)
#   TD   = LuxAlgo Reversal Signals "momentum phase": 9 closes in a row below (or above) the close 4 candles back
#          (DeMark's TD Sequential setup). Only fade the fake move once the move shows this exhaustion count:
#          a completed 9 in the fake move's direction within 4 candles either side of it, else skip the setup.
#   Zone = Flux Consolidation Zones (candle mode): 4+ candles in a row with small bodies and a range under ATR(4).
#          Only trade if the accumulation actually contained such a tight zone.
#   Exits, pairs and everything else are unchanged from the Flux Power Of Three (stop 5 x ATR, target 0.86R).
# ----------------------------------------------------------------------------
PO3C_TD_WINDOW = _i("PO3C_TD_WINDOW", 4)
PO3C_ZONE_MIN = _i("PO3C_ZONE_MIN", 4)


def td_counts(c):
    """LuxAlgo momentum-phase counters: buy count (close < close 4 back) and sell count, each resetting after 9."""
    n = len(c)
    bsc = np.zeros(n, dtype=int)
    ssc = np.zeros(n, dtype=int)
    b = s = 0
    for i in range(4, n):
        if c[i] < c[i - 4]:
            b = 1 if b == 9 else b + 1
            s = 0
        else:
            s = 1 if s == 9 else s + 1
            b = 0
        bsc[i], ssc[i] = b, s
    return bsc, ssc


def consolidation_runs(d):
    """Flux 'Candles' consolidation: length of the current run of small-body candles with range under ATR(4)."""
    o, h, l, c = (d[k].to_numpy() for k in ("o", "h", "l", "c"))
    atr4 = calc_atr(d, 4).to_numpy()
    tight = (np.abs(c - o) < (h - l) * 0.5) & ((h - l) < atr4)
    runs = np.zeros(len(d), dtype=int)
    r = 0
    for i in range(len(d)):
        r = r + 1 if tight[i] else 0
        runs[i] = r
    return runs


def simulate_po3_combo(d, pair, start_ts, use_td, use_zone):
    acc_len, acc_mult, manip_mult = PO3_PRESETS.get(PO3_MODE, PO3_PRESETS["small"])
    t = d["time"].tolist()
    h, l, c = d["h"].to_numpy(), d["l"].to_numpy(), d["c"].to_numpy()
    bo, bh, bl, bc = (d[k].to_numpy() for k in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[k].to_numpy() for k in ("ao", "ah", "al", "ac"))
    atr = calc_atr(d, 50).to_numpy()
    hh = d["h"].rolling(acc_len).max().to_numpy()
    ll = d["l"].rolling(acc_len).min().to_numpy()
    bsc, ssc = td_counts(c)
    runs = consolidation_runs(d)
    n = len(d)
    trades, rejected = [], {"td": 0, "zone": 0}
    state, i = 0, 0
    acc_top = acc_bot = None
    acc_end = acc_start = 0
    W = PO3C_TD_WINDOW
    while i < n - 1:
        a = atr[i]
        if np.isnan(hh[i]) or np.isnan(a):
            i += 1
            continue
        if state == 0 and hh[i] - ll[i] <= acc_mult * a:
            state, acc_end, acc_start = 1, i, max(0, i - acc_len + 1)
            acc_top, acc_bot = hh[i] + 0.5 * a, ll[i] - 0.5 * a
        if state == 1:
            if h[i] > acc_top or l[i] < acc_bot:
                state = 2
            else:
                acc_end = i
        direction = 0
        if state == 2 and i > acc_end:
            if h[i] > acc_top + manip_mult * a:
                direction = -1
            elif l[i] < acc_bot - manip_mult * a:
                direction = 1
        if direction == 0:
            i += 1
            continue
        state = 0
        if use_zone and runs[acc_start:i + 1].max() < PO3C_ZONE_MIN:
            rejected["zone"] += 1
            i += 1
            continue
        signal_bar = i
        if use_td:
            # fading a drop needs a completed BUY count (9 closes lower); fading a spike needs a SELL count
            count = bsc if direction == 1 else ssc
            found = None
            for m in range(max(0, i - W), min(i + W, n - 2) + 1):
                if count[m] == 9:
                    found = m
                    break
            if found is None:
                rejected["td"] += 1
                i += W + 1          # we would have waited W candles for it
                continue
            signal_bar = max(i, found)
        j = signal_bar + 1
        entry = ao[j] if direction == 1 else bo[j]
        risk = PO3_SL_ATR * atr[signal_bar]
        sl, tp = entry - direction * risk, entry + direction * PO3_RR * risk
        r, k = _fixed_exit(direction, j, entry, sl, tp, PO3_RR, risk, bh, bl, bc, ah, al, ac, n)
        if t[j] >= start_ts:
            trades.append({"pair": pair, "entry": t[j], "exit": t[k], "dir": direction, "r": float(r),
                           "risk_pips": risk / pip(pair), "spread_frac": (ao[j] - bo[j]) / risk})
        i = k + 1
    return trades, rejected


def run_backtest_po3c(api, days):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ts = pd.Timestamp(start)
    log.info(f"BACKTEST | Power Of Three + Reversal Signals (TD 9) + Consolidation Zones | {TIMEFRAME} | "
             f"last {days} days | pairs {PO3S_PAIRS}")
    rows = {"Flux ref": [], "+TD": [], "+Zone": [], "All 3": []}
    rej = {"td": 0, "zone": 0}
    for pair in PO3S_PAIRS:
        try:
            sig = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=10))
            if len(sig) < 300:
                log.info(f"{pair} | not enough history, skipped")
                continue
            rows["Flux ref"] += simulate_po3(sig, pair, start_ts)
            tr, _ = simulate_po3_combo(sig, pair, start_ts, use_td=True, use_zone=False)
            rows["+TD"] += tr
            tr, _ = simulate_po3_combo(sig, pair, start_ts, use_td=False, use_zone=True)
            rows["+Zone"] += tr
            tr, rj = simulate_po3_combo(sig, pair, start_ts, use_td=True, use_zone=True)
            rows["All 3"] += tr
            for k2 in rej:
                rej[k2] += rj[k2]
            log.info(f"{pair} | done | All 3: {len(tr)} trades")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    log.info("")
    log.info(f"===== PO3 COMBO | {TIMEFRAME} | last {days} days | results in R =====")
    log.info(f"{'':<8}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    for label, tr in rows.items():
        log.info(_summary_line(label, tr))
    log.info("Flux ref = plain Power Of Three. +TD / +Zone = one filter each. All 3 = both filters (the main candidate).")
    log.info(f"All 3 skipped setups: no TD exhaustion {rej['td']}, no tight zone {rej['zone']}.")
    log.info(f"Targets {PO3_RR:g}R per 1R risked, so it needs {100 / (1 + PO3_RR):.0f}%+ wins to break even.")
    log.info("Need: avgR above 0 in BOTH halves, PF above 1.2, a few hundred trades.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: Power Of Three + LuxAlgo Echo Forecast + EMA break of structure (rules fixed before testing)
#   Echo = the script's forecast, recomputed at the signal candle from past prices only: find the older
#          49-candle window most correlated with the latest 49 closes, and use the direction it projects.
#          Only take the fade if the forecast points the same way.
#   BOS  = after the fake move, wait up to 20 candles for a close back through the latest swing point
#          (a 3-candle swing high for buys, swing low for sells) that is also beyond the 50 EMA; enter then.
#   Exits and pairs are unchanged from the Flux Power Of Three (stop 5 x ATR, target 0.86R, 3 tight-spread pairs).
# ----------------------------------------------------------------------------
ECHO_LEN = _i("ECHO_LEN", 50)        # evaluation window
ECHO_FCAST = _i("ECHO_FCAST", 50)    # forecast window
PO3E_EMA = _i("PO3E_EMA", 50)
PO3E_BOS_WINDOW = _i("PO3E_BOS_WINDOW", 20)
PO3E_SWING = _i("PO3E_SWING", 3)


def echo_direction(c, i):
    """Echo Forecast [LuxAlgo] at candle i ('Similarity' + 'Cumulative'): +1 up, -1 down, 0 unknown."""
    L, F = ECHO_LEN, ECHO_FCAST
    need = 2 * F + L
    if i < need:
        return 0
    A = c[i - need:i + 1][::-1]          # A[0] = current close, A[j] = close j candles ago
    a = A[0:F - 1]
    best_r, k = None, 0
    if np.std(a) == 0:
        return 0
    for j in range(L):
        b = A[F + j:2 * F + j - 1]
        if np.std(b) == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if best_r is None or r >= best_r:
            best_r, k = r, j
    move = A[F + k] - A[2 * F + k]       # the forecast path adds up these price changes
    return 1 if move > 0 else (-1 if move < 0 else 0)


def last_swings(h, l, s):
    """Price of the most recent confirmed swing high / low known at each candle (needs s candles each side)."""
    n = len(h)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    cur_h = cur_l = np.nan
    for i in range(n):
        p = i - s                         # candidate swing, confirmed now
        if p - s >= 0:
            if h[p] > h[p - s:p].max() and h[p] > h[p + 1:i + 1].max():
                cur_h = h[p]
            if l[p] < l[p - s:p].min() and l[p] < l[p + 1:i + 1].min():
                cur_l = l[p]
        sh[i], sl[i] = cur_h, cur_l
    return sh, sl


def simulate_po3_echo(d, pair, start_ts, use_echo, use_bos):
    acc_len, acc_mult, manip_mult = PO3_PRESETS.get(PO3_MODE, PO3_PRESETS["small"])
    t = d["time"].tolist()
    h, l, c = d["h"].to_numpy(), d["l"].to_numpy(), d["c"].to_numpy()
    bo, bh, bl, bc = (d[k].to_numpy() for k in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[k].to_numpy() for k in ("ao", "ah", "al", "ac"))
    atr = calc_atr(d, 50).to_numpy()
    hh = d["h"].rolling(acc_len).max().to_numpy()
    ll = d["l"].rolling(acc_len).min().to_numpy()
    ema = d["c"].ewm(span=PO3E_EMA, adjust=False).mean().to_numpy()
    sh, sl_ = last_swings(h, l, PO3E_SWING) if use_bos else (None, None)
    n = len(d)
    trades, rejected = [], {"echo": 0, "bos": 0}
    state, i = 0, 0
    acc_top = acc_bot = None
    acc_end = 0
    while i < n - 1:
        a = atr[i]
        if np.isnan(hh[i]) or np.isnan(a):
            i += 1
            continue
        if state == 0 and hh[i] - ll[i] <= acc_mult * a:
            state, acc_end = 1, i
            acc_top, acc_bot = hh[i] + 0.5 * a, ll[i] - 0.5 * a
        if state == 1:
            if h[i] > acc_top or l[i] < acc_bot:
                state = 2
            else:
                acc_end = i
        direction = 0
        if state == 2 and i > acc_end:
            if h[i] > acc_top + manip_mult * a:
                direction = -1
            elif l[i] < acc_bot - manip_mult * a:
                direction = 1
        if direction == 0:
            i += 1
            continue
        state = 0
        signal_bar = i
        if use_bos:
            found = None
            for m in range(i + 1, min(i + PO3E_BOS_WINDOW, n - 2) + 1):
                level = sh[m - 1] if direction == 1 else sl_[m - 1]
                if np.isnan(level):
                    continue
                if direction == 1 and c[m] > level and c[m] > ema[m]:
                    found = m
                    break
                if direction == -1 and c[m] < level and c[m] < ema[m]:
                    found = m
                    break
            if found is None:
                rejected["bos"] += 1
                i += PO3E_BOS_WINDOW + 1
                continue
            signal_bar = found
        if use_echo and echo_direction(c, signal_bar) != direction:
            rejected["echo"] += 1
            i = signal_bar + 1
            continue
        j = signal_bar + 1
        entry = ao[j] if direction == 1 else bo[j]
        risk = PO3_SL_ATR * atr[signal_bar]
        sl, tp = entry - direction * risk, entry + direction * PO3_RR * risk
        r, k = _fixed_exit(direction, j, entry, sl, tp, PO3_RR, risk, bh, bl, bc, ah, al, ac, n)
        if t[j] >= start_ts:
            trades.append({"pair": pair, "entry": t[j], "exit": t[k], "dir": direction, "r": float(r),
                           "risk_pips": risk / pip(pair), "spread_frac": (ao[j] - bo[j]) / risk})
        i = k + 1
    return trades, rejected


def run_backtest_po3e(api, days):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ts = pd.Timestamp(start)
    log.info(f"BACKTEST | Power Of Three + Echo Forecast | {TIMEFRAME} | last {days} days | pairs {PO3S_PAIRS}")
    rows = {"Flux ref": [], "PO3+Echo": []}
    rej = {"echo": 0, "bos": 0}
    for pair in PO3S_PAIRS:
        try:
            sig = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=10))
            if len(sig) < 300:
                log.info(f"{pair} | not enough history, skipped")
                continue
            rows["Flux ref"] += simulate_po3(sig, pair, start_ts)
            tr, rj = simulate_po3_echo(sig, pair, start_ts, use_echo=True, use_bos=False)
            rows["PO3+Echo"] += tr
            for k2 in rej:
                rej[k2] += rj[k2]
            log.info(f"{pair} | done | PO3+Echo: {len(tr)} trades")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    log.info("")
    log.info(f"===== PO3 + ECHO | {TIMEFRAME} | last {days} days | results in R =====")
    log.info(f"{'':<8}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    for label, tr in rows.items():
        log.info(_summary_line(label, tr))
    log.info("Flux ref = plain Power Of Three (for comparison). PO3+Echo = the candidate.")
    log.info(f"Setups skipped because Echo disagreed: {rej['echo']}.")
    log.info(f"Targets {PO3_RR:g}R per 1R risked, so it needs {100 / (1 + PO3_RR):.0f}%+ wins to break even.")
    log.info("Need: avgR above 0 in BOTH halves, PF above 1.2, a few hundred trades.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: London breakout (rules fixed before testing)
#   Exactly the session Power Of Three setup (Asian range, break at the London open, stop 5 x ATR,
#   target 0.86R, 3 tight-spread pairs) but trading WITH the break. The idea came from the fade losing
#   on the last 2 years, so only the OLDER half of a longer run counts as a fair test.
# ----------------------------------------------------------------------------
def run_backtest_lbo(api, days):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ts = pd.Timestamp(start)
    log.info(f"BACKTEST | London breakout vs fade | {TIMEFRAME} | last {days} days | pairs {PO3S_PAIRS}")
    fade, brk, per_pair = [], [], []
    for pair in PO3S_PAIRS:
        try:
            sig = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=10))
            dly = fetch_range(api, pair, "D", start - timedelta(days=20))
            if len(sig) < 300:
                log.info(f"{pair} | not enough history, skipped")
                continue
            f1 = simulate_po3_session(sig, dly, pair, start_ts, need_sweep=False, follow=False)
            b1 = simulate_po3_session(sig, dly, pair, start_ts, need_sweep=False, follow=True)
            fade += f1
            brk += b1
            per_pair.append((pair, stats(b1)))
            log.info(f"{pair} | done | breakout {len(b1)} trades")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")
    log.info("")
    log.info(f"===== LONDON BREAKOUT | {TIMEFRAME} | last {days} days | results in R =====")
    log.info(f"{'':<8}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    log.info(_summary_line("Fade", fade))
    log.info(_summary_line("Breakout", brk))
    log.info("Judge Breakout on its 1st (older) half only: the newer half is where the idea came from.")
    log.info(f"{'pair':<9}{'n':>5}{'totalR':>8}")
    for pair, s in per_pair:
        log.info(f"{pair:<9}{s['n']:>5}{s['total']:>8.1f}")
    log.info(f"Targets {PO3_RR:g}R per 1R risked, so it needs {100 / (1 + PO3_RR):.0f}%+ wins to break even.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: Power Of Three + Echo with structural exits (rules fixed before testing)
#   Base    = unchanged: stop 5 x ATR(50), target 0.86R
#   +Stop   = stop just beyond the fake move's extreme (0.25 ATR past it, at least 0.5 ATR from entry);
#             target price kept the same as Base, so only the stop changes
#   +Target = Base stop, target = the far side of the quiet range (where ICT says price is heading)
#   Both    = structural stop + range target
#   Window  = BACKTEST_DAYS long, ending BACKTEST_END_DAYS ago (0 = today). Tune on recent data,
#             then run the final version ONCE on an older window nobody has looked at.
# ----------------------------------------------------------------------------
BACKTEST_END_DAYS = _i("BACKTEST_END_DAYS", 0)
PO3X_STOP_PAD_ATR = 0.25
PO3X_MIN_RISK_ATR = 0.5


def simulate_po3x(d, pair, start_ts, end_ts, struct_stop, range_target):
    acc_len, acc_mult, manip_mult = PO3_PRESETS.get(PO3_MODE, PO3_PRESETS["small"])
    t = d["time"].tolist()
    h, l, c = d["h"].to_numpy(), d["l"].to_numpy(), d["c"].to_numpy()
    bo, bh, bl, bc = (d[k].to_numpy() for k in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[k].to_numpy() for k in ("ao", "ah", "al", "ac"))
    atr = calc_atr(d, 50).to_numpy()
    hh = d["h"].rolling(acc_len).max().to_numpy()
    ll = d["l"].rolling(acc_len).min().to_numpy()
    n = len(d)
    trades = []
    state, i = 0, 0
    acc_top = acc_bot = rng_hi = rng_lo = None
    acc_end = 0
    while i < n - 1:
        a = atr[i]
        if np.isnan(hh[i]) or np.isnan(a):
            i += 1
            continue
        if state == 0 and hh[i] - ll[i] <= acc_mult * a:
            state, acc_end = 1, i
            rng_hi, rng_lo = hh[i], ll[i]
            acc_top, acc_bot = hh[i] + 0.5 * a, ll[i] - 0.5 * a
        if state == 1:
            if h[i] > acc_top or l[i] < acc_bot:
                state = 2
            else:
                acc_end = i
        direction = 0
        if state == 2 and i > acc_end:
            if h[i] > acc_top + manip_mult * a:
                direction = -1
            elif l[i] < acc_bot - manip_mult * a:
                direction = 1
        if direction == 0:
            i += 1
            continue
        state = 0
        if echo_direction(c, i) != direction:
            i += 1
            continue
        j = i + 1
        entry = ao[j] if direction == 1 else bo[j]
        base_risk = PO3_SL_ATR * a
        base_tp = entry + direction * PO3_RR * base_risk
        if struct_stop:
            extreme = l[i] if direction == 1 else h[i]          # the fake move's furthest point
            sl = extreme - direction * PO3X_STOP_PAD_ATR * a
            risk = direction * (entry - sl)
            if risk < PO3X_MIN_RISK_ATR * a:
                risk = PO3X_MIN_RISK_ATR * a
                sl = entry - direction * risk
        else:
            risk = base_risk
            sl = entry - direction * risk
        tp = (rng_hi if direction == 1 else rng_lo) if range_target else base_tp
        reward = direction * (tp - entry)
        if risk <= 0 or reward <= 0:
            i += 1
            continue
        rr = reward / risk
        r, k = _fixed_exit(direction, j, entry, sl, tp, rr, risk, bh, bl, bc, ah, al, ac, n)
        if start_ts <= t[j] < end_ts:
            trades.append({"pair": pair, "entry": t[j], "exit": t[k], "dir": direction, "r": float(r),
                           "rr": rr, "risk_pips": risk / pip(pair), "spread_frac": (ao[j] - bo[j]) / risk})
        i = k + 1
    return trades


def run_backtest_po3x(api, days):
    now = datetime.now(timezone.utc)
    end = now - timedelta(days=BACKTEST_END_DAYS)
    start = end - timedelta(days=days)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    log.info(f"BACKTEST | PO3 + Echo exit tuning | {TIMEFRAME} | {start:%Y-%m-%d} to {end:%Y-%m-%d} | pairs {PO3S_PAIRS}")
    variants = {"Base": (False, False), "+Stop": (True, False), "+Target": (False, True), "Both": (True, True)}
    rows = {k: [] for k in variants}
    for pair in PO3S_PAIRS:
        try:
            sig = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=max(10, SIG_WARMUP[TIMEFRAME])))
            if len(sig) < 300:
                log.info(f"{pair} | not enough history, skipped")
                continue
            for name, (ss, rt) in variants.items():
                rows[name] += simulate_po3x(sig, pair, start_ts, end_ts, ss, rt)
            first = sig["time"].iloc[0]
            log.info(f"{pair} | done | data from {first:%Y-%m-%d} | Both: {sum(1 for x in rows['Both'] if x['pair'] == pair)} trades")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    log.info("")
    log.info(f"===== PO3 + ECHO EXITS | {TIMEFRAME} | {start:%Y} to {end:%Y} | results in R =====")
    log.info(f"{'':<8}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    for name, tr in rows.items():
        log.info(_summary_line(name, tr))
    for name, tr in rows.items():
        if tr:
            log.info(f"{name}: average target {np.mean([x['rr'] for x in tr]):.2f}R, stop "
                     f"{np.mean([x['risk_pips'] for x in tr]):.0f} pips, spread {np.mean([x['spread_frac'] for x in tr]) * 100:.1f}% of risk")
    log.info("Compare PF (avgR is per each version's own stop size). Need avgR above 0 in BOTH halves and PF above 1.2.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: your strategy (rules fixed before testing)
#   Trend   = H4, H1 and M15 all making higher highs + higher lows (bullish) or lower highs + lower lows
#             (bearish), judged from the last two confirmed swing highs and lows (3 candles each side)
#   BOS     = an M15 close beyond the latest swing high (buys) / swing low (sells), in the trend's direction
#   Entry   = limit order on the pullback: the fair value gap left by the breakout move, or a retest of the
#             broken level, whichever price reaches first; valid for 8 hours (32 candles)
#   Stop    = beyond the pullback low/high that launched the break (+0.1 ATR), never tighter than 15 pips
#   Target  = fixed risk-reward (1:2, 1:3, 1:4 shown side by side)
#   +Lvl    = the entry must also sit on a key H4 support/resistance level (within 0.5 H4 ATR)
#   No new entries in the late New York hours. Pairs: EUR_USD, USD_JPY, GBP_USD (SMC_PAIRS to change).
# ----------------------------------------------------------------------------
SMC_PAIRS = [p.strip().upper() for p in os.getenv("SMC_PAIRS", "EUR_USD,USD_JPY,GBP_USD").split(",") if p.strip()]
SMC_SWING = _i("SMC_SWING", 3)
SMC_WAIT = _i("SMC_WAIT", 32)
SMC_MIN_STOP_PIPS = _f("SMC_MIN_STOP_PIPS", 15)
SMC_LEVEL_ATR = 0.5


def structure_trend(h, l, n):
    """+1 = higher highs and higher lows, -1 = lower highs and lower lows, 0 = mixed (confirmed swings only)."""
    N = len(h)
    tr = np.zeros(N, dtype=int)
    sh = [None, None]
    sl = [None, None]
    for i in range(N):
        p = i - n
        if p - n >= 0:
            if h[p] > h[p - n:p].max() and h[p] >= h[p + 1:i + 1].max():
                sh = [sh[1], h[p]]
            if l[p] < l[p - n:p].min() and l[p] <= l[p + 1:i + 1].min():
                sl = [sl[1], l[p]]
        if None not in sh and None not in sl:
            if sh[1] > sh[0] and sl[1] > sl[0]:
                tr[i] = 1
            elif sh[1] < sh[0] and sl[1] < sl[0]:
                tr[i] = -1
    return tr


def _on_m15(d, frame, values, minutes):
    """Values from a higher-timeframe frame, known when each M15 candle closes (completed candles only)."""
    f = pd.DataFrame({"avail": frame["time"] + pd.Timedelta(minutes=minutes), "v": values})
    ct = (d["time"] + pd.Timedelta(minutes=TF_MIN[TIMEFRAME])).to_frame("ct")
    m = pd.merge_asof(ct, f.sort_values("avail"), left_on="ct", right_on="avail", direction="backward")
    return m["v"].to_numpy()


def h4_levels(h4, n):
    """Confirmed H4 swing highs/lows as (time the swing became known, price), sorted by time."""
    h, l = h4["h"].to_numpy(), h4["l"].to_numpy()
    times = h4["time"].tolist()
    out_t, out_p = [], []
    for i in range(len(h4)):
        p = i - n
        if p - n >= 0:
            known = (times[i] + pd.Timedelta(hours=4)).value
            if h[p] > h[p - n:p].max() and h[p] >= h[p + 1:i + 1].max():
                out_t.append(known)
                out_p.append(h[p])
            if l[p] < l[p - n:p].min() and l[p] <= l[p + 1:i + 1].min():
                out_t.append(known)
                out_p.append(l[p])
    return np.array(out_t, dtype=np.int64), np.array(out_p)


def simulate_smc(d, pair, start_ts, end_ts, ctx, rr, use_level, entry_mode="limit"):
    t = d["time"].tolist()
    h, l, c = d["h"].to_numpy(), d["l"].to_numpy(), d["c"].to_numpy()
    bo, bh, bl, bc = (d[k].to_numpy() for k in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[k].to_numpy() for k in ("ao", "ah", "al", "ac"))
    atr = calc_atr(d, 14).to_numpy()
    tr15, tr1, tr4, atr4 = ctx["tr15"], ctx["tr1"], ctx["tr4"], ctx["atr4"]
    lvl_t, lvl_p, ct_ns, blocked = ctx["lvl_t"], ctx["lvl_p"], ctx["ct_ns"], ctx["blocked"]
    n = len(d)
    S = SMC_SWING
    minr = SMC_MIN_STOP_PIPS * pip(pair)
    trades = []
    skip = {"expired": 0, "spread": 0, "no key level": 0}
    sh_p = sh_i = sl_p = sl_i = None
    sh_done = sl_done = True
    pending = pos = None

    def record(p, i, r):
        if p["record"]:
            trades.append({"pair": pair, "entry": p["t"], "exit": t[i], "dir": p["dir"], "r": float(r),
                           "risk_pips": p["risk"] / pip(pair), "spread_frac": p["spread"] / p["risk"]})

    def near_level(price, i):
        if len(lvl_t) == 0 or np.isnan(atr4[i]):
            return False
        hi = np.searchsorted(lvl_t, ct_ns[i], side="right")
        lo = np.searchsorted(lvl_t, ct_ns[i] - 30 * 86400 * 10**9)
        return bool(len(lvl_p[lo:hi])) and np.min(np.abs(lvl_p[lo:hi] - price)) <= SMC_LEVEL_ATR * atr4[i]

    for i in range(n):
        # 1) manage the open trade
        if pos is not None and i > pos["fill"]:
            r = None
            if pos["dir"] == 1:
                if bl[i] <= pos["sl"]:
                    r = -1.0
                elif bh[i] >= pos["tp"]:
                    r = rr
            else:
                if ah[i] >= pos["sl"]:
                    r = -1.0
                elif al[i] <= pos["tp"]:
                    r = rr
            if r is None and i - pos["fill"] >= SB_MAX_BARS:
                r = pos["dir"] * ((bc[i] if pos["dir"] == 1 else ac[i]) - pos["entry"]) / pos["risk"]
            if r is not None:
                record(pos, i, r)
                pos = None
        # 2) pending entry: limit (pullback), market (at the break) or confirm (touch + rejection candle)
        elif pos is None and pending is not None and i > pending["created"]:
            dirn, E, mode = pending["dir"], pending["entry"], pending.get("mode", "limit")
            fill = None
            if mode == "market":
                if not blocked[i]:
                    fill = ao[i] if dirn == 1 else bo[i]          # next candle's open, paying the spread
                else:
                    skip["expired"] += 1
                if fill is None:
                    pending = None
            elif i - pending["created"] > SMC_WAIT and not pending.get("go"):
                skip["expired"] += 1
                pending = None
            elif mode == "confirm":
                if pending.get("go"):
                    if not blocked[i]:
                        fill = ao[i] if dirn == 1 else bo[i]
                    else:
                        skip["expired"] += 1
                        pending = None
                else:
                    sl0 = pending["sl"]
                    if (dirn == 1 and bl[i] <= sl0) or (dirn == -1 and ah[i] >= sl0):
                        skip["stopped before confirm"] = skip.get("stopped before confirm", 0) + 1
                        pending = None
                    else:
                        if (dirn == 1 and al[i] <= E) or (dirn == -1 and bh[i] >= E):
                            pending["touched"] = True
                        if pending.get("touched"):
                            # rejection: candle closes back on the trend side of the zone, in the trend direction
                            if (dirn == 1 and bc[i] > E and bc[i] > bo[i]) or (dirn == -1 and ac[i] < E and ac[i] < ao[i]):
                                pending["go"] = True                # enter at the next candle's open
            elif not blocked[i]:
                if (dirn == 1 and al[i] <= E) or (dirn == -1 and bh[i] >= E):
                    fill = min(E, ao[i]) if dirn == 1 else max(E, bo[i])
            if fill is not None and pending is not None:
                sl = pending["sl"]
                risk = dirn * (fill - sl)
                if risk < minr:
                    risk = minr
                    sl = fill - dirn * risk
                spread = ao[i] - bo[i]
                pending = None
                if spread > MAX_SPREAD_RISK * risk:
                    skip["spread"] += 1
                else:
                    p = {"dir": dirn, "entry": fill, "sl": sl, "tp": fill + dirn * rr * risk, "risk": risk,
                         "fill": i, "t": t[i], "spread": spread, "record": start_ts <= t[i] < end_ts}
                    if (dirn == 1 and bl[i] <= sl) or (dirn == -1 and ah[i] >= sl):
                        record(p, i, -1.0)      # filled and stopped in the same candle: count the loss
                    else:
                        pos = p
        # 3) swings confirmed this candle
        q = i - S
        if q - S >= 0:
            if h[q] > h[q - S:q].max() and h[q] >= h[q + 1:i + 1].max():
                sh_p, sh_i, sh_done = h[q], q, False
            if l[q] < l[q - S:q].min() and l[q] <= l[q + 1:i + 1].min():
                sl_p, sl_i, sl_done = l[q], q, False
        # 4) break of structure
        for dirn in (1, -1):
            lvl, idx, done = (sh_p, sh_i, sh_done) if dirn == 1 else (sl_p, sl_i, sl_done)
            if lvl is None or done or not ((dirn == 1 and c[i] > lvl) or (dirn == -1 and c[i] < lvl)):
                continue
            if dirn == 1:
                sh_done = True
            else:
                sl_done = True
            if pending is not None and pending["dir"] != dirn:
                pending = None                      # structure broke the other way: cancel
            if pos is not None or pending is not None or np.isnan(atr[i]):
                continue
            if not (tr15[i] == dirn and tr1[i] == dirn and tr4[i] == dirn):
                continue
            seg = slice(idx, i + 1)
            launch = idx + (int(np.argmin(l[seg])) if dirn == 1 else int(np.argmax(h[seg])))
            stop_ref = l[launch] if dirn == 1 else h[launch]
            gap = None
            for k in range(i, launch + 1, -1):     # newest fair value gap in the breakout move
                if k - 2 < launch:
                    break
                if dirn == 1 and l[k] > h[k - 2]:
                    gap = l[k]
                    break
                if dirn == -1 and h[k] < l[k - 2]:
                    gap = h[k]
                    break
            E = lvl if gap is None else (max(lvl, gap) if dirn == 1 else min(lvl, gap))
            if dirn * (c[i] - E) <= 0 or dirn * (E - stop_ref) <= 0:
                continue
            if use_level and not near_level(E, i):
                skip["no key level"] += 1
                continue
            pending = {"dir": dirn, "entry": E, "sl": stop_ref - dirn * 0.1 * atr[i], "created": i, "mode": entry_mode}
    if pos is not None:
        record(pos, n - 1, pos["dir"] * ((bc[n - 1] if pos["dir"] == 1 else ac[n - 1]) - pos["entry"]) / pos["risk"])
    return trades, skip


def build_smc_ctx(m15, h1, h4):
    lt = m15["time"].dt.tz_convert("America/New_York")
    nym = (lt.dt.hour * 60 + lt.dt.minute).to_numpy()
    a, b = (NY_BLOCK or "0:0-0:0").split("-")
    s0 = int(a.split(":")[0]) * 60 + int(a.split(":")[1])
    e0 = int(b.split(":")[0]) * 60 + int(b.split(":")[1])
    ctx = {
        "tr15": structure_trend(m15["h"].to_numpy(), m15["l"].to_numpy(), SMC_SWING),
        "tr1": np.nan_to_num(_on_m15(m15, h1, structure_trend(h1["h"].to_numpy(), h1["l"].to_numpy(), SMC_SWING), 60)).astype(int),
        "tr4": np.nan_to_num(_on_m15(m15, h4, structure_trend(h4["h"].to_numpy(), h4["l"].to_numpy(), SMC_SWING), 240)).astype(int),
        "atr4": _on_m15(m15, h4, calc_atr(h4, 14).to_numpy(), 240),
        "ct_ns": np.array([(x + pd.Timedelta(minutes=TF_MIN[TIMEFRAME])).value for x in m15["time"]], dtype=np.int64),
        "blocked": (nym >= s0) & (nym < e0),
    }
    ctx["lvl_t"], ctx["lvl_p"] = h4_levels(h4, SMC_SWING)
    return ctx


def run_backtest_smc(api, days):
    now = datetime.now(timezone.utc)
    end = now - timedelta(days=BACKTEST_END_DAYS)
    start = end - timedelta(days=days)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    log.info(f"BACKTEST | your strategy (trend x3 + BOS + FVG/retest) | {TIMEFRAME} | {start:%Y-%m-%d} to {end:%Y-%m-%d} | pairs {SMC_PAIRS}")
    variants = {"1:2": (2.0, False), "1:3": (3.0, False), "1:4": (4.0, False), "1:3 +Lvl": (3.0, True)}
    rows = {k: [] for k in variants}
    skips = {k: {} for k in variants}
    for pair in SMC_PAIRS:
        try:
            m15 = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=10))
            h1 = fetch_range(api, pair, "H1", start - timedelta(days=40))
            h4 = fetch_range(api, pair, "H4", start - timedelta(days=120))
            if len(m15) < 500 or len(h1) < 100 or len(h4) < 100:
                log.info(f"{pair} | not enough history, skipped")
                continue
            lt = m15["time"].dt.tz_convert("America/New_York")
            nym = (lt.dt.hour * 60 + lt.dt.minute).to_numpy()
            a, b = (NY_BLOCK or "0:0-0:0").split("-")
            s0 = int(a.split(":")[0]) * 60 + int(a.split(":")[1])
            e0 = int(b.split(":")[0]) * 60 + int(b.split(":")[1])
            ctx = {
                "tr15": structure_trend(m15["h"].to_numpy(), m15["l"].to_numpy(), SMC_SWING),
                "tr1": _on_m15(m15, h1, structure_trend(h1["h"].to_numpy(), h1["l"].to_numpy(), SMC_SWING), 60),
                "tr4": _on_m15(m15, h4, structure_trend(h4["h"].to_numpy(), h4["l"].to_numpy(), SMC_SWING), 240),
                "atr4": _on_m15(m15, h4, calc_atr(h4, 14).to_numpy(), 240),
                "ct_ns": np.array([(x + pd.Timedelta(minutes=TF_MIN[TIMEFRAME])).value for x in m15["time"]], dtype=np.int64),
                "blocked": (nym >= s0) & (nym < e0),
            }
            ctx["tr1"] = np.nan_to_num(ctx["tr1"]).astype(int)
            ctx["tr4"] = np.nan_to_num(ctx["tr4"]).astype(int)
            ctx["lvl_t"], ctx["lvl_p"] = h4_levels(h4, SMC_SWING)
            for name, (rr, use_lvl) in variants.items():
                tr, sk = simulate_smc(m15, pair, start_ts, end_ts, ctx, rr, use_lvl)
                rows[name] += tr
                for k2, v in sk.items():
                    skips[name][k2] = skips[name].get(k2, 0) + v
            log.info(f"{pair} | done | 1:3: {sum(1 for x in rows['1:3'] if x['pair'] == pair)} trades")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    log.info("")
    log.info(f"===== YOUR STRATEGY | {TIMEFRAME} | {start:%Y} to {end:%Y} | results in R =====")
    log.info(f"{'':<8}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    for name, tr in rows.items():
        log.info(_summary_line(name, tr))
    log.info("Coin-flip win rates to beat: 1:2 = 33%, 1:3 = 25%, 1:4 = 20% (before spread).")
    base = rows["1:3"]
    if base:
        months = max(days / 30.4, 1)
        log.info(f"1:3: average stop {np.mean([x['risk_pips'] for x in base]):.1f} pips, spread "
                 f"{np.mean([x['spread_frac'] for x in base]) * 100:.1f}% of risk, about {len(base) / months:.1f} trades a month")
    sk = skips["1:3 +Lvl"]
    log.info("Setups not taken (1:3 +Lvl): " + (", ".join(f"{k2} {v}" for k2, v in sk.items() if v) or "none"))
    log.info("Need: avgR above 0 in BOTH halves, PF above 1.2, a few hundred trades.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: entry-method test for your strategy (rules fixed before testing)
#   Same setups as BACKTEST_STRATEGY=smc (H4+H1+M15 trend, M15 break of structure, structural stop).
#   Only the ENTRY changes:
#     Limit   = current rule: limit order at the FVG/retest, waits up to SMC_WAIT candles
#     Market  = enter at the open of the candle right after the break (no waiting, no missed trades)
#     Confirm = wait for price to touch the FVG/retest, then a candle that closes back in the trend
#               direction (a rejection); enter at the next open. Cancelled if the stop is hit first.
#   Shown at 1:2 and 1:3. Pass bar: Market or Confirm must beat Limit AND have avgR > 0 in both halves, PF >= 1.2.
# ----------------------------------------------------------------------------
def run_backtest_smce(api, days):
    now = datetime.now(timezone.utc)
    end = now - timedelta(days=BACKTEST_END_DAYS)
    start = end - timedelta(days=days)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    log.info(f"BACKTEST | your strategy - ENTRY TEST (limit vs market vs confirm) | {TIMEFRAME} | "
             f"{start:%Y-%m-%d} to {end:%Y-%m-%d} | pairs {SMC_PAIRS}")
    variants = {}
    short = {"limit": "Lim", "market": "Mkt", "confirm": "Conf"}
    for mode in ("limit", "market", "confirm"):
        for rr in (2.0, 3.0):
            variants[f"{short[mode]} 1:{int(rr)}"] = (mode, rr)
    rows = {k: [] for k in variants}
    skips = {k: {} for k in variants}
    for pair in SMC_PAIRS:
        try:
            m15 = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=10))
            h1 = fetch_range(api, pair, "H1", start - timedelta(days=40))
            h4 = fetch_range(api, pair, "H4", start - timedelta(days=120))
            if len(m15) < 500 or len(h1) < 100 or len(h4) < 100:
                log.info(f"{pair} | not enough history, skipped")
                continue
            ctx = build_smc_ctx(m15, h1, h4)
            for name, (mode, rr) in variants.items():
                tr, sk = simulate_smc(m15, pair, start_ts, end_ts, ctx, rr, False, mode)
                rows[name] += tr
                for k2, v in sk.items():
                    skips[name][k2] = skips[name].get(k2, 0) + v
            log.info(f"{pair} | done")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    log.info("")
    log.info(f"===== ENTRY TEST | {TIMEFRAME} | {start:%Y} to {end:%Y} | results in R =====")
    log.info(f"{'':<8}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    for name, tr in rows.items():
        log.info(_summary_line(name, tr))
    log.info("Coin-flip win rates to beat: 1:2 = 33%, 1:3 = 25% (before spread).")
    months = max(days / 30.4, 1)
    for mode in ("Lim", "Mkt", "Conf"):
        base = rows[f"{mode} 1:3"]
        if base:
            log.info(f"{mode} 1:3: avg stop {np.mean([x['risk_pips'] for x in base]):.1f} pips, spread "
                     f"{np.mean([x['spread_frac'] for x in base]) * 100:.1f}% of risk, {len(base) / months:.1f} trades/month")
        sk = skips[f"{mode} 1:3"]
        log.info(f"{mode} setups not taken: " + (", ".join(f"{k2} {v}" for k2, v in sk.items() if v) or "none"))
    log.info("Pass: Market or Confirm beats Limit AND avgR > 0 in BOTH halves, PF >= 1.2.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: your strategy + Power Of Three combined (rules fixed before testing)
#   1. H4 and H1 trends agree (higher highs + higher lows for buys, the reverse for sells)
#   2. Power Of Three finds a quiet range (40 candles under 5 ATR) and a fake move out of it that goes
#      AGAINST that trend (a sweep below the range in an uptrend, above it in a downtrend)
#   3. Within 5 hours (20 candles) an M15 break of structure back in the trend's direction (your rule)
#   4. Entry on the pullback into the fair value gap / retest (your rule), valid 8 hours
#   5. Stop beyond the fake move's extreme (+0.1 ATR, at least 15 pips); target 1:3 (1:2 and 1:4 shown)
#   No entries in the late New York hours; one trade at a time per pair.
# ----------------------------------------------------------------------------
COMBO_BOS_WINDOW = _i("COMBO_BOS_WINDOW", 20)


def simulate_combo(d, pair, start_ts, end_ts, ctx, rr, entry_mode="limit"):
    acc_len, acc_mult, manip_mult = PO3_PRESETS["small"]
    t = d["time"].tolist()
    h, l, c = d["h"].to_numpy(), d["l"].to_numpy(), d["c"].to_numpy()
    bo, bh, bl, bc = (d[k].to_numpy() for k in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[k].to_numpy() for k in ("ao", "ah", "al", "ac"))
    atr50 = calc_atr(d, 50).to_numpy()
    atr = calc_atr(d, 14).to_numpy()
    hh = d["h"].rolling(acc_len).max().to_numpy()
    ll = d["l"].rolling(acc_len).min().to_numpy()
    tr1, tr4, blocked = ctx["tr1"], ctx["tr4"], ctx["blocked"]
    n = len(d)
    S = SMC_SWING
    minr = SMC_MIN_STOP_PIPS * pip(pair)
    trades = []
    count = {"fake moves": 0, "against trend": 0, "no BOS in time": 0, "entry not reached": 0, "spread": 0}
    sh_p = sl_p = None
    pending = pos = None
    state = 0
    acc_top = acc_bot = None
    acc_end = 0
    setup = None   # waiting for the break of structure: dir, extreme, extreme index, deadline

    def record(p, i, r):
        if p["record"]:
            trades.append({"pair": pair, "entry": p["t"], "exit": t[i], "dir": p["dir"], "r": float(r),
                           "risk_pips": p["risk"] / pip(pair), "spread_frac": p["spread"] / p["risk"]})

    for i in range(n):
        # 1) open trade
        if pos is not None and i > pos["fill"]:
            r = None
            if pos["dir"] == 1:
                if bl[i] <= pos["sl"]:
                    r = -1.0
                elif bh[i] >= pos["tp"]:
                    r = rr
            else:
                if ah[i] >= pos["sl"]:
                    r = -1.0
                elif al[i] <= pos["tp"]:
                    r = rr
            if r is None and i - pos["fill"] >= SB_MAX_BARS:
                r = pos["dir"] * ((bc[i] if pos["dir"] == 1 else ac[i]) - pos["entry"]) / pos["risk"]
            if r is not None:
                record(pos, i, r)
                pos = None
        # 2) pending pullback entry
        elif pos is None and pending is not None and i > pending["created"]:
            if pending.get("mode") == "market" and blocked[i]:
                count["entry not reached"] += 1
                pending = None
            elif i - pending["created"] > SMC_WAIT:
                count["entry not reached"] += 1
                pending = None
            elif not blocked[i]:
                dirn, E = pending["dir"], pending["entry"]
                market = pending.get("mode") == "market"
                if market or (dirn == 1 and al[i] <= E) or (dirn == -1 and bh[i] >= E):
                    if market:
                        fill = ao[i] if dirn == 1 else bo[i]      # next candle's open, paying the spread
                    else:
                        fill = min(E, ao[i]) if dirn == 1 else max(E, bo[i])
                    sl = pending["sl"]
                    risk = dirn * (fill - sl)
                    if risk < minr:
                        risk = minr
                        sl = fill - dirn * risk
                    spread = ao[i] - bo[i]
                    pending = None
                    if spread > MAX_SPREAD_RISK * risk:
                        count["spread"] += 1
                    else:
                        p = {"dir": dirn, "entry": fill, "sl": sl, "tp": fill + dirn * rr * risk, "risk": risk,
                             "fill": i, "t": t[i], "spread": spread, "record": start_ts <= t[i] < end_ts}
                        if (dirn == 1 and bl[i] <= sl) or (dirn == -1 and ah[i] >= sl):
                            record(p, i, -1.0)
                        else:
                            pos = p
        # 3) swings confirmed this candle
        q = i - S
        if q - S >= 0:
            if h[q] > h[q - S:q].max() and h[q] >= h[q + 1:i + 1].max():
                sh_p = h[q]
            if l[q] < l[q - S:q].min() and l[q] <= l[q + 1:i + 1].min():
                sl_p = l[q]
        if pos is not None or pending is not None:
            continue
        a = atr50[i]
        if np.isnan(hh[i]) or np.isnan(a) or np.isnan(atr[i]):
            continue
        # 4) waiting for your break of structure after the fake move
        if setup is not None:
            dirn = setup["dir"]
            if dirn == 1 and l[i] < setup["ext"]:
                setup["ext"], setup["ext_i"] = l[i], i
            if dirn == -1 and h[i] > setup["ext"]:
                setup["ext"], setup["ext_i"] = h[i], i
            lvl = sh_p if dirn == 1 else sl_p
            if lvl is not None and i > setup["ext_i"] and (
                    (dirn == 1 and c[i] > lvl >= c[i - 1]) or (dirn == -1 and c[i] < lvl <= c[i - 1])):
                launch, ext = setup["ext_i"], setup["ext"]
                gap = None
                for k in range(i, launch + 1, -1):
                    if k - 2 < launch:
                        break
                    if dirn == 1 and l[k] > h[k - 2]:
                        gap = l[k]
                        break
                    if dirn == -1 and h[k] < l[k - 2]:
                        gap = h[k]
                        break
                E = lvl if gap is None else (max(lvl, gap) if dirn == 1 else min(lvl, gap))
                setup = None
                if entry_mode == "market":
                    pending = {"dir": dirn, "entry": c[i], "sl": ext - dirn * 0.1 * atr[i], "created": i, "mode": "market"}
                elif dirn * (c[i] - E) > 0 and dirn * (E - ext) > 0:
                    pending = {"dir": dirn, "entry": E, "sl": ext - dirn * 0.1 * atr[i], "created": i}
                continue
            if i > setup["deadline"]:
                count["no BOS in time"] += 1
                setup = None
            continue
        # 5) Power Of Three: quiet range, then a fake move out of it
        if state == 0 and hh[i] - ll[i] <= acc_mult * a:
            state, acc_end = 1, i
            acc_top, acc_bot = hh[i] + 0.5 * a, ll[i] - 0.5 * a
        if state == 1:
            if h[i] > acc_top or l[i] < acc_bot:
                state = 2
            else:
                acc_end = i
        if state == 2 and i > acc_end:
            fade = 0
            if h[i] > acc_top + manip_mult * a:
                fade = -1
            elif l[i] < acc_bot - manip_mult * a:
                fade = 1
            if fade != 0:
                state = 0
                count["fake moves"] += 1
                if tr1[i] == fade and tr4[i] == fade:   # the fake move went against the H4/H1 trend
                    setup = {"dir": fade, "ext": l[i] if fade == 1 else h[i], "ext_i": i,
                             "deadline": i + COMBO_BOS_WINDOW}
                else:
                    count["against trend"] += 1
    if pos is not None:
        record(pos, n - 1, pos["dir"] * ((bc[n - 1] if pos["dir"] == 1 else ac[n - 1]) - pos["entry"]) / pos["risk"])
    return trades, count


def run_backtest_combo(api, days):
    now = datetime.now(timezone.utc)
    end = now - timedelta(days=BACKTEST_END_DAYS)
    start = end - timedelta(days=days)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    log.info(f"BACKTEST | your strategy + Power Of Three | {TIMEFRAME} | {start:%Y-%m-%d} to {end:%Y-%m-%d} | pairs {SMC_PAIRS}")
    rows = {"Yours": [], "PO3": [], "Combo1:2": [], "Combo1:3": [], "Combo1:4": []}
    counts = {}
    for pair in SMC_PAIRS:
        try:
            m15 = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=10))
            h1 = fetch_range(api, pair, "H1", start - timedelta(days=40))
            h4 = fetch_range(api, pair, "H4", start - timedelta(days=120))
            if len(m15) < 500 or len(h1) < 100 or len(h4) < 100:
                log.info(f"{pair} | not enough history, skipped")
                continue
            ctx = build_smc_ctx(m15, h1, h4)
            rows["Yours"] += simulate_smc(m15, pair, start_ts, end_ts, ctx, 3.0, False)[0]
            rows["PO3"] += [x for x in simulate_po3(m15, pair, start_ts) if x["entry"] < end_ts]
            for rr in (2.0, 3.0, 4.0):
                tr, cnt = simulate_combo(m15, pair, start_ts, end_ts, ctx, rr)
                rows[f"Combo1:{rr:g}"] += tr
                if rr == 3.0:
                    for k2, v in cnt.items():
                        counts[k2] = counts.get(k2, 0) + v
            log.info(f"{pair} | done | Combo 1:3: {sum(1 for x in rows['Combo1:3'] if x['pair'] == pair)} trades")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    log.info("")
    log.info(f"===== YOURS + PO3 | {TIMEFRAME} | {start:%Y} to {end:%Y} | results in R =====")
    log.info(f"{'':<8}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    for name, tr in rows.items():
        log.info(_summary_line(name, tr))
    log.info("Yours = your strategy alone (1:3). PO3 = plain Power Of Three. Combo = both together (1:3 is the main one).")
    log.info("Coin-flip win rates to beat: 1:2 = 33%, 1:3 = 25%, 1:4 = 20% (PO3 targets 0.86R: 54%).")
    log.info("Combo 1:3 setups: " + ", ".join(f"{k2} {v}" for k2, v in counts.items()))
    base = rows["Combo1:3"]
    if base:
        log.info(f"Combo 1:3: average stop {np.mean([x['risk_pips'] for x in base]):.1f} pips, spread "
                 f"{np.mean([x['spread_frac'] for x in base]) * 100:.1f}% of risk, about {len(base) / max(days / 30.4, 1):.1f} trades a month")
    log.info("Need: avgR above 0 in BOTH halves, PF above 1.2, a few hundred trades.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: Power Of Three v2 (rules fixed before testing)
#   PO3      = plain Power Of Three (fade the fake move at once, target 0.86R) - reference
#   PO3+Tr   = same, but only fade fake moves that went AGAINST the H4+H1 trend
#   Conf1:2 / Conf1:3 = PO3 + trend + confirmation: wait (up to COMBO_BOS_WINDOW candles) for an M15
#              break of structure back in the fade direction, enter at the NEXT candle's open (market,
#              no pullback limit), stop beyond the fake move's extreme (+0.1 ATR, min 15 pips)
#   Pass: avgR > 0 in BOTH halves, PF >= 1.2, a few hundred trades -> then confirm on untouched data.
# ----------------------------------------------------------------------------
def run_backtest_po3v2(api, days):
    now = datetime.now(timezone.utc)
    end = now - timedelta(days=BACKTEST_END_DAYS)
    start = end - timedelta(days=days)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    log.info(f"BACKTEST | Power Of Three v2 (trend filter + confirmation) | {TIMEFRAME} | "
             f"{start:%Y-%m-%d} to {end:%Y-%m-%d} | pairs {SMC_PAIRS}")
    rows = {"PO3": [], "PO3+Tr": [], "Conf1:2": [], "Conf1:3": []}
    counts = {}
    for pair in SMC_PAIRS:
        try:
            m15 = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=10))
            h1 = fetch_range(api, pair, "H1", start - timedelta(days=40))
            h4 = fetch_range(api, pair, "H4", start - timedelta(days=120))
            if len(m15) < 500 or len(h1) < 100 or len(h4) < 100:
                log.info(f"{pair} | not enough history, skipped")
                continue
            ctx = build_smc_ctx(m15, h1, h4)
            rows["PO3"] += [x for x in simulate_po3(m15, pair, start_ts) if x["entry"] < end_ts]
            rows["PO3+Tr"] += [x for x in simulate_po3(m15, pair, start_ts, ctx["tr1"], ctx["tr4"]) if x["entry"] < end_ts]
            for rr in (2.0, 3.0):
                tr, cnt = simulate_combo(m15, pair, start_ts, end_ts, ctx, rr, "market")
                rows[f"Conf1:{rr:g}"] += tr
                if rr == 2.0:
                    for k2, v in cnt.items():
                        counts[k2] = counts.get(k2, 0) + v
            log.info(f"{pair} | done")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    log.info("")
    log.info(f"===== PO3 v2 | {TIMEFRAME} | {start:%Y} to {end:%Y} | results in R =====")
    log.info(f"{'':<8}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    for name, tr in rows.items():
        log.info(_summary_line(name, tr))
    log.info(f"Coin-flip win rates to beat: PO3 and PO3+Tr ({PO3_RR:g}R target) = {100 / (1 + PO3_RR):.0f}%, "
             f"Conf1:2 = 33%, Conf1:3 = 25% (before spread).")
    months = max(days / 30.4, 1)
    for name, tr in rows.items():
        if tr:
            log.info(f"{name}: avg stop {np.mean([x['risk_pips'] for x in tr]):.1f} pips, spread "
                     f"{np.mean([x['spread_frac'] for x in tr]) * 100:.1f}% of risk, {len(tr) / months:.1f} trades/month")
    log.info("Conf setups: " + ", ".join(f"{k2} {v}" for k2, v in counts.items()))
    log.info("Pass: avgR > 0 in BOTH halves, PF >= 1.2, a few hundred trades. Then confirm on untouched data.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: PO3 + Echo, final wide test (rules fixed before testing)
#   Base     = PO3 + Echo exactly as before (fade at once, stop 5 x ATR(50), target 0.86R)
#   Conf1:2 / Conf1:3 = same setup and Echo filter, then wait up to PO3E_BOS_WINDOW candles for a break of
#              structure back in the fade direction (close beyond the last confirmed swing), enter at the
#              next open, stop beyond the fake move's extreme (+0.1 ATR). No trend filter (it hurt on M15).
#   Run on many pairs and the full history so the sample is big enough to decide.
#   Pass: avgR > 0 in BOTH halves, PF >= 1.2, 300+ trades, and most pairs positive.
# ----------------------------------------------------------------------------
PO3W_PAIRS = [p.strip().upper() for p in os.getenv(
    "PO3W_PAIRS", "EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CAD,NZD_USD,USD_CHF,EUR_JPY,GBP_JPY,EUR_GBP,AUD_JPY,EUR_AUD"
).split(",") if p.strip()]


def simulate_po3_echo_conf(d, pair, start_ts, end_ts, rr):
    acc_len, acc_mult, manip_mult = PO3_PRESETS.get(PO3_MODE, PO3_PRESETS["small"])
    t = d["time"].tolist()
    h, l, c = d["h"].to_numpy(), d["l"].to_numpy(), d["c"].to_numpy()
    bo, bh, bl, bc = (d[k].to_numpy() for k in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[k].to_numpy() for k in ("ao", "ah", "al", "ac"))
    atr = calc_atr(d, 50).to_numpy()
    hh = d["h"].rolling(acc_len).max().to_numpy()
    ll = d["l"].rolling(acc_len).min().to_numpy()
    sh, sl_ = last_swings(h, l, PO3E_SWING)
    n = len(d)
    trades = []
    count = {"echo disagreed": 0, "no BOS in time": 0}
    state, i = 0, 0
    acc_top = acc_bot = None
    acc_end = 0
    while i < n - 1:
        a = atr[i]
        if np.isnan(hh[i]) or np.isnan(a):
            i += 1
            continue
        if state == 0 and hh[i] - ll[i] <= acc_mult * a:
            state, acc_end = 1, i
            acc_top, acc_bot = hh[i] + 0.5 * a, ll[i] - 0.5 * a
        if state == 1:
            if h[i] > acc_top or l[i] < acc_bot:
                state = 2
            else:
                acc_end = i
        direction = 0
        if state == 2 and i > acc_end:
            if h[i] > acc_top + manip_mult * a:
                direction = -1
            elif l[i] < acc_bot - manip_mult * a:
                direction = 1
        if direction == 0:
            i += 1
            continue
        state = 0
        if echo_direction(c, i) != direction:
            count["echo disagreed"] += 1
            i += 1
            continue
        ext = l[i] if direction == 1 else h[i]
        found = None
        for m in range(i + 1, min(i + PO3E_BOS_WINDOW, n - 2) + 1):
            if direction == 1:
                ext = min(ext, l[m])
            else:
                ext = max(ext, h[m])
            level = sh[m - 1] if direction == 1 else sl_[m - 1]
            if np.isnan(level):
                continue
            if (direction == 1 and c[m] > level) or (direction == -1 and c[m] < level):
                found = m
                break
        if found is None:
            count["no BOS in time"] += 1
            i += PO3E_BOS_WINDOW + 1
            continue
        j = found + 1
        entry = ao[j] if direction == 1 else bo[j]
        sl = ext - direction * 0.1 * atr[found]
        risk = direction * (entry - sl)
        if risk <= 0:
            i = j
            continue
        tp = entry + direction * rr * risk
        r, k = _fixed_exit(direction, j, entry, sl, tp, rr, risk, bh, bl, bc, ah, al, ac, n)
        if start_ts <= t[j] < end_ts:
            trades.append({"pair": pair, "entry": t[j], "exit": t[k], "dir": direction, "r": float(r),
                           "risk_pips": risk / pip(pair), "spread_frac": (ao[j] - bo[j]) / risk})
        i = k + 1
    return trades, count


def run_backtest_po3w(api, days):
    now = datetime.now(timezone.utc)
    end = now - timedelta(days=BACKTEST_END_DAYS)
    start = end - timedelta(days=days)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    log.info(f"BACKTEST | PO3 + Echo WIDE TEST | {TIMEFRAME} | {start:%Y-%m-%d} to {end:%Y-%m-%d} | "
             f"{len(PO3W_PAIRS)} pairs {PO3W_PAIRS}")
    rows = {"Base": [], "Conf1:2": [], "Conf1:3": []}
    counts = {}
    for pair in PO3W_PAIRS:
        try:
            sig = fetch_range(api, pair, TIMEFRAME, start - timedelta(days=60))
            if len(sig) < 300:
                log.info(f"{pair} | not enough history, skipped")
                continue
            tr, _ = simulate_po3_echo(sig, pair, start_ts, use_echo=True, use_bos=False)
            rows["Base"] += [x for x in tr if x["entry"] < end_ts]
            for rr in (2.0, 3.0):
                tr2, cnt = simulate_po3_echo_conf(sig, pair, start_ts, end_ts, rr)
                rows[f"Conf1:{rr:g}"] += tr2
                if rr == 2.0:
                    for k2, v in cnt.items():
                        counts[k2] = counts.get(k2, 0) + v
            log.info(f"{pair} | done | first candle {sig['time'].iloc[0]:%Y-%m-%d}")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    log.info("")
    log.info(f"===== PO3+ECHO WIDE | {TIMEFRAME} | {start:%Y} to {end:%Y} | results in R =====")
    log.info(f"{'':<8}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    for name, tr in rows.items():
        log.info(_summary_line(name, tr))
    log.info(f"Coin-flip win rates: Base ({PO3_RR:g}R) = {100 / (1 + PO3_RR):.0f}%, Conf1:2 = 33%, Conf1:3 = 25%.")
    months = max(days / 30.4, 1)
    for name, tr in rows.items():
        if not tr:
            continue
        per = {}
        for x in tr:
            per.setdefault(x["pair"], []).append(x["r"])
        pos = sum(1 for v in per.values() if np.mean(v) > 0)
        log.info(f"{name}: {pos} of {len(per)} pairs positive | spread {np.mean([x['spread_frac'] for x in tr]) * 100:.1f}% "
                 f"of risk | {len(tr) / months:.1f} trades/month")
    for name in ("Base", "Conf1:2"):
        per = {}
        for x in rows[name]:
            per.setdefault(x["pair"], []).append(x["r"])
        log.info(f"{name} by pair: " + ", ".join(f"{k} {np.mean(v):+.2f} ({len(v)})" for k, v in per.items()))
    log.info("Conf setups: " + ", ".join(f"{k2} {v}" for k2, v in counts.items()))
    log.info("Pass: avgR > 0 in BOTH halves, PF >= 1.2, 300+ trades, most pairs positive.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# BACKTEST: daily trend-following / time-series momentum (rules fixed before testing)
#   Daily candles. Every TSM_CHECK trading days (default 5 = weekly) look at the close vs the close
#   L days ago: higher -> be long, lower -> be short. If that differs from the current position,
#   close it and reverse at the next day's open (real bid/ask, so spread is paid). Always in the market.
#   No stop and no target: trades end only when the trend signal flips.
#   1R = 3 x daily ATR(20) at entry (a size unit so results compare with the other tests).
#   Lookbacks: 60 / 120 / 250 days (about 3, 6, 12 months). Overnight financing (swap) is NOT included.
#   Pass: avgR > 0 in BOTH halves, PF >= 1.2, 300+ trades, most pairs positive, neighbouring lookbacks agree.
# ----------------------------------------------------------------------------
TSM_LOOKBACKS = [int(x) for x in os.getenv("TSM_LOOKBACKS", "60,120,250").split(",") if x.strip()]
TSM_CHECK = _i("TSM_CHECK", 5)


def simulate_tsm(d, pair, start_ts, end_ts, lookback):
    t = d["time"].tolist()
    c = d["c"].to_numpy()
    bo, ao = d["bo"].to_numpy(), d["ao"].to_numpy()
    bc, ac = d["bc"].to_numpy(), d["ac"].to_numpy()
    atr = calc_atr(d, 20).to_numpy()
    n = len(d)
    trades = []
    pos = None

    def close(p, k, price):
        r = p["dir"] * (price - p["entry"]) / p["risk"]
        if start_ts <= p["t"] < end_ts:
            trades.append({"pair": pair, "entry": p["t"], "exit": t[k], "dir": p["dir"], "r": float(r),
                           "risk_pips": p["risk"] / pip(pair), "spread_frac": p["spread"] / p["risk"],
                           "days": k - p["fill"]})

    for i in range(lookback, n - 1, TSM_CHECK):
        if np.isnan(atr[i]):
            continue
        sig = 1 if c[i] > c[i - lookback] else (-1 if c[i] < c[i - lookback] else 0)
        if sig == 0 or (pos is not None and pos["dir"] == sig):
            continue
        j = i + 1
        if pos is not None:
            close(pos, j, bo[j] if pos["dir"] == 1 else ao[j])
        entry = ao[j] if sig == 1 else bo[j]
        pos = {"dir": sig, "entry": entry, "risk": 3 * atr[i], "fill": j, "t": t[j], "spread": ao[j] - bo[j]}
    if pos is not None:
        close(pos, n - 1, bc[n - 1] if pos["dir"] == 1 else ac[n - 1])
    return trades


def run_backtest_tsm(api, days):
    now = datetime.now(timezone.utc)
    end = now - timedelta(days=BACKTEST_END_DAYS)
    start = end - timedelta(days=days)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    log.info(f"BACKTEST | daily trend-following (time-series momentum) | D | {start:%Y-%m-%d} to {end:%Y-%m-%d} | "
             f"{len(PO3W_PAIRS)} pairs {PO3W_PAIRS}")
    rows = {f"TS{L}": [] for L in TSM_LOOKBACKS}
    for pair in PO3W_PAIRS:
        try:
            dly = fetch_range(api, pair, "D", start - timedelta(days=int(max(TSM_LOOKBACKS) * 1.6) + 60))
            if len(dly) < max(TSM_LOOKBACKS) + 100:
                log.info(f"{pair} | not enough history, skipped")
                continue
            for L in TSM_LOOKBACKS:
                rows[f"TS{L}"] += simulate_tsm(dly, pair, start_ts, end_ts, L)
            log.info(f"{pair} | done | first candle {dly['time'].iloc[0]:%Y-%m-%d}")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    log.info("")
    log.info(f"===== TREND-FOLLOWING | D | {start:%Y} to {end:%Y} | results in R (1R = 3 x ATR) =====")
    log.info(f"{'':<8}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    for name, tr in rows.items():
        log.info(_summary_line(name, tr))
    years = max(days / 365.25, 1)
    for name, tr in rows.items():
        if not tr:
            continue
        st = stats(tr)
        per = {}
        for x in tr:
            per.setdefault(x["pair"], []).append(x["r"])
        pos = sum(1 for v in per.values() if np.sum(v) > 0)
        log.info(f"{name}: {pos} of {len(per)} pairs positive | total {st['total']:+.0f}R ({st['total'] / years:+.1f}R/yr) | "
                 f"worst drawdown {st['dd']:.0f}R | avg hold {np.mean([x['days'] for x in tr]):.0f} days | "
                 f"spread {np.mean([x['spread_frac'] for x in tr]) * 100:.1f}% of risk")
    mid = rows.get("TS120") or next(iter(rows.values()))
    per = {}
    for x in mid:
        per.setdefault(x["pair"], []).append(x["r"])
    log.info("TS120 by pair (total R): " + ", ".join(f"{k} {np.sum(v):+.0f}" for k, v in per.items()))
    log.info("Win rate is low by design: trend-following loses small and often, wins big and rarely.")
    log.info("Swap/financing not included. Pass: avgR > 0 in BOTH halves, PF >= 1.2, 300+ trades, most pairs "
             "positive, lookbacks agree.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# M15 SIGNAL SCANNER (rules fixed before testing)
#   Tests a batch of simple M15 timing/level signals at once, with a strict split:
#     SEARCH  = SCAN_START to SCAN_SPLIT (default 2014-01-01 to 2020-01-01)
#     HOLDOUT = SCAN_SPLIT to today (never looked at during the search)
#   Holdout results are ONLY printed for signals that pass the search, so nothing can be cherry-picked.
#   Cost control: EUR_USD + USD_JPY only, entries only in the London-New York overlap (08:00-12:00 New York),
#   stop = 0.5 x daily ATR(20) (wide, so spread is a small share), target 1:1 or 1:2,
#   every trade closed by 15:00 New York (clear of the 17:00 rollover spread; no swap). One trade per signal per day per pair.
#   Signals (each tested as "f" = follow and "x" = fade/opposite, at 1:1 and 1:2):
#     LDN   London session direction 03:00->08:00 NY, enter 08:00
#     NY1H  first New York hour 08:00->09:00, enter 09:00
#     PDHL  first M15 close beyond the previous day's high/low (08:00-12:00), enter next open
#     ASIA  first M15 close beyond today's Asian range (19:00-03:00 NY), enter next open
#     RND   first M15 close across a round number (EUR_USD 100 pips, USD_JPY 1 yen), enter next open
#     FIX   move into the London 4pm fix (15:00->16:00 London), enter at 16:00 London
#     H08..H11  fixed-hour drift: "f" = buy at that hour, "x" = sell
#   SEARCH pass: 300+ trades, PF >= 1.3, avgR > 0 in both search halves, both pairs avgR > 0.
#   HOLDOUT pass: avgR > 0 in both holdout halves, PF >= 1.2, both pairs avgR > 0.
# ----------------------------------------------------------------------------
SCAN_PAIRS = [p.strip().upper() for p in os.getenv("SCAN_PAIRS", "EUR_USD,USD_JPY").split(",") if p.strip()]
SCAN_START = os.getenv("SCAN_START", "2014-01-01")
SCAN_SPLIT = os.getenv("SCAN_SPLIT", "2020-01-01")
SCAN_STOP_DATR = _f("SCAN_STOP_DATR", 0.5)
SCAN_EXIT_MOD = _i("SCAN_EXIT_HOUR", 15) * 60 - 15   # last candle held: 14:45 NY, so trades close at 15:00 (away from the 17:00 rollover spread)


def scan_entries(d, pair):
    """For each signal name: list of (entry bar index, follow direction, session date)."""
    ny = d["time"].dt.tz_convert("America/New_York")
    ld = d["time"].dt.tz_convert("Europe/London")
    mod = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
    lmod = (ld.dt.hour * 60 + ld.dt.minute).to_numpy()
    sd = (ny + pd.Timedelta(hours=7)).dt.date.to_numpy()
    o, h, l, c = (d[k].to_numpy() for k in ("o", "h", "l", "c"))
    step = 1.0 if "JPY" in pair else 0.01
    out = {k: [] for k in ("LDN", "NY1H", "PDHL", "ASIA", "RND", "FIX", "H08", "H09", "H10", "H11")}
    days = pd.Series(np.arange(len(d))).groupby(sd).agg(["min", "max"])
    prev_hi = prev_lo = None
    for day, (a, b) in zip(days.index, days.to_numpy()):
        idx = np.arange(a, b + 1)
        m = mod[idx]
        at = {int(mm): int(ii) for mm, ii in zip(m, idx)}
        lat = {int(mm): int(ii) for mm, ii in zip(lmod[idx], idx)}
        win = [ii for ii in idx if 480 <= mod[ii] < 720]          # 08:00-11:45 NY bars
        def add(name, e, dirn):
            if dirn != 0 and e is not None and 480 <= mod[e] <= 720 and e <= b:
                out[name].append((e, dirn, day))
        if 180 in at and 480 in at:
            add("LDN", at[480], int(np.sign(o[at[480]] - o[at[180]])))
        if 480 in at and 540 in at:
            add("NY1H", at[540], int(np.sign(o[at[540]] - o[at[480]])))
        for hh, name in ((480, "H08"), (540, "H09"), (600, "H10"), (660, "H11")):
            if hh in at:
                add(name, at[hh], 1)
        if 900 in lat and 960 in lat and 480 <= mod[lat[960]] <= 720:
            add("FIX", lat[960], int(np.sign(o[lat[960]] - o[lat[900]])))
        asia = [ii for ii in idx if mod[ii] >= 1140 or mod[ii] < 180]
        a_hi = h[asia].max() if len(asia) >= 20 else None
        a_lo = l[asia].min() if len(asia) >= 20 else None
        done = {"PDHL": False, "ASIA": False, "RND": False}
        for ii in win:
            if ii + 1 > b:
                break
            if not done["PDHL"] and prev_hi is not None:
                if c[ii] > prev_hi:
                    add("PDHL", ii + 1, 1); done["PDHL"] = True
                elif c[ii] < prev_lo:
                    add("PDHL", ii + 1, -1); done["PDHL"] = True
            if not done["ASIA"] and a_hi is not None:
                if c[ii] > a_hi:
                    add("ASIA", ii + 1, 1); done["ASIA"] = True
                elif c[ii] < a_lo:
                    add("ASIA", ii + 1, -1); done["ASIA"] = True
            if not done["RND"] and ii - 1 >= a:
                if np.floor(c[ii - 1] / step) != np.floor(c[ii] / step):
                    add("RND", ii + 1, int(np.sign(c[ii] - c[ii - 1]))); done["RND"] = True
        prev_hi, prev_lo = h[idx].max(), l[idx].min()
    return out, mod


def scan_simulate(d, pair, entries, mod, datr, rr, flip):
    t = d["time"].tolist()
    bo, bh, bl, bc = (d[k].to_numpy() for k in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[k].to_numpy() for k in ("ao", "ah", "al", "ac"))
    n = len(d)
    trades = []
    for e, dirn, day in entries:
        dirn = -dirn if flip else dirn
        a = datr.get(day)
        if a is None or np.isnan(a):
            continue
        risk = SCAN_STOP_DATR * a
        entry = ao[e] if dirn == 1 else bo[e]
        spread = ao[e] - bo[e]
        if spread > MAX_SPREAD_RISK * risk:
            continue
        sl, tp = entry - dirn * risk, entry + dirn * rr * risk
        r, k = None, e
        while k < n:
            if k > e and (mod[k] > SCAN_EXIT_MOD or mod[k] < mod[e]):   # past the exit candle or into the next day
                k -= 1
                break
            if dirn == 1:
                if bl[k] <= sl:
                    r = -1.0; break
                if bh[k] >= tp:
                    r = rr; break
            else:
                if ah[k] >= sl:
                    r = -1.0; break
                if al[k] <= tp:
                    r = rr; break
            if mod[k] == SCAN_EXIT_MOD:                           # the exit candle: close at its close
                break
            k += 1
        k = min(max(k, e), n - 1)
        if r is None:
            r = dirn * ((bc[k] if dirn == 1 else ac[k]) - entry) / risk
        trades.append({"pair": pair, "entry": t[e], "exit": t[k], "dir": dirn, "r": float(r),
                       "risk_pips": risk / pip(pair), "spread_frac": spread / risk,
                       "exit_spread": (ac[k] - bc[k]) / risk, "timed": r not in (-1.0, rr)})
    return trades


def _scan_line(label, tr):
    s = stats(tr)
    o = sorted(tr, key=lambda x: x["entry"])
    h2 = len(o) // 2
    a1, a2 = stats(o[:h2])["avg"], stats(o[h2:])["avg"]
    return f"{label:<10}{s['n']:>5}{s['win']:>5.0f}%{s['avg']:>6.2f}{s['pf']:>5.2f}{a1:>6.2f}{a2:>6.2f}"


def _scan_checks(tr, pf_min, n_min):
    s = stats(tr)
    o = sorted(tr, key=lambda x: x["entry"])
    h2 = len(o) // 2
    if s["n"] < n_min or s["pf"] < pf_min or h2 == 0:
        return False
    if stats(o[:h2])["avg"] <= 0 or stats(o[h2:])["avg"] <= 0:
        return False
    for p in SCAN_PAIRS:
        rp = [x["r"] for x in tr if x["pair"] == p]
        if not rp or np.mean(rp) <= 0:
            return False
    return True


def run_scan(api, days):
    t0 = pd.Timestamp(SCAN_START, tz="UTC")
    split = pd.Timestamp(SCAN_SPLIT, tz="UTC")
    log.info(f"SCAN | M15 signal scanner | search {t0:%Y-%m-%d} to {split:%Y-%m-%d} | holdout {split:%Y-%m-%d} "
             f"to today | pairs {SCAN_PAIRS}")
    res = {}
    for pair in SCAN_PAIRS:
        try:
            d = fetch_range(api, pair, "M15", (t0 - pd.Timedelta(days=40)).to_pydatetime())
            if len(d) < 5000:
                log.info(f"{pair} | not enough history, skipped")
                continue
            entries, mod = scan_entries(d, pair)
            ny = d["time"].dt.tz_convert("America/New_York")
            sd = (ny + pd.Timedelta(hours=7)).dt.date
            dly = d.groupby(sd.to_numpy()).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
            pc = dly["c"].shift(1)
            trr = np.maximum(dly["h"] - dly["l"], np.maximum((dly["h"] - pc).abs(), (dly["l"] - pc).abs()))
            datr = trr.rolling(20).mean().shift(1).to_dict()          # known before the day starts
            for name, ent in entries.items():
                for rr in (1.0, 2.0):
                    for flip, tag in ((False, "f"), (True, "x")):
                        key = f"{name} {tag}{rr:g}"
                        res.setdefault(key, []).extend(
                            [x for x in scan_simulate(d, pair, ent, mod, datr, rr, flip) if x["entry"] >= t0])
            log.info(f"{pair} | done | {len(d)} candles from {d['time'].iloc[0]:%Y-%m-%d}")
            del d, entries, mod, dly, datr
            gc.collect()
        except Exception as e:
            log.error(f"{pair} | SCAN ERROR | {e}")

    search = {k: [x for x in v if x["entry"] < split] for k, v in res.items()}
    hold = {k: [x for x in v if x["entry"] >= split] for k, v in res.items()}
    log.info("")
    log.info(f"===== M15 SCAN | SEARCH {t0:%Y}-{split:%Y} | {len(search)} signals tested =====")
    log.info("f = follow, x = fade, number = target in R. Coin flip: 1:1 = 50%, 1:2 = 33%.")
    log.info(f"{'':<10}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
    ranked = sorted(search.items(), key=lambda kv: -stats(kv[1])["avg"] if kv[1] else 0)
    for k, v in ranked[:8]:
        log.info(_scan_line(k, v))
    passed = [k for k, v in search.items() if _scan_checks(v, 1.3, 300)]
    log.info(f"(top 8 of {len(search)} by avgR shown) | SEARCH passes: {len(passed)} -> {passed if passed else 'none'}")
    if passed:
        log.info("")
        log.info(f"===== HOLDOUT {split:%Y}-today (passers only, first time seen) =====")
        log.info(f"{'':<10}{'n':>5}{'win':>6}{'avgR':>6}{'PF':>5}{'1st':>6}{'2nd':>6}")
        for k in passed:
            ok = _scan_checks(hold[k], 1.2, 1)
            log.info(_scan_line(k, hold[k]) + ("  PASS" if ok else "  fail"))
            per = ", ".join(f"{p} {np.mean([x['r'] for x in hold[k] if x['pair'] == p]):+.2f}"
                            for p in SCAN_PAIRS if any(x["pair"] == p for x in hold[k]))
            log.info(f"   {k} holdout by pair: {per}")
    allr = [x for v in search.values() for x in v]
    if allr:
        log.info(f"Avg stop {np.mean([x['risk_pips'] for x in allr]):.1f} pips | entry spread "
                 f"{np.mean([x['spread_frac'] for x in allr]) * 100:.1f}% of risk | spread at timed exits "
                 f"{np.mean([x['exit_spread'] for x in allr if x['timed']] or [0]) * 100:.1f}% of risk | "
                 f"{np.mean([x['timed'] for x in allr]) * 100:.0f}% of trades closed by the time exit")
    log.info("A signal counts only if it passes SEARCH and then HOLDOUT. Then it goes to the demo account first.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# LIVE
# ----------------------------------------------------------------------------
_NEWS = {"t": 0.0, "events": None}


def news_events():
    """High-impact events this week from the free Forex Factory feed, refreshed every 3 hours."""
    now = time.time()
    if now - _NEWS["t"] < 3 * 3600:
        return _NEWS["events"]
    try:
        r = requests.get(NEWS_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0 (trading bot)"})
        r.raise_for_status()
        events = []
        for e in r.json():
            if str(e.get("impact", "")).lower() != "high":
                continue
            try:
                ts = pd.Timestamp(e["date"]).tz_convert("UTC")
            except Exception:
                continue
            events.append((str(e.get("country", "")).upper(), ts, str(e.get("title", ""))))
        _NEWS.update(t=now, events=events)
        log.info(f"news calendar loaded | {len(events)} high-impact events this week")
    except Exception as ex:
        _NEWS["t"] = now - 3 * 3600 + 900   # try again in 15 minutes
        log.error(f"NEWS FEED ERROR | {ex} | keeping the last calendar" if _NEWS["events"] is not None
                  else f"NEWS FEED ERROR | {ex} | no calendar yet ({NEWS_FAIL_MODE} trading)")
    return _NEWS["events"]


def news_block(pair, now):
    if not NEWS_FILTER:
        return None
    events = news_events()
    if events is None:
        return "news feed unavailable" if NEWS_FAIL_MODE == "block" else None
    ccys = pair.split("_")
    for ccy, ts, title in events:
        if ccy in ccys and abs((ts - now).total_seconds()) <= NEWS_BLOCK_MIN * 60:
            return f"news ({ccy} {title} at {ts:%H:%M} UTC)"
    return None


def ny_block(now):
    if not NY_BLOCK:
        return None
    try:
        a, b = NY_BLOCK.split("-")
        start = int(a.split(":")[0]) * 60 + int(a.split(":")[1])
        end = int(b.split(":")[0]) * 60 + int(b.split(":")[1])
    except (ValueError, IndexError):
        return None
    t = now.tz_convert("America/New_York")
    m = t.hour * 60 + t.minute
    return f"late New York hours ({NY_BLOCK} NY time)" if start <= m < end else None


def exposure(trades):
    """Net same-direction count per currency (long EUR_USD = +1 EUR, -1 USD)."""
    ex = {}
    for t in trades:
        base, quote = t["instrument"].split("_")
        d = 1 if float(t["currentUnits"]) > 0 else -1
        ex[base] = ex.get(base, 0) + d
        ex[quote] = ex.get(quote, 0) - d
    return ex


def exposure_block(ex, pair, direction):
    base, quote = pair.split("_")
    if abs(ex.get(base, 0) + direction) > MAX_PER_CURRENCY:
        return f"too much {base} exposure"
    if abs(ex.get(quote, 0) - direction) > MAX_PER_CURRENCY:
        return f"too much {quote} exposure"
    return None


def add_exposure(ex, pair, direction):
    base, quote = pair.split("_")
    ex[base] = ex.get(base, 0) + direction
    ex[quote] = ex.get(quote, 0) - direction


def size_units(api, pair, risk_price):
    """Units so that hitting the stop loses about RISK_PCT% of the balance (falls back to UNITS)."""
    if RISK_PCT <= 0:
        return UNITS, "fixed"
    try:
        acct = api.get(f"/v3/accounts/{api.account}/summary")["account"]
        balance, ccy = float(acct["balance"]), acct["currency"]
        quote = pair.split("_")[1]
        conv = 1.0 if quote == ccy else _HOME_CONV.get(quote)
        if conv is None:
            raise RuntimeError(f"no {quote}->{ccy} rate")
        units = int(balance * RISK_PCT / 100 / (risk_price * conv))
        return max(0, min(units, MAX_UNITS)), f"{RISK_PCT:g}% risk"
    except Exception as e:
        log.info(f"{pair} | sizing fell back to {UNITS} units ({e})")
        return UNITS, "fixed (fallback)"


def place_order(api, pair, direction, sl, tp=None, size=None):
    size = UNITS if size is None else size
    units = size if direction == 1 else -size
    order = {
        "type": "MARKET",
        "instrument": pair,
        "units": str(units),
        "timeInForce": "FOK",
        "positionFill": "DEFAULT",
        "stopLossOnFill": {"price": fmt_price(pair, sl), "timeInForce": "GTC"},
        "tradeClientExtensions": {"tag": MANAGED_TAG, "comment": f"UT+ {TIMEFRAME} {EXIT_MODE}"},
    }
    if tp is not None:  # flip mode has no target: the bot closes the trade itself when UT flips
        order["takeProfitOnFill"] = {"price": fmt_price(pair, tp), "timeInForce": "GTC"}
    body = {"order": order}
    status, j = api.post(f"/v3/accounts/{api.account}/orders", body)
    fill = j.get("orderFillTransaction")
    if fill:
        trade_id = (fill.get("tradeOpened") or {}).get("tradeID", "?")
        return True, f"ORDER FILLED | trade {trade_id} @ {fill.get('price')}"
    cancel = j.get("orderCancelTransaction")
    if cancel:
        return False, f"ORDER CANCELLED | {cancel.get('reason')}"
    return False, f"ORDER REJECTED | HTTP {status} | {str(j)[:200]}"


def manage_flip_exits(api, pair, d, trades_by_pair, open_pairs):
    """EXIT_MODE=flip: close this bot's trades on a pair once UT Bot has flipped the other way since they opened."""
    mine = [t for t in trades_by_pair.get(pair, []) if (t.get("clientExtensions") or {}).get("tag") == MANAGED_TAG]
    if not mine:
        return
    flips = d[d["ut_buy"] | d["ut_sell"]]
    if flips.empty:
        return
    last = flips.iloc[-1]
    last_dir = 1 if last["ut_buy"] else -1
    for t in mine:
        trade_dir = 1 if float(t["currentUnits"]) > 0 else -1
        opened = pd.Timestamp(t["openTime"])
        # only a flip that closed AFTER the trade opened counts (the flip that opened it does not)
        if last_dir == trade_dir or last["close_time"] <= opened:
            continue
        status, j = api.put(f"/v3/accounts/{api.account}/trades/{t['id']}/close", {"units": "ALL"})
        fill = j.get("orderFillTransaction")
        if fill:
            log.info(f"{pair} | FLIP EXIT | closed trade {t['id']} @ {fill.get('price')} | P/L {fill.get('pl')}")
            trades_by_pair[pair] = [x for x in trades_by_pair[pair] if x["id"] != t["id"]]
        else:
            log.info(f"{pair} | FLIP EXIT FAILED | trade {t['id']} | HTTP {status} | {str(j)[:200]}")
    if not trades_by_pair.get(pair):
        trades_by_pair.pop(pair, None)
        open_pairs.discard(pair)


def process_pair(api, pair, prices, open_pairs, trades_by_pair=None, ex=None):
    sig = fetch_recent(api, pair, TIMEFRAME, 500)
    if len(sig) < 200:
        log.info(f"{pair} | NO TRADE | blocked: not enough candles")
        return
    htf = fetch_recent(api, pair, HTF, 300)
    d = add_htf_trend(prepare(sig), htf)
    row = d.iloc[-1]
    now = pd.Timestamp(datetime.now(timezone.utc))
    if EXIT_MODE == "flip" and trades_by_pair:
        manage_flip_exits(api, pair, d, trades_by_pair, open_pairs)  # runs even on a stale candle, so a missed flip is caught up
    if (now - row["close_time"]).total_seconds() > STALE_SECONDS:
        return False  # no fresh candle for this pair yet (market closed, or between candle closes)

    trend = trend_label(row)
    direction = 1 if row["ut_buy"] else (-1 if row["ut_sell"] else 0)
    if direction == 0:
        log.info(f"{pair} | {trend} | NO TRADE | blocked: no_signal")
        return
    side = "BUY" if direction == 1 else "SELL"

    if pair in open_pairs:
        log.info(f"{pair} | {trend} | {side} | blocked: position already open")
        return
    safety = ny_block(now) or news_block(pair, now) or (exposure_block(ex, pair, direction) if ex is not None else None)
    if safety:
        log.info(f"{pair} | {trend} | {side} | blocked: {safety}")
        return
    if len(open_pairs) >= MAX_OPEN_TRADES:
        log.info(f"{pair} | {trend} | {side} | blocked: max open trades")
        return
    if pair not in prices:
        log.info(f"{pair} | {trend} | {side} | blocked: no live price / not tradeable")
        return

    bid, ask = prices[pair]
    entry = ask if direction == 1 else bid
    risk = calc_risk(pair, row["atr"], row["ut_loss"])
    dly = fetch_recent(api, pair, "D", 120)
    res, sup = sr_levels(dly, now)
    reason = filter_reason(row, direction, risk, entry, res, sup)
    if reason:
        log.info(f"{pair} | {trend} | NO TRADE | blocked: {reason}")
        return
    spread_ratio = (ask - bid) / risk
    if spread_ratio > MAX_SPREAD_RISK:
        log.info(f"{pair} | {trend} | {side} | SKIPPED: spread too wide ({spread_ratio * 100:.0f}% of risk)")
        return

    sl = entry - direction * risk
    if EXIT_MODE == "flip":
        tp = None
        log.info(
            f"{pair} | {trend} | {side} | SIGNAL ENTRY {fmt_price(pair, entry)} "
            f"SL {fmt_price(pair, sl)} | exit when UT flips"
        )
    else:
        tp = entry + direction * RR * risk
        log.info(
            f"{pair} | {trend} | {side} | SIGNAL ENTRY {fmt_price(pair, entry)} "
            f"SL {fmt_price(pair, sl)} TP {fmt_price(pair, tp)} | RR 1:{RR:.1f}"
        )
    units, how = size_units(api, pair, risk)
    if units < 1:
        log.info(f"{pair} | {trend} | {side} | SKIPPED: size rounds to 0 units")
        return
    ok, msg = place_order(api, pair, direction, sl, tp, units)
    log.info(f"{pair} | {msg} | {units} units ({how})")
    if ok:
        open_pairs.add(pair)
        if ex is not None:
            add_exposure(ex, pair, direction)


def live_cycle(api):
    trades = api.get(f"/v3/accounts/{api.account}/openTrades").get("trades", [])
    trades_by_pair = {}
    for t in trades:
        trades_by_pair.setdefault(t["instrument"], []).append(t)
    open_pairs = set(trades_by_pair)
    ex = exposure(trades)
    prices = get_prices(api)
    log.info(f"--- scan | {len(trades)} open trade(s) | exit {EXIT_MODE} ---")
    evaluated = 0
    for pair in PAIRS:
        try:
            if process_pair(api, pair, prices, open_pairs, trades_by_pair, ex) is not False:
                evaluated += 1
        except Exception as e:
            log.error(f"{pair} | ERROR | {e}")
    if evaluated == 0:
        log.info(f"no new {TIMEFRAME} candle yet (market closed, or waiting for the next close)")


def seconds_to_next_bar():
    now = datetime.now(timezone.utc)
    mins = 15 - (now.minute % 15)
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=mins)
    return max(1.0, (nxt - now).total_seconds() + 5)


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("MODE", "live")).lower()
    api = Oanda()
    if mode == "backtest":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else _i("BACKTEST_DAYS", 90)
        strategy = os.getenv("BACKTEST_STRATEGY", "ut").lower()
        if strategy == "srbb":
            run_backtest_srbb(api, days)
        elif strategy == "flux":
            run_backtest_flux(api, days)
        elif strategy == "po3s":
            run_backtest_po3s(api, days)
        elif strategy == "po3c":
            run_backtest_po3c(api, days)
        elif strategy == "po3e":
            run_backtest_po3e(api, days)
        elif strategy == "lbo":
            run_backtest_lbo(api, days)
        elif strategy == "po3x":
            run_backtest_po3x(api, days)
        elif strategy == "smc":
            run_backtest_smc(api, days)
        elif strategy == "combo":
            run_backtest_combo(api, days)
        elif strategy == "smce":
            run_backtest_smce(api, days)
        elif strategy == "po3v2":
            run_backtest_po3v2(api, days)
        elif strategy == "po3w":
            run_backtest_po3w(api, days)
        elif strategy == "tsm":
            run_backtest_tsm(api, days)
        elif strategy == "scan":
            run_scan(api, days)
        else:
            run_backtest(api, days)
        while True:  # idle so Railway does not keep re-running it
            time.sleep(3600)

    stop_txt = f"{SL_ATR_MULT:g}xATR" if STOP_MODE == "atr" else "UT trail"
    exit_txt = f"fixed (RR 1:{RR:g})" if EXIT_MODE == "fixed" else "flip"
    log.info(
        f"UT+ bot live | {TIMEFRAME} | UT_KEY {UT_KEY:g} | ADX_MIN {ADX_MIN:g} | {len(PAIRS)} pairs | "
        f"units {UNITS} | stop {stop_txt} (min {MIN_STOP_PIPS} pips) | exit {exit_txt}"
    )
    log.info(
        f"safety | size {'%g%% risk (max %d units)' % (RISK_PCT, MAX_UNITS) if RISK_PCT > 0 else 'fixed %d units' % UNITS} | "
        f"max {MAX_PER_CURRENCY} per currency | news filter {'on, %d min' % NEWS_BLOCK_MIN if NEWS_FILTER else 'off'} | "
        f"no trades {NY_BLOCK or 'none'} NY time"
    )
    while True:
        wait = seconds_to_next_bar()
        log.info(f"Waiting {wait:.0f}s for the next 15-minute mark...")
        time.sleep(wait)
        try:
            live_cycle(api)
        except Exception as e:
            log.error(f"CYCLE ERROR | {e}")


if __name__ == "__main__":
    main()
