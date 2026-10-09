"""
UT+ bot for OANDA  (bot.py)

Signal : UT Bot style ATR trailing stop flip on M15, H1 or H4 (TIMEFRAME variable, default M15)
Filters: ADX (no ranges), Bollinger squeeze, tick volume, higher-timeframe trend (EMA50/EMA200),
         daily support/resistance, spread vs risk
Risk   : stop = 1.5 x ATR (STOP_MODE=atr) or the UT trailing-stop distance (STOP_MODE=ut), with a minimum size
Modes  : live (default)  |  backtest  (UT baseline vs UT+ on OANDA history)
         BACKTEST_STRATEGY=srbb backtests your original bot instead (daily S/R + Bollinger bounce)
         BACKTEST_STRATEGY=flux backtests the two Flux Charts scripts (ICT Power Of Three, Volumized Order Blocks)
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
FLIP_MAX_BARS = _i("FLIP_MAX_BARS", {"M15": 2880, "H1": 960, "H4": 240}.get(os.getenv("TIMEFRAME", "M15").upper(), 960))  # backtest only: ~30-40 days
SR_TP_BUFFER_ATR = _f("SR_TP_BUFFER_ATR", 0.25)  # "sr" exit: take profit this many ATR before the level (the zone edge)
SR_MIN_R = _f("SR_MIN_R", 1.0)               # "sr" exit: ignore levels closer than this many R
SR_MAX_R = _f("SR_MAX_R", 6.0)               # "sr" exit: levels farther than this are ignored (trade just rides to the flip)
STALE_SECONDS = _i("STALE_SECONDS", 300)     # ignore signals older than this (live)


TIMEFRAME = os.getenv("TIMEFRAME", "M15").upper()
TF_MIN = {"M15": 15, "H1": 60, "H4": 240}            # candle length in minutes
HTF_OF = {"M15": "H1", "H1": "H4", "H4": "D"}        # trend-filter timeframe
HTF_MIN = {"H1": 60, "H4": 240, "D": 1440}
SIG_WARMUP = {"M15": 5, "H1": 15, "H4": 40}          # backtest: extra days of history for indicators
HTF_WARMUP = {"M15": 25, "H1": 70, "H4": 320}        # backtest: extra days for the trend EMAs
if TIMEFRAME not in TF_MIN:
    raise RuntimeError(f"TIMEFRAME must be one of {list(TF_MIN)}, got {TIMEFRAME!r}")
HTF = HTF_OF[TIMEFRAME]
EXIT_MODE = os.getenv("EXIT_MODE", "fixed").lower()  # live exits: "fixed" = stop + RR target, "flip" = stop + close on UT flip
if EXIT_MODE not in ("fixed", "flip"):
    raise RuntimeError(f"EXIT_MODE must be 'fixed' or 'flip', got {EXIT_MODE!r}")
STOP_MODE = os.getenv("STOP_MODE", "atr").lower()    # "atr": stop = SL_ATR_MULT x ATR(14); "ut": stop = the UT trailing-stop distance (UT_KEY x ATR(UT_ATR))
if STOP_MODE not in ("atr", "ut"):
    raise RuntimeError(f"STOP_MODE must be 'atr' or 'ut', got {STOP_MODE!r}")
MANAGED_TAG = "utflip"                               # marks trades opened by this bot, so old/manual trades are never touched


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
    raw_all = []
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
        raw_all.extend(raw)
        if len(raw) < 5000:
            break
        cur = pd.Timestamp(raw[-1]["time"]).to_pydatetime() + timedelta(seconds=1)
        time.sleep(0.2)
    return parse_candles(raw_all)


def get_prices(api):
    data = api.get(f"/v3/accounts/{api.account}/pricing", {"instruments": ",".join(PAIRS)})
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


def simulate_po3(d, pair, start_ts):
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
# LIVE
# ----------------------------------------------------------------------------
def place_order(api, pair, direction, sl, tp=None):
    units = UNITS if direction == 1 else -UNITS
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


def process_pair(api, pair, prices, open_pairs, trades_by_pair=None):
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
    ok, msg = place_order(api, pair, direction, sl, tp)
    log.info(f"{pair} | {msg}")
    if ok:
        open_pairs.add(pair)


def live_cycle(api):
    trades = api.get(f"/v3/accounts/{api.account}/openTrades").get("trades", [])
    trades_by_pair = {}
    for t in trades:
        trades_by_pair.setdefault(t["instrument"], []).append(t)
    open_pairs = set(trades_by_pair)
    prices = get_prices(api)
    log.info(f"--- scan | {len(trades)} open trade(s) | exit {EXIT_MODE} ---")
    evaluated = 0
    for pair in PAIRS:
        try:
            if process_pair(api, pair, prices, open_pairs, trades_by_pair) is not False:
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
