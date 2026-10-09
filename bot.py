"""
UT+ bot for OANDA  (bot.py)

Signal : UT Bot style ATR trailing stop flip on M15 (same maths as the TradingView script)
Filters: ADX (no ranges), Bollinger squeeze, tick volume, H1 trend (EMA50/EMA200),
         daily support/resistance, spread vs risk
Risk   : stop = ATR x multiplier (with a minimum), take profit = 3 x risk (1:3)
Modes  : live (default)  |  backtest  (UT baseline vs UT+ on OANDA history)

Run:      python bot.py
Backtest: set Railway variable MODE=backtest (optionally BACKTEST_DAYS=90) and redeploy,
          or run: python bot.py backtest 90
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

MAX_BARS_HELD = _i("MAX_BARS_HELD", 96)      # backtest only: close trade after 24h
STALE_SECONDS = _i("STALE_SECONDS", 300)     # ignore signals older than this (live)


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
                if r.status_code in (429, 500, 502, 503, 504):
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


def prepare(m15):
    d = m15.copy()
    d["ut_stop"], d["ut_buy"], d["ut_sell"] = ut_bot(d, UT_KEY, UT_ATR)
    d["atr"] = calc_atr(d, ATR_PERIOD)
    d["adx"] = calc_adx(d, 14)
    mid = d["c"].rolling(20).mean()
    sd = d["c"].rolling(20).std(ddof=0)
    d["bw"] = (4 * sd) / mid
    d["bw_cut"] = d["bw"].rolling(100).quantile(SQUEEZE_PCT)
    d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()
    d["close_time"] = d["time"] + pd.Timedelta(minutes=15)
    return d


def add_htf_trend(d, h1):
    """Attach the H1 trend (1 up, -1 down, 0 unknown) using only COMPLETED H1 candles."""
    h = h1.copy().reset_index(drop=True)
    h["ema50"] = h["c"].ewm(span=50, adjust=False).mean()
    h["ema200"] = h["c"].ewm(span=200, adjust=False).mean()
    h["htf"] = np.where(h["ema50"] > h["ema200"], 1, -1)
    h.loc[h.index < 200, "htf"] = 0  # not enough history yet
    h["avail"] = h["time"] + pd.Timedelta(hours=1)
    m = pd.merge_asof(
        d.sort_values("close_time"),
        h[["avail", "htf"]].sort_values("avail"),
        left_on="close_time",
        right_on="avail",
        direction="backward",
    )
    m["htf"] = m["htf"].fillna(0).astype(int)
    return m.reset_index(drop=True)


def sr_levels(daily, as_of, lookback=90):
    """Confirmed daily swing highs (resistance) and lows (support) known at time `as_of`."""
    if daily is None or daily.empty:
        return [], []
    d = daily[daily["time"] + pd.Timedelta(days=1) <= as_of].tail(lookback).reset_index(drop=True)
    h = d["h"].to_numpy()
    l = d["l"].to_numpy()
    res, sup = [], []
    for i in range(2, len(d) - 2):
        if h[i] >= h[i - 2:i + 3].max():
            res.append(h[i])
        if l[i] <= l[i - 2:i + 3].min():
            sup.append(l[i])
    return res, sup


def calc_risk(pair, atr):
    return max(SL_ATR_MULT * atr, MIN_STOP_PIPS * pip(pair))


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
def simulate(d, pair, daily, start_ts, use_filters):
    times = d["time"].tolist()
    bo, bh, bl, bc = (d[c].to_numpy() for c in ("bo", "bh", "bl", "bc"))
    ao, ah, al, ac = (d[c].to_numpy() for c in ("ao", "ah", "al", "ac"))
    atr = d["atr"].to_numpy()
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
        risk = calc_risk(pair, atr[i])
        reason = None
        if use_filters:
            res, sup = sr_levels(daily, d["close_time"].iloc[i])
            reason = filter_reason(d.iloc[i], direction, risk, entry, res, sup)
        if reason is None and (ao[j] - bo[j]) / risk > MAX_SPREAD_RISK:
            reason = "spread"
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        sl = entry - direction * risk
        tp = entry + direction * RR * risk
        end = min(j + MAX_BARS_HELD, n - 1)
        r = None
        k = j
        for k in range(j, end + 1):
            if direction == 1:  # long exits at the bid
                if bl[k] <= sl:
                    r = -1.0
                    break
                if bh[k] >= tp:
                    r = RR
                    break
            else:  # short exits at the ask
                if ah[k] >= sl:
                    r = -1.0
                    break
                if al[k] <= tp:
                    r = RR
                    break
        if r is None:
            exit_p = bc[k] if direction == 1 else ac[k]
            r = direction * (exit_p - entry) / risk
        busy = k
        trades.append({"pair": pair, "entry": times[j], "exit": times[k], "dir": direction, "r": float(r)})
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
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ts = pd.Timestamp(start)
    log.info(f"BACKTEST | last {days} days | {len(PAIRS)} pairs | fetching OANDA history...")
    base_all, plus_all, skip_all = [], [], {}
    per_pair = []
    for pair in PAIRS:
        try:
            m15 = fetch_range(api, pair, "M15", start - timedelta(days=5))
            h1 = fetch_range(api, pair, "H1", start - timedelta(days=25))
            dly = fetch_range(api, pair, "D", start - timedelta(days=120))
            if len(m15) < 300 or len(h1) < 250:
                log.info(f"{pair} | not enough history, skipped")
                continue
            d = add_htf_trend(prepare(m15), h1)
            base, _ = simulate(d, pair, dly, start_ts, use_filters=False)
            plus, sk = simulate(d, pair, dly, start_ts, use_filters=True)
            base_all += base
            plus_all += plus
            for k, v in sk.items():
                skip_all[k] = skip_all.get(k, 0) + v
            per_pair.append((pair, stats(base), stats(plus)))
            log.info(f"{pair} | done | baseline {len(base)} trades | UT+ {len(plus)} trades")
        except Exception as e:
            log.error(f"{pair} | BACKTEST ERROR | {e}")

    log.info("")
    log.info(f"===== BACKTEST RESULTS (last {days} days, all results in R = multiples of risk) =====")
    log.info(f"{'':<18}{'trades':>6}{'win%':>9}{'avgR':>8}{'totalR':>9}{'PF':>7}{'maxDD':>9}")
    log.info(stat_line("UT baseline", base_all))
    log.info(stat_line("UT+ (filters)", plus_all))
    ordered = sorted(plus_all, key=lambda t: t["entry"])
    half = len(ordered) // 2
    log.info(stat_line("UT+ 1st half", ordered[:half]))
    log.info(stat_line("UT+ 2nd half", ordered[half:]))
    log.info("")
    log.info("Signals blocked by UT+ filters: " + (", ".join(f"{k} {v}" for k, v in sorted(skip_all.items(), key=lambda x: -x[1])) or "none"))
    log.info("")
    log.info(f"{'pair':<10}{'base n':>8}{'base R':>9}{'UT+ n':>8}{'UT+ R':>9}")
    for pair, b, p in per_pair:
        log.info(f"{pair:<10}{b['n']:>8}{b['total']:>9.1f}{p['n']:>8}{p['total']:>9.1f}")
    log.info("")
    be = 100.0 / (1.0 + RR)
    log.info(f"Break-even win rate at 1:{RR:g} is {be:.0f}%. Look for: positive avgR in BOTH halves, PF above 1.2, enough trades (50+).")
    log.info("Backtests ignore slippage, swap fees and news gaps. Treat them as a sanity check, not a promise.")
    log.info("Backtest finished. Remove the MODE variable (or set MODE=live) and redeploy to go live.")


# ----------------------------------------------------------------------------
# LIVE
# ----------------------------------------------------------------------------
def place_order(api, pair, direction, sl, tp):
    units = UNITS if direction == 1 else -UNITS
    body = {
        "order": {
            "type": "MARKET",
            "instrument": pair,
            "units": str(units),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": fmt_price(pair, sl), "timeInForce": "GTC"},
            "takeProfitOnFill": {"price": fmt_price(pair, tp), "timeInForce": "GTC"},
        }
    }
    status, j = api.post(f"/v3/accounts/{api.account}/orders", body)
    fill = j.get("orderFillTransaction")
    if fill:
        trade_id = (fill.get("tradeOpened") or {}).get("tradeID", "?")
        return True, f"ORDER FILLED | trade {trade_id} @ {fill.get('price')}"
    cancel = j.get("orderCancelTransaction")
    if cancel:
        return False, f"ORDER CANCELLED | {cancel.get('reason')}"
    return False, f"ORDER REJECTED | HTTP {status} | {str(j)[:200]}"


def process_pair(api, pair, prices, open_pairs):
    m15 = fetch_recent(api, pair, "M15", 500)
    if len(m15) < 200:
        log.info(f"{pair} | NO TRADE | blocked: not enough candles")
        return
    h1 = fetch_recent(api, pair, "H1", 300)
    d = add_htf_trend(prepare(m15), h1)
    row = d.iloc[-1]
    now = pd.Timestamp(datetime.now(timezone.utc))
    if (now - row["close_time"]).total_seconds() > STALE_SECONDS:
        log.info(f"{pair} | NO TRADE | blocked: stale data (market closed?)")
        return

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
    risk = calc_risk(pair, row["atr"])
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
    open_pairs = {t["instrument"] for t in trades}
    prices = get_prices(api)
    log.info(f"--- scan | {len(open_pairs)} open trade(s) ---")
    for pair in PAIRS:
        try:
            process_pair(api, pair, prices, open_pairs)
        except Exception as e:
            log.error(f"{pair} | ERROR | {e}")


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
        run_backtest(api, days)
        while True:  # idle so Railway does not keep re-running it
            time.sleep(3600)

    log.info(
        f"UT+ bot live | {len(PAIRS)} pairs | units {UNITS} | SL {SL_ATR_MULT}xATR (min {MIN_STOP_PIPS} pips) | RR 1:{RR:g}"
    )
    while True:
        wait = seconds_to_next_bar()
        log.info(f"Waiting {wait:.0f}s for the next 15-minute candle close...")
        time.sleep(wait)
        try:
            live_cycle(api)
        except Exception as e:
            log.error(f"CYCLE ERROR | {e}")


if __name__ == "__main__":
    main()
