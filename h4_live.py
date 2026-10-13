"""
H4 Power Of Three + LuxAlgo Echo Forecast — LIVE FORWARD TEST on the OANDA PRACTICE account.

Frozen rules (identical to bot.py's simulate_po3_echo with use_echo=True, use_bos=False, the "Base"
row of the po3w wide test). Do not change them during the forward test.
  - H4 candles, 12 pairs (the wide-test list)
  - Accumulation: 40-candle high-low range <= 5 x ATR(50); box = that range +/- 0.5 ATR
  - Fake move: price leaves the box by more than 0.6 ATR -> fade it (break up = SELL, break down = BUY)
  - Echo Forecast (50/50, similarity) must point the same way, else skip
  - Enter at market on the next candle's open; stop 5 x ATR(50); target 0.86R; hold until stop or target
  - One position per pair; no new signal while the simulated trade for that pair would still be open

Safety
  - Practice only: refuses to start unless OANDA_ENV is practice (or unset) and H4_LIVE_CONFIRM=yes
  - Every order carries a stop loss and take profit, and a unique client ID per pair + candle, so OANDA
    itself rejects a duplicate if the bot restarts
  - Orders are never retried automatically
  - Each trade's 1R distance is stored on the trade itself (OANDA client comment), so results in R
    survive restarts without needing a Railway Volume

Railway variables (new service, start command: python h4_live.py)
  OANDA_TOKEN / OANDA_API_KEY, OANDA_ACCOUNT_ID   (same as the existing bot)
  OANDA_ENV=practice
  H4_LIVE_CONFIRM=yes
  H4_RISK_PCT=0.5        percent of NAV risked per trade (0 = use H4_UNITS fixed size)
  H4_UNITS=1000          only used when H4_RISK_PCT=0
  H4_MAX_UNITS=100000    hard cap on any order
"""
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("h4live")

# ---------------------------------------------------------------------------- frozen strategy settings
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD",
         "USD_CHF", "EUR_JPY", "GBP_JPY", "EUR_GBP", "AUD_JPY", "EUR_AUD"]
GRAN = "H4"
ACC_LEN, ACC_MULT, MANIP_MULT = 40, 5.0, 0.6   # PO3 "small" preset
SL_ATR, RR = 5.0, 0.86
ECHO_LEN = ECHO_FCAST = 50
HISTORY = 1000                                 # H4 candles fetched each check (~8 months of warm-up)
TAG = "h4po3e"


def _f(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


RISK_PCT = _f("H4_RISK_PCT", 0.5)
UNITS = int(_f("H4_UNITS", 1000))
MAX_UNITS = int(_f("H4_MAX_UNITS", 100000))


def pip(pair):
    return 0.01 if "JPY" in pair else 0.0001


def px_round(pair, x):
    return f"{x:.3f}" if "JPY" in pair else f"{x:.5f}"


# ---------------------------------------------------------------------------- OANDA
class Oanda:
    def __init__(self):
        token = next((os.getenv(n).strip() for n in ("OANDA_API_KEY", "OANDA_API_TOKEN", "OANDA_TOKEN",
                                                       "OANDA_ACCESS_TOKEN") if os.getenv(n)), None)
        account = next((os.getenv(n).strip() for n in ("OANDA_ACCOUNT_ID", "OANDA_ACCOUNT") if os.getenv(n)), None)
        if not token or not account:
            raise RuntimeError("Missing OANDA token or account id in Railway variables.")
        if os.getenv("OANDA_ENV", "practice").strip().lower() != "practice":
            raise RuntimeError("This forward test runs on PRACTICE only. Set OANDA_ENV=practice.")
        self.base = "https://api-fxpractice.oanda.com"   # hard-coded: this file can never reach real money
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
        r = self.s.post(self.base + path, json=body, timeout=30)   # never retried
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"raw": r.text[:300]}


def parse_candles(raw):
    rows = []
    for c in raw:
        if not c.get("complete", False):
            continue
        rows.append({
            "time": c["time"],
            "o": float(c["mid"]["o"]), "h": float(c["mid"]["h"]), "l": float(c["mid"]["l"]), "c": float(c["mid"]["c"]),
            "bo": float(c["bid"]["o"]), "bh": float(c["bid"]["h"]), "bl": float(c["bid"]["l"]), "bc": float(c["bid"]["c"]),
            "ao": float(c["ask"]["o"]), "ah": float(c["ask"]["h"]), "al": float(c["ask"]["l"]), "ac": float(c["ask"]["c"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def fetch_recent(api, pair, count=HISTORY):
    data = api.get(f"/v3/instruments/{pair}/candles", {"granularity": GRAN, "price": "BAM", "count": count})
    return parse_candles(data.get("candles", []))


# ---------------------------------------------------------------------------- signal (copied from bot.py)
def calc_atr(df, n):
    pc = df["c"].shift(1)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - pc).abs(), (df["l"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def echo_direction(c, i):
    L, F = ECHO_LEN, ECHO_FCAST
    need = 2 * F + L
    if i < need:
        return 0
    A = c[i - need:i + 1][::-1]
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
    move = A[F + k] - A[2 * F + k]
    return 1 if move > 0 else (-1 if move < 0 else 0)


def _exit_bar(direction, j, sl, tp, bh, bl, ah, al, n):
    """Index of the candle that hits stop or target, or None if the trade would still be open."""
    for k in range(j, n):
        if direction == 1 and (bl[k] <= sl or bh[k] >= tp):
            return k
        if direction == -1 and (ah[k] >= sl or al[k] <= tp):
            return k
    return None


def scan_last_bar(d):
    """Run the backtest's state machine over the candles. Returns (direction, atr) if the LAST closed candle
    is a signal (enter at the next open), else (0, None). Mirrors simulate_po3_echo(use_echo=True, use_bos=False),
    including skipping signals while that pair's simulated trade would still be open."""
    h, l, c = d["h"].to_numpy(), d["l"].to_numpy(), d["c"].to_numpy()
    bo, bh, bl = d["bo"].to_numpy(), d["bh"].to_numpy(), d["bl"].to_numpy()
    ao, ah, al = d["ao"].to_numpy(), d["ah"].to_numpy(), d["al"].to_numpy()
    atr = calc_atr(d, 50).to_numpy()
    hh = d["h"].rolling(ACC_LEN).max().to_numpy()
    ll = d["l"].rolling(ACC_LEN).min().to_numpy()
    n = len(d)
    state, i, acc_top, acc_bot, acc_end = 0, 0, None, None, 0
    while i < n:
        a = atr[i]
        if np.isnan(hh[i]) or np.isnan(a):
            i += 1
            continue
        if state == 0 and hh[i] - ll[i] <= ACC_MULT * a:
            state, acc_end = 1, i
            acc_top, acc_bot = hh[i] + 0.5 * a, ll[i] - 0.5 * a
        if state == 1:
            if h[i] > acc_top or l[i] < acc_bot:
                state = 2
            else:
                acc_end = i
        direction = 0
        if state == 2 and i > acc_end:
            if h[i] > acc_top + MANIP_MULT * a:
                direction = -1
            elif l[i] < acc_bot - MANIP_MULT * a:
                direction = 1
        if direction == 0:
            i += 1
            continue
        state = 0
        if echo_direction(c, i) != direction:
            i += 1
            continue
        if i == n - 1:
            return direction, float(a)
        j = i + 1
        entry = ao[j] if direction == 1 else bo[j]
        risk = SL_ATR * a
        sl, tp = entry - direction * risk, entry + direction * RR * risk
        k = _exit_bar(direction, j, sl, tp, bh, bl, ah, al, n)
        if k is None:
            return 0, None          # simulated trade still open: no new signal on this pair
        i = k + 1
    return 0, None


# ---------------------------------------------------------------------------- trading
def open_tagged_trades(api):
    data = api.get(f"/v3/accounts/{api.account}/openTrades")
    return [t for t in data.get("trades", []) if t.get("clientExtensions", {}).get("tag") == TAG]


def account_nav(api):
    return float(api.get(f"/v3/accounts/{api.account}/summary")["account"]["NAV"])


def quote_and_conv(api, pair):
    data = api.get(f"/v3/accounts/{api.account}/pricing", {"instruments": pair, "includeHomeConversions": "true"})
    p = data["prices"][0]
    if not p.get("tradeable", True):
        return None
    bid, ask = float(p["bids"][0]["price"]), float(p["asks"][0]["price"])
    quote_ccy = pair.split("_")[1]
    conv = next((float(hc["accountLoss"]) for hc in data.get("homeConversions", [])
                 if hc.get("currency") == quote_ccy), None)
    return bid, ask, conv


def place(api, pair, direction, atr, bar_time):
    q = quote_and_conv(api, pair)
    if q is None:
        log.info(f"{pair} | signal skipped: not tradeable right now")
        return
    bid, ask, conv = q
    entry = ask if direction == 1 else bid
    risk = SL_ATR * atr
    sl = entry - direction * risk
    tp = entry + direction * RR * risk
    if RISK_PCT > 0:
        if not conv:
            log.error(f"{pair} | signal skipped: no currency conversion rate from OANDA")
            return
        units = int(account_nav(api) * RISK_PCT / 100 / (risk * conv))
    else:
        units = UNITS
    units = min(units, MAX_UNITS)
    if units < 1:
        log.info(f"{pair} | signal skipped: size rounds to 0 units")
        return
    cid = f"{TAG}-{pair}-{bar_time:%Y%m%d%H}"
    body = {"order": {
        "type": "MARKET", "instrument": pair, "units": str(units * direction),
        "timeInForce": "FOK", "positionFill": "DEFAULT",
        "stopLossOnFill": {"price": px_round(pair, sl), "timeInForce": "GTC"},
        "takeProfitOnFill": {"price": px_round(pair, tp), "timeInForce": "GTC"},
        "clientExtensions": {"id": cid, "tag": TAG},
        "tradeClientExtensions": {"id": cid, "tag": TAG, "comment": f"risk={risk:.6f}"},
    }}
    side = "BUY" if direction == 1 else "SELL"
    status, resp = api.post(f"/v3/accounts/{api.account}/orders", body)
    fill = resp.get("orderFillTransaction")
    if status == 201 and fill:
        log.info(f"{pair} | {side} FILLED {units} @ {fill.get('price')} | SL {px_round(pair, sl)} | "
                 f"TP {px_round(pair, tp)} | 1R {risk / pip(pair):.1f} pips | id {cid}")
    else:
        reason = (resp.get("orderRejectTransaction") or resp.get("orderCancelTransaction") or {}).get("reason") \
                 or resp.get("errorMessage") or resp
        log.error(f"{pair} | {side} NOT FILLED (HTTP {status}) | {reason}")


def performance(api):
    """Closed trades from this runner, in R, using the 1R distance stored on each trade."""
    data = api.get(f"/v3/accounts/{api.account}/trades", {"state": "CLOSED", "count": 500})
    rs = []
    for t in data.get("trades", []):
        ce = t.get("clientExtensions", {})
        if ce.get("tag") != TAG or "risk=" not in ce.get("comment", ""):
            continue
        try:
            risk = float(ce["comment"].split("risk=")[1])
            d = 1 if float(t["initialUnits"]) > 0 else -1
            rs.append(d * (float(t["averageClosePrice"]) - float(t["price"])) / risk)
        except (KeyError, ValueError, ZeroDivisionError):
            continue
    if not rs:
        log.info("PERFORMANCE | no closed trades yet")
        return
    r = np.array(rs)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    pf = gains / losses if losses > 0 else float("inf")
    log.info(f"PERFORMANCE | {len(r)} closed | win {np.mean(r > 0) * 100:.0f}% | avgR {r.mean():+.3f} | "
             f"PF {pf:.2f} | total {r.sum():+.2f}R | backtest expected about +0.04R, PF 1.10")


def check(api, last_seen):
    open_pairs = {t["instrument"] for t in open_tagged_trades(api)}
    for pair in PAIRS:
        try:
            d = fetch_recent(api, pair)
            if len(d) < 300:
                log.info(f"{pair} | not enough candles")
                continue
            bar_time = d["time"].iloc[-1]
            if last_seen.get(pair) == bar_time:
                continue                       # this H4 candle was already checked
            last_seen[pair] = bar_time
            if datetime.now(timezone.utc) - bar_time.to_pydatetime() > timedelta(hours=8):
                continue                       # market closed / stale data
            direction, atr = scan_last_bar(d)
            if direction == 0:
                log.info(f"{pair} | H4 {bar_time:%m-%d %H:%M} | no signal")
                continue
            side = "BUY" if direction == 1 else "SELL"
            if pair in open_pairs:
                log.info(f"{pair} | {side} signal | skipped: position already open")
                continue
            place(api, pair, direction, atr, bar_time)
        except Exception as e:
            log.error(f"{pair} | ERROR | {e}")


def main():
    if os.getenv("H4_LIVE_CONFIRM", "").strip().lower() != "yes":
        log.error("Refusing to trade: set H4_LIVE_CONFIRM=yes in Railway to start the H4 forward test.")
        while True:
            time.sleep(3600)
    api = Oanda()
    size = f"{RISK_PCT:g}% of NAV per trade (max {MAX_UNITS} units)" if RISK_PCT > 0 else f"fixed {UNITS} units"
    log.info(f"H4 PO3+Echo forward test | PRACTICE | {len(PAIRS)} pairs | stop {SL_ATR:g}xATR50 | "
             f"target {RR:g}R | {size}")
    last_seen, checks = {}, 0
    while True:
        try:
            check(api, last_seen)
            checks += 1
            if checks % 6 == 1:                # about once a day
                performance(api)
        except Exception as e:
            log.error(f"CHECK ERROR | {e}")
        now = time.time()
        time.sleep(3600 - now % 3600 + 20)     # 20 s after each hour: catches every OANDA H4 close, DST included


if __name__ == "__main__":
    main()
