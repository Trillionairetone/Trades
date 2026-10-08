"""
OANDA Practice trading bot - M15 strategy with daily S/R, Bollinger Bands(20,2),
volume and candlestick confirmation. SL/TP at 1:3 risk-reward, fixed size.

Settings are read from environment variables (no code editing needed):
  OANDA_TOKEN       (required) your API token
  OANDA_ACCOUNT_ID  (required) e.g. 101-001-1234567-001
Optional:
  UNITS             fallback size in units if risk-based sizing fails (default 1000)
  RISK_PCT          % of balance risked per trade (default 0.5)
  MAX_UNITS         hard cap on trade size (default 50000)
  MAX_PER_CURRENCY  max net same-direction trades per currency (default 2)
  MAX_RISK_ATR      skip if stop is wider than this many ATRs (default 2.5)
  MAX_SPREAD_FRAC   skip if spread is more than this share of the risk (default 0.2)
  MAX_POSITIONS     max open trades at once (default 5)
  PAIRS             comma list (default major pairs)
"""
import os
import sys
import time
import datetime as dt
import requests

BASE = "https://api-fxpractice.oanda.com"  # PRACTICE ONLY
TOKEN = "".join(os.environ.get("OANDA_TOKEN", "").split())
ACCOUNT = "".join(os.environ.get("OANDA_ACCOUNT_ID", "").split())
UNITS = int(os.environ.get("UNITS", "1000"))
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", "5"))
PAIRS = os.environ.get(
    "PAIRS",
    "EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CAD,USD_CHF,NZD_USD,EUR_JPY,GBP_JPY",
).split(",")
RR = 3.0
RISK_PCT = float(os.environ.get("RISK_PCT", "0.5"))
MAX_UNITS = int(os.environ.get("MAX_UNITS", "50000"))
MAX_PER_CURRENCY = int(os.environ.get("MAX_PER_CURRENCY", "2"))
MAX_RISK_ATR = float(os.environ.get("MAX_RISK_ATR", "2.5"))
MAX_SPREAD_FRAC = float(os.environ.get("MAX_SPREAD_FRAC", "0.2"))

if not TOKEN or not ACCOUNT:
    sys.exit("Missing OANDA_TOKEN or OANDA_ACCOUNT_ID environment variables.")

HEAD = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def api(method, path, **kw):
    r = requests.request(method, BASE + path, headers=HEAD, timeout=20, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.text[:300]}")
    return r.json()


def candles(pair, gran, count):
    d = api("GET", f"/v3/instruments/{pair}/candles",
            params={"granularity": gran, "count": count, "price": "M"})
    out = []
    for c in d["candles"]:
        if c["complete"]:
            m = c["mid"]
            out.append({"o": float(m["o"]), "h": float(m["h"]),
                        "l": float(m["l"]), "c": float(m["c"]),
                        "v": c["volume"]})
    return out


def sma(x, n):
    return sum(x[-n:]) / n


def ema(x, n):
    k = 2 / (n + 1)
    e = sum(x[:n]) / n
    for v in x[n:]:
        e = v * k + e * (1 - k)
    return e


def atr(cs, n=14):
    trs = [max(c["h"] - c["l"], abs(c["h"] - p["c"]), abs(c["l"] - p["c"]))
           for p, c in zip(cs[:-1], cs[1:])]
    return sum(trs[-n:]) / n


def bollinger(closes, n=20, k=2):
    w = closes[-n:]
    m = sum(w) / n
    sd = (sum((v - m) ** 2 for v in w) / n) ** 0.5
    return m - k * sd, m, m + k * sd


def bullish_candle(p, c):
    body = abs(c["c"] - c["o"])
    rng = (c["h"] - c["l"]) or 1e-9
    lower = min(c["o"], c["c"]) - c["l"]
    engulf = (c["c"] > c["o"] and p["c"] < p["o"]
              and c["c"] >= p["o"] and c["o"] <= p["c"])
    hammer = c["c"] > c["o"] and lower >= 2 * body and body / rng > 0.1
    return engulf or hammer


def bearish_candle(p, c):
    body = abs(c["c"] - c["o"])
    rng = (c["h"] - c["l"]) or 1e-9
    upper = c["h"] - max(c["o"], c["c"])
    engulf = (c["c"] < c["o"] and p["c"] > p["o"]
              and c["c"] <= p["o"] and c["o"] >= p["c"])
    star = c["c"] < c["o"] and upper >= 2 * body and body / rng > 0.1
    return engulf or star


def analyse(pair):
    m15 = candles(pair, "M15", 250)
    daily = candles(pair, "D", 30)
    if len(m15) < 210 or len(daily) < 12:
        return None
    closes = [c["c"] for c in m15]
    last, prev = m15[-1], m15[-2]
    price = last["c"]
    a = atr(m15)

    # Daily support / resistance: previous day and 10-day extremes
    supports = [daily[-1]["l"], min(c["l"] for c in daily[-10:])]
    resists = [daily[-1]["h"], max(c["h"] for c in daily[-10:])]

    # M15 trend
    e50, e200 = ema(closes, 50), ema(closes, 200)
    e50_prev = ema(closes[:-8], 50)
    if price > e50 > e200 and e50 > e50_prev:
        trend = "Bullish"
    elif price < e50 < e200 and e50 < e50_prev:
        trend = "Bearish"
    else:
        trend = "Sideways"

    lower, mid, upper = bollinger(closes)
    vol_ok = last["v"] > sma([c["v"] for c in m15[-21:-1]], 20)
    zone = 1.0 * a  # "near" a level

    signal = "NO TRADE"
    sl = tp = None
    if trend == "Bullish" and vol_ok and bullish_candle(prev, last):
        near_sup = any(abs(last["l"] - s) <= zone or last["l"] < s for s in supports)
        touched = min(last["l"], prev["l"]) <= lower * 1.0 + 0.2 * a
        if (near_sup or touched) and price > lower:
            sl = min(last["l"], prev["l"]) - 0.3 * a
            risk = price - sl
            if risk > 0:
                signal, tp = "BUY", price + RR * risk
    elif trend == "Bearish" and vol_ok and bearish_candle(prev, last):
        near_res = any(abs(last["h"] - r) <= zone or last["h"] > r for r in resists)
        touched = max(last["h"], prev["h"]) >= upper - 0.2 * a
        if (near_res or touched) and price < upper:
            sl = max(last["h"], prev["h"]) + 0.3 * a
            risk = sl - price
            if risk > 0:
                signal, tp = "SELL", price - RR * risk

    return {"pair": pair, "trend": trend, "signal": signal, "entry": price,
            "sl": sl, "tp": tp, "atr": a}


def fmt(pair, x):
    return f"{x:.3f}" if pair.endswith("JPY") else f"{x:.5f}"


def open_trades():
    return api("GET", f"/v3/accounts/{ACCOUNT}/openTrades")["trades"]


def account_info():
    a = api("GET", f"/v3/accounts/{ACCOUNT}/summary")["account"]
    return float(a["balance"]), a["currency"]


def live_price(pair):
    """Current bid/ask plus the rate to convert quote-currency losses to account currency."""
    d = api("GET", f"/v3/accounts/{ACCOUNT}/pricing",
            params={"instruments": pair, "includeHomeConversions": "true"})
    p = d["prices"][0]
    if p.get("tradeable") is False:
        raise RuntimeError("market not tradeable right now")
    bid = float(p["bids"][0]["price"])
    ask = float(p["asks"][0]["price"])
    quote = pair.split("_")[1]
    conv = None
    for h in d.get("homeConversions", []):
        if h["currency"] == quote:
            conv = float(h["accountLoss"])
    return bid, ask, quote, conv


def size_units(risk_price, quote, conv):
    """Units so that hitting the stop loses about RISK_PCT% of the balance."""
    balance, acct_ccy = account_info()
    if conv is None:
        if quote != acct_ccy:
            raise RuntimeError(f"no conversion rate {quote}->{acct_ccy}")
        conv = 1.0
    units = int((balance * RISK_PCT / 100) / (risk_price * conv))
    return max(0, min(units, MAX_UNITS))


def exposure(trades):
    """Net same-direction count per currency (long EUR_USD = +1 EUR, -1 USD)."""
    ex = {}
    for t in trades:
        base, quote = t["instrument"].split("_")
        d = 1 if float(t["currentUnits"]) > 0 else -1
        ex[base] = ex.get(base, 0) + d
        ex[quote] = ex.get(quote, 0) - d
    return ex


def exposure_ok(ex, pair, signal):
    base, quote = pair.split("_")
    d = 1 if signal == "BUY" else -1
    return (abs(ex.get(base, 0) + d) <= MAX_PER_CURRENCY
            and abs(ex.get(quote, 0) - d) <= MAX_PER_CURRENCY)


def add_exposure(ex, pair, signal):
    base, quote = pair.split("_")
    d = 1 if signal == "BUY" else -1
    ex[base] = ex.get(base, 0) + d
    ex[quote] = ex.get(quote, 0) - d


def place(sig, units):
    pair = sig["pair"]
    units = units if sig["signal"] == "BUY" else -units
    body = {"order": {
        "type": "MARKET", "instrument": pair, "units": str(units),
        "timeInForce": "FOK", "positionFill": "DEFAULT",
        "stopLossOnFill": {"price": fmt(pair, sig["sl"])},
        "takeProfitOnFill": {"price": fmt(pair, sig["tp"])},
    }}
    r = api("POST", f"/v3/accounts/{ACCOUNT}/orders", json=body)
    if "orderFillTransaction" in r:
        print(f"  -> FILLED {units} units @ {r['orderFillTransaction'].get('price')}", flush=True)
        return True
    why = (r.get("orderCancelTransaction") or r.get("orderRejectTransaction") or {}).get(
        "reason", list(r.keys()))
    print(f"  -> NOT filled: {why}", flush=True)
    return False


def execute(s):
    """Re-price on live bid/ask, check stop and spread, size by risk, place the order."""
    pair = s["pair"]
    bid, ask, quote, conv = live_price(pair)
    buy = s["signal"] == "BUY"
    entry = ask if buy else bid
    risk = (entry - s["sl"]) if buy else (s["sl"] - entry)
    if risk <= 0:
        print("  skipped: price already beyond the stop", flush=True)
        return False
    if risk > MAX_RISK_ATR * s["atr"]:
        print(f"  skipped: stop too wide ({risk / s['atr']:.1f} ATR)", flush=True)
        return False
    if (ask - bid) > MAX_SPREAD_FRAC * risk:
        print(f"  skipped: spread too wide ({(ask - bid) / risk:.0%} of risk)", flush=True)
        return False
    s["entry"] = entry
    s["tp"] = entry + RR * risk if buy else entry - RR * risk
    try:
        units = size_units(risk, quote, conv)
    except Exception as e:
        units = UNITS
        print(f"  sizing failed ({e}); using fixed {UNITS} units", flush=True)
    if units < 1:
        print("  skipped: risk-based size rounds to 0 units", flush=True)
        return False
    print(f"  re-priced: ENTRY {fmt(pair, entry)} SL {fmt(pair, s['sl'])} "
          f"TP {fmt(pair, s['tp'])} | {units} units", flush=True)
    return place(s, units)


def cycle():
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n=== Scan {now} ===", flush=True)
    trades = open_trades()
    busy = {t["instrument"] for t in trades}
    count = len(trades)
    ex = exposure(trades)
    for pair in PAIRS:
        try:
            s = analyse(pair)
            if not s:
                continue
            line = f"{pair} | {s['trend']} | {s['signal']}"
            if s["signal"] != "NO TRADE":
                risk = abs(s["entry"] - s["sl"])
                line += (f" | ENTRY {fmt(pair, s['entry'])} SL {fmt(pair, s['sl'])}"
                         f" TP {fmt(pair, s['tp'])} | RR 1:{abs(s['tp']-s['entry'])/risk:.1f}")
            print(line, flush=True)
            if s["signal"] != "NO TRADE":
                if pair in busy:
                    print("  skipped: already have a trade on this pair")
                elif count >= MAX_POSITIONS:
                    print("  skipped: max positions reached")
                elif not exposure_ok(ex, pair, s["signal"]):
                    print("  skipped: too much exposure to one currency")
                elif execute(s):
                    busy.add(pair)
                    count += 1
                    add_exposure(ex, pair, s["signal"])
        except Exception as e:
            print(f"{pair} error: {e}", flush=True)


def main():
    print("Bot started (PRACTICE). Pairs:", ", ".join(PAIRS), flush=True)
    while True:
        try:
            cycle()
        except Exception as e:
            print("cycle error:", e, flush=True)
        # sleep until 10 seconds after the next 15-minute mark
        t = time.time()
        time.sleep(900 - (t % 900) + 10)


if __name__ == "__main__":
    main()
