"""
kalshi_backtest.py - does a fair-value model beat Kalshi's 15-minute crypto prices?

Run on Railway by setting MODE=backtest on the Kalshi service (kalshi_bot.py
dispatches here). Read-only: it never places orders and uses PUBLIC market
data from Kalshi's production exchange (no key needed) plus Coinbase prices.

THE MODEL
  Each market asks: will the coin's settlement value be at or above the
  target price K when the window closes? At a checkpoint with tau seconds
  left and spot S:
      fair P(up) = N( ln(S/K) / (sigma * sqrt(tau_eff)) )
  sigma    = realized volatility of the last 60 one-minute returns
  tau_eff  = tau - 40s, because settlement is a 60-second AVERAGE, which
             removes about two thirds of the last minute's variance.
  Zero drift: over 15 minutes, drift is noise.

THE TEST (rules fixed before seeing results)
  * Checkpoints at 10, 7, 5, 3 and 2 minutes before close; at most one
    trade per market (the first checkpoint whose edge clears the bar).
  * Buy 1 contract at the real ask from Kalshi's per-minute candles,
    pay the taker fee (0.07 * P * (1-P)), settle on the real result.
  * Chronological split: the first 60% of markets choose the edge
    threshold from a small grid; the last 40% (holdout) is scored ONCE
    with that threshold frozen.
  * PASS requires, on the holdout: >= 60 trades, positive total P&L,
    t-stat >= 2.0, and still profitable when every fill is delayed by one
    minute (a realism check against stale quotes).

LIMITS (be honest about what this can and cannot show)
  * Coinbase is a proxy for the CF Benchmarks index Kalshi settles on.
    The difference is usually a few basis points but matters when spot
    sits right at the target.
  * Candles give top-of-book only, once per minute; real size at the ask is
    unknown. Results assume small size (1 contract).
"""
from __future__ import annotations

import logging
import math
import os
import statistics
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import requests

from kalshi_client import KalshiClient, KalshiError, fee_estimate, market_close_ts

log = logging.getLogger("backtest")

PRODUCTS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
            "XRP": "XRP-USD", "DOGE": "DOGE-USD"}


def env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def asset_of(series: str) -> str:
    s = series.upper()
    s = s[2:] if s.startswith("KX") else s
    return s[:-3] if s.endswith("15M") else s


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# Spot prices (Coinbase, public)
# --------------------------------------------------------------------------
class Coinbase:
    BASE = "https://api.exchange.coinbase.com"

    def __init__(self, session: Optional[requests.Session] = None):
        self.http = session or requests.Session()

    def minute_closes(self, product: str, start: int, end: int) -> Dict[int, float]:
        """{minute_start_ts: close}. Coinbase returns max 300 candles per call."""
        out: Dict[int, float] = {}
        t = start - start % 60
        while t < end:
            chunk_end = min(end, t + 300 * 60)
            params = {"granularity": 60,
                      "start": datetime.fromtimestamp(t, timezone.utc).isoformat(),
                      "end": datetime.fromtimestamp(chunk_end, timezone.utc).isoformat()}
            for attempt in range(6):
                try:
                    r = self.http.get(f"{self.BASE}/products/{product}/candles",
                                      params=params, timeout=15,
                                      headers={"User-Agent": "kalshi-backtest"})
                    if r.status_code == 429 or r.status_code >= 500:
                        raise requests.HTTPError(f"HTTP {r.status_code}")
                    r.raise_for_status()
                    for row in r.json():
                        out[int(row[0])] = float(row[4])
                    break
                except (requests.RequestException, ValueError) as e:
                    if attempt == 5:
                        raise
                    time.sleep(min(20, 1.5 * 2 ** attempt))
                    log.warning("coinbase retry %s: %s", product, e)
            time.sleep(0.12)  # stay under public rate limits
            t = chunk_end
        return out


def spot_at(closes: Dict[int, float], t: int) -> Optional[float]:
    """Price at time t = close of the minute candle that ends at t."""
    return closes.get(t - 60)


def realized_sigma(closes: Dict[int, float], t: int, minutes: int = 60
                   ) -> Optional[float]:
    """Per-sqrt-second volatility from the last `minutes` one-minute returns."""
    pts = [closes.get(t - 60 * (i + 1)) for i in range(minutes + 1)]
    if any(p is None or p <= 0 for p in pts):
        pts = [p for p in pts if p and p > 0]
        if len(pts) < minutes * 0.8:
            return None
    rets = [math.log(pts[i] / pts[i + 1]) for i in range(len(pts) - 1)]
    if len(rets) < 20:
        return None
    sd = statistics.pstdev(rets)
    return max(sd, 1e-7) / math.sqrt(60.0)


def fair_up(spot: float, strike: float, sigma_s: float, tau: float) -> float:
    tau_eff = max(tau - 40.0, 20.0)
    z = math.log(spot / strike) / (sigma_s * math.sqrt(tau_eff))
    return min(max(norm_cdf(z), 0.001), 0.999)


# --------------------------------------------------------------------------
# Kalshi data (public, production)
# --------------------------------------------------------------------------
def strike_of(m: Dict[str, Any]) -> Optional[float]:
    for key in ("floor_strike", "strike", "cap_strike"):
        v = m.get(key)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    import re
    for key in ("yes_sub_title", "subtitle", "rules_primary"):
        txt = str(m.get(key) or "")
        mm = re.search(r"\$([\d,]+(?:\.\d+)?)", txt)
        if mm:
            return float(mm.group(1).replace(",", ""))
    return None


def load_markets(k: KalshiClient, series: str, since: int) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    try:
        raw = list(k.iter_markets(series_ticker=series, min_close_ts=since))
    except KalshiError as e:
        log.warning("%s: min_close_ts query failed (%s); using status=settled", series, e)
        raw = list(k.iter_markets(series_ticker=series, status="settled", max_items=20000))
    out, seen = [], set()
    now = time.time()
    for m in raw:
        r = str(m.get("result", "")).lower()
        close = market_close_ts(m)
        K = strike_of(m)
        if r not in ("yes", "no") or not close or close < since or close > now or not K:
            continue
        if m["ticker"] in seen:
            continue
        seen.add(m["ticker"])
        out.append({"ticker": m["ticker"], "close": int(close), "open": int(close) - 900,
                    "K": K, "up": r == "yes", "series": series})
    out.sort(key=lambda x: x["close"])
    return out


def _px(d: Any) -> Optional[float]:
    if not isinstance(d, dict):
        return None
    v = d.get("close_dollars")
    if v is None:
        v = d.get("close")
        if isinstance(v, int):
            return v / 100.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if 0.0 < f < 1.0 else None


def _parse_candles(cands: List[Dict[str, Any]]) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    q = {}
    for c in cands or []:
        ts = c.get("end_period_ts")
        if ts is None:
            continue
        q[int(ts)] = (_px(c.get("yes_bid")), _px(c.get("yes_ask")))
    return q


def load_quotes(k: KalshiClient, markets: List[Dict[str, Any]], batch: int = 10
                ) -> Dict[str, Dict[int, Tuple[Optional[float], Optional[float]]]]:
    """Per-minute (yes_bid, yes_ask) for every market, keyed by candle end ts."""
    out: Dict[str, Dict[int, Tuple[Optional[float], Optional[float]]]] = {}
    use_batch = True
    for i in range(0, len(markets), batch):
        group = markets[i:i + batch]
        start, end = min(m["open"] for m in group) - 60, max(m["close"] for m in group)
        if use_batch:
            try:
                data = k.request("GET", "/markets/candlesticks", {
                    "market_tickers": ",".join(m["ticker"] for m in group),
                    "start_ts": start, "end_ts": end, "period_interval": 1})
                for entry in data.get("markets") or []:
                    out[entry.get("market_ticker")] = _parse_candles(entry.get("candlesticks"))
                missing = [m for m in group if m["ticker"] not in out]
                if not missing:
                    continue
                group = missing
            except KalshiError as e:
                log.warning("batch candlesticks unavailable (%s); switching to per-market", e)
                use_batch = False
        for m in group:
            params = {"start_ts": m["open"] - 60, "end_ts": m["close"], "period_interval": 1}
            try:
                data = k.request("GET", f"/series/{m['series']}/markets/{m['ticker']}/candlesticks",
                                 params)
            except KalshiError:
                try:
                    data = k.request("GET", f"/historical/markets/{m['ticker']}/candlesticks",
                                     params)
                except KalshiError as e:
                    log.debug("no candles for %s: %s", m["ticker"], e)
                    continue
            out[m["ticker"]] = _parse_candles(data.get("candlesticks"))
        if (i // batch) % 20 == 0:
            log.info("  quotes %d/%d markets", min(i + batch, len(markets)), len(markets))
    return out


def quote_at(q: Dict[int, Tuple[Optional[float], Optional[float]]], t: int
             ) -> Tuple[Optional[float], Optional[float]]:
    return q.get(t, (None, None))


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def build_samples(markets, spots, quotes, checkpoints, delay: int = 0):
    """One row per (market, checkpoint) with a usable fair value and quote."""
    rows = []
    for m in markets:
        closes = spots[asset_of(m["series"])]
        q = quotes.get(m["ticker"]) or {}
        for tau in checkpoints:
            t = m["close"] - tau
            S = spot_at(closes, t)
            sig = realized_sigma(closes, t)
            if not S or not sig:
                continue
            p = fair_up(S, m["K"], sig, tau)
            bid, ask = quote_at(q, t + delay)
            rows.append({"m": m, "tau": tau, "t": t, "p": p, "bid": bid, "ask": ask,
                         "up": m["up"]})
    return rows


def per_contract_fee(price: float) -> float:
    return float(fee_estimate(10, Decimal(f"{price:.4f}"))) / 10.0


def simulate(rows, threshold: float) -> List[Dict[str, Any]]:
    """First checkpoint per market whose edge >= threshold; 1 contract."""
    trades, done = [], set()
    for r in sorted(rows, key=lambda r: (r["m"]["close"], -r["tau"])):
        tk = r["m"]["ticker"]
        if tk in done:
            continue
        best = None
        if r["ask"] is not None:                       # buy YES (up)
            edge = r["p"] - r["ask"] - per_contract_fee(r["ask"])
            best = ("UP", r["ask"], edge, r["up"])
        if r["bid"] is not None:                       # buy NO (down) at 1 - yes_bid
            no_ask = 1.0 - r["bid"]
            edge = (1 - r["p"]) - no_ask - per_contract_fee(no_ask)
            if best is None or edge > best[2]:
                best = ("DOWN", no_ask, edge, not r["up"])
        if best and best[2] >= threshold:
            side, price, edge, win = best
            fee = per_contract_fee(price)
            pnl = (1.0 if win else 0.0) - price - fee
            trades.append({"ticker": tk, "series": r["m"]["series"], "tau": r["tau"],
                           "side": side, "price": price, "fee": fee, "edge": edge,
                           "p": r["p"], "win": win, "pnl": pnl, "close": r["m"]["close"]})
            done.add(tk)
    return trades


def summarize(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(trades)
    if n == 0:
        return {"n": 0, "pnl": 0.0, "win": 0.0, "be": 0.0, "t": 0.0, "avg": 0.0}
    pnls = [t["pnl"] for t in trades]
    sd = statistics.pstdev(pnls) if n > 1 else 0.0
    tstat = (statistics.mean(pnls) / (sd / math.sqrt(n))) if sd > 0 else 0.0
    return {"n": n, "pnl": sum(pnls), "avg": statistics.mean(pnls),
            "win": sum(t["win"] for t in trades) / n,
            "be": statistics.mean(t["price"] + t["fee"] for t in trades), "t": tstat}


def fmt(s: Dict[str, float]) -> str:
    if s["n"] == 0:
        return "n=0"
    return (f"n={s['n']:4d}  win {s['win'] * 100:5.1f}% vs breakeven {s['be'] * 100:5.1f}%  "
            f"pnl ${s['pnl']:+8.2f}  per trade {s['avg'] * 100:+5.2f}c  t={s['t']:+.2f}")


def calibration(rows) -> None:
    """Is the model's probability honest, and is it sharper than the market mid?"""
    usable = [r for r in rows if r["bid"] is not None and r["ask"] is not None]
    if not usable:
        log.info("calibration: no rows with two-sided quotes")
        return
    bm = statistics.mean((r["p"] - r["up"]) ** 2 for r in usable)
    bk = statistics.mean((((r["bid"] + r["ask"]) / 2) - r["up"]) ** 2 for r in usable)
    log.info("BRIER (lower is better) over %d checkpoints: model %.4f | market mid %.4f %s",
             len(usable), bm, bk, "<- model sharper" if bm < bk else "<- market sharper")
    by_tau: Dict[int, List] = {}
    for r in usable:
        by_tau.setdefault(r["tau"], []).append(r)
    for tau in sorted(by_tau, reverse=True):
        rs = by_tau[tau]
        b1 = statistics.mean((r["p"] - r["up"]) ** 2 for r in rs)
        b2 = statistics.mean((((r["bid"] + r["ask"]) / 2) - r["up"]) ** 2 for r in rs)
        log.info("   %2d min left: model %.4f | market %.4f  (n=%d)", tau // 60, b1, b2, len(rs))
    log.info("CALIBRATION model p-bucket -> actual UP rate:")
    for lo in [i / 10 for i in range(10)]:
        rs = [r for r in usable if lo <= r["p"] < lo + 0.1]
        if len(rs) >= 30:
            log.info("   %.1f-%.1f: predicted %.3f actual %.3f (n=%d)", lo, lo + 0.1,
                     statistics.mean(r["p"] for r in rs),
                     statistics.mean(1.0 if r["up"] else 0.0 for r in rs), len(rs))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run() -> int:
    days = int(env("BT_DAYS", "30"))
    series = [s.strip().upper() for s in
              env("BT_SERIES", "KXBTC15M,KXETH15M,KXSOL15M,KXXRP15M,KXDOGE15M").split(",")
              if s.strip()]
    checkpoints = [int(x) for x in env("BT_CHECKPOINTS", "600,420,300,180,120").split(",")]
    grid = [float(x) for x in env("BT_GRID", "0.03,0.05,0.08,0.12").split(",")]
    design_frac = float(env("BT_DESIGN_FRAC", "0.6"))
    since = int(time.time()) - days * 86400

    log.info("== Fair-value backtest | %d days | %s | checkpoints %s min",
             days, ",".join(series), ",".join(str(c // 60) for c in checkpoints))
    k = KalshiClient(env="prod", public=True)   # public market data only
    cb = Coinbase()

    markets: List[Dict[str, Any]] = []
    for s in series:
        ms = load_markets(k, s, since)
        log.info("%s: %d settled markets with target prices", s, len(ms))
        markets += ms
    if not markets:
        log.error("no markets loaded - check series tickers")
        return 1
    markets.sort(key=lambda m: m["close"])

    spots: Dict[str, Dict[int, float]] = {}
    for s in series:
        a = asset_of(s)
        log.info("fetching %s 1-minute spot from Coinbase ...", PRODUCTS.get(a, a))
        spots[a] = cb.minute_closes(PRODUCTS.get(a, f"{a}-USD"), since - 7200, int(time.time()))
        log.info("  %d minutes", len(spots[a]))

    log.info("fetching Kalshi per-minute quotes for %d markets ...", len(markets))
    quotes = load_quotes(k, markets)
    with_q = sum(1 for m in markets if quotes.get(m["ticker"]))
    log.info("quotes found for %d/%d markets", with_q, len(markets))

    rows = build_samples(markets, spots, quotes, checkpoints)
    rows_late = build_samples(markets, spots, quotes, checkpoints, delay=60)
    if not rows:
        log.error("no usable samples (spot or quotes missing)")
        return 1
    calibration(rows)

    cut = markets[int(len(markets) * design_frac)]["close"]
    design = [r for r in rows if r["m"]["close"] < cut]
    hold = [r for r in rows if r["m"]["close"] >= cut]
    hold_late = [r for r in rows_late if r["m"]["close"] >= cut]
    log.info("split: design = markets closing before %s UTC, holdout after",
             datetime.fromtimestamp(cut, timezone.utc).strftime("%Y-%m-%d %H:%M"))

    log.info("DESIGN (choosing the threshold):")
    best_th, best_pnl = None, -1e9
    for th in grid:
        s = summarize(simulate(design, th))
        log.info("   edge >= %.2f: %s", th, fmt(s))
        if s["n"] >= 50 and s["pnl"] > best_pnl:
            best_th, best_pnl = th, s["pnl"]
    if best_th is None:
        log.info("VERDICT: FAIL - no threshold produced 50+ design trades")
        return 0
    log.info("frozen threshold: edge >= %.2f", best_th)

    ht = simulate(hold, best_th)
    hs = summarize(ht)
    ls = summarize(simulate(hold_late, best_th))
    log.info("HOLDOUT (scored once): %s", fmt(hs))
    log.info("HOLDOUT, fills delayed 1 min: %s", fmt(ls))
    for s in series:
        sub = summarize([t for t in ht if t["series"] == s])
        log.info("   %-10s %s", s, fmt(sub))
    for tau in checkpoints:
        sub = summarize([t for t in ht if t["tau"] == tau])
        log.info("   entry %2d min left: %s", tau // 60, fmt(sub))

    checks = {
        "60+ trades": hs["n"] >= 60,
        "positive P&L": hs["pnl"] > 0,
        "t-stat >= 2.0": hs["t"] >= 2.0,
        "profitable with 1-min delay": ls["pnl"] > 0,
    }
    for name, ok in checks.items():
        log.info("   [%s] %s", "x" if ok else " ", name)
    log.info("VERDICT: %s", "PASS - qualifies for a paper forward test" if all(checks.values())
             else "FAIL - no tradeable edge under these rules")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(run())
