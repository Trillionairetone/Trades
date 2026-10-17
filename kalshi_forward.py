"""
kalshi_forward.py - FROZEN prospective paper test of the fair-value model.

Run with MODE=forward (kalshi_bot.py dispatches here).

WHAT IT IS
  A forward test on data nobody has seen: live Kalshi PRODUCTION order books
  (public endpoints, no key) and live Coinbase prices (public, no account).
  It records paper decisions and simulated fills. It NEVER places orders:
  it uses an unauthenticated client that cannot sign a request.

WHY "FROZEN"
  The rules below are fixed before the test starts and fingerprinted
  (RULES_HASH). The hash is written to every ledger row and to the state
  file. If the rules in the code ever differ from the ones a running test
  started with, the bot refuses to resume - you cannot quietly tune a test
  that is already running. Interim stats are printed for monitoring only;
  the verdict is computed once, at the pre-registered end.

PER MARKET (15-minute up/down, per coin)
  At 10, 7, 5, 3 and 2 minutes before close (+/- tolerance):
    1. CHECKPOINT: spot (Coinbase ticker), sigma (last 60 one-minute
       returns), target K (Kalshi), fair P(up) - identical model code to the
       backtest - plus Kalshi's live bid/ask. Logged even when no trade, so
       the model can be scored against the market price at every checkpoint.
    2. DECISION (first checkpoint only where edge >= threshold, one per
       market, no retries): side, decision-time ask, and limit L = the
       highest price that still clears the edge threshold after fees.
    3. EXECUTION: a FRESH order book is fetched right after the decision
       (real network latency). Best ask <= L -> FILLED at that ask;
       ask > L -> UNFILLED_PRICE; no asks on that side -> NO_QUOTE.
  After close: the real result settles every checkpoint and decision.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import requests

from kalshi_backtest import (PRODUCTS, Coinbase, asset_of, fair_up, realized_sigma,
                             strike_of)
from kalshi_client import KalshiClient, KalshiError, fee_estimate, market_close_ts

log = logging.getLogger("forward")

# ---------------------------------------------------------------------------
# PRE-REGISTERED RULES - changing ANY of these changes RULES_HASH and blocks
# resuming a test that started under different rules.
# ---------------------------------------------------------------------------
RULES: Dict[str, Any] = {
    "version": "fwd-1",
    "series": ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M"],
    "checkpoints_s": [600, 420, 300, 180, 120],
    "checkpoint_tolerance_s": 20,
    "model": "N(ln(S/K)/(sigma*sqrt(tau-40))), sigma=60x1m realized, zero drift",
    "edge_threshold": 0.12,
    "fee_model": "0.07*C*P*(1-P) ceiled to $0.0001 (kalshi_client.fee_estimate)",
    "contracts": 1,
    "one_decision_per_market": True,
    "execution": "fresh book after decision; fill at best ask if <= limit, else UNFILLED_PRICE; "
                 "empty side NO_QUOTE",
    "end": {"filled_trades": 1000, "max_days": 42},
    "pass": "filled P&L > 0 AND t-stat >= 2.0, evaluated once at the end",
}
RULES_HASH = hashlib.sha256(json.dumps(RULES, sort_keys=True).encode()).hexdigest()[:12]

LEDGER_FIELDS = ["at", "event", "series", "ticker", "tau_s", "spot", "strike", "sigma",
                 "p_up", "yes_bid", "yes_ask", "side", "decision_ask", "limit",
                 "fill_price", "fee", "status", "latency_ms", "result", "pnl",
                 "rules_hash"]


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_dir() -> str:
    return os.getenv("FWD_DIR") or ("/data/kalshi_forward" if os.path.isdir("/data")
                                    else "kalshi_forward")


def limit_price(p_side: float, threshold: float) -> float:
    """Highest price L (on a 1c tick) with p_side - L - fee(L) >= threshold."""
    L = p_side - threshold
    for _ in range(4):
        L = p_side - threshold - float(fee_estimate(1, Decimal(f"{max(L, 0.01):.4f}")))
    return math.floor(L * 100 + 1e-9) / 100.0


class CoinbaseLive(Coinbase):
    """Adds the live ticker to the backtest's (frozen) candle fetcher."""

    def ticker(self, product: str) -> Optional[float]:
        try:
            r = self.http.get(f"{self.BASE}/products/{product}/ticker", timeout=5,
                              headers={"User-Agent": "kalshi-forward"})
            r.raise_for_status()
            return float(r.json()["price"])
        except (requests.RequestException, ValueError, KeyError) as e:
            log.warning("coinbase ticker %s failed: %s", product, e)
            return None


class Forward:
    def __init__(self, kalshi: Optional[KalshiClient] = None,
                 coinbase: Optional[CoinbaseLive] = None, out_dir: Optional[str] = None):
        self.k = kalshi or KalshiClient(env="prod", public=True)  # cannot sign = cannot trade
        self.cb = coinbase or CoinbaseLive()
        self.dir = out_dir or default_dir()
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "forward_state.json")
        self.ledger_path = os.path.join(self.dir, "forward_ledger.csv")
        self.closes: Dict[str, Dict[int, float]] = {}
        self.closes_at: Dict[str, float] = {}
        self.market_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_at: Dict[str, float] = {}
        self.last_stats = 0.0
        self.state = self._load_state()

    # ---------------- persistence ----------------
    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                st = json.load(f)
            if st.get("rules_hash") != RULES_HASH:
                raise SystemExit(
                    f"REFUSING TO RESUME: state was created under rules {st.get('rules_hash')}, "
                    f"code has {RULES_HASH}. A running test cannot change its rules. "
                    f"Start a new test by pointing FWD_DIR at a fresh folder.")
            return st
        return {"rules_hash": RULES_HASH, "started": time.time(), "final": False,
                "done_checkpoints": [], "markets": {},
                "t": {"cp_n": 0, "brier_model": 0.0, "brier_market": 0.0,
                      "decisions": 0, "filled": 0, "unfilled": 0, "no_quote": 0,
                      "missed_cp": 0, "pnl": 0.0, "pnl_sq": 0.0, "wins": 0, "be": 0.0,
                      "unfilled_cf_wins": 0, "unfilled_settled": 0}}

    def save(self) -> None:
        self.state["done_checkpoints"] = self.state["done_checkpoints"][-2000:]
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.state_path)

    def write(self, now: float, **row: Any) -> None:
        new = not os.path.exists(self.ledger_path)
        row.update(at=iso(now), rules_hash=RULES_HASH)
        with open(self.ledger_path, "a", newline="") as f:
            w = csv.DictWriter(f, LEDGER_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})

    # ---------------- data ----------------
    def open_market(self, series: str, now: float) -> Optional[Dict[str, Any]]:
        if now - self.cache_at.get(series, 0) > 60 or series not in self.market_cache:
            ms = [m for m in self.k.iter_markets(series_ticker=series, min_close_ts=int(now))
                  if str(m.get("status", "")).lower() in ("open", "active")
                  and (market_close_ts(m) or 0) > now]
            self.market_cache[series] = min(ms, key=market_close_ts) if ms else {}
            self.cache_at[series] = now
        m = self.market_cache.get(series) or {}
        return m if m and (market_close_ts(m) or 0) > now else None

    def sigma(self, asset: str, now: float) -> Optional[float]:
        product = PRODUCTS.get(asset, f"{asset}-USD")
        if now - self.closes_at.get(asset, 0) > 55:
            try:
                self.closes[asset] = self.cb.minute_closes(product, int(now) - 7200, int(now))
                self.closes_at[asset] = now
            except requests.RequestException as e:
                log.warning("coinbase candles %s failed: %s", product, e)
        closes = self.closes.get(asset) or {}
        return realized_sigma(closes, int(now) // 60 * 60) if closes else None

    # ---------------- one checkpoint ----------------
    def checkpoint(self, series: str, m: Dict[str, Any], tau_nominal: int, now: float) -> None:
        ticker, close = m["ticker"], market_close_ts(m)
        asset = asset_of(series)
        K = strike_of(m)
        spot = self.cb.ticker(PRODUCTS.get(asset, f"{asset}-USD"))
        sig = self.sigma(asset, now)
        mk = self.state["markets"].setdefault(ticker, {"series": series, "close": close,
                                                       "rows": [], "decision": None})
        if not (K and spot and sig):
            self.write(now, event="SKIP", series=series, ticker=ticker, tau_s=tau_nominal,
                       status=f"missing {'strike' if not K else 'spot' if not spot else 'sigma'}")
            return
        tau = close - now
        p = fair_up(spot, K, sig, tau)
        ob = self.k.get_orderbook(ticker)
        yb = float(ob.best_yes_bid) if ob.best_yes_bid is not None else None
        ya = float(ob.yes_ask) if ob.yes_ask is not None else None
        mk["rows"].append({"tau": tau_nominal, "p": p, "bid": yb, "ask": ya})
        self.write(now, event="CHECKPOINT", series=series, ticker=ticker, tau_s=tau_nominal,
                   spot=spot, strike=K, sigma=f"{sig:.3e}", p_up=f"{p:.4f}", yes_bid=yb,
                   yes_ask=ya)
        if mk["decision"] is not None:
            return  # one decision per market, no retries

        cands = []
        if ya is not None:
            cands.append(("UP", p, ya))
        if yb is not None:
            cands.append(("DOWN", 1 - p, 1 - yb))
        best = None
        for side, ps, ask in cands:
            edge = ps - ask - float(fee_estimate(1, Decimal(f"{ask:.4f}")))
            if best is None or edge > best[3]:
                best = (side, ps, ask, edge)
        if not best or best[3] < RULES["edge_threshold"]:
            return

        side, ps, dec_ask, edge = best
        L = limit_price(ps, RULES["edge_threshold"])
        t0 = time.time()
        ob2 = self.k.get_orderbook(ticker)  # execution snapshot: real latency
        latency = int((time.time() - t0) * 1000)
        levels = ob2.ask_levels("yes" if side == "UP" else "no")
        if not levels:
            status, fill, fee = "NO_QUOTE", None, None
        elif float(levels[0][0]) <= L:
            status, fill = "FILLED", float(levels[0][0])
            fee = float(fee_estimate(RULES["contracts"], Decimal(f"{fill:.4f}")))
        else:
            status, fill, fee = "UNFILLED_PRICE", None, None
        mk["decision"] = {"side": side, "p_side": ps, "decision_ask": dec_ask, "limit": L,
                          "status": status, "fill": fill, "fee": fee, "tau": tau_nominal}
        t = self.state["t"]
        t["decisions"] += 1
        t[{"FILLED": "filled", "UNFILLED_PRICE": "unfilled", "NO_QUOTE": "no_quote"}[status]] += 1
        log.info("DECISION %s %s %s p=%.3f ask=%.2f limit=%.2f -> %s%s (%dms)", ticker,
                 f"{tau_nominal // 60}m", side, ps, dec_ask, L, status,
                 f" @ {fill:.2f}" if fill else "", latency)
        self.write(now, event="DECISION", series=series, ticker=ticker, tau_s=tau_nominal,
                   spot=spot, strike=K, p_up=f"{p:.4f}", side=side, decision_ask=dec_ask,
                   limit=L, fill_price=fill, fee=fee, status=status, latency_ms=latency)

    # ---------------- settlement ----------------
    def settle(self, now: float) -> None:
        t = self.state["t"]
        for ticker in list(self.state["markets"]):
            mk = self.state["markets"][ticker]
            if now < mk["close"] + 30:
                continue
            try:
                r = str(self.k.get_market(ticker).get("result", "")).lower()
            except KalshiError as e:
                log.warning("settle %s: %s", ticker, e)
                continue
            if r not in ("yes", "no"):
                if now > mk["close"] + 6 * 3600:
                    log.error("no result for %s after 6h; dropping", ticker)
                    self.state["markets"].pop(ticker)
                continue
            up = 1.0 if r == "yes" else 0.0
            for row in mk["rows"]:
                if row["bid"] is not None and row["ask"] is not None:
                    t["cp_n"] += 1
                    t["brier_model"] += (row["p"] - up) ** 2
                    t["brier_market"] += ((row["bid"] + row["ask"]) / 2 - up) ** 2
            d = mk["decision"]
            pnl = None
            if d:
                won = (d["side"] == "UP") == (up == 1.0)
                if d["status"] == "FILLED":
                    pnl = (1.0 if won else 0.0) - d["fill"] - d["fee"]
                    t["pnl"] += pnl
                    t["pnl_sq"] += pnl * pnl
                    t["wins"] += 1 if won else 0
                    t["be"] += d["fill"] + d["fee"]
                elif d["status"] == "UNFILLED_PRICE":  # diagnostic only, never in P&L
                    t["unfilled_settled"] += 1
                    t["unfilled_cf_wins"] += 1 if won else 0
            self.write(now, event="SETTLED", series=mk["series"], ticker=ticker, result=r,
                       side=d["side"] if d else None, status=d["status"] if d else "NO_DECISION",
                       pnl=f"{pnl:.4f}" if pnl is not None else None)
            if d and d["status"] == "FILLED":
                log.info("SETTLED %s %s -> %s pnl %+.4f", ticker, d["side"], r.upper(), pnl)
            self.state["markets"].pop(ticker)

    # ---------------- reporting ----------------
    def summary(self) -> Dict[str, float]:
        t = self.state["t"]
        nf = t["filled"] - self._open_filled()   # fills whose market has settled
        mean = t["pnl"] / nf if nf else 0.0
        var = (t["pnl_sq"] / nf - mean * mean) if nf > 1 else 0.0
        tstat = mean / math.sqrt(var / nf) if nf > 1 and var > 0 else 0.0
        return {"settled_filled": nf, "pnl": t["pnl"], "mean": mean, "t": tstat,
                "win": t["wins"] / nf if nf else 0.0, "be": t["be"] / nf if nf else 0.0}

    def _open_filled(self) -> int:
        return sum(1 for mk in self.state["markets"].values()
                   if mk["decision"] and mk["decision"]["status"] == "FILLED")

    def stats(self, now: float, final: bool = False) -> None:
        t, s = self.state["t"], self.summary()
        days = (now - self.state["started"]) / 86400
        bm = t["brier_model"] / t["cp_n"] if t["cp_n"] else float("nan")
        bk = t["brier_market"] / t["cp_n"] if t["cp_n"] else float("nan")
        log.info("%s day %.1f | rules %s | decisions %d: filled %d, unfilled %d, no-quote %d | "
                 "missed checkpoints %d", "FINAL" if final else "STATS", days, RULES_HASH,
                 t["decisions"], t["filled"], t["unfilled"], t["no_quote"], t["missed_cp"])
        log.info("   settled fills %d: win %.1f%% vs breakeven %.1f%% | P&L $%+.2f "
                 "(%+.2fc/trade) t=%+.2f | progress %d/%d fills", s["settled_filled"],
                 s["win"] * 100, s["be"] * 100, s["pnl"], s["mean"] * 100, s["t"],
                 s["settled_filled"], RULES["end"]["filled_trades"])
        log.info("   BRIER over %d checkpoints: model %.4f | market %.4f %s", t["cp_n"], bm, bk,
                 "" if not t["cp_n"] else ("<- model sharper" if bm < bk else "<- market sharper"))
        if t["unfilled_settled"]:
            log.info("   unfilled decisions would have won %d/%d (diagnostic, not P&L)",
                     t["unfilled_cf_wins"], t["unfilled_settled"])
        if final:
            ok = s["pnl"] > 0 and s["t"] >= 2.0
            log.info("VERDICT (pre-registered, rules %s): %s", RULES_HASH,
                     "PASS" if ok else "FAIL")

    def finished(self, now: float) -> bool:
        s = self.summary()
        return (s["settled_filled"] >= RULES["end"]["filled_trades"]
                or now - self.state["started"] >= RULES["end"]["max_days"] * 86400)

    # ---------------- loop ----------------
    def step(self, now: float) -> None:
        tol = RULES["checkpoint_tolerance_s"]
        done = set(self.state["done_checkpoints"])
        for series in RULES["series"]:
            try:
                m = self.open_market(series, now)
                if not m:
                    continue
                close = market_close_ts(m)
                for tau in RULES["checkpoints_s"]:
                    key = f"{m['ticker']}|{tau}"
                    if key in done:
                        continue
                    cp = close - tau
                    if now < cp:
                        continue
                    self.state["done_checkpoints"].append(key)
                    done.add(key)
                    if now <= cp + tol:
                        self.checkpoint(series, m, tau, now)
                    else:
                        self.state["t"]["missed_cp"] += 1
                        self.write(now, event="MISSED", series=series, ticker=m["ticker"],
                                   tau_s=tau, status="checkpoint window passed")
            except KalshiError as e:
                log.warning("%s: %s", series, e)
        self.settle(now)
        if now - self.last_stats > 3600:
            self.stats(now)
            self.last_stats = now
        self.save()


def run() -> int:
    fw = Forward()
    log.info("== FORWARD PAPER TEST | rules %s | dir %s | no orders (unauthenticated client)",
             RULES_HASH, fw.dir)
    log.info("rules: %s", json.dumps(RULES, sort_keys=True))
    if not fw.dir.startswith("/data"):
        log.warning("state is on the container's temporary disk: any redeploy (e.g. a commit "
                    "to main) RESETS this test. Mount a Railway volume at /data to persist it.")
    if fw.state.get("final"):
        log.info("this test already finished; final stats follow")
        fw.stats(time.time(), final=True)
        return 0
    while True:
        now = time.time()
        try:
            fw.step(now)
        except SystemExit:
            raise
        except Exception:
            log.exception("step failed; continuing")
        if fw.finished(now):
            fw.state["final"] = True
            fw.save()
            fw.stats(now, final=True)
            return 0
        time.sleep(2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(run())
