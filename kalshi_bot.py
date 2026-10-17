"""
kalshi_bot.py - Kalshi 15-minute up/down bot for Railway.

MODES (env MODE):
  probe  (default) read-only: checks keys, balance, positions, and prints the
         live order books + streak statistics for each series, then exits.
  paper  forward test with REAL order books and FAKE money: fills are
         simulated by walking the actual asks (not "at the ask" for any
         size), fees use Kalshi's formula, settlement uses the real result.
  live   sends real IOC orders. Demo env runs freely; prod also requires
         LIVE_PROD_CONFIRM=YES_REAL_MONEY.

STRATEGY (frozen rules - change them only before a test, never after):
  The pattern your Clean Odds log describes ("after 2 ups in a row, BTC
  ended UP 46.9% of 6,852 times"), done properly:
    1. Build P(next 15m is UP | last direction, run length) from Kalshi's
       own settled markets for each series (Laplace-smoothed).
    2. Take a side ONLY if the LOWER confidence bound of its win
       probability beats the all-in cost (ask + fee) by MIN_EDGE.
       Using the lower bound - not the point estimate - is what stops the
       bot buying a 66c contract on a 56% "edge".
    3. Size by dollars at risk, capped by the depth actually available
       at or below the price cap.

Every skipped market is logged with the reason ("blocked by ...").
Every trade and settlement is appended to LEDGER_PATH as CSV.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, List, Optional, Tuple

from kalshi_client import (ONE, ZERO, D, KalshiClient, KalshiError, Orderbook,
                           fee_estimate, market_close_ts)

log = logging.getLogger("bot")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass
class Config:
    mode: str = field(default_factory=lambda: env("MODE", "probe").lower())
    series: List[str] = field(default_factory=lambda: [
        s.strip().upper() for s in
        env("SERIES", "KXBTC15M,KXETH15M,KXSOL15M,KXXRP15M,KXDOGE15M").split(",") if s.strip()])
    yes_means_up: bool = field(default_factory=lambda: env("YES_MEANS_UP", "1") == "1")
    poll_seconds: float = field(default_factory=lambda: float(env("POLL_SECONDS", "20")))
    # entry window, measured as seconds before the market closes
    entry_min_secs: float = field(default_factory=lambda: float(env("ENTRY_MIN_SECS_TO_CLOSE", "300")))
    entry_max_secs: float = field(default_factory=lambda: float(env("ENTRY_MAX_SECS_TO_CLOSE", "870")))
    # model
    history_days: int = field(default_factory=lambda: int(env("HISTORY_DAYS", "45")))
    max_run: int = field(default_factory=lambda: int(env("MAX_RUN", "4")))
    min_samples: int = field(default_factory=lambda: int(env("MIN_SAMPLES", "300")))
    conf_z: float = field(default_factory=lambda: float(env("CONF_Z", "1.645")))  # 90% lower bound
    model_refresh_secs: float = field(default_factory=lambda: float(env("MODEL_REFRESH_SECS", "3600")))
    # execution gate
    min_edge: Decimal = field(default_factory=lambda: D(env("MIN_EDGE", "0.02")))  # after fees
    max_spread: Decimal = field(default_factory=lambda: D(env("MAX_SPREAD", "0.06")))
    min_price: Decimal = field(default_factory=lambda: D(env("MIN_PRICE", "0.10")))
    max_price: Decimal = field(default_factory=lambda: D(env("MAX_PRICE", "0.85")))
    taker_fee_coef: Decimal = field(default_factory=lambda: D(env("TAKER_FEE_COEF", "0.07")))
    # sizing / risk
    stake_dollars: Decimal = field(default_factory=lambda: D(env("STAKE_DOLLARS", "20")))  # per trade
    max_contracts: int = field(default_factory=lambda: int(env("MAX_CONTRACTS", "50")))
    hard_order_cap: Decimal = field(default_factory=lambda: D(env("HARD_ORDER_CAP_DOLLARS", "100")))
    daily_loss_limit: Decimal = field(default_factory=lambda: D(env("DAILY_LOSS_LIMIT", "60")))
    max_trades_per_day: int = field(default_factory=lambda: int(env("MAX_TRADES_PER_DAY", "40")))
    max_open_positions: int = field(default_factory=lambda: int(env("MAX_OPEN_POSITIONS", "4")))
    loss_streak_pause: int = field(default_factory=lambda: int(env("LOSS_STREAK_PAUSE", "6")))
    paper_start_cash: Decimal = field(default_factory=lambda: D(env("PAPER_START_CASH", "1000")))
    # files (point these at a Railway volume, e.g. /data/..., to survive redeploys)
    ledger_path: str = field(default_factory=lambda: env("LEDGER_PATH", "kalshi_ledger.csv"))
    state_path: str = field(default_factory=lambda: env("STATE_PATH", "kalshi_state.json"))
    prod_confirm: str = field(default_factory=lambda: env("LIVE_PROD_CONFIRM", ""))


def asset_of(series: str) -> str:
    s = series.upper()
    s = s[2:] if s.startswith("KX") else s
    return s[:-3] if s.endswith("15M") else s


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(ts, timezone.utc) if ts else utc_now()
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def is_open(m: Dict[str, Any]) -> bool:
    return str(m.get("status", "")).lower() in ("open", "active")


def result_of(m: Dict[str, Any]) -> Optional[str]:
    r = str(m.get("result", "")).lower()
    return r if r in ("yes", "no") else None


# --------------------------------------------------------------------------
# Streak model
# --------------------------------------------------------------------------
def wilson_lower(p_hat: float, n: int, z: float) -> float:
    if n <= 0:
        return 0.0
    denom = 1 + z * z / n
    centre = p_hat + z * z / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


@dataclass
class StreakModel:
    series: str
    max_run: int
    counts: Dict[str, List[int]] = field(default_factory=dict)  # state -> [ups, total]
    n_outcomes: int = 0
    built_at: float = 0.0

    @staticmethod
    def state_key(direction: str, run: int, max_run: int) -> str:
        return f"{direction}{min(run, max_run)}"

    @staticmethod
    def current_state(outcomes: List[Tuple[float, str]], max_run: int,
                      max_gap: float = 16 * 60) -> Optional[str]:
        """outcomes: [(close_ts, 'U'|'D')] oldest->newest. Run resets on gaps."""
        if not outcomes:
            return None
        last_dir = outcomes[-1][1]
        run = 0
        prev_ts = None
        for ts, d in reversed(outcomes):
            if prev_ts is not None and prev_ts - ts > max_gap:
                break
            if d != last_dir:
                break
            run += 1
            prev_ts = ts
        return StreakModel.state_key(last_dir, run, max_run)

    @classmethod
    def build(cls, series: str, outcomes: List[Tuple[float, str]], max_run: int,
              max_gap: float = 16 * 60) -> "StreakModel":
        m = cls(series, max_run, built_at=time.time())
        last_dir, run, prev_ts = None, 0, None
        for ts, d in outcomes:
            contiguous = prev_ts is not None and ts - prev_ts <= max_gap
            if contiguous and last_dir is not None:
                key = cls.state_key(last_dir, run, max_run)
                c = m.counts.setdefault(key, [0, 0])
                c[0] += 1 if d == "U" else 0
                c[1] += 1
                m.n_outcomes += 1
            if contiguous and d == last_dir:
                run += 1
            else:
                run = 1
            last_dir, prev_ts = d, ts
        return m

    def p_up(self, state: str) -> Tuple[float, int]:
        ups, n = self.counts.get(state, [0, 0])
        return (ups + 1) / (n + 2), n

    def summary(self) -> str:
        parts = []
        for k in sorted(self.counts):
            ups, n = self.counts[k]
            parts.append(f"{k}:{(ups + 1) / (n + 2):.3f}/{n}")
        return " ".join(parts)


# --------------------------------------------------------------------------
# Ledger + persistent state
# --------------------------------------------------------------------------
LEDGER_FIELDS = ["at", "mode", "env", "event", "asset", "ticker", "side", "contracts",
                 "price", "fee", "payout", "pnl", "cash_after", "model_pct",
                 "model_lb_pct", "edge", "n_samples", "state", "reason", "order_id"]


class Ledger:
    def __init__(self, path: str, mode: str, env_name: str):
        self.path, self.mode, self.env = path, mode, env_name
        new = not os.path.exists(path)
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if new:
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, LEDGER_FIELDS).writeheader()

    def write(self, **row: Any) -> None:
        row = {k: ("" if v is None else str(v)) for k, v in row.items()}
        row.update(at=iso(), mode=self.mode, env=self.env)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, LEDGER_FIELDS, extrasaction="ignore").writerow(row)


@dataclass
class Trade:
    ticker: str
    series: str
    side: str              # "yes" | "no"
    direction: str         # "UP" | "DOWN"
    contracts: str
    avg_price: str
    fee: str
    close_ts: float
    model_p: float
    model_lb: float
    order_id: str = ""


@dataclass
class State:
    day: str = ""
    trades_today: int = 0
    realized_today: str = "0"
    loss_streak: int = 0
    paper_cash: str = "0"
    open: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    traded_tickers: List[str] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=lambda: {
        "n": 0, "wins": 0, "pnl": "0", "cost": "0", "sum_p": 0.0, "sum_be": 0.0})

    @classmethod
    def load(cls, path: str, start_cash: Decimal) -> "State":
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return cls(**json.load(f))
            except Exception as e:  # corrupt file -> start clean but say so
                log.error("state file unreadable (%s); starting fresh", e)
        return cls(paper_cash=str(start_cash))

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=1)
        os.replace(tmp, path)

    def roll_day(self) -> None:
        today = utc_now().strftime("%Y-%m-%d")
        if self.day != today:
            self.day, self.trades_today, self.realized_today = today, 0, "0"
            self.loss_streak = 0
            self.traded_tickers = self.traded_tickers[-200:]


# --------------------------------------------------------------------------
# Bot
# --------------------------------------------------------------------------
class Bot:
    def __init__(self, cfg: Config, client: KalshiClient):
        self.cfg, self.k = cfg, client
        self.models: Dict[str, StreakModel] = {}
        self.ledger = Ledger(cfg.ledger_path, cfg.mode, client.env)
        self.state = State.load(cfg.state_path, cfg.paper_start_cash)
        self.running = True
        self.last_stats = 0.0

    # ---------------- data ----------------
    def fetch_outcomes(self, series: str, since_ts: float) -> Tuple[List[Tuple[float, str]],
                                                                    List[Dict[str, Any]]]:
        """Settled outcomes since `since_ts` (oldest->newest) + raw markets."""
        try:
            markets = list(self.k.iter_markets(series_ticker=series,
                                               min_close_ts=int(since_ts)))
        except KalshiError as e:
            log.warning("%s: min_close_ts query failed (%s); falling back", series, e)
            markets = list(self.k.iter_markets(series_ticker=series, status="settled",
                                               max_items=20000))
            markets += list(self.k.iter_markets(series_ticker=series, status="open"))
        seen, outs = set(), []
        for m in markets:
            r, ts = result_of(m), market_close_ts(m)
            if r and ts and ts >= since_ts and m.get("ticker") not in seen:
                seen.add(m.get("ticker"))
                up = (r == "yes") == self.cfg.yes_means_up
                outs.append((ts, "U" if up else "D"))
        outs.sort()
        return outs, markets

    def refresh_model(self, series: str, force: bool = False) -> Optional[StreakModel]:
        m = self.models.get(series)
        if m and not force and time.time() - m.built_at < self.cfg.model_refresh_secs:
            return m
        since = time.time() - self.cfg.history_days * 86400
        outs, _ = self.fetch_outcomes(series, since)
        model = StreakModel.build(series, outs, self.cfg.max_run)
        self.models[series] = model
        log.info("model %s: %d outcomes over %dd | %s", series, model.n_outcomes,
                 self.cfg.history_days, model.summary())
        return model

    # ---------------- decision ----------------
    def evaluate(self, series: str, market: Dict[str, Any], ob: Orderbook,
                 state_key: str, model: StreakModel) -> Dict[str, Any]:
        cfg = self.cfg
        p_up, n = model.p_up(state_key)
        best: Dict[str, Any] = {"ok": False, "reason": "", "n": n, "p_up": p_up}
        if n < cfg.min_samples:
            best["reason"] = f"blocked by samples: state {state_key} has {n} < {cfg.min_samples}"
            return best
        cands = []
        for direction in ("UP", "DOWN"):
            side = "yes" if (direction == "UP") == cfg.yes_means_up else "no"
            p = p_up if direction == "UP" else 1 - p_up
            lb = wilson_lower(p, n, cfg.conf_z)
            ask = ob.ask(side)
            if ask is None:
                cands.append((None, direction, side, p, lb, None, "no asks"))
                continue
            fee_pc = fee_estimate(100, ask, cfg.taker_fee_coef) / 100
            edge = D(f"{lb:.6f}") - ask - fee_pc
            cands.append((edge, direction, side, p, lb, ask, ""))
        valid = [c for c in cands if c[0] is not None]
        if not valid:
            best["reason"] = "blocked by book: no asks on either side"
            return best
        edge, direction, side, p, lb, ask, _ = max(valid, key=lambda c: c[0])
        best.update(direction=direction, side=side, p=p, lb=lb, ask=ask, edge=edge)
        spread = ob.spread(side)
        if edge < cfg.min_edge:
            best["reason"] = (f"blocked by edge: best {direction} lb {lb:.3f} vs ask {ask} "
                              f"+fee -> edge {edge:+.3f} < {cfg.min_edge}")
        elif spread is None or spread > cfg.max_spread:
            best["reason"] = f"blocked by spread: {spread} > {cfg.max_spread}"
        elif not (cfg.min_price <= ask <= cfg.max_price):
            best["reason"] = f"blocked by price band: ask {ask} outside {cfg.min_price}-{cfg.max_price}"
        else:
            best["ok"] = True
        return best

    def size_order(self, ob: Orderbook, side: str, lb: float) -> Tuple[Decimal, Decimal, Decimal]:
        """Returns (contracts, avg_price, price_cap). Cap keeps min_edge after fees."""
        cfg = self.cfg
        cap = D(f"{lb:.4f}") - cfg.min_edge
        cap -= fee_estimate(100, cap, cfg.taker_fee_coef) / 100
        cap = min(cap, cfg.max_price).quantize(D("0.01"), rounding=ROUND_DOWN)
        if cap <= 0:
            return ZERO, ZERO, cap
        budget = min(cfg.stake_dollars, cfg.hard_order_cap)
        want = min(D(cfg.max_contracts), (budget / cap).to_integral_value(rounding=ROUND_DOWN))
        got, avg = ob.fill_plan(side, want, cap)
        got = got.to_integral_value(rounding=ROUND_DOWN)
        return got, avg, cap

    # ---------------- risk ----------------
    def risk_block(self) -> Optional[str]:
        cfg, st = self.cfg, self.state
        st.roll_day()
        if env("PAUSE", "0") == "1":
            return "blocked by PAUSE=1"
        open_cost = sum(D(t["avg_price"]) * D(t["contracts"]) + D(t["fee"])
                        for t in st.open.values())
        if D(st.realized_today) - open_cost <= -cfg.daily_loss_limit:
            return (f"blocked by daily loss limit: realized {st.realized_today} "
                    f"- open risk {open_cost:.2f}")
        if st.trades_today >= cfg.max_trades_per_day:
            return f"blocked by max trades/day ({cfg.max_trades_per_day})"
        if len(st.open) >= cfg.max_open_positions:
            return f"blocked by max open positions ({cfg.max_open_positions})"
        if st.loss_streak >= cfg.loss_streak_pause:
            return f"blocked by loss streak ({st.loss_streak}) - paused for the day"
        return None

    # ---------------- execution ----------------
    def execute(self, series: str, market: Dict[str, Any], ob: Orderbook,
                dec: Dict[str, Any], state_key: str) -> None:
        cfg, st, ticker = self.cfg, self.state, market["ticker"]
        contracts, avg, cap = self.size_order(ob, dec["side"], dec["lb"])
        common = dict(asset=asset_of(series), ticker=ticker, side=dec["direction"],
                      model_pct=f"{dec['p'] * 100:.1f}", model_lb_pct=f"{dec['lb'] * 100:.1f}",
                      n_samples=dec["n"], state=state_key)
        if contracts < 1:
            log.info("%s %s: blocked by depth: nothing at or below cap %s", series, ticker, cap)
            self.ledger.write(event="SKIPPED", reason=f"no depth at or below {cap}", **common)
            return

        order_id = ""
        if cfg.mode == "paper":
            fee = fee_estimate(contracts, avg, cfg.taker_fee_coef)
            cost = contracts * avg + fee
            st.paper_cash = str(D(st.paper_cash) - cost)
            cash_after = st.paper_cash
        else:
            fn = self.k.buy_yes if dec["side"] == "yes" else self.k.buy_no
            resp = fn(ticker, contracts, cap, time_in_force="immediate_or_cancel")
            order_id = resp.get("order_id", "")
            filled = D(resp.get("fill_count", "0"))
            if resp.get("recovered"):
                o = resp.get("order", {})
                filled = D(o.get("fill_count_fp") or o.get("fill_count") or "0")
            if filled <= 0:
                log.info("%s %s: IOC got no fill at cap %s", series, ticker, cap)
                self.ledger.write(event="NOFILL", price=cap, order_id=order_id,
                                  reason="IOC unfilled", **common)
                st.traded_tickers.append(ticker)
                return
            if resp.get("average_fill_price") is not None:
                # V2 quotes the YES side; a NO buy is a YES sell at 1 - price
                yes_avg = D(resp["average_fill_price"])
                avg = yes_avg if dec["side"] == "yes" else ONE - yes_avg
            else:
                avg = cap  # conservative: assume we paid our limit
            fee_pc = D(resp.get("average_fee_paid") or "0")
            fee = ((fee_pc * filled).quantize(D("0.0001")) if fee_pc > 0
                   else fee_estimate(filled, avg, cfg.taker_fee_coef))
            contracts = filled
            cash_after = str(self.k.get_balance())

        t = Trade(ticker=ticker, series=series, side=dec["side"], direction=dec["direction"],
                  contracts=str(contracts), avg_price=str(avg.quantize(D("0.0001"))),
                  fee=str(fee), close_ts=market_close_ts(market) or time.time(),
                  model_p=dec["p"], model_lb=dec["lb"], order_id=order_id)
        st.open[ticker] = asdict(t)
        st.traded_tickers.append(ticker)
        st.trades_today += 1
        edge_real = D(f"{dec['lb']:.6f}") - avg - fee / contracts
        log.info("OPENED %s %s %s x%s @ %s fee %s | p %.3f lb %.3f edge %+.3f",
                 cfg.mode.upper(), ticker, dec["direction"], contracts, t.avg_price, fee,
                 dec["p"], dec["lb"], edge_real)
        self.ledger.write(event="OPENED", contracts=contracts, price=t.avg_price, fee=fee,
                          cash_after=cash_after, edge=f"{edge_real:.4f}", order_id=order_id,
                          reason=f"lb {dec['lb']:.3f} > ask+fee by {edge_real:.3f}", **common)

    def settle(self) -> None:
        st, now = self.state, time.time()
        for ticker in list(st.open):
            t = st.open[ticker]
            if now < t["close_ts"] + 20:
                continue
            try:
                m = self.k.get_market(ticker)
            except KalshiError as e:
                log.warning("settle %s: %s", ticker, e)
                continue
            r = result_of(m)
            if r is None:
                if now > t["close_ts"] + 6 * 3600:
                    log.error("settle %s: no result 6h after close; dropping from open", ticker)
                    st.open.pop(ticker)
                continue
            c, px, fee = D(t["contracts"]), D(t["avg_price"]), D(t["fee"])
            won = r == t["side"]
            payout = c if won else ZERO
            pnl = payout - c * px - fee
            st.realized_today = str(D(st.realized_today) + pnl)
            st.loss_streak = 0 if won else st.loss_streak + 1
            tot = st.totals
            tot["n"] += 1
            tot["wins"] += 1 if won else 0
            tot["pnl"] = str(D(tot["pnl"]) + pnl)
            tot["cost"] = str(D(tot["cost"]) + c * px + fee)
            tot["sum_p"] += t["model_p"]
            tot["sum_be"] += float(px + fee / c)
            if self.cfg.mode == "paper":
                st.paper_cash = str(D(st.paper_cash) + payout)
                cash_after = st.paper_cash
            else:
                cash_after = str(self.k.get_balance())
            st.open.pop(ticker)
            log.info("%s %s %s x%s @ %s -> pnl %+.2f", "WON" if won else "LOST", ticker,
                     t["direction"], c, px, pnl)
            self.ledger.write(event="WON" if won else "LOST", asset=asset_of(t["series"]),
                              ticker=ticker, side=t["direction"], contracts=c, price=px, fee=fee,
                              payout=payout, pnl=f"{pnl:.2f}", cash_after=cash_after,
                              model_pct=f"{t['model_p'] * 100:.1f}",
                              model_lb_pct=f"{t['model_lb'] * 100:.1f}",
                              order_id=t.get("order_id", ""), reason=f"result {r}")

    def stats(self) -> None:
        tot = self.state.totals
        n = tot["n"]
        if n == 0:
            log.info("STATS no settled trades yet | open %d", len(self.state.open))
            return
        win = tot["wins"] / n
        be = tot["sum_be"] / n
        se = math.sqrt(max(win * (1 - win), 1e-9) / n)
        roi = D(tot["pnl"]) / D(tot["cost"]) if D(tot["cost"]) > 0 else ZERO
        log.info("STATS n=%d win=%.1f%% breakeven=%.1f%% (gap %+.1f pts, %.1f SE) "
                 "model avg=%.1f%% pnl=%s ROI=%.1f%% open=%d", n, win * 100, be * 100,
                 (win - be) * 100, (win - be) / se, tot["sum_p"] / n * 100, tot["pnl"],
                 float(roi) * 100, len(self.state.open))

    # ---------------- main cycle ----------------
    def cycle(self) -> None:
        self.settle()
        now = time.time()
        for series in self.cfg.series:
            try:
                self.scan_series(series, now)
            except KalshiError as e:
                log.error("%s: API error %s", series, e)
        if now - self.last_stats > 3600:
            self.stats()
            self.last_stats = now
        self.state.save(self.cfg.state_path)

    def scan_series(self, series: str, now: float) -> None:
        cfg = self.cfg
        model = self.refresh_model(series)
        outs, markets = self.fetch_outcomes(series, now - 6 * 3600)
        live = [m for m in markets if is_open(m) and market_close_ts(m)]
        live = [m for m in live
                if cfg.entry_min_secs <= market_close_ts(m) - now <= cfg.entry_max_secs]
        if not live:
            return
        market = min(live, key=market_close_ts)
        ticker = market["ticker"]
        if ticker in self.state.traded_tickers or ticker in self.state.open:
            return
        close = market_close_ts(market)
        # stale-state guard: the previous 15m result must already be settled
        if not outs or outs[-1][0] < close - 15 * 60 - 60:
            log.info("%s %s: blocked by stale data: previous result not settled yet",
                     series, ticker)
            return
        state_key = StreakModel.current_state(outs, cfg.max_run)
        block = self.risk_block()
        if block:
            log.info("%s %s: %s", series, ticker, block)
            return
        ob = self.k.get_orderbook(ticker)
        dec = self.evaluate(series, market, ob, state_key, model)
        if not dec["ok"]:
            log.info("%s %s [%s]: %s", series, ticker, state_key, dec["reason"])
            self.state.traded_tickers.append(ticker)  # decide once per market
            return
        self.execute(series, market, ob, dec, state_key)

    def run(self) -> None:
        log.info("starting %s on %s | series %s", self.cfg.mode, self.k.env,
                 ",".join(self.cfg.series))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "running", False))
        while self.running:
            started = time.time()
            try:
                self.cycle()
            except Exception:
                log.exception("cycle failed; continuing")
            time.sleep(max(1.0, self.cfg.poll_seconds - (time.time() - started)))
        self.state.save(self.cfg.state_path)
        log.info("stopped cleanly")


# --------------------------------------------------------------------------
# Probe (read-only)
# --------------------------------------------------------------------------
def probe(cfg: Config, k: KalshiClient) -> int:
    print(f"== Kalshi probe | env={k.env} | {k.base}")
    try:
        print("exchange:", k.exchange_status())
    except KalshiError as e:
        print("exchange status failed:", e)
    try:
        bal = k.get_balance()
        print(f"AUTH OK | cash ${bal}")
    except KalshiError as e:
        print("AUTH FAILED:", e)
        print("Check: key ID matches the key, KALSHI_ENV matches where the key was made "
              "(demo keys only work on demo), and the PEM includes the BEGIN/END lines.")
        return 1
    try:
        lim = k.account_limits()
        print("api tier:", lim.get("usage_tier"), "| read", lim.get("read"),
              "| write", lim.get("write"))
    except KalshiError as e:
        print("limits unavailable:", e)
    pos = k.get_positions()
    print(f"open positions: {len(pos)}")
    for p in pos[:10]:
        print("  ", p.get("ticker"), "pos", p.get("position_fp"),
              "exposure $", p.get("market_exposure_dollars"))
    resting = k.get_orders(status="resting", max_items=50)
    print(f"resting orders: {len(resting)}")

    bot = Bot(cfg, k)
    now = time.time()
    for s in cfg.series:
        print(f"\n-- {s}")
        try:
            model = bot.refresh_model(s, force=True)
            outs, markets = bot.fetch_outcomes(s, now - 6 * 3600)
        except KalshiError as e:
            print("   error:", e)
            continue
        live = sorted([m for m in markets if is_open(m) and market_close_ts(m)],
                      key=market_close_ts)
        print("   last results:", "".join(d for _, d in outs[-12:]) or "(none)",
              "| state", StreakModel.current_state(outs, cfg.max_run))
        print("   model:", model.summary() or "(no history)")
        if not live:
            print("   no open market right now")
            continue
        m = live[0]
        ob = k.get_orderbook(m["ticker"])
        mins = (market_close_ts(m) - now) / 60
        print(f"   open: {m['ticker']} closes in {mins:.1f} min")
        print(f"   title: {m.get('title')} | yes = {m.get('yes_sub_title') or m.get('subtitle')}")
        print(f"   YES bid/ask {ob.best_yes_bid}/{ob.yes_ask}  NO bid/ask {ob.best_no_bid}/{ob.no_ask}"
              f"  top depth yes {ob.yes_bids[0][1] if ob.yes_bids else 0}"
              f" no {ob.no_bids[0][1] if ob.no_bids else 0}")
        st = StreakModel.current_state(outs, cfg.max_run)
        if st:
            dec = bot.evaluate(s, m, ob, st, model)
            print("   decision:", "TRADE " + dec.get("direction", "") if dec["ok"] else dec["reason"])
    print("\nprobe finished (read-only, no orders sent)")
    return 0


# --------------------------------------------------------------------------
# Order test (demo only): proves the trading plumbing end to end
# --------------------------------------------------------------------------
def ordertest(cfg: Config, k: KalshiClient) -> int:
    """
    1. resting post-only order far from the market -> confirm it rests -> cancel it
    2. buy 1 contract at the ask (NO side if available, to test the conversion)
    3. confirm position + fill, wait for settlement, reconcile cash to the cent
    Places at most ONE filled contract. Refuses to run on prod.
    """
    checks: Dict[str, bool] = {}

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks[name] = ok
        print(f"   [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    print(f"== ORDER TEST | env={k.env}")
    if k.env != "demo":
        print("refusing: order test only runs on KALSHI_ENV=demo")
        return 0
    series = env("ORDERTEST_SERIES", "KXBTC15M")
    bal0 = k.get_balance()
    print(f"cash ${bal0}")
    if bal0 < 5:
        print("demo balance under $5 - add mock funds at demo.kalshi.co first")
        return 0

    # find an open market with at least 4 minutes left
    market, ob = None, None
    for _ in range(12):
        now = time.time()
        live = [m for m in k.iter_markets(series_ticker=series, min_close_ts=int(now))
                if is_open(m) and (market_close_ts(m) or 0) - now >= 240]
        if live:
            market = min(live, key=market_close_ts)
            ob = k.get_orderbook(market["ticker"])
            if ob.yes_ask is not None or ob.no_ask is not None:
                break
        print("   waiting for an open market with asks ...")
        time.sleep(15)
        market = None
    if not market:
        print("no tradeable market found - try again in a few minutes")
        return 0
    ticker = market["ticker"]
    close = market_close_ts(market)
    print(f"market {ticker} | closes in {(close - time.time()) / 60:.1f} min | "
          f"YES {ob.best_yes_bid}/{ob.yes_ask}  NO {ob.best_no_bid}/{ob.no_ask}")
    if k.get_positions(ticker=ticker):
        print("already holding this market (earlier run?) - not adding another order")
        return 0

    # 1) resting order + cancel
    print("\n1) resting order + cancel")
    try:
        r = k.create_order(ticker, "bid", 1, D("0.01"), time_in_force="good_till_canceled",
                           post_only=True)
        oid = r.get("order_id", "")
        check("order accepted", bool(oid), f"id={oid}")
        time.sleep(1.5)
        resting = [o for o in k.get_orders(ticker=ticker, status="resting")
                   if o.get("order_id") == oid]
        check("order is resting on the book", bool(resting))
        k.cancel_order(oid, ticker)
        time.sleep(1.5)
        still = [o for o in k.get_orders(ticker=ticker, status="resting")
                 if o.get("order_id") == oid]
        check("cancel removed it", not still)
    except KalshiError as e:
        check("resting order round-trip", False, str(e))

    # 2) one real fill
    print("\n2) buy 1 contract at the ask")
    side = "no" if ob.no_ask is not None else "yes"
    ask = ob.ask(side)
    try:
        fn = k.buy_no if side == "no" else k.buy_yes
        r = fn(ticker, 1, ask, time_in_force="immediate_or_cancel")
        print("   raw response:", json.dumps({kk: r.get(kk) for kk in
              ("order_id", "fill_count", "remaining_count", "average_fill_price",
               "average_fee_paid", "recovered")}))
        filled = D(r.get("fill_count") or "0")
        check(f"filled 1 {side.upper()} at <= {ask}", filled >= 1, f"fill_count={filled}")
    except KalshiError as e:
        check("fill", False, str(e))
        filled = ZERO
    if filled < 1:
        print("no fill (thin demo book?) - run again next window")
        return 0
    yes_avg = r.get("average_fill_price")
    paid = (D(yes_avg) if side == "yes" else ONE - D(yes_avg)) if yes_avg else ask
    fee = D(r.get("average_fee_paid") or "0") or fee_estimate(1, paid)
    print(f"   paid {paid} + fee {fee} for 1 {side.upper()}")
    time.sleep(2)
    pos = k.get_positions(ticker=ticker)
    pfp = D(pos[0].get("position_fp", "0")) if pos else ZERO
    # Kalshi reports YES as positive, NO as negative
    check("position shows on the account",
          bool(pos) and ((pfp > 0) if side == "yes" else (pfp < 0)), f"position_fp={pfp}")
    fills = k.get_fills(ticker=ticker, max_items=5)
    check("fill appears in fill history", bool(fills))
    if fills:
        print("   fill record:", json.dumps({kk: fills[0].get(kk) for kk in list(fills[0])[:12]}))
    bal1 = k.get_balance()
    check("cash dropped by cost + fee", abs((bal0 - bal1) - (paid + fee)) <= D("0.011"),
          f"{bal0} -> {bal1}")

    # 3) settlement
    print(f"\n3) waiting for settlement (~{(close - time.time()) / 60:.0f} min) ...")
    result = None
    deadline = close + 1800
    while time.time() < deadline:
        time.sleep(30)
        if time.time() < close:
            continue
        try:
            result = result_of(k.get_market(ticker))
        except KalshiError:
            continue
        if result:
            break
    if not result:
        check("market settled within 30 min of close", False)
    else:
        won = result == side
        time.sleep(20)  # let the cash credit post
        bal2 = k.get_balance()
        expected = bal1 + (ONE if won else ZERO)
        print(f"   result: {result.upper()} -> you {'WON' if won else 'LOST'}")
        check("settled cash matches expectation", abs(bal2 - expected) <= D("0.011"),
              f"expected {expected}, got {bal2}")
        print(f"   net on the test: {bal2 - bal0:+}")

    ok = all(checks.values())
    print(f"\nORDER TEST: {'ALL PASS - trading plumbing verified' if ok else 'SOME CHECKS FAILED'}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config()
    if cfg.mode == "backtest":
        import kalshi_backtest
        return kalshi_backtest.run()
    k = KalshiClient()
    if cfg.mode == "probe":
        return probe(cfg, k)
    if cfg.mode == "ordertest":
        try:
            return ordertest(cfg, k)
        except Exception:
            # exit 0 on purpose: a crash would make Railway restart us and
            # place the test order again
            log.exception("order test aborted")
            return 0
    if cfg.mode == "live" and k.env == "prod" and cfg.prod_confirm != "YES_REAL_MONEY":
        log.error("refusing live trading on PROD without LIVE_PROD_CONFIRM=YES_REAL_MONEY")
        return 2
    if cfg.mode not in ("paper", "live"):
        log.error("MODE must be probe, paper or live")
        return 2
    Bot(cfg, k).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
