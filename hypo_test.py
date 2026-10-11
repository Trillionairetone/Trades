"""Fixed-rule, candle-only hypothesis backtests. Run HYPO=csm|shock python hypo_test.py.

Optional HYPO_END is an ISO-8601 exclusive endpoint (date-only = midnight UTC).
Default endpoint is startup UTC time; it is printed for reproducibility. No tuning
parameters are exposed. Credentials: OANDA_API_KEY (also usual token aliases),
OANDA_ENV=practice|live. Only public instrument-candle GETs are implemented; no
bot import, account ID, or trading interface is needed. Railway restart policy
should be Never: this is a one-shot job, exiting after the report.

Conventions fixed before results: sample SD (ddof=1); Wilder ATR seeded with the
mean of 20 true ranges (first TR is high-low); daily candle's trading date is
its New York close date. Split by elapsed UTC time, not trade count, and close
positions at each period's last completed M15 close. Each leg has one constant
initial cash-risk unit, so monthly R is the sum, not the average, of leg R.
R uses fixed entry-time quote conversion (no subsequent FX translation).
Intrabar timestamps are candle starts, explicitly approximate; rollover and
holding estimates therefore have at most a candle of timing uncertainty.
Spread cost is attribution, NOT another deduction: entry half-spread plus exit
half-spread (OHLC component at fill, open spread as proxy for intrabar fills).
Stress fixes accepted baseline entry IDs, direction, risk and scheduled exit;
levels are re-anchored at stressed entry prices, exits are re-simulated. Old
ordering also fixes baseline entries, isolating the exit-engine difference.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone, time as walltime
import gc
import json
import math
import os
import statistics
import time
from zoneinfo import ZoneInfo

UTC = timezone.utc
NY = ZoneInfo('America/New_York')
CSM_PAIRS = ('EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD',
             'USD_JPY', 'USD_CAD', 'USD_CHF')
SHOCK_PAIRS = ('EUR_USD', 'GBP_USD', 'USD_JPY')


def stamp(value):
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def iso(value):
    return value.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def pip_size(pair):
    return 0.01 if pair.endswith('_JPY') else 0.0001


def close_time(start, gran):
    if gran == 'D':
        local = start.astimezone(NY)
        return datetime.combine(local.date() + timedelta(days=1), walltime(17), NY).astimezone(UTC)
    return start + timedelta(minutes={'H1': 60, 'M15': 15}[gran])


@dataclass(frozen=True)
class Candle:
    t: datetime
    end: datetime
    mid: tuple  # o,h,l,c
    bid: tuple
    ask: tuple

    def widened(self):
        return replace(self,
                       bid=tuple(m + 2 * (b - m) for m, b in zip(self.mid, self.bid)),
                       ask=tuple(m + 2 * (a - m) for m, a in zip(self.mid, self.ask)))


def truncate(candles, end):
    return (c for c in candles if c.end <= end and c.t < end)


class CandleClient:
    """Intentionally exposes ONLY a fixed instrument candle endpoint."""
    def __init__(self):
        import requests
        token = next((os.getenv(k) for k in ('OANDA_API_KEY', 'OANDA_API_TOKEN',
                     'OANDA_TOKEN', 'OANDA_ACCESS_TOKEN') if os.getenv(k)), None)
        if not token:
            raise RuntimeError('Set OANDA_API_KEY to a candle-data API token')
        env = os.getenv('OANDA_ENV', 'practice').lower()
        if env not in ('practice', 'live'):
            raise ValueError('OANDA_ENV must be practice or live')
        self.base = 'https://api-fxpractice.oanda.com' if env == 'practice' else 'https://api-fxtrade.oanda.com'
        self.session = requests.Session()
        self.session.headers['Authorization'] = 'Bearer ' + token.strip()

    def page(self, pair, gran, **params):
        if pair not in CSM_PAIRS or gran not in ('D', 'H1', 'M15'):
            raise ValueError('Unsupported candle request')
        params.update(granularity=gran, price='MBA', smooth='false',
                      dailyAlignment=17, alignmentTimezone='America/New_York')
        for attempt in range(5):
            response = self.session.get(self.base + '/v3/instruments/' + pair + '/candles',
                                        params=params, timeout=60)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            raw = response.json()['candles']
            result = []
            for row in raw:
                if row.get('complete'):
                    t = stamp(row['time'])
                    values = [tuple(float(row[k][x]) for x in 'ohlc') for k in ('mid', 'bid', 'ask')]
                    if any(not math.isfinite(v) for group in values for v in group):
                        raise ValueError('Nonfinite candle data')
                    result.append(Candle(t, close_time(t, gran), *values))
            return sorted(result, key=lambda c: c.t)
        raise RuntimeError('Candle API retry limit reached')

    def history(self, pair, gran, end):
        """Backward paging discovers history without an arbitrary start date."""
        cursor, rows = end, []
        while True:
            page = self.page(pair, gran, to=iso(cursor), count=5000)
            page = [c for c in page if c.t < cursor and c.end <= end]
            if not page:
                break
            rows.extend(page)
            cursor = page[0].t
            print(f'Download {pair} {gran}: {len(rows)} candles, back to {iso(cursor)}', flush=True)
        unique = {c.t: c for c in rows}
        return sorted(unique.values(), key=lambda c: c.t)

    def stream(self, pair, start, end):
        """Forward M15 pages, at most 5,000 candles resident; end clipped first."""
        cursor = start
        while cursor < end:
            page = self.page(pair, 'M15', **{'from': iso(cursor), 'count': 5000})
            page = [c for c in page if c.t >= cursor]
            if not page:
                return
            for c in truncate(page, end):
                yield c
            if page[-1].end >= end:
                return
            next_cursor = page[-1].t + timedelta(seconds=1)
            if next_cursor <= cursor:
                raise RuntimeError('Candle pagination made no progress')
            cursor = next_cursor

    def first_m15(self, start, end, pair):
        rows = self.page(pair, 'M15', **{'from': iso(start), 'count': 1})
        if not rows or rows[0].end > end:
            raise RuntimeError(f'No M15 history for {pair}')
        return rows[0].t


def wilder(candles, n=20):
    result, seed, previous, atr = [], [], None, None
    for c in candles:
        _, high, low, closing = c.mid
        tr = high - low if previous is None else max(high-low, abs(high-previous), abs(low-previous))
        previous = closing
        if atr is None:
            seed.append(tr)
            if len(seed) == n:
                atr = statistics.fmean(seed)
        else:
            atr = ((n-1)*atr + tr)/n
        result.append(atr)
    return result


def rollovers(entry, exit_):
    """17:00 NY boundaries in (entry, exit], calendar nights including weekends."""
    if exit_ < entry:
        raise ValueError('Exit precedes entry')
    def bucket(t):
        local = t.astimezone(NY)
        return local.date().toordinal() - (local.time() < walltime(17))
    return bucket(exit_) - bucket(entry)


def currency_direction(pair, buy_currency):
    direction = 1 if buy_currency else -1
    return -direction if pair.startswith('USD_') else direction


@dataclass(frozen=True)
class Plan:
    pair: str
    when: datetime
    direction: int
    risk: float
    rr: float
    expiry: datetime | None = None
    bars: int | None = None
    month: str | None = None


def csm_plans(history, start, end, session_dates=None):
    atrs = {p: wilder(rows) for p, rows in history.items()}
    ends = {p: [c.end for c in rows] for p, rows in history.items()}
    # The session close date identifies Monday's Sunday-17:00 daily candle.
    dates = sorted(session_dates if session_dates is not None else
                   {c.end.astimezone(NY).date() for rows in history.values() for c in rows})
    first = {}
    for date in dates:
        if date.weekday() < 5:
            first.setdefault((date.year, date.month), date)
    schedule = [datetime.combine(d, walltime(9), NY).astimezone(UTC) for d in first.values()]
    schedule.sort()
    plans = []
    for j, when in enumerate(schedule):
        if not start <= when < end:
            continue
        expiry = schedule[j+1] if j+1 < len(schedule) else end
        scores, risks = {}, {}
        for pair, rows in history.items():
            i = bisect_right(ends[pair], when)-1
            if i < 63 or atrs[pair][i] is None:
                break
            returns = [math.log(rows[k].mid[3]/rows[k-1].mid[3]) for k in range(i-62, i+1)]
            sd = statistics.stdev(returns)
            if sd <= 0 or atrs[pair][i] <= 0:
                break
            scores[pair] = sum(returns)/sd * (-1 if pair.startswith('USD_') else 1)
            risks[pair] = 2*atrs[pair][i]
        if len(scores) != len(history):
            continue
        strongest = max(sorted(scores), key=scores.get)
        weakest = min(sorted(scores), key=scores.get)
        if scores[strongest] > 0 and scores[weakest] < 0:
            for pair, buy in ((strongest, True), (weakest, False)):
                plans.append(Plan(pair, when, currency_direction(pair, buy), risks[pair], 3,
                                  expiry=expiry, month=when.astimezone(NY).strftime('%Y-%m')))
    return plans


def shock_plans(pair, candles, start, end):
    atrs = wilder(candles)
    plans = []
    for i, c in enumerate(candles):
        if i == 0 or atrs[i-1] is None or not start <= c.end < end:
            continue
        local = c.end.astimezone(NY).time()
        if not walltime(8) <= local <= walltime(12):
            continue
        o, h, l, closing = c.mid
        risk = atrs[i-1]
        if h <= l or risk <= 0 or abs(closing-o) <= 1.5*risk:
            continue
        direction = 1 if closing > o else -1
        if (direction == 1 and closing >= h-0.2*(h-l)) or (direction == -1 and closing <= l+0.2*(h-l)):
            plans.append(Plan(pair, c.end, direction, risk, 2, bars=16))
    return plans


class Position:
    def __init__(self, plan, candle, variant):
        self.plan, self.variant = plan, variant
        c = candle.widened() if variant == 'stress' else candle
        self.entry = c.ask[0] if plan.direction == 1 else c.bid[0]
        self.entry_time = c.t
        self.stop = self.entry-plan.direction*plan.risk
        self.target = self.entry+plan.direction*plan.risk*plan.rr
        self.entry_half_spread = (c.ask[0]-c.bid[0])/2
        self.held = 0
        self.result = None

    def finish(self, candle, price, reason, timestamp, approximate=False, ambiguous=False, component=0):
        p = self.plan
        nights = rollovers(self.entry_time, timestamp)
        slip = (0.4 if self.variant == 'stress' else 0.2)*pip_size(p.pair)
        swap = nights*0.5*pip_size(p.pair)
        spread = self.entry_half_spread + (candle.ask[component]-candle.bid[component])/2
        self.result = dict(pair=p.pair, entry=iso(self.entry_time), exit=iso(timestamp),
                           direction=p.direction, risk=p.risk, entry_price=self.entry, exit_price=price,
                           r=(p.direction*(price-self.entry)-slip-swap)/p.risk,
                           cost_r=(spread+slip+swap)/p.risk, spread_r=spread/p.risk,
                           slippage_r=slip/p.risk, swap_r=swap/p.risk, nights=nights,
                           holding_hours=(timestamp-self.entry_time).total_seconds()/3600,
                           reason=reason, approximate=approximate, ambiguous=int(ambiguous),
                           end_forced=reason == 'end', month=p.month, variant=self.variant)
        return self.result

    def step(self, raw):
        if self.result is not None:
            return self.result
        c = raw.widened() if self.variant == 'stress' else raw
        p = self.plan
        o, h, l, closing = c.bid if p.direction == 1 else c.ask
        # Monthly scheduled liquidation precedes any new monthly entry.
        if p.expiry is not None and c.t >= p.expiry:
            return self.finish(c, o, 'month', c.t)
        self.held += 1
        stop_open = o <= self.stop if p.direction == 1 else o >= self.stop
        target_open = o >= self.target if p.direction == 1 else o <= self.target
        stop_hit = l <= self.stop if p.direction == 1 else h >= self.stop
        target_hit = h >= self.target if p.direction == 1 else l <= self.target
        if self.variant != 'old':
            if stop_open:
                return self.finish(c, o, 'stop_gap', c.t)
            if target_open:
                return self.finish(c, self.target, 'target_gap', c.t)
        if stop_hit:
            # Legacy ordering still applied adverse stop gaps, but checked the
            # whole candle's stop before an already-executable target at open.
            price = o if stop_open else self.stop
            return self.finish(c, price, 'stop', c.t, not stop_open, target_hit)
        if target_hit:
            return self.finish(c, self.target, 'target', c.t, True)
        if p.bars is not None and self.held >= p.bars:
            return self.finish(c, closing, 'time', c.end, component=3)
        return None

    def force_end(self, raw):
        c = raw.widened() if self.variant == 'stress' else raw
        px = c.bid[3] if self.plan.direction == 1 else c.ask[3]
        return self.finish(c, px, 'end', c.end, component=3)


def simulate(candles, plans, start, end, strategy):
    """Single pass; baseline eligibility freezes the trade set for all variants."""
    plans = sorted((p for p in plans if start <= p.when < end), key=lambda p: p.when)
    records = {v: [] for v in ('base', 'old', 'stress')}
    active, used_dates, skips, cursor, last = [], set(), 0, 0, None
    for c in truncate(candles, end):
        if c.t < start:
            continue
        last = c
        # At a monthly boundary, liquidate first, even in a no-trade month.
        for pos in active[:]:
            if pos.plan.expiry is not None and c.t >= pos.plan.expiry:
                records[pos.variant].append(pos.step(c))
                active.remove(pos)
        due = []
        while cursor < len(plans) and plans[cursor].when <= c.t:
            due.append(plans[cursor])
            cursor += 1
        for p in due:
            if p.expiry is not None and c.t >= p.expiry:
                continue
            date = c.t.astimezone(NY).date()
            if any(pos.variant == 'base' for pos in active):
                continue
            if strategy == 'shock' and date in used_dates:
                continue
            if c.ask[0]-c.bid[0] > 0.05*p.risk + 1e-12:
                skips += 1
                continue
            used_dates.add(date)
            active.extend(Position(p, c, v) for v in records)
        for pos in active[:]:
            result = pos.step(c)
            if result is not None:
                records[pos.variant].append(result)
                active.remove(pos)
    if last is not None:
        for pos in active:
            records[pos.variant].append(pos.force_end(last))
    for values in records.values():
        values.sort(key=lambda r: (r['entry'], r['pair']))
    return records, skips


def metrics(values):
    values = list(values)
    positive = sum(x for x in values if x > 0)
    negative = -sum(x for x in values if x < 0)
    equity = peak = dd = 0.0
    for x in values:
        equity += x
        peak = max(peak, equity)
        dd = max(dd, peak-equity)
    return dict(trades=len(values), win_rate=sum(x > 0 for x in values)/len(values) if values else 0,
                avgR=statistics.fmean(values) if values else 0,
                PF=positive/negative if negative else ('infinity' if positive else None), max_drawdown_R=dd)


def summarize(rows, strategy, skips):
    # Aggregate simultaneous closes to avoid a spurious arbitrary pair ordering.
    exits = defaultdict(float)
    for r in rows:
        exits[r['exit']] += r['r']
    report = metrics(r['r'] for r in rows)
    report['max_drawdown_R'] = metrics(exits[t] for t in sorted(exits))['max_drawdown_R']
    report.update(average_holding_hours=statistics.fmean(r['holding_hours'] for r in rows) if rows else 0,
                  average_cost_R=statistics.fmean(r['cost_r'] for r in rows) if rows else 0,
                  ambiguous_candles=sum(r['ambiguous'] for r in rows), spread_filter_skips=skips,
                  end_forced=sum(r['end_forced'] for r in rows),
                  approximate_exits=sum(r['approximate'] for r in rows))
    report['per_pair'] = {p: dict(metrics(r['r'] for r in rows if r['pair'] == p),
                                  total_R=sum(r['r'] for r in rows if r['pair'] == p))
                          for p in (CSM_PAIRS if strategy == 'csm' else SHOCK_PAIRS)}
    if strategy == 'csm':
        months = defaultdict(float)
        for r in rows:
            months[r['month']] += r['r']
        report['monthly'] = metrics(months[m] for m in sorted(months))
        report['monthly']['months_traded'] = report['monthly'].pop('trades')
        report['monthly_R'] = dict(sorted(months.items()))
    return report


def changed_count(base, old):
    def key(r):
        return r['pair'], r['entry']
    other = {key(r): r for r in old}
    return sum(r['exit'] != other[key(r)]['exit'] or
               not math.isclose(r['exit_price'], other[key(r)]['exit_price'], abs_tol=1e-12) or
               not math.isclose(r['r'], other[key(r)]['r'], abs_tol=1e-12)
               for r in base)


def emit(tag, value):
    print(tag + ' ' + json.dumps(value, allow_nan=False, sort_keys=True), flush=True)


def main():
    strategy = os.getenv('HYPO', '').lower().strip()
    if strategy not in ('csm', 'shock'):
        raise ValueError('Set HYPO=csm or HYPO=shock')
    end = stamp(os.environ['HYPO_END']) if os.getenv('HYPO_END') else datetime.now(UTC)
    if end > datetime.now(UTC):
        raise ValueError('HYPO_END cannot be in the future')
    api = CandleClient()
    pairs = CSM_PAIRS if strategy == 'csm' else SHOCK_PAIRS
    gran = 'D' if strategy == 'csm' else 'H1'
    histories, starts = {}, []
    # H1 discovery retains just one pair at a time, then reloads that pair for
    # simulation. Daily history for all seven pairs is small and is retained.
    for pair in pairs:
        rows = api.history(pair, gran, end)
        warm = 63 if strategy == 'csm' else 19
        if len(rows) <= warm:
            raise RuntimeError(f'Insufficient {gran} history for {pair}')
        first = api.first_m15(rows[0].t, end, pair)
        starts.append(max(rows[warm].end, first))
        if strategy == 'csm':
            histories[pair] = rows
        del rows
        gc.collect()
    start = max(starts)
    split = start + (end-start)*2/3
    periods = [('evaluation', start, split), ('holdout', split, end)]
    emit('WINDOWS', dict(strategy=strategy, start=iso(start), split=iso(split), end=iso(end),
                         holdout_label='previously inspected (related trend-following tests)' if strategy == 'csm' else 'holdout'))
    print('Fixed rules. No parameter search. Equal initial risk units; realized-close drawdown. '
          'Intrabar times/costs approximate. Periods flat at boundaries. Stress and old ordering use baseline entries.', flush=True)
    if strategy == 'csm':
        print('The two legs are correlated; months are the independent sample. '
              'Monthly statistics include months with >=1 accepted leg; portfolio R=sum(leg R).', flush=True)
    all_rows = {name: {v: [] for v in ('base', 'old', 'stress')} for name, _, _ in periods}
    skips = defaultdict(int)
    for pair in pairs:
        rows = histories if strategy == 'csm' else api.history(pair, gran, end)
        for name, a, b in periods:
            if strategy == 'csm':
                # Truncate daily data at THIS period endpoint before planning.
                clipped = {p: list(truncate(ds, b)) for p, ds in rows.items()}
                # Session-calendar metadata uses only session opens before the
                # cutoff; no incomplete daily OHLC enters rankings or ATR.
                dates = {c.end.astimezone(NY).date() for ds in rows.values() for c in ds if c.t < b}
                plans = [p for p in csm_plans(clipped, a, b, dates) if p.pair == pair]
                del clipped
                # Fetch only selected monthly holding windows for this pair.
                def selected_candles():
                    for plan in plans:
                        # Include the first available expiry open, even after a
                        # long data gap. Adjacent holding windows are deduped.
                        for candle in api.stream(pair, plan.when, b):
                            yield candle
                            if candle.t >= plan.expiry:
                                break
                def unique_candles():
                    previous = None
                    for candle in selected_candles():
                        if previous is None or candle.t > previous:
                            previous = candle.t
                            yield candle
                candles = unique_candles()
            else:
                plans = shock_plans(pair, list(truncate(rows, b)), a, b)
                candles = api.stream(pair, a, b)
            results, skipped = simulate(candles, plans, a, b, strategy)
            skips[name] += skipped
            for variant, trades in results.items():
                all_rows[name][variant].extend(trades)
            print(f'Finished {pair} {name}: {len(results["base"])} accepted legs, {skipped} spread skips', flush=True)
        del rows, plans, results, candles
        gc.collect()
    pooled = {v: all_rows['evaluation'][v]+all_rows['holdout'][v] for v in ('base', 'old', 'stress')}
    reports = {}
    for name, variants in list(all_rows.items()) + [('pooled', pooled)]:
        skip = sum(skips.values()) if name == 'pooled' else skips[name]
        reports[name] = {}
        for variant, trades in variants.items():
            reports[name][variant] = summarize(trades, strategy, skip)
            emit(f'REPORT {name} {variant}', reports[name][variant])
        emit(f'ORDERING_CHANGE {name}', dict(changed_trades=changed_count(variants['base'], variants['old']),
                                             compared_trades=len(variants['base'])))
    # Leg-level audit trail is printed once (pooled), including stress and legacy.
    for variant, trades in pooled.items():
        for r in sorted(trades, key=lambda r: (r['entry'], r['pair'])):
            emit('TRADE', r)
    for variant in ('base', 'stress'):
        p = reports['pooled'][variant]
        pf_ok = p['PF'] == 'infinity' or (isinstance(p['PF'], (float, int)) and p['PF'] >= 1.2)
        passed = (reports['evaluation'][variant]['avgR'] > 0 and reports['holdout'][variant]['avgR'] > 0
                  and pf_ok and p['trades'] >= 300)
        emit('PASS_BAR', dict(strategy=strategy, variant=variant, result='PASS' if passed else 'FAIL',
                              closed_trades=p['trades'], months=p.get('monthly', {}).get('months_traded')))


if __name__ == '__main__':
    main()
