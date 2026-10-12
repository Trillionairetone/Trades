"""Frozen daily pullback specification revision 2; historical simulation only.

Run: PB_END=2026-10-09T21:00:00Z python pullback_test.py
Optional PB_PAIRS is a comma-separated subset of the seven specified pairs.
Requires Python 3.10+ and zoneinfo timezone data. Standard library only.
Credentials: OANDA_API_KEY or OANDA_TOKEN. Practice instrument candle GETs only.
No account ID, bot imports, trading interfaces, or per-trade log output.

Implementation choices where the specification is silent:
* Intrabar stops get a conservative candle-CLOSE timestamp and an approximate
  flag, with the candle start retained. Thus no completed candle is inspected
  early, including at a split falling inside an M15 candle. Re-entry requires a
  daily completion strictly after this recorded exit time.
* At shared timestamps: candle close/stop/endpoint liquidation, daily signals,
  candle open and gap/scheduled exit/entry, portfolio mark, missed-entry expiry.
* The first scheduled exit is latched; recovery takes precedence if recovery
  and holding-day five trigger on the same daily close. Delayed exits retain
  their underlying trigger as well as their delay flag.
* Coverage assigns candles by NY OPEN date; [06:00,10:00) London defines presence
  for missing-entry classification. This is the specification's heuristic for
  'market closed', not independent proof of a closure. Partial classification
  windows ending after PB_END are excluded from skipped-entry counts.
* Gaps count missing 15-minute slots in the normal Sun-17:00 to Fri-17:00 NY
  weekly market window. Weekday holidays are included. Long trade-gap flags use
  >288 missing market-open M15 slots (three 24-hour trading days).
* Zero/nonfinite ATR on an eligible signal, malformed quotes, insufficient warm-up, empty M15 data,
  failed downloads or no full weekdays make the run INCOMPLETE, never PASS.
* Close costs use half entry-open plus half exit-close spread; intrabar and gap
  fills use exit-open spread as the attribution proxy. No extra spread deduction.
* Same-timestamp realized exits are aggregated for trade drawdown; open equity
  drawdown is sampled at the prescribed weekly/split/endpoint marks only. Position
  and USD exposure averages/maxima are sampled at those marks, not continuous peaks.
* Price-based R and entry-time currency conversion are frozen at the BASELINE
  entry for stress as well. This is not revalued USD P&L for USD-base pairs.
* Marks at the split use only fully completed candle quotes. A source quote
  older than four hours is flagged; its exact age is retained in the result.
* The final completed M15 close is known from one-candle lookahead and forces
  endpoint liquidation there. Later marks cannot accrue costs on closed trades.
* RSI/ATR are seeded from the oldest completed daily history returned by OANDA.
  The common start and split are based only on availability and warm-up, not returns.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, time as walltime, timezone
import gc
import heapq
import json
import math
import os
import statistics
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo

UTC = timezone.utc
NY = ZoneInfo('America/New_York')
LONDON = ZoneInfo('Europe/London')
M15 = timedelta(minutes=15)
PAIRS = ('EUR_USD', 'GBP_USD', 'AUD_USD', 'NZD_USD', 'USD_JPY', 'USD_CAD', 'USD_CHF')
FROZEN_END = datetime(2026, 10, 9, 21, tzinfo=UTC)
VARIANTS = ('baseline', 'stress')


def ts(value):
    if isinstance(value, datetime):
        v = value
    else:
        v = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if v.tzinfo is None:
        raise ValueError('Timestamps must include a timezone')
    return v.astimezone(UTC)


def iso(value):
    return value.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def pip(pair):
    return .01 if pair == 'USD_JPY' else .0001


def local_time(day, hour, zone):
    return datetime.combine(day, walltime(hour), zone).astimezone(UTC)


def next_london_open(completion):
    local = completion.astimezone(LONDON)
    day = local.date()
    candidate = local_time(day, 8, LONDON)
    if candidate <= completion:
        day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return local_time(day, 8, LONDON)


def nights(entry, end):
    def bucket(t):
        v = t.astimezone(NY)
        return v.date().toordinal() - int(v.time() < walltime(17))
    if end < entry:
        raise ValueError('Valuation before entry')
    return bucket(end) - bucket(entry)


@dataclass(frozen=True)
class Candle:
    t: datetime
    end: datetime
    mid: tuple
    bid: tuple
    ask: tuple

    def quote(self, side, index, stress=False):
        raw = (self.bid if side == 'bid' else self.ask)[index]
        return self.mid[index] + 2*(raw-self.mid[index]) if stress else raw

    def spread(self, index, stress=False):
        return (self.ask[index]-self.bid[index])*(2 if stress else 1)


def parse(raw, gran):
    rows = []
    for r in raw:
        if not r.get('complete'):
            continue
        t = ts(r['time'])
        end = (local_time(t.astimezone(NY).date()+timedelta(days=1), 17, NY)
               if gran == 'D' else t+M15)
        values = [tuple(float(r[k][f]) for f in 'ohlc') for k in ('mid', 'bid', 'ask')]
        if any(not math.isfinite(x) or x <= 0 for q in values for x in q):
            raise ValueError('Invalid candle prices')
        for q in values:
            o, h, l, c = q
            if not l <= min(o, c) <= max(o, c) <= h:
                raise ValueError('Invalid candle OHLC ordering')
        if any(b > a for b, a in zip(values[1], values[2])):
            raise ValueError('Negative spread')
        rows.append(Candle(t, end, *values))
    return sorted({r.t: r for r in rows}.values(), key=lambda r: r.t)


class CandleClient:
    """No arbitrary URL/path methods: fixed practice instrument-candle reads only."""
    def __init__(self):
        self._token = (os.getenv('OANDA_API_KEY') or os.getenv('OANDA_TOKEN') or '').strip()
        if not self._token:
            raise ValueError('Set OANDA_API_KEY or OANDA_TOKEN')

    def page(self, pair, gran, **params):
        if pair not in PAIRS or gran not in ('D', 'M15'):
            raise ValueError('Unsupported instrument or granularity')
        params.update(granularity=gran, price='MBA', smooth='false',
                      dailyAlignment=17, alignmentTimezone='America/New_York')
        url = 'https://api-fxpractice.oanda.com/v3/instruments/'+pair+'/candles?'+urlencode(params)
        for attempt in range(5):
            req = Request(url, headers={'Authorization': 'Bearer '+self._token}, method='GET')
            try:
                with urlopen(req, timeout=60) as response:
                    return parse(json.load(response)['candles'], gran)
            except HTTPError as exc:
                status = exc.code
                if status not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise RuntimeError(f'Candle HTTP failure {status}') from None
            except (URLError, TimeoutError):
                if attempt == 4:
                    raise RuntimeError('Candle connection failed') from None
            time.sleep(2**attempt)
        raise RuntimeError('Candle retry limit')

    def daily(self, pair, end):
        cursor, rows = end, []
        while True:
            page = [c for c in self.page(pair, 'D', to=iso(cursor), count=5000)
                    if c.t < cursor and c.end <= end]
            if not page:
                break
            rows.extend(page)
            cursor = page[0].t
        return sorted({c.t: c for c in rows}.values(), key=lambda c: c.t)

    def first_m15(self, pair, a, end):
        rows = self.page(pair, 'M15', **{'from': iso(a), 'count': 1})
        if not rows or rows[0].end > end:
            raise ValueError('No M15 history')
        return rows[0].t

    def stream(self, pair, start, end):
        cursor = start
        previous = None
        while cursor < end:
            limit = min(cursor+timedelta(days=14), end)
            # Fourteen elapsed days contain <=1344 M15 slots (<OANDA's 5000 cap).
            rows = self.page(pair, 'M15', **{'from': iso(cursor), 'to': iso(limit)})
            for c in rows:
                if cursor <= c.t and c.end <= limit and (previous is None or c.t > previous):
                    yield c
                    previous = c.t
            cursor = limit


@dataclass(frozen=True)
class Daily:
    end: datetime
    close: float
    sma200: float | None
    sma5: float | None
    rsi: float | None
    atr: float | None

    @property
    def eligible(self):
        return all(x is not None for x in (self.sma200, self.sma5, self.rsi, self.atr))

    def signal(self):
        if not self.eligible:
            return 0
        if self.close > self.sma200 and self.rsi < 10:
            return 1
        if self.close < self.sma200 and self.rsi > 90:
            return -1
        return 0


def indicators(candles):
    closes, gains, losses, trs, result = [], [], [], [], []
    gain = loss = atr = None
    previous = None
    for c in candles:
        o, high, low, close = c.mid
        if previous is not None:
            change = close-previous
            g, l = max(change, 0), max(-change, 0)
            if gain is None:
                gains.append(g); losses.append(l)
                if len(gains) == 2:
                    gain, loss = statistics.fmean(gains), statistics.fmean(losses)
            else:
                gain, loss = (gain+g)/2, (loss+l)/2
            tr = max(high-low, abs(high-previous), abs(low-previous))
            if atr is None:
                trs.append(tr)
                if len(trs) == 20:
                    atr = statistics.fmean(trs)
            else:
                atr = (19*atr+tr)/20
        previous = close
        closes.append(close)
        rsi = (None if gain is None else 50. if gain == loss == 0 else
               100. if loss == 0 else 0. if gain == 0 else 100-100/(1+gain/loss))
        result.append(Daily(c.end, close,
                            statistics.fmean(closes[-200:]) if len(closes) >= 200 else None,
                            statistics.fmean(closes[-5:]) if len(closes) >= 5 else None,
                            rsi, atr))
    return result


@dataclass
class Position:
    pair: str
    signal: Daily
    entry_time: datetime
    direction: int
    baseline_entry: float
    entry: float
    risk: float
    stop: float
    entry_half: float
    variant: str
    held: int = 0
    scheduled: datetime | None = None
    trigger: str | None = None
    long_gap: bool = False

    @property
    def stress(self):
        return self.variant == 'stress'

    @property
    def side(self):
        return 'bid' if self.direction == 1 else 'ask'

    @property
    def id(self):
        return self.pair+'|'+iso(self.entry_time)+'|'+iso(self.signal.end)

    def costs(self, at):
        p = pip(self.pair)
        return (0.4 if self.stress else 0.2)*p, nights(self.entry_time, at)*(1.0 if self.stress else .5)*p

    def net_r(self, price, at):
        slip, swap = self.costs(at)
        return (self.direction*(price-self.entry)-slip-swap)/self.risk

    def daily(self, d):
        if d.end <= self.entry_time:
            return
        self.held += 1
        if self.scheduled is not None:
            return
        recovery = d.sma5 is not None and (d.close > d.sma5 if self.direction == 1 else d.close < d.sma5)
        if recovery or self.held >= 5:
            self.scheduled = next_london_open(d.end)
            self.trigger = 'recovery' if recovery else 'five_day'

    def stopped(self, price):
        return price <= self.stop if self.direction == 1 else price >= self.stop

    def record(self, c, at, price, reason, index, approximate=False):
        slip, swap = self.costs(at)
        delayed = self.scheduled is not None and at > self.scheduled and reason in ('recovery', 'five_day')
        # Frozen baseline quote conversion and units per one USD of nominal cash risk.
        q = 1/self.baseline_entry if self.pair.startswith('USD_') else 1.
        return dict(id=self.id, pair=self.pair, variant=self.variant,
                    entry=self.entry_time, exit=at, signal=self.signal.end,
                    direction=self.direction, entry_price=self.entry, stop=self.stop,
                    baseline_entry=self.baseline_entry, risk=self.risk, units_per_cash_R=1/(self.risk*q),
                    exit_price=price, r=self.net_r(price, at), reason='delayed' if delayed else reason,
                    trigger=self.trigger, scheduled=self.scheduled, approximate=approximate,
                    candle_start=c.t, held=self.held, long_gap=self.long_gap,
                    holding_hours=(at-self.entry_time).total_seconds()/3600,
                    spread_r=(self.entry_half+c.spread(index,self.stress)/2)/self.risk,
                    slippage_r=slip/self.risk, swap_r=swap/self.risk)


def open_positions(pair, d, c):
    direction = d.signal()
    risk = 2*d.atr
    if direction == 0 or not math.isfinite(risk) or risk <= 0:
        raise ValueError('Invalid entry risk or signal')
    baseline = c.ask[0] if direction == 1 else c.bid[0]
    stop = baseline-direction*risk
    return [Position(pair, d, c.t, direction, baseline,
                     c.quote('ask' if direction == 1 else 'bid',0, v=='stress'),
                     risk, stop, c.spread(0,v=='stress')/2, v) for v in VARIANTS]


def full_weekdays(start, end):
    day = start.astimezone(NY).date()
    while local_time(day,0,NY) < start:
        day += timedelta(days=1)
    result = set()
    while local_time(day+timedelta(days=1),0,NY) <= end:
        if day.weekday() < 5:
            result.add(day)
        day += timedelta(days=1)
    return result


def marks(start, split, end):
    result = {start, split, end}
    day = start.astimezone(NY).date()
    while local_time(day,17,NY) <= end:
        t = local_time(day,17,NY)
        if day.weekday() == 4 and start < t <= end:
            result.add(t)
        day += timedelta(days=1)
    return sorted(result)


def missing_slots(a, b):
    """Expected market-open M15 slots between one candle close and next open."""
    count = 0
    day = a.astimezone(NY).date()
    while local_time(day,0,NY) < b:
        wd = day.weekday()
        lo = local_time(day,17 if wd == 6 else 0,NY)
        hi = local_time(day,17,NY) if wd == 4 else local_time(day+timedelta(days=1),0,NY)
        if wd != 5:
            left, right = max(a,lo), min(b,hi)
            if right > left:
                count += math.ceil((right-left).total_seconds()/900)
        day += timedelta(days=1)
    return count


def candle_events(candles, start, end):
    it = iter(c for c in candles if c.t >= start and c.end <= end)
    current = next(it, None)
    while current is not None:
        following = next(it, None)
        if following is not None and following.t < current.end:
            raise ValueError('Duplicate, overlapping or unordered M15 candles')
        yield (current.t,2,'open',current)
        yield (current.end,0,'close',(current,following is None))
        current = following


def simulate_pair(pair, daily, candles, start, split, end):
    """Event stream retains one pair's daily indicators and two M15 candles only."""
    if not start < split < end:
        raise ValueError('Invalid test window')
    # A signal completed before the common start may still have its one
    # entry opportunity inside the window (e.g. M15 history starts overnight).
    daily = [d for d in daily if d.end <= end and
             (d.end >= start or next_london_open(d.end) >= start)]
    times = marks(start,split,end)
    opportunities = sorted({next_london_open(d.end) for d in daily if start <= next_london_open(d.end) < end})
    events = heapq.merge(candle_events(candles,start,end),
                        ((d.end,1,'daily',d) for d in daily),
                        ((t,3,'mark',None) for t in times),
                        ((t,4,'expiry',None) for t in opportunities))
    records = {v: [] for v in VARIANTS}
    snapshots = {v: {} for v in VARIANTS}
    active, pending, missed = {}, {}, []
    last_exit = None
    latest = None
    previous = None
    present, mornings = set(), set()
    gaps = gap_slots = long_gaps = 0
    for at, _, kind, item in events:
        if kind == 'daily':
            d = item
            for pos in active.values():
                pos.daily(d)
            when = next_london_open(d.end)
            # Only the latest daily signal can own a given morning.
            pending.pop(when,None)
            if ('baseline' not in active and (last_exit is None or d.end > last_exit)
                    and d.signal() and start <= when < end):
                if d.atr <= 0 or not math.isfinite(d.atr):
                    raise ValueError('Invalid ATR on eligible signal')
                pending[when] = d
        elif kind == 'open':
            c = item
            present.add(c.t.astimezone(NY).date())
            local = c.t.astimezone(LONDON)
            if walltime(6) <= local.time() < walltime(10):
                mornings.add(local.date())
            if previous is not None and c.t > previous.end:
                missing = missing_slots(previous.end,c.t)
                if missing:
                    gaps += 1; gap_slots += missing
                    if active and missing > 288:
                        long_gaps += 1
                        for pos in active.values():
                            pos.long_gap = True
            for v,pos in list(active.items()):
                price = c.quote(pos.side,0,pos.stress)
                reason = ('stop_gap' if pos.stopped(price) else
                          pos.trigger if pos.scheduled is not None and c.t >= pos.scheduled else None)
                if reason:
                    records[v].append(pos.record(c,c.t,price,reason,0))
                    del active[v]
                    if v == 'baseline': last_exit = c.t
            d = pending.pop(c.t,None)
            if d is not None and 'baseline' not in active and (last_exit is None or d.end > last_exit):
                if 'stress' in active:
                    raise ValueError('Stress overlap violates frozen trade set')
                for pos in open_positions(pair,d,c):
                    active[pos.variant] = pos
                # Entry-candle exit-side open can itself be beyond the stop.
                for v,pos in list(active.items()):
                    price = c.quote(pos.side,0,pos.stress)
                    if pos.stopped(price):
                        records[v].append(pos.record(c,c.t,price,'stop_gap',0))
                        del active[v]
                        if v == 'baseline': last_exit=c.t
        elif kind == 'close':
            c,islast = item
            for v,pos in list(active.items()):
                extreme = c.quote(pos.side,2 if pos.direction==1 else 1,pos.stress)
                reason = 'stop' if pos.stopped(extreme) else 'end' if islast else None
                if reason:
                    price = pos.stop if reason=='stop' else c.quote(pos.side,3,pos.stress)
                    records[v].append(pos.record(c,c.end,price,reason,0 if reason=='stop' else 3,reason=='stop'))
                    del active[v]
                    if v=='baseline': last_exit=c.end
            latest=previous=c
        elif kind == 'expiry':
            if pending.pop(at,None) is not None:
                day=at.astimezone(LONDON).date()
                if local_time(day,10,LONDON) <= end:
                    missed.append(at)
        elif kind == 'mark':
            for v in VARIANTS:
                pos=active.get(v)
                if pos is None:
                    snapshots[v][at] = dict(open_r=0.,count=0,usd=0,stale=0,max_age_hours=0.)
                else:
                    if latest is None:
                        raise ValueError('No completed quote for an open-position mark')
                    age=(at-latest.end).total_seconds()/3600
                    snapshots[v][at] = dict(open_r=pos.net_r(latest.quote(pos.side,3,pos.stress),at),
                                           count=1,usd=pos.direction*(1 if pair.startswith('USD_') else -1),
                                           stale=int(age>4),max_age_hours=age)
    if latest is None:
        raise ValueError('Empty M15 history')
    if active:
        raise ValueError('Unliquidated endpoint position')
    keys=lambda r:(r['id'],r['entry'],r['direction'],r['stop'],r['risk'])
    for v in VARIANTS:
        records[v].sort(key=lambda r:r['entry'])
    if list(map(keys,records['baseline'])) != list(map(keys,records['stress'])):
        raise ValueError('Stress trade set differs')
    denominator=full_weekdays(start,end)
    coverage=len(present&denominator)/len(denominator) if denominator else 0.
    skipped=Counter('data_gap' if t.astimezone(LONDON).date() in mornings else 'market_closed' for t in missed)
    skipped_periods={label:dict(Counter('data_gap' if t.astimezone(LONDON).date() in mornings else 'market_closed'
                     for t in missed if (t<split)==(label=='evaluation')))
                     for label in ('evaluation','second')}
    return dict(pair=pair,records=records,snapshots=snapshots,coverage=coverage,
                covered_dates=len(present&denominator),expected_dates=len(denominator),
                skipped=dict(skipped),skipped_periods=skipped_periods,intraday_gaps=gaps,missing_slots=gap_slots,long_trade_gaps=long_gaps)


def metrics(values):
    values=list(values)
    gp=sum(x for x in values if x>0)
    gl=-sum(x for x in values if x<0)
    equity=peak=drawdown=0.
    for x in values:
        equity+=x; peak=max(peak,equity); drawdown=max(drawdown,peak-equity)
    return dict(n=len(values),win_rate=sum(x>0 for x in values)/len(values) if values else 0.,
                avgR=statistics.fmean(values) if values else 0.,totalR=sum(values),
                PF=gp/gl if gl else 'infinity' if gp else None,max_drawdown_R=drawdown)


def trade_summary(rows):
    report=metrics(r['r'] for r in rows)
    exits=defaultdict(float)
    for r in rows: exits[r['exit']]+=r['r']
    report['max_drawdown_R']=metrics(exits[t] for t in sorted(exits))['max_drawdown_R']
    for key in ('holding_hours','spread_r','slippage_r','swap_r'):
        report['average_'+key]=statistics.fmean(r[key] for r in rows) if rows else 0.
    report['average_cost_R']=sum(report['average_'+k] for k in ('spread_r','slippage_r','swap_r'))
    report['exits']=dict(Counter(r['reason'] for r in rows))
    report['approximate_exits']=sum(r['approximate'] for r in rows)
    report['per_pair']={p:metrics(r['r'] for r in rows if r['pair']==p) for p in sorted({r['pair'] for r in rows})}
    return report


def portfolio(results,variant,start,split,end):
    times=marks(start,split,end)
    trades=sorted((r for pair in results for r in pair['records'][variant]),key=lambda r:r['exit'])
    intervals=[]
    prev_open=0.; index=0
    for k,at in enumerate(times):
        states=[p['snapshots'][variant][at] for p in results]
        opening=sum(s['open_r'] for s in states)
        realized=0.
        while index<len(trades) and trades[index]['exit']<=at:
            realized+=trades[index]['r'];index+=1
        if k:
            intervals.append(dict(start=times[k-1],end=at,
                                  period='evaluation' if at<=split else 'second',
                                  r=realized+opening-prev_open,open_r=opening,
                                  positions=sum(s['count'] for s in states),usd_risk_count=sum(s['usd'] for s in states),
                                  stale_marks=sum(s['stale'] for s in states),
                                  max_quote_age_hours=max((s['max_age_hours'] for s in states),default=0.)))
        elif realized:
            raise ValueError('Closed trade at initial portfolio boundary')
        prev_open=opening
    total=sum(t['r'] for t in trades)
    if not math.isclose(sum(x['r'] for x in intervals),total,rel_tol=1e-9,abs_tol=1e-8):
        raise ValueError('Portfolio R does not reconcile to trade R')
    return intervals


def verdict(rows,split,incomplete):
    if incomplete: return 'INCOMPLETE'
    early=metrics(r['r'] for r in rows if r['entry']<split)
    late=metrics(r['r'] for r in rows if r['entry']>=split)
    pooled=metrics(r['r'] for r in rows)
    pf=pooled['PF']
    return 'PASS' if (early['avgR']>0 and late['avgR']>0 and pooled['n']>=300 and
                      (pf=='infinity' or isinstance(pf,(int,float)) and pf>=1.2)) else 'FAIL'


def emit(label,data):
    def encode(value):
        if isinstance(value,datetime): return iso(value)
        raise TypeError('Unsupported report type')
    print(label+' '+json.dumps(data,default=encode,allow_nan=False,sort_keys=True),flush=True)


def report(results,pairs,start,split,end,incomplete):
    verdicts={}
    for variant in VARIANTS:
        rows=[r for p in results for r in p['records'][variant]]
        weeks=portfolio(results,variant,start,split,end)
        for period in ('evaluation','second','pooled'):
            chosen=[r for r in rows if period=='pooled' or (r['entry']<split)==(period=='evaluation')]
            intervals=[x for x in weeks if period=='pooled' or x['period']==period]
            summary=trade_summary(chosen)
            summary['skipped_entries_by_pair']={p['pair']:p['skipped'] if period=='pooled'
                                                else p['skipped_periods'][period] for p in results}
            weekly=metrics(x['r'] for x in intervals)
            weekly['intervals']=weekly.pop('n')
            weekly.update(average_positions=statistics.fmean(x['positions'] for x in intervals) if intervals else 0.,
                          max_positions=max((x['positions'] for x in intervals),default=0),
                          usd_risk_count_min=min((x['usd_risk_count'] for x in intervals),default=0),
                          usd_risk_count_max=max((x['usd_risk_count'] for x in intervals),default=0),
                          stale_position_marks=sum(x['stale_marks'] for x in intervals),
                          max_quote_age_hours=max((x['max_quote_age_hours'] for x in intervals),default=0.))
            summary['weekly_portfolio']=weekly
            summary['evaluation_trades_closing_after_split']=sum(r['entry']<split<r['exit'] for r in chosen)
            emit(f'REPORT {variant} {period}',summary)
        # Weekly results are chunked into one summary line per calendar year,
        # rather than a log per trade or a burst of thousands of weekly lines.
        years=defaultdict(list)
        for x in weeks:
            years[x['end'].year].append({k:v for k,v in x.items() if k!='open_r'})
        for year,items in sorted(years.items()): emit(f'WEEKLY {variant} {year}',items)
        verdicts[variant]=verdict(rows,split,incomplete)
        emit('VERDICT',dict(variant=variant,status=verdicts[variant]))
    overall='INCOMPLETE' if incomplete else 'PASS' if all(v=='PASS' for v in verdicts.values()) else 'FAIL'
    emit('OVERALL',dict(status=overall,incomplete=incomplete,pairs=pairs,
                        qualification='Separate untouched forward demo only; no execution enabled'))
    return overall


def execute(api,pairs,end):
    starts=[]
    errors=[]
    for pair in pairs:
        try:
            candles=api.daily(pair,end)
            days=indicators(candles)
            eligible=next((d for d in days if d.eligible),None)
            if eligible is None: raise ValueError('Insufficient daily warm-up')
            start=max(eligible.end,api.first_m15(pair,candles[0].t,end))
            if start>=end: raise ValueError('No usable window')
            starts.append(start)
            del candles,days
            gc.collect()
        except Exception as exc:
            # Never log exception bodies from APIs or credentials.
            errors.append(pair+': history discovery failed ('+type(exc).__name__+')')
    if errors:
        for v in VARIANTS: emit('VERDICT',dict(variant=v,status='INCOMPLETE'))
        emit('OVERALL',dict(status='INCOMPLETE',incomplete=errors))
        return 'INCOMPLETE'
    start=max(starts)
    split=start+(end-start)*2/3
    emit('WINDOW',dict(start=start,split=split,end=end,pairs=pairs,
                       periods='evaluation and second: both previously inspected',
                       R='price-based; baseline entry conversion frozen',subset=set(pairs)!=set(PAIRS)))
    results=[]
    for pair in pairs:
        try:
            days=indicators(api.daily(pair,end))
            result=simulate_pair(pair,days,api.stream(pair,start,end),start,split,end)
            results.append(result)
            if result['coverage']<.95:
                errors.append(pair+': weekday presence below 95%')
            emit('DATA', {k:v for k,v in result.items() if k not in ('records','snapshots')})
            del days
            gc.collect()
        except Exception as exc:
            errors.append(pair+': processing failed ('+type(exc).__name__+')')
    return report(results,pairs,start,split,end,errors)


def main():
    try:
        end=ts(os.environ['PB_END'])
        if end!=FROZEN_END:
            raise ValueError('PB_END must equal the frozen endpoint 2026-10-09T21:00:00Z')
        pairs=[p.strip().upper() for p in os.getenv('PB_PAIRS',','.join(PAIRS)).split(',') if p.strip()]
        if not pairs or len(pairs)!=len(set(pairs)) or any(p not in PAIRS for p in pairs):
            raise ValueError('PB_PAIRS must contain unique supported pairs')
        emit('CONFIG',dict(end=end,pairs=pairs,mode='historical candles only',
                           caveat='Fixed financing proxy; price-based R; inspected history; correlated trades and weeks'))
        return execute(CandleClient(),pairs,end)
    except Exception as exc:
        for v in VARIANTS: emit('VERDICT',dict(variant=v,status='INCOMPLETE'))
        emit('OVERALL',dict(status='INCOMPLETE',reason='Configuration or processing error: '+type(exc).__name__))
        return 'INCOMPLETE'


if __name__=='__main__':
    main()
