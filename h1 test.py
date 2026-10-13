"""Read-only H1 PO3+Echo research test. No account endpoints or trading methods.

Requires numpy and pandas. Dates are REQUIRED: use the H4 comparison's exact
calendar window. The strategy is frozen; this script does not optimize it.
"""
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = ('EUR_USD','GBP_USD','USD_JPY','AUD_USD','USD_CAD','NZD_USD',
         'USD_CHF','EUR_JPY','GBP_JPY','EUR_GBP','AUD_JPY','EUR_AUD')
ACC_LEN, ACC_MULT, MANIP_MULT = 40, 5.0, 0.6
SL_ATR, RR = 5.0, 0.86
ECHO_LEN = ECHO_FCAST = 50
SLIPPAGE = 0.1  # estimated pips per side, same baseline as uploaded bot.py
SWAP = 0.5     # estimated pips per calendar rollover, NOT broker financing
COLS = ['o','h','l','c','bo','bh','bl','bc','ao','ah','al','ac']


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def download(pair, start, end, gran):
    if pair not in PAIRS or gran not in ('H1', 'H4'):
        raise ValueError('Unapproved instrument or timeframe')
    token = next((os.environ[k].strip() for k in
                  ('OANDA_API_KEY','OANDA_API_TOKEN','OANDA_TOKEN','OANDA_ACCESS_TOKEN')
                  if os.environ.get(k, '').strip()), None)
    if not token:
        raise RuntimeError('No OANDA token available; use --data-dir with cached candles')
    opener = urllib.request.build_opener(NoRedirect())
    cur, rows = start, []
    while cur < end:
        query = urllib.parse.urlencode({'granularity':gran, 'price':'BAM',
                                        'from':cur.isoformat(), 'count':5000})
        url = f'https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles?{query}'
        req = urllib.request.Request(url, headers={'Authorization':f'Bearer {token}'}, method='GET')
        payload = None
        for attempt in range(4):
            try:
                with opener.open(req, timeout=30) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise RuntimeError(f'Candle GET failed: HTTP {exc.code}') from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise RuntimeError('Candle GET failed: network error') from None
            time.sleep(2 ** attempt)
        raw = payload.get('candles', [])
        if not raw:
            break
        for candle in raw:
            ts = pd.Timestamp(candle['time'])
            if not candle.get('complete') or ts >= end:
                continue
            row = {'time':ts.isoformat()}
            for source, prefix in [('mid',''), ('bid','b'), ('ask','a')]:
                row.update({prefix+k:float(candle[source][k]) for k in 'ohlc'})
            rows.append(row)
        # OANDA returns the candle CONTAINING 'from', so start the next page one full
        # candle after the last one received (H1: 1 hour, H4: 4 hours), not 1 second.
        nxt = pd.Timestamp(raw[-1]['time']) + pd.Timedelta(hours=1 if gran == 'H1' else 4)
        if nxt <= cur:
            raise RuntimeError('Candle pagination failed to advance')
        cur = nxt
        if len(raw) < 5000:
            break
        time.sleep(0.25)
    return pd.DataFrame(rows)


def validate(d, start, end, gran):
    d = d.copy()
    if any(c not in d for c in ['time'] + COLS):
        raise ValueError('Missing time or bid/ask/mid OHLC columns')
    d['time'] = pd.to_datetime(d['time'], utc=True)
    if d['time'].duplicated().any() or not d['time'].is_monotonic_increasing:
        raise ValueError('Candles must be unique and chronological')
    step = pd.Timedelta(hours=1 if gran == 'H1' else 4)
    d = d[d.time + step <= end].reset_index(drop=True)
    if not np.isfinite(d[COLS].to_numpy(dtype=float)).all() or (d[COLS] <= 0).any().any():
        raise ValueError('Non-finite or non-positive OHLC data')
    for prefix in ('','b','a'):
        if ((d[prefix+'h'] < d[[prefix+'o',prefix+'c',prefix+'l']].max(axis=1)) |
            (d[prefix+'l'] > d[[prefix+'o',prefix+'c',prefix+'h']].min(axis=1))).any():
            raise ValueError('Invalid OHLC ordering')
    for k in 'ohlc':
        if ((d['b'+k] > d[k]) | (d[k] > d['a'+k])).any():
            raise ValueError('Bid/mid/ask ordering invalid')
    if int((d.time < start).sum()) < 1000:
        raise ValueError('Need at least 1000 warm-up candles before start')
    if not (d.time >= start).any() or d.time.iloc[-1] + step < end - pd.Timedelta(days=4):
        raise ValueError('Missing evaluation data or stale dataset end')
    if ((d.time.diff().dropna() < step)).any():
        raise ValueError('Candle spacing does not match requested timeframe')
    return d


def exit_trade(direction, j, entry, risk, bo, bh, bl, bc, ao, ah, al, ac):
    sl, tp = entry-direction*risk, entry+direction*RR*risk
    for k in range(j, len(bo)):
        if direction == 1:
            if bl[k] <= sl:
                return k, min(sl, bo[k]), 'stop'
            if bh[k] >= tp:
                return k, tp, 'target'
        else:
            if ah[k] >= sl:
                return k, max(sl, ao[k]), 'stop'
            if al[k] <= tp:
                return k, tp, 'target'
    k = len(bo)-1
    return k, bc[k] if direction == 1 else ac[k], 'open_at_end'


def rollovers(entry, exit_time):
    # Approximation only; actual broker financing can differ by direction,
    # rate, holidays and multi-day charges. Includes weekend calendar days.
    def day(ts):
        local = ts.tz_convert('America/New_York')
        return local.date().toordinal() - int(local.hour < 17)
    return max(0, day(exit_time)-day(entry))


def simulate(d, pair, start, end, gran='H1', cost_mult=1.0):
    h,l,c = (d[k].to_numpy() for k in ('h','l','c'))
    # Stress both spread and additional execution/financing estimates.
    quote = {prefix+k:d[k].to_numpy() + cost_mult*(d[prefix+k].to_numpy()-d[k].to_numpy())
             for prefix in ('b','a') for k in 'ohlc'}
    bo,bh,bl,bc,ao,ah,al,ac = (quote[k] for k in ('bo','bh','bl','bc','ao','ah','al','ac'))
    atr = calc_atr(d,50).to_numpy()
    hh,ll = d.h.rolling(ACC_LEN).max().to_numpy(), d.l.rolling(ACC_LEN).min().to_numpy()
    times = d.time.tolist()
    step = pd.Timedelta(hours=1 if gran == 'H1' else 4)
    state,i,acc_end,top,bot = 0,0,0,None,None
    records=[]
    while i < len(d)-1:
        a=atr[i]
        if np.isnan(hh[i]) or not np.isfinite(a) or a <= 0:
            i+=1; continue
        if state==0 and hh[i]-ll[i] <= ACC_MULT*a:
            state,acc_end,top,bot=1,i,hh[i]+.5*a,ll[i]-.5*a
        if state==1:
            if h[i]>top or l[i]<bot: state=2
            else: acc_end=i
        direction=0
        if state==2 and i>acc_end:
            if h[i]>top+MANIP_MULT*a: direction=-1
            elif l[i]<bot-MANIP_MULT*a: direction=1
        if direction==0:
            i+=1; continue
        state=0
        if echo_direction(c,i)!=direction:
            i+=1; continue
        j=i+1
        entry=ao[j] if direction==1 else bo[j]
        risk=SL_ATR*a
        k,price,reason=exit_trade(direction,j,entry,risk,bo,bh,bl,bc,ao,ah,al,ac)
        if start <= times[j] < end:
            # Intrabar exit time is unknown: use bar close as a stated convention.
            exit_time=times[k]+step
            p=.01 if 'JPY' in pair else .0001
            nights=rollovers(times[j],exit_time)
            gross=direction*(price-entry)/risk
            extra=cost_mult*(2*SLIPPAGE+nights*SWAP)*p/risk
            records.append(dict(pair=pair,entry=times[j].isoformat(),exit=exit_time.isoformat(),
                direction=direction,entry_price=entry,exit_price=price,risk_pips=risk/p,
                gross_r=gross,net_r=gross-extra,estimated_rollovers=nights,reason=reason,
                entry_gap_hours=(times[j]-times[i]-step).total_seconds()/3600))
        if reason=='open_at_end': break
        i=k+1
    return records


def metrics(records):
    closed=sorted((t for t in records if t['reason']!='open_at_end'), key=lambda t:t['exit'])
    r=np.array([t['net_r'] for t in closed],dtype=float)
    losses=-r[r<0].sum(); curve=np.r_[0.,r.cumsum()]
    return dict(closed_trades=len(r),open_at_end=sum(t['reason']=='open_at_end' for t in records),
        win_pct=float(100*(r>0).mean()) if len(r) else None,
        avg_r=float(r.mean()) if len(r) else None,total_r=float(r.sum()),
        profit_factor=float(r[r>0].sum()/losses) if losses>0 else None,
        realized_max_dd_r=float((np.maximum.accumulate(curve)-curve).max()),
        open_mark_to_market_r=sum(t['net_r'] for t in records if t['reason']=='open_at_end'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',required=True,help='Inclusive date, same window as H4 reference')
    parser.add_argument('--end',required=True,help='Exclusive date, no future data')
    parser.add_argument('--gran',choices=['H1','H4'],default='H1')
    parser.add_argument('--data-dir',type=Path,help='Read local PAIR_GRAN.csv files; no network')
    parser.add_argument('--out',type=Path,default=Path('h1_results'))
    args=parser.parse_args()
    start,end=pd.Timestamp(args.start,tz='UTC'),pd.Timestamp(args.end,tz='UTC')
    if start>=end or end>pd.Timestamp.now(tz='UTC'):
        raise ValueError('Require start < end <= now')
    args.out.mkdir(parents=True,exist_ok=False)
    manifest={'timeframe':args.gran,'start':start.isoformat(),'end':end.isoformat(),
              'pairs':list(PAIRS),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'data':{},'status':'INCOMPLETE','strategy':{'acc_len':40,'acc_mult':5,'manip_mult':.6,
              'atr':50,'sl_atr':5,'rr':.86,'echo_len':50,'echo_forecast':50},
              'note':'Research only. No automatic pass declaration. Halves are descriptive, not untouched holdouts.'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    all_records={1.0:[],2.0:[]}
    for pair in PAIRS:
        filename=f'{pair}_{args.gran}.csv'
        if args.data_dir:
            d=pd.read_csv(args.data_dir/filename)
        else:
            warmup=start-pd.Timedelta(days=90 if args.gran=='H1' else 365)
            d=download(pair,warmup,end,args.gran)
        d=validate(d,start,end,args.gran)
        cache=args.out/filename; d.to_csv(cache,index=False)
        gaps=d.time.diff().dt.total_seconds()/3600
        manifest['data'][pair]={'sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),
            'bars':len(d),'first':d.time.iloc[0].isoformat(),'last':d.time.iloc[-1].isoformat(),
            'gaps_over_one_bar':int((gaps>(1 if args.gran=='H1' else 4)).sum())}
        for cost in all_records:
            all_records[cost].extend(simulate(d,pair,start,end,args.gran,cost))
        print(f'{pair}: {len(d)} candles evaluated',flush=True)
    split=start+(end-start)/2
    summary={}
    for cost,records in all_records.items():
        label='baseline' if cost==1 else 'double_costs'
        pd.DataFrame(records).to_csv(args.out/f'{label}_trades.csv',index=False)
        summary[label]={'all':metrics(records),
            'first_half':metrics([t for t in records if pd.Timestamp(t['entry'])<split]),
            'second_half':metrics([t for t in records if pd.Timestamp(t['entry'])>=split]),
            'by_pair':{pair:metrics([t for t in records if t['pair']==pair]) for pair in PAIRS}}
    manifest['status']='COMPLETED'
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2,allow_nan=False))
    print('No pass/fail asserted: compare with your unchanged frozen criteria and inspect data gaps.')


# The following two pure indicator functions are copied verbatim from h4_live.py.

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


if __name__ == "__main__":
    main()
