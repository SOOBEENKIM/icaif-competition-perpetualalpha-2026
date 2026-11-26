"""Complete initial cash and preserve actual millisecond settlement times.

The common seed was saved just after03UTC with funding through02UTC. Starting
a replay at03:05 must account for the03UTC payment; skipping it would retain
the wrong starting cash. This is an evaluator correction, not a new strategy.
"""
import copy,json
from pathlib import Path
from .cash_replay import run_cash as original_run_cash


def timed_units(units,records,team_wallet):
    groups={}
    for wallet,r in records:
        d=r['delta'];coin=d.get('coin','')
        if not coin.startswith('pas:') or not float(d.get('szi',0)):continue
        groups.setdefault((coin,r['time']//3600000),[]).append((wallet.lower(),r['time']))
    out=[]
    for u in units:
        observations=groups.get((u['coin'],u['hour']),[])
        if not observations:raise ValueError('Verified cash unit has no timestamp evidence')
        own={t for w,t in observations if w==team_wallet.lower()}
        times=[t for _,t in observations]
        if len(own)>1:raise ValueError('Multiple team settlement timestamps in one market/hour')
        chosen=next(iter(own)) if own else max(times)
        out.append(dict(u,settlement_time_ms=chosen,timing_bounds_ms=[min(times),max(times)],
                        team_timestamp_observed=bool(own)))
    return out


def synchronize_initial(seed,units,start,expected_settlements=()):
    p=copy.deepcopy(seed);a=p.account
    latest={}
    for event in a.physical.funding_events:
        latest[event['coin']]=max(latest.get(event['coin'],-1),event['time'])
    additions=[];missing=[]
    available={(u['coin'],u['hour']):u for u in units}
    for f in expected_settlements:
        coin=f['coin'];key=(coin,f['time']//3600000);unit=available.get(key)
        at=(unit['settlement_time_ms'] if unit else f['time'])/1000
        if not latest.get(coin,-1)<at<=start:continue
        j=a.physical.symbols.index(coin[4:])
        qty=sum(r['signed_qty'] for r in a.physical.fills if r['asset']==j and r['time']<at)
        if qty and float(f['fundingRate']) and unit is None:
            missing.append(dict(kind='missing_initial_funding_cash',coin=coin,time=at,qty=qty))
    p.research_initial_cash_missing=missing
    checkpoint=a.checkpoint;a.checkpoint=lambda *args:None
    try:
        for u in sorted(units,key=lambda x:x['settlement_time_ms']):
            at=u['settlement_time_ms']/1000;coin=u['coin']
            if not latest.get(coin,-1)<at<=start:continue
            j=a.physical.symbols.index(coin[4:])
            qty=sum(f['signed_qty'] for f in a.physical.fills if f['asset']==j and f['time']<at)
            if not qty:continue
            cash=qty*u['cash_per_unit']
            a.apply_actual_funding([dict(time=u['settlement_time_ms'],delta=dict(coin=coin,szi=str(qty),usdc=str(cash)),synthetic=True)])
            additions.append(dict(coin=coin,time=at,qty=qty,cash=cash,maximum_rounding_error=abs(qty)*u['maximum_unit_error']))
            latest[coin]=at
    finally:a.checkpoint=checkpoint
    return p,additions


def run_exact(streams,source,profile,scenario,start,end,seed,arm,funding,*,cash_units,records,initial_settlements=()):
    units=timed_units(cash_units,records,profile['team_wallet'])
    p,initial=synchronize_initial(seed,units,start,[*initial_settlements,*funding])
    lookup={(u['coin'],u['hour']):u for u in units}
    settled=[]
    for f in funding:
        u=lookup.get((f['coin'],f['time']//3600000))
        settled.append(dict(f,time=u['settlement_time_ms'] if u is not None else f['time']))
    result=original_run_cash(streams,source,profile,scenario,start,end,p,arm,settled,cash_units=units)
    result['funding_mode']='verified_cash_actual_settlement_time_complete_initial_cash'
    result['initial_cash_synchronization']=initial
    result['failures'].extend(p.research_initial_cash_missing)
    for u in units:
        if u['team_timestamp_observed']:continue
        lo,hi=u['timing_bounds_ms']
        if lo==hi:continue
        for o in result['orders']:
            if o['filled'] and o['symbol']==u['coin'][4:] and lo<=o['arrival']*1000<=hi:
                result['failures'].append(dict(kind='ambiguous_counterfactual_funding_inventory',coin=u['coin'],hour=u['hour']))
    return result


def read_records(directory):
    result=[]
    for f in sorted(Path(directory).glob('*.json')):
        raw=json.loads(f.read_text());rows=raw.get('data',raw.get('rows',raw.get('response'))) if isinstance(raw,dict) else raw
        if rows is None:raise ValueError('Unknown public funding archive format')
        result.extend((f.stem,r) for r in rows)
    return result
