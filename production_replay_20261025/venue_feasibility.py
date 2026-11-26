"""Post-end OI capacity audit of every recorded hypothetical order.

Counterfactual fills may add OI on top of the observed book. Retain that full
extra opening amount without assuming later closes reduce aggregate OI.
This is conservative capacity evidence, not an exchange acceptance receipt.
"""
import copy,gzip,json,time,sys
from decimal import Decimal
from types import SimpleNamespace
import numpy as np
from .protocol import ROOT,OUT,save,sha
from .venue_guard import opening_quantity,check_oi_strict

DEST=OUT/'venue_feasibility';END=1792915500


def assess(result,streams,profile,scenario,reserved):
    from .book_replay import BookVenue
    symbols=profile['trading']['symbols'];n=len(symbols)
    qty=np.array(result['snapshots'][0]['qty'],float)+np.asarray(reserved,float)
    extra=np.zeros(n);venue=BookVenue(streams,profile,scenario,0,END,[])
    meta=dict(universe=[dict(name='pas:'+s) for s in symbols])
    limits=dict(coinToOiCap=[['pas:'+s,str(profile['venue_caps']['per_market'])] for s in symbols],
                totalOiCap=str(profile['venue_caps']['total']))
    failures=[];largest=0.;checks=0
    for o in result['orders']:
        venue.now=o['time'];contexts=[venue.mark(s) for s in symbols]
        if any(c is None or 'openInterest' not in c for c in contexts):
            failures.append(dict(time=o['time'],symbol=o['symbol'],reason='OI context unavailable at decision'))
        else:
            adjusted=copy.deepcopy(contexts)
            for j,c in enumerate(adjusted):c['openInterest']=str(Decimal(c['openInterest'])+Decimal(str(extra[j])))
            positions=[dict(position=dict(coin='pas:'+s,szi=str(qty[j]))) for j,s in enumerate(symbols)]
            order=SimpleNamespace(coin='pas:'+o['symbol'],is_buy=o['side']>0,size=str(o['size']),reduce_only=False)
            try:
                r=check_oi_strict([order],positions,meta,adjusted,limits)
                largest=max(largest,r['current_oi']+r['increment_upper_bound'])
            except ValueError as e:failures.append(dict(time=o['time'],symbol=o['symbol'],reason=str(e)))
            checks+=1
        j=symbols.index(o['symbol']);delta=o['side']*o['filled']
        extra[j]+=float(opening_quantity(Decimal(str(qty[j])),Decimal(str(delta))))
        qty[j]+=delta
    return dict(complete=not failures,checks=checks,failures=failures,maximum_conservative_global_oi=largest,
                actual_orders_sent=0,teammate_inventory_assumption='same fixed reserve as comparative experiment',
                margin_or_exchange_acceptance_proven=False)


def freeze():
    assert time.time()<END and not (DEST/'protocol.json').exists()
    save(DEST/'protocol.json',dict(at=time.time(),rule='Every requested order must fit observed all-market OI plus conservative prior counterfactual opening increments and its own possible opening amount, including net reversal. Missing contexts are unresolved. No PnL tuning or relaxation of announced caps.',
        hashes={f'production_replay_20261025/{f}':sha(ROOT/'production_replay_20261025'/f) for f in ['venue_feasibility.py','venue_guard.py']},orders_sent=0))


def run():
    contract=json.loads((DEST/'protocol.json').read_text())
    for p,h in contract['hashes'].items():assert sha(ROOT/p)==h,p
    while time.time()<END+1650:
        save(DEST/'status.json',dict(at=time.time(),state='waiting_for_fixed_end_and_corrected_paths',orders_sent=0))
        time.sleep(min(30,max(1,END+1650-time.time())))
    while True:
        state=OUT/'exact_forward_review/status.json'
        if state.exists() and json.loads(state.read_text()).get('state')=='complete':break
        if time.time()>END+2400:raise ValueError('Corrected paper paths unavailable by audit deadline')
        time.sleep(10)
    from competition_live.profile import load_profile
    from competition_live.runtime import private_load
    from .book_replay import load_books
    profile=load_profile();streams,_=load_books((OUT/'capture').glob('*.jsonl.gz'),END)
    reserve=private_load(OUT/'forward_seed.pkl')['portfolio'].account.reserved.qty
    rows=[]
    for f in sorted((OUT/'exact_forward_review').glob('*/*.json.gz')):
        with gzip.open(f,'rt') as fp:r=json.load(fp)
        sc=dict(fee_bps=12. if 'stress' in f.stem else 9.,depth_fraction=.5 if 'stress' in f.stem else 1.,latency_ms=3000 if 'stress' in f.stem else 1000,receipt_multiplier=.5 if 'stress' in f.stem else 1.)
        rows.append(dict(file=str(f.relative_to(OUT)),**assess(r,streams,profile,sc,reserve)))
    save(DEST/'results.json',rows);save(DEST/'status.json',dict(at=time.time(),state='complete',orders_sent=0))


if __name__=='__main__':globals()[sys.argv[1]]()
