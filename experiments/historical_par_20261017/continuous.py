"""Ongoing-policy comparison; participation sensitivity is not model selection.

The original finite-horizon replay is retained, including its losing results.
This is a separate declared diagnostic: arbitrary sample boundaries should not
suppress the final 96 hours of entries for a continuously operated strategy.
"""
import copy
import difflib
import gzip
import json
import time
import numpy as np
from strategy.agent import TAIL
from experiments.complementary_alpha_20261017.research import protected
from experiments.performance_attribution_20261016.engine import simulate
from experiments.cost_horizon_20261016.research import SecondController
from replay_lab.ema_rebuild.engine import Controller
from replay_lab.common_comparison.models import Market
from .data import HERE,OUT,ROOT,save,sha,ts
from .continuous_engine import run


def main():
    dest=OUT/'continuous';dest.mkdir(exist_ok=True)
    if (dest/'frozen.json').exists():raise FileExistsError('Continuous evaluation already run')
    cfg=json.loads((HERE/'protocol.json').read_text());p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    cfg['terminal_reduce_hours']=None
    cfg['scenarios']={
        'participation_1pct':dict(fee_bps=9,impact_bps=10,participation=.01,adverse_funding_bps_hour=0.,funding_receipt_multiplier=1.),
        'participation_10pct':dict(fee_bps=9,impact_bps=10,participation=.1,adverse_funding_bps_hour=0.,funding_receipt_multiplier=1.),
        'stress_10pct':dict(fee_bps=12,impact_bps=20,participation=.1,adverse_funding_bps_hour=0.,funding_receipt_multiplier=.5)}
    save(dest/'protocol.json',dict(cfg=cfg,history_reuse=True,
        reason='Continuous mark-to-market valuation; retain full holding horizon even when it extends past sample end. No entry rule or model fit altered. Participation rates are sensitivity assumptions, not verified IOC capacity.',
        old_protocol_selection=json.loads((OUT/'selection.json').read_text()),
        note='Every sensitivity is disclosed. Do not pick whichever assumed capacity produces the best return.'))
    diff=''.join(difflib.unified_diff((HERE/'engine.py').read_text().splitlines(True),
                                      (HERE/'continuous_engine.py').read_text().splitlines(True)))
    (dest/'engine.diff').write_text(diff)
    sources={str(q.relative_to(ROOT)):sha(q) for q in [HERE/'continuous.py',HERE/'continuous_engine.py',dest/'protocol.json',OUT/'inputs.npz',OUT/'signals.npz']}
    save(dest/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    with np.load(OUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(OUT/'signals.npz') as z:signals={k:z[k].copy() for k in z.files}
    rows=[]; (dest/'runs').mkdir(exist_ok=True)
    for phase,start,end in [('full',cfg['start_utc'],cfg['end_utc']),('selection',cfg['start_utc'],cfg['split_utc']),('later',cfg['split_utc'],cfg['end_utc'])]:
        for arm in cfg['variants']:
            for scenario in cfg['scenarios']:
                r=run(d,signals,cfg,p,arm,scenario,ts(start),ts(end));r['phase']=phase
                key=f'{phase}-{arm}-{scenario}'
                with gzip.open(dest/'runs'/f'{key}.json.gz','wt') as f:
                    json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
                rows.append(dict(phase=phase,arm=arm,scenario=scenario,metrics=r['metrics']))
                m=r['metrics'];print(key,'PnL',round(m['net_pnl'],4),'Q',round(m['financial_score'],8),
                    'fills',m['fills'],'pairs',m['completed_pairs'],'aborts',m['aborted_pairs'],flush=True)
    # Same dates, incumbent signals, and reference-market liquidity benchmark.
    # This isolates a liquid reference simulation from actual par volume limits.
    with np.load(ROOT/cfg['reference_run']/'inputs.npz') as z:ref={k:z[k].copy() for k in z.files}
    ix=np.searchsorted(ref['market_times'],d['times'])
    m=Market(d['times'],ref['close'][ix],ref['volume'][ix],ref['trades'][ix],ref['funding'][ix],ref['observed'][ix],tuple(p['symbols']))
    reference=[]
    for arm in ('first_strategy','second_strategy'):
        for scenario in ('base','stress'):
            control=Controller() if arm=='first_strategy' else SecondController(d['times'],d['retention'])
            r=simulate(m,d['first'] if arm=='first_strategy' else d['second'],d['sigma'],p,
                       m.exact(ts(cfg['start_utc'])),m.exact(ts(cfg['end_utc'])),scenario,control)
            reference.append(dict(arm=arm,scenario=scenario,metrics=r['metrics'],reference_market_only=True))
            print('reference',arm,scenario,r['metrics']['net_pnl'],flush=True)
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    protected()
    save(dest/'results.json',dict(rows=rows,reference=reference,historical_independence=False,
                                 orders_sent=0,protected_sources_unchanged=True,evaluations=len(rows)+len(reference)))
    print('DONE',len(rows)+len(reference),'additional historical replays',flush=True)


if __name__=='__main__':main()
