"""Finite rule development with shared execution and explicit reused data."""
import copy
import gzip
import json
from itertools import combinations
from pathlib import Path
import numpy as np

from .engine import CandidateSimulation
from ..pairs_rl.engine import PairSimulation
from ..pairs_rl.public_data import load_market
from ..pairs_rl.readiness import completed_trades
from ..pairs_rl.verify import independent_accounting
from ..common_comparison.experiment import ROOT,write,digest,stamp,json_value,audit_run
from ..common_comparison.models import timestamp

DEST=ROOT/'replay_lab/output/pairs_development/20261013-v3'
PREVIOUS=ROOT/'replay_lab/output/pairs_rl/20261013-filter-repair-v2'
GROUPS=((0,1,2),(3,4,5),(6,7),(8,9))


def configuration(name,universe='groups',entry=3,mode='cross',fixed=False,carry=False,hold=168,beta='levels'):
    p=json.loads((PREVIOUS/'paper_rank_variogram-protocol.json').read_text())
    p['id']=name
    p['strategy'].update(entry_z=entry,max_hold_hours=hold,stop_decay_start_hours=hold/2)
    p['development']=dict(universe=universe,entry_mode=mode,fixed_mean=fixed,funding_hurdle=carry,
        cooldown_hours=24 if mode=='band' else 0,beta_mode=beta,
        selection='All economically grouped candidates evaluated each hour; no claim of confirmed cointegration. Disjoint live assets, net-cost opportunity priority, existing risk ceiling.',
        carry='When enabled: last 24 settled hours, ignore credits, reserve maximum holding-period debit plus scenario surcharge.')
    return p


def configurations():
    return [configuration('all45_cross3',universe='all'),configuration('groups8_cross3'),
        configuration('groups8_cross2',entry=2),configuration('groups8_band2',entry=2,mode='band'),
        configuration('groups8_band2_fixed',entry=2,mode='band',fixed=True),
        configuration('groups8_band2_fixed_carry',entry=2,mode='band',fixed=True,carry=True),
        configuration('groups8_band1.5_fixed_carry',entry=1.5,mode='band',fixed=True,carry=True),
        configuration('groups8_band2.5_fixed_carry',entry=2.5,mode='band',fixed=True,carry=True),
        configuration('groups8_band2_fixed_carry_hold72',entry=2,mode='band',fixed=True,carry=True,hold=72)]


def formations(market,start,universe='groups',only_pair=None):
    pairs=list(combinations(range(10),2)) if universe=='all' else [p for g in GROUPS for p in combinations(g,2)]
    rows=[dict(pair='/'.join(market.symbols[j] for j in pair),assets=list(pair)) for pair in pairs]
    if only_pair is not None:
        rows=[r for r in rows if r['pair']==only_pair]
    return [dict(time=int(market.times[start]),close_index=start,rows=rows,selected=[r['pair'] for r in rows])]


def run_rule(market,forms,p,start,end,scenario='base',features=None,sigmas=None,old=False):
    cls=PairSimulation if old else CandidateSimulation
    sim=cls(market,forms,p,start,end,scenario,features=features,sigmas=sigmas)
    while sim.i<sim.last:
        sim.advance(sim.rule_actions())
    return sim.result(p['id'])


def record(run,forms,phase,market):
    filename=f"{phase}-{run['policy']}-{run['scenario']}.json.gz"
    if (DEST/filename).exists():
        raise RuntimeError('Refusing to replace completed rule path '+filename)
    verification=dict(shared=audit_run(run),independent=independent_accounting(run,market))
    trades=completed_trades(run,forms)
    q=np.array([s['equity'] for s in run['snapshots']]);cuts=np.linspace(0,len(q)-1,4,dtype=int)
    thirds=[float((q[b]-q[a])/1e6) for a,b in zip(cuts,cuts[1:])]
    with gzip.open(DEST/filename,'wt') as f:
        json.dump(run,f,default=json_value,allow_nan=False)
    row=dict(policy=run['policy'],phase=phase,scenario=run['scenario'],file=filename,
        start=run['start'],end=run['end'],metrics=run['metrics'],gates=run['gates'],thirds=thirds,
        counts=dict(intents=trades['entry_intents'],completed=len(trades['completed']),
                    rejected=len(trades['rejected']),open=len(trades['open_entries'])),verification=verification)
    write(DEST/(filename+'.summary.json'),row)
    print(json.dumps({k:row[k] for k in ('policy','phase','scenario','counts','thirds')}|
                     dict(roi=run['metrics']['roi'],daily_vol=run['metrics']['daily_vol'])),flush=True)
    return row


def pick(rows):
    eligible=[r for r in rows if all(r['gates'].values()) and r['metrics']['roi']>0 and sum(x>0 for x in r['thirds'])>=2]
    passing=[r for r in rows if all(r['gates'].values())]
    if not passing:
        raise RuntimeError('No integrity-passing rule path; investigate before selection')
    chosen=max(eligible or passing,key=lambda r:(r['metrics']['roi'],-r['metrics']['daily_vol']))
    return dict(policy=chosen['policy'],has_positive_training_evidence=bool(eligible),
                criterion='Positive ROI and >=2 of 3 chronological training contributions positive; max ROI among passing, daily volatility breaks tie.',row=chosen)


def stage1():
    if (DEST/'stage1_manifest.json').exists():
        raise RuntimeError('Stage 1 already started')
    market,original,data=load_market()
    start=1439;end=market.exact(timestamp('2026-07-29T00:00:00Z'))
    configs=configurations()
    write(DEST/'stage1_manifest.json',dict(time=stamp(),data=data,configs=configs,
        plan_sha256=digest(DEST/'plan.json'),sources={str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).parent.glob('*.py')}))
    sigmas=PairSimulation.make_sigmas(market,configs[0]);rows=[]
    oldp=json.loads((PREVIOUS/'paper_rank_variogram-protocol.json').read_text());oldp['id']='v2_control'
    oldf=json.loads((PREVIOUS/'paper_rank_variogram-training-screen.json').read_text())
    control=run_rule(market,oldf,oldp,start,end,sigmas=sigmas,old=True)
    with gzip.open(PREVIOUS/'paper_rank_variogram-train-base-rules.json.gz','rt') as f:old=json.load(f)
    assert control['metrics']==old['metrics'],'Original result must reproduce'
    rows.append(record(control,oldf,'train',market))
    for p in configs:
        fs=formations(market,start,p['development']['universe'])
        write(DEST/(p['id']+'-protocol.json'),p)
        r=run_rule(market,fs,p,start,end,sigmas=sigmas)
        rows.append(record(r,fs,'train',market))
        write(DEST/'stage1_progress.json',rows)
    choice=pick([r for r in rows if r['policy'] not in ('v2_control','all45_cross3')])
    write(DEST/'stage1_selection.json',choice)
    write(DEST/'stage1_results.json',rows)
    print('STAGE1_PARENT',choice['policy'],choice['has_positive_training_evidence'],flush=True)


if __name__=='__main__':
    stage1()
