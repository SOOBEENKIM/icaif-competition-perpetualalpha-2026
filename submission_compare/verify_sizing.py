import copy
import hashlib
import json
import numpy as np
from competition_live.profile import ROOT,BUNDLE,load_profile,verify_deployment
from competition_live.scoring import evaluate
from experiments.decision_gate_pairs_20261017.portfolio import replay
from .run import OUT,OLD,save
from .sizing_candidate import transform,MULTIPLIER


def main():
    z=dict(np.load(OLD/'targets.npz',allow_pickle=False))
    scaled=transform(z['incumbent'])
    error=float(np.max(abs(scaled-z['size_only'])))
    assert error<1e-12
    p=load_profile();d0=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
    state=dict(np.load(BUNDLE/'models/reference_gate_state.npz',allow_pickle=False))
    ids=np.searchsorted(z['times'],d0['times'])
    old=json.loads((OLD/'reference_results.json').read_text())
    rows=[]
    for expected in [r for r in old if r['variant']=='size_only']:
        d=dict(d0);d['first']=scaled[ids]
        cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
        result=replay(d,state,None,cfg,p['trading'],p['candidate'],expected['scenario'],expected['start'],expected['end'])
        m=result['metrics']
        daily=[dict(time=s['time'],equity=s['equity']+50000) for s in result['snapshots'] if s['time']%86400==43200]
        sc=evaluate(daily,trading_days=len({f['time']//86400 for f in result['fills']}))
        assert abs(m['net_pnl']-expected['net_pnl'])<1e-7
        assert abs(sc['Q']-expected['Q'])<1e-12
        assert m['risk_breach_bars']==0 and m['maximum_accounting_error']<1e-6
        assert all(f['source_available']<=f['decision_time']<f['time'] for f in result['fills'])
        rows.append(dict(phase=expected['phase'],scenario=expected['scenario'],net_pnl=m['net_pnl'],Q=sc['Q'],
                        matches_previous_backtest=True))
        print(json.dumps(rows[-1]),flush=True)
    source=ROOT/'submission_compare/sizing_candidate.py'
    save(OUT/'sizing_submission_candidate.json',dict(
        name='original_size_with_current_pairs',multiplier=MULTIPLIER,
        adapter=str(source.relative_to(ROOT)),adapter_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        user_allocation=50000,reserve=50000,pair_policy=p['candidate'],
        risk_limits_unchanged=True,reference_target_max_error=error,reproduced_windows=rows,
        basis='Highest continuous historical net PnL and Q among the compared candidates. '
              'This is greater exposure, not demonstrated direction-forecast improvement.',
        caveat='Higher drawdown and weaker Sep-Oct stress PnL/Q than current. Reused reference history, not PAS actual fills.',
        integration='Apply transform to default base targets BEFORE building and hashing the published signal frame. '
                    'Keep current recovery/freshness fixes, pair settings and caps. No integration has been activated.',
        active=False,orders_sent=0,deployment_hashes=verify_deployment()['files']))


if __name__=='__main__':main()
