"""Longer reference-market exposure robustness, explicitly NOT PAS PnL."""
import copy,gzip,json
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from competition_live.profile import load_profile,BUNDLE
from pas_economic_allocation.replay import replay
from pas_allocation_study.study import summarize
from pas_performance_research.selected import controller
from .protocol import ROOT,OUT,save,sha

DEST=OUT/'reference_budget'

def job(multiplier,w):
    p=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));ix=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times'])
    source=json.loads((ROOT/'reports/universe_repair_20261022/protocol.json').read_text())['input']['inputs']
    raw=dict(np.load(ROOT/source))
    np.testing.assert_array_equal(f['times'][fi],d['times']);np.testing.assert_array_equal(z['times'][ix],d['times'])
    c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][ix]*multiplier,
                 z['all_market_signed_core_v1_protected'][ix],price_times=raw['market_times'],prices=raw['close'])
    cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    r=replay(d,states,None,cfg,p['trading'],p['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
    row=summarize(r,'budget_'+str(multiplier),w['phase'],w['scenario'])
    row.update(venue='reference_only',independent_test=False,budget_multiplier=multiplier)
    with gzip.open(DEST/f"{multiplier:g}_{w['phase']}_{w['scenario']}.json.gz",'wt') as fp:
        json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    return row

def main():
    if (DEST/'protocol.json').exists():raise FileExistsError('Keep original run')
    p=load_profile();windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==p['candidate']]
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary=2.,control=1.,
        definition='Same frozen all-market core, pullback_any3, weekly forecast, selected retained pair expert, cost/partial-fill/funding engine. Only anchor budget2x, same50kcapital and hardlimits. No model refit or coefficient selection.',
        windows=[{k:w[k] for k in ['phase','start','end','scenario']} for w in windows],
        scope='Aug1-Sep1,Sep1-Oct15 and continuous are previously exposed reference development; do not label them untouched or PAS returns. Purpose is capital/risk/overturnover sensitivity, not proof of live superiority.',
        source_hash=sha(__file__),orders_sent=0))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,m,w) for w in windows for m in [1.,2.]]):
            r=f.result();rows.append(r);save(DEST/'results.json',rows)
            print({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','volatility','max_drawdown','mean_notional','fees','risk_breach_bars','accounting_error']},flush=True)

if __name__=='__main__':main()
