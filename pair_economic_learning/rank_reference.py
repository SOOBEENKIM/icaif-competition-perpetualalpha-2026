"""Matched replay helper for forecast versus execution ablations."""
import copy,gzip,json
import numpy as np
from pair_value_learning.study import ROOT,INPUT,read,save
from competition_live.profile import load_profile,BUNDLE
from auto_pair_selection.selection import PAIRS


def run(out,statepath,arm,phase,scenario,*,cap_bps=None,include_base=True):
    from .ranked_replay import replay
    from pas_performance_research.selected import controller
    from pas_allocation_study.study import summarize
    profile=load_profile();w=next(x for x in read(BUNDLE/'expected.json')['runs'] if x['venue']=='reference' and x['arm']==profile['candidate'] and x['phase']==phase and x['scenario']==scenario)
    d=dict(np.load(BUNDLE/'data/reference.npz'));s=dict(np.load(statepath));t=s.pop('times');ii=np.searchsorted(t,d['times']);assert np.array_equal(t[ii],d['times'])
    s={k:(v[ii] if v.ndim and len(v)==len(t) else v) for k,v in s.items()}
    cfg=copy.deepcopy(profile['configs']['reference']);cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.}
    if cap_bps is not None:cfg['entry_slippage_cap_bps']=cap_bps
    c=None
    if include_base:
        z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));zi=np.searchsorted(z['times'],d['times']);f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times']);raw=dict(np.load(INPUT))
        c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][zi],z['all_market_signed_core_v1_protected'][zi],price_times=raw['market_times'],prices=raw['close'])
    name=('first' if include_base else 'pairs')+'_dense_hourlyhedge_individual_72'
    r=replay(d,s,None,cfg,profile['trading'],name,scenario,w['start'],w['end'],base_policy=c)
    path=out/'reference'/f'{arm}_{phase}_{scenario}.json.gz';path.parent.mkdir(exist_ok=True)
    with gzip.open(path,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    row=summarize(r,arm,phase,scenario);row.update(include_base=include_base,entry_slippage_cap_bps=cfg['entry_slippage_cap_bps'])
    assert row['timing_pass'] and row['accounting_error']<1e-6
    save(out/'rows'/f'{arm}_{phase}_{scenario}.json',row);return row
