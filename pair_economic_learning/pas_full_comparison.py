"""Matched full-strategy PAS48h comparison, restricted by frozen core coverage."""
import copy,gzip,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from pair_value_learning.study import ROOT,read,save,sha
from pair_value_learning.pas_extension import OUT as VENUE,verify as verify_venue,START
from pair_value_learning.pas_intraday_replay import replay
from .execution_study import verify as verify_parent
from competition_live.profile import load_profile
from auto_pair_selection.selection import PAIRS
from auto_pair_selection.signals import state_for_arm
from pas_performance_research.selected import controller
from pas_allocation_study.study import summarize
OUT=ROOT/'reports/pair_economic_learning_20261023/pas_full48'
END=START+48*3600
ARMS=['incumbent_full','coherent_full','coherent_pairs']
FILES={
 'venue':VENUE/'dataset.npz',
 'coherent':ROOT/'reports/pair_economic_learning_20261023/wave3/unbounded_pairs_72.npz',
 'incumbent':ROOT/'reports/auto_pair_selection_20261023/states.npz',
 'targets':ROOT/'reports/universe_repair_20261022/targets.npz',
 'forecast':ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'}


def freeze():
    verify_parent();verify_venue()
    if (OUT/'protocol.json').exists():raise FileExistsError('Frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),start=START,end=END,arms=ARMS,
        purpose='Compare full incumbent with candidate, not just isolated pair modules. Endpoint48h is last noon covered by frozen core input ending21Oct14UTC, not PnL-selected.',
        prices='Same observed PAS completedhour candles,volume and actual funding events. PAS entrytiminghistory uses completedvenuecloses with lastactiveage<=1h. No prelaunch24hprice invention. Core weeklyforecast+anchors from unchanged stored causalpipeline.',
        controls='Each starts flat50k+50kreserve. Incumbent two rebuilt24h gates; candidate coherent45pairs72h with/without unchangedcore. Same quantities/costs/risk/lifecycle. No historicalpositions imported.',
        limitations='48h insufficient for72hpairmaturity; show marked and exitreserve-adjustedPnL. Hourlyclose execution is conditional, not realL2. Core is evaluated at hourly venue closes, not full live minute feed. Whole-strategy counterfactual, not walletactual.',
        history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'protocol.json',*FILES.values()]})


def verify():
    verify_parent();verify_venue()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def mapped(path,t):
    d=dict(np.load(path));st=d.pop('times');ii=np.searchsorted(st,t)
    assert np.array_equal(st[ii],t)
    return {k:(v[ii] if v.ndim and len(v)==len(st) else v) for k,v in d.items()}


def run():
    verify();raw=dict(np.load(FILES['venue']));t=raw['times'][raw['times']<=END];n=len(raw['times'])
    d={k:(v[:len(t)].copy() if v.ndim and len(v)==n else v.copy()) for k,v in raw.items()}
    core=mapped(FILES['targets'],t);forecast=mapped(FILES['forecast'],t)
    incumbent=state_for_arm(mapped(FILES['incumbent'],t),'fixed_two_rebuilt');coherent=mapped(FILES['coherent'],t)
    active=(d['par_volume']>0)&(d['par_trades']>0)
    seen=np.full(10,-np.inf);age=np.empty_like(d['par_close'])
    for i,now in enumerate(t):
        seen=np.where(active[i],now,seen);age[i]=now-seen
    age=np.minimum(age,1e20)
    p=load_profile();cfg=copy.deepcopy(p['configs']['reference']);cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.}
    cfg['scenarios']['no_receipts']={**cfg['scenarios']['base'],'funding_receipt_multiplier':0.}
    for arm in ARMS:
        s={k:v.copy() for k,v in (incumbent if arm=='incumbent_full' else coherent).items()}
        visible=np.array([active[:,a]&active[:,b] for a,b in s['pairs']]).T
        s['directions']*=visible
        for scenario in ['base','stress','no_receipts']:
            c=None
            if arm!='coherent_pairs':
                c=controller(t,forecast['forecast'],forecast['carry'],core['all_market_signed_core_v1'],core['all_market_signed_core_v1_protected'],price_times=t,prices=d['par_close'],age=age)
            name=p['candidate'] if arm=='incumbent_full' else ('first' if c is not None else 'pairs')+'_dense_hourlyhedge_individual_72'
            r=replay(d,s,None,cfg,p['trading'],name,scenario,START,END,base_policy=c)
            row=summarize(r,arm,'PAS48h_full_strategy',scenario)
            row.update(official_account_result=False,conditional_execution=True)
            assert row['timing_pass'] and row['accounting_error']<1e-6
            path=OUT/'runs'/f'{arm}_{scenario}.json.gz';path.parent.mkdir(exist_ok=True)
            with gzip.open(path,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
            save(OUT/'rows'/f'{arm}_{scenario}.json',row)
            print(json.dumps({k:row[k] for k in ['arm','scenario','net_pnl','terminal_pnl','Q','risk_breach_bars','completed_pairs']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
