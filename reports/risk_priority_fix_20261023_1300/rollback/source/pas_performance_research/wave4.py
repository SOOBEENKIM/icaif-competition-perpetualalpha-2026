"""Inventory-aware entry timing: retain base thesis and mandatory exits."""
import copy,gzip,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,BASE,save,sha,live_files
from .extension import OUT as EXT,START,END,SPLIT,SYMBOLS
from .inventory import EntryController

OUT=ROOT/'reports/pas_performance_research_20261022/wave4'
ARMS=[BASE,'entry_pullback6','entry_pullback24','entry_trend6','entry_trend24','entry_carry']

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,live_files=live_files(),orders_sent=0,live_changed=False,history_reused=True,independent_test=False,
        reason='Wave3 price confirmation changed absolute targets and repeatedly liquidated held inventory. Separate conditions for adding inventory from reasons to exit. Require favourable entry timing without discarding a surviving base thesis.',
        rules='Only block additional same-side size or reopening opposite side. Retain all baseline reductions/exits/hard risk caps.6h/24h venue pullback vs continuation, plus existing weekly economic-edge ablation. No forced asset exclusions.',
        fixed='Same budget/caps/pairs/forecasts/fees/schedule,all10,symmetric sides.',
        selection='Development search. All five variants reported, including failures; no independent holdout claim.'))
    files=[Path(__file__),Path(__file__).with_name('inventory.py'),OUT/'protocol.json'];save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})

def verify():
    for f,h in json.loads((OUT/'freeze.json').read_text()).items():assert sha(ROOT/f)==h,f
    assert live_files()==json.loads((OUT/'protocol.json').read_text())['live_files']

def build(f,z,ix,times,price_times,prices,arm,age=None):
    from pas_economic_allocation.lifecycle import LifecycleController
    args=(times,f['forecast'],f['carry'],z[BASE][ix],z[BASE+'_protected'][ix],5.)
    return LifecycleController(*args) if arm==BASE else EntryController(*args,price_times=price_times,prices=prices,entry_rule=arm,age=age)

def reference():
    from competition_live.profile import load_profile,BUNDLE
    from pas_economic_allocation.replay import replay
    from pas_allocation_study.study import summarize
    verify();p=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));ix=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times']);f={k:v[fi] for k,v in f.items()}
    source=json.loads((ROOT/'reports/universe_repair_20261022/protocol.json').read_text())['input']['inputs'];raw=dict(np.load(ROOT/source))
    cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==p['candidate']];rows=[]
    for w in windows:
        for arm in ARMS:
            c=build(f,z,ix,d['times'],raw['market_times'],raw['close'],arm)
            r=replay(d,states,None,cfg,p['trading'],p['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
            row=summarize(r,arm,w['phase'],w['scenario']);rows.append(row)
            file=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz";file.parent.mkdir(exist_ok=True)
            with gzip.open(file,'wt') as fp:json.dump(r,fp,default=lambda x:x.tolist(),allow_nan=False)
            save(OUT/'reference_results.json',rows);print(json.dumps({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q']}),flush=True)
    verify()

def pas():
    # Same engine and dataset as wave3; only the controller factory differs.
    from . import wave3
    wave3.OUT=OUT;wave3.ARMS=ARMS;wave3.build=build;wave3.verify=verify
    wave3.pas()

if __name__=='__main__':globals()[sys.argv[1]]()
