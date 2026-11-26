"""Ablations declared after the first improvement, not new selected candidates."""
import copy,gzip,json
import numpy as np
from .study import ROOT,BASE,save,sha,live_files
from .wave2 import OUT as PARENT,verify

OUT=PARENT/'controls'

def main():
    from competition_live.profile import load_profile,BUNDLE
    from pas_economic_allocation.replay import replay
    from pas_economic_allocation.lifecycle import LifecycleController
    from pas_allocation_study.study import summarize
    verify()
    save(OUT/'protocol.json',dict(arms=['pair_only','uniform_half_core'],purpose='Separate removal of unprofitable core exposure from newly added carry alpha. Not an independent test or candidate selection.',
        scale=.5,scale_chosen_from_returns=False,live_changed=False,orders_sent=0))
    p=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));ix=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times'])
    cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==p['candidate']]
    rows=[]
    for w in windows:
        for arm in ['pair_only','uniform_half_core']:
            if arm=='pair_only':anchor=np.zeros_like(z[BASE][ix]);guard=np.ones_like(z[BASE+'_protected'][ix],bool)
            else:anchor=.5*z[BASE][ix];guard=z[BASE+'_protected'][ix]
            c=LifecycleController(d['times'],f['forecast'][fi],f['carry'][fi],anchor,guard,5.)
            r=replay(d,states,None,cfg,p['trading'],p['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
            row=summarize(r,arm,w['phase'],w['scenario']);rows.append(row)
            file=OUT/f"{arm}_{w['phase']}_{w['scenario']}.json.gz"
            with gzip.open(file,'wt') as fp:json.dump(r,fp,default=lambda x:x.tolist(),allow_nan=False)
            save(OUT/'results.json',rows);print(json.dumps({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q']}),flush=True)
    verify()

if __name__=='__main__':main()
