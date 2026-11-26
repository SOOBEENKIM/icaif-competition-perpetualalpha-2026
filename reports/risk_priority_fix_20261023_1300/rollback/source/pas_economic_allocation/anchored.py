"""Preserve corrected incumbent alpha; allocate only genuinely unused slots."""
import copy
import gzip
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from pas_adaptive_development.study import ROOT,save,sha,native
from .study import OUT as ECONOMIC
from .policy import EconomicController,solve,utility
from replay_lab.common_comparison.engine import project,rounded,risk_breaches
from experiments.risk_preserving_20261016.policy import RiskCapController

OUT=ROOT/'reports/pas_economic_anchor_20261022'
RATIO=1.44117372816997


class AnchoredController:
    def __init__(self,times,forecast,carry,anchors,protected,lam,augment=True):
        self.times=np.asarray(times);self.forecast=forecast;self.carry=carry
        self.anchors=anchors;self.protected=protected;self.lam=lam;self.augment=augment
        self.base=RiskCapController();self.trace=[]

    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        if time%21600 or time<=self.base.last_decision:return None
        i=int(np.searchsorted(self.times,time))
        if i>=len(self.times) or self.times[i]!=time:raise ValueError('Unaligned target')
        anchor=self.anchors[i];protected=self.protected[i]
        action=self.base.propose(time,prices,qty,equity,anchor,sigma,volume,trades,protocol,
                                 retention=np.where(protected,.5,1.),buffer=.2)
        if action is None:return None
        target=np.asarray(action['target_qty'])*prices/equity
        occupied=abs(anchor)>1e-12
        free=~occupied & ~protected & (volume>0) & (trades>0)
        old=qty*prices/equity
        mu=np.nan_to_num(self.forecast[i],nan=0.)-.75*self.carry[i]
        cost=.0019;exit_cost=.0024+.25*abs(self.carry[i]);cov=168*sigma
        if self.augment and np.any(free):
            ids=np.flatnonzero(free);risk=protocol['risk']
            cap=min(risk['asset_fraction'],risk['asset_notional']/equity)*risk['sizing_headroom']
            # Cross covariance accounts for the anchored positions, not just
            # standalone opportunity risk. Anchors remain fixed in this solve.
            fixed=target.copy();fixed[free]=0
            residual_mu=mu[ids]-self.lam*(cov[ids]@fixed)
            chosen=solve(old[ids],residual_mu,cov[np.ix_(ids,ids)],self.lam,cost,exit_cost[ids],np.full(len(ids),cap))
            target[ids]=chosen
        # Exactly one final downward projection; it cannot undo a risk ceiling.
        target=project(target,equity,sigma,protocol['risk'])
        if np.any(abs(target[protected])>abs(anchor[protected])+1e-10):
            raise AssertionError('Alpha allocation restored a protected risk cut')
        q=np.array([rounded(v,d) for v,d in zip(target*equity/prices,protocol['execution']['decimals'])])
        if risk_breaches(q,equity,prices,sigma,protocol['risk']):raise AssertionError('Final risk violation')
        self.trace.append(dict(time=int(time),anchor=anchor.tolist(),protected=protected.tolist(),
            free=free.tolist(),mu=mu.tolist(),old=old.tolist(),target=(q*prices/equity).tolist()))
        action['target_qty']=q.tolist();return action


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already sealed')
    primary=json.loads((ECONOMIC/'protocol.json').read_text())
    selected=json.loads((ECONOMIC/'selected.json').read_text())
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),nominee='repaired_augmented',
      reason='Complete replacement lost working incumbent price PnL. Preserve its alpha with final risk gates; add cost-positive opportunities only where the base has no signal and no protective restriction.',
      revision='Both final risk reductions and shared-capacity paid trims preserved. Restore the pre-existing original scale BEFORE final risk caps. All ten assets may enter through the symmetric forecast/carry allocator, subject to active incumbent thesis.',
      model='Reuse the June-trained/June-calibrated168h model and July-selected risk aversion without refitting after August-October outcomes.',
      risk_aversion=selected['selected']['risk_aversion'],ratio=RATIO,
      arms=['incumbent','size_only','repaired_anchor','repaired_augmented','incumbent_augmented'],
      selection='Only repaired_augmented is primary; incumbent_augmented diagnoses the cost of enforcing the final gates. No later winner substitution.',
      promotion=primary['promotion'],independent=False,history_reused=True,
      changed_after_previous_results=True,parent='pas_economic_allocation_20261022',
      final_risk_inputs='reports/strategy_repairs_20261021/final_risk_targets.npz',
      forecasts='reports/pas_economic_allocation_20261022/forecasts.npz',
      live_changed=False,orders_sent=0,deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    files=[Path(__file__),Path(__file__).with_name('policy.py'),Path(__file__).with_name('replay.py'),
           OUT/'protocol.json',ROOT/p['final_risk_inputs'],ROOT/p['forecasts'],ECONOMIC/'selected.json']
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from .replay import replay
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text())
    for name,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/name)==h,name
    profile=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'))
    state=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(ROOT/p['forecasts']));ix=np.searchsorted(z['times'],d['times'])
    r=dict(np.load(ROOT/p['final_risk_inputs']));ir=np.searchsorted(r['times'],d['times'])
    np.testing.assert_array_equal(z['times'][ix],d['times']);np.testing.assert_array_equal(r['times'][ir],d['times'])
    np.testing.assert_allclose(r['incumbent'][ir],d['first'],atol=1e-12)
    cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==profile['candidate']]
    rows=[]
    controls=json.loads((ROOT/'reports/pas_covariance_v2_20261021/reference_results.json').read_text())
    for w in windows:
        rows.extend([a for a in controls if a['phase']==w['phase'] and a['scenario']==w['scenario'] and a['arm'] in ['incumbent','size_only']])
        for arm in p['arms'][2:]:
            file=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz"
            if file.exists():
                with gzip.open(file,'rt') as f:result=json.load(f)
            else:
                if arm=='incumbent_augmented':anchors=d['first']*RATIO;mask=np.zeros_like(anchors,dtype=bool)
                else:anchors=r['final_mix_risk'][ir]*RATIO;mask=r['protected'][ir]
                c=AnchoredController(d['times'],z['forecast'][ix],z['carry'][ix],anchors,mask,p['risk_aversion'],augment=arm!='repaired_anchor')
                result=replay(d,state,None,cfg,profile['trading'],profile['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
                result['allocation_trace']=c.trace
                file.parent.mkdir(exist_ok=True)
                with gzip.open(file,'wt') as f:json.dump(result,f,default=native,allow_nan=False)
            row=summarize(result,arm,w['phase'],w['scenario']);rows.append(row)
            assert row['timing_pass'] and row['accounting_error']<1e-6
            save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
