"""Recheck incremental utility AFTER joint caps and lot rounding.

Earlier anchored studies are preserved. A box-optimal addition need not remain
economic after a shared net/gross/variance projection changes the portfolio.
"""
import copy,gzip,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .anchored import AnchoredController,RATIO
from .policy import utility
from .study import OUT as ECONOMIC
from pas_adaptive_development.study import ROOT,save,sha,native
from replay_lab.common_comparison.engine import project,rounded,risk_breaches

OUT=ROOT/'reports/pas_economic_projected_20261022'


def choose(projected,reference,old,mu,cov,lam,cost,exit_cost):
    gain=utility(projected,old,mu,cov,lam,cost,exit_cost)-utility(reference,old,mu,cov,lam,cost,exit_cost)
    return (projected if gain>1e-12 else reference).copy(),float(gain)


class ProjectedController(AnchoredController):
    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        action=super().propose(time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if action is None or not self.augment:return action
        trace=self.trace[-1];free=np.array(trace['free']);old=qty*prices/equity
        selected=np.array(action['target_qty'])*prices/equity
        reference=selected.copy();reference[free]=old[free]
        reference=project(reference,equity,sigma,protocol['risk'])
        q=np.array([rounded(v,d) for v,d in zip(reference*equity/prices,protocol['execution']['decimals'])])
        reference=q*prices/equity
        i=int(np.searchsorted(self.times,time));mu=np.array(trace['mu'])
        chosen,gain=choose(selected,reference,old,mu,168*sigma,self.lam,.0019,.0024+.25*abs(self.carry[i]))
        q=np.array([rounded(v,d) for v,d in zip(chosen*equity/prices,protocol['execution']['decimals'])])
        if risk_breaches(q,equity,prices,sigma,protocol['risk']):raise AssertionError('Risk violation')
        protected=self.protected[i]
        if np.any(abs(q*prices/equity)[protected]>abs(self.anchors[i])[protected]+1e-10):raise AssertionError('Gate override')
        action['target_qty']=q.tolist();trace.update(target=(q*prices/equity).tolist(),
            projection_utility_gain=gain,addition_rejected_after_projection=bool(gain<=1e-12),
            feasible_hold_target=reference.tolist())
        return action


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),nominee='budget_projected',
      hypothesis='An added sleeve must still improve forecast utility after combined risk projection and lot rounding. Otherwise retain the feasible actual inventory alongside mandatory base risk changes. Remove projection-driven uneconomic turnover.',
      arms=['incumbent','size_only','budget_projected','original_projected','slow_projected'],
      correction='Only the incremental utility acceptance check changes. Full costed independent fills, final risk caps, pair expert, forecast coefficients and July-selectedlambda5 unchanged.',
      controls='Compare each corrected arm against its exact preserved parent and current incumbent. budget_projected is the primary, fixed before results; other forecasts are diagnostic.',
      promotion='PnL and Q above incumbent in full and later periods under both costs; zero accounting/timing/risk issues. No retrospective primary substitution.',
      independent=False,history_reused=True,orders_sent=0,live_changed=False,deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    files=[Path(__file__),Path(__file__).with_name('anchored.py'),Path(__file__).with_name('policy.py'),Path(__file__).with_name('replay.py'),OUT/'protocol.json']
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from .replay import replay
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text())
    for rel,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/rel)==h
    profile=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(ECONOMIC/'forecasts.npz'));ix=np.searchsorted(z['times'],d['times'])
    slow=dict(np.load(ROOT/'reports/pas_economic_slow_20261022/forecasts.npz'))
    r=dict(np.load(ROOT/'reports/strategy_repairs_20261021/final_risk_targets.npz'));ir=np.searchsorted(r['times'],d['times'])
    np.testing.assert_array_equal(z['times'][ix],d['times']);np.testing.assert_array_equal(r['times'][ir],d['times'])
    cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    lam=json.loads((ECONOMIC/'selected.json').read_text())['selected']['risk_aversion']
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==profile['candidate']]
    controls=json.loads((ROOT/'reports/pas_covariance_v2_20261021/reference_results.json').read_text());rows=[]
    for w in windows:
        rows.extend([r for r in controls if r['arm'] in ['incumbent','size_only'] and r['phase']==w['phase'] and r['scenario']==w['scenario']])
        for arm in p['arms'][2:]:
            file=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz"
            if file.exists():
                with gzip.open(file,'rt') as f:result=json.load(f)
            else:
                pred=slow['forecast'] if arm=='slow_projected' else z['forecast']
                ratio=RATIO*(2 if arm=='budget_projected' else 1)
                c=ProjectedController(d['times'],pred[ix],z['carry'][ix],r['final_mix_risk'][ir]*ratio,r['protected'][ir],lam)
                result=replay(d,states,None,cfg,profile['trading'],profile['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
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
