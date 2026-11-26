"""Keep acquired additional units while their signed thesis survives.

Preserve mandatory anchor changes and every protected risk cap. The change only
prevents repeated same-direction quantity optimization of the added sleeve.
"""
import copy,gzip,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .anchored import RATIO
from .projected import ProjectedController
from .study import OUT as ECONOMIC
from pas_adaptive_development.study import ROOT,save,sha,native
from replay_lab.common_comparison.engine import project,rounded,risk_breaches

OUT=ROOT/'reports/pas_economic_lifecycle_20261022'


class LifecycleController(ProjectedController):
    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        result=super().propose(time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if result is None:return None
        trace=self.trace[-1];free=np.array(trace['free']);old=np.asarray(qty)*prices/equity
        target=np.array(result['target_qty'])*prices/equity
        i=int(np.searchsorted(self.times,time))
        # Full forecast payments and half receipts, matching the optimizer.
        direction=np.sign(old)
        expected=direction*self.forecast[i]-np.where(direction*self.carry[i]>0,direction*self.carry[i],.5*direction*self.carry[i])
        held=free&(old!=0)
        maintain=held&np.isfinite(expected)&(expected>0)
        target[maintain]=old[maintain]
        target[held&~maintain]=0.  # Finish the old side before opening its opposite.
        target=project(target,equity,sigma,protocol['risk'])
        q=np.array([rounded(v,d) for v,d in zip(target*equity/prices,protocol['execution']['decimals'])])
        if risk_breaches(q,equity,prices,sigma,protocol['risk']):raise AssertionError('Risk violation')
        protected=self.protected[i]
        if np.any(abs(q*prices/equity)[protected]>abs(self.anchors[i])[protected]+1e-10):raise AssertionError('Risk cap restored')
        result['target_qty']=q.tolist();trace.update(target=(q*prices/equity).tolist(),same_thesis_units_preserved=maintain.tolist())
        return result


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),nominee='slow_lifecycle',
      hypothesis='Slow signals produced positive standalone gross price PnL but excessive paid rebalancing. Preserve actual acquired free-sleeve units while the refreshed weekly thesis remains positive, instead of resizing every6h. Base alpha/gate changes and hard risk trims still take effect.',
      arms=['incumbent','size_only','slow_lifecycle','budget_lifecycle'],
      changes='Only inventory lifecycle changes. No new forecast, no new horizon fitting, no paper-based return substitution. Close before a sign reversal; actual partial execution and funding remain charged.',
      promotion='Primary PnL and Q exceed incumbent in full and later windows base/stress with zero timing/accounting/hard-risk violations. All prior runs retained; budget variant diagnostic only.',
      independent=False,history_reused=True,orders_sent=0,live_changed=False,deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    files=[Path(__file__),Path(__file__).with_name('anchored.py'),Path(__file__).with_name('projected.py'),Path(__file__).with_name('replay.py'),OUT/'protocol.json']
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
                pred=slow['forecast'] if arm=='slow_lifecycle' else z['forecast']
                ratio=RATIO*(2 if arm=='budget_lifecycle' else 1)
                c=LifecycleController(d['times'],pred[ix],z['carry'][ix],r['final_mix_risk'][ir]*ratio,r['protected'][ir],lam)
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
