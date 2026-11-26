"""No-rebalance lifecycle applied to the all-market economic model itself."""
import copy,gzip,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .policy import EconomicController
from .study import OUT as ECONOMIC
from pas_adaptive_development.study import ROOT,save,sha,native
from replay_lab.common_comparison.engine import project,rounded,risk_breaches

OUT=ROOT/'reports/pas_unified_held_20261022'


class HeldController(EconomicController):
    def propose(self,time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        action=super().propose(time,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if action is None:return None
        i=int(np.searchsorted(self.times,time));old=qty*prices/equity;direction=np.sign(old)
        mu=self.forecast[i] if self.mode!='carry_only' else np.zeros_like(old)
        expected=direction*mu-np.where(direction*self.carry[i]>0,direction*self.carry[i],.5*direction*self.carry[i])
        target=np.asarray(action['target_qty'])*prices/equity
        held=old!=0;keep=held&np.isfinite(expected)&(expected>0)
        target[keep]=old[keep];target[held&~keep]=0.
        target=project(target,equity,sigma,protocol['risk'])
        q=np.array([rounded(v,d) for v,d in zip(target*equity/prices,protocol['execution']['decimals'])])
        if risk_breaches(q,equity,prices,sigma,protocol['risk']):raise AssertionError('Final cap violation')
        action['target_qty']=q.tolist();self.trace[-1].update(target=(q*prices/equity).tolist(),same_thesis_units_preserved=keep.tolist())
        return action


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),nominee='unified_held',
      arms=['incumbent','size_only','unified_held','carry_held'],
      reason='ETH exclusion invalidated the apparent improved anchored candidate. Test economic model with actual-unit holding to reduce recurrent carry trading costs, under the same ETH exclusion for EVERY arm.',
      hypothesis='Do not recreate the lost ETH price alpha by increasing BTC. The standalone pooled price/carry model can evaluate every nonreserved market without the legacy long-only base; preserve quantities while its directional thesis is positive.',
      model='Exact sealed May-end ridge, June attenuation, July risk aversion; no refit. Existing forecast/carry entries and costs. Only same-thesis inventory sizing changes; close before reverse.',
      risk='One hard-cap projection, paid proactive trims, ownership accounting, independent pair legs, 50k own capital, 50k reserve, ETH exclusion. This replacement does not use the old base Chronos gate; it has no downstream upscaler that can undo a gate.',
      promotion='PnL and Q above matched incumbent AND size_only in full and later windows base/stress. Report same model without lifecycle, same-price matched costs, and all earlier failures.',
      independent=False,history_reused=True,orders_sent=0,live_changed=False,deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    files=[Path(__file__),Path(__file__).with_name('policy.py'),Path(__file__).with_name('coexist_replay.py'),OUT/'protocol.json',ECONOMIC/'forecasts.npz']
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})


def run():
    from competition_live.profile import load_profile,BUNDLE,verify_deployment
    from .coexist_replay import replay
    from pas_allocation_study.study import summarize
    p=json.loads((OUT/'protocol.json').read_text())
    for rel,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/rel)==h
    profile=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));d['first'][:,1]=0.
    states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'));z=dict(np.load(ECONOMIC/'forecasts.npz'))
    ix=np.searchsorted(z['times'],d['times']);np.testing.assert_array_equal(z['times'][ix],d['times'])
    cfg=copy.deepcopy(profile['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    lam=json.loads((ECONOMIC/'selected.json').read_text())['selected']['risk_aversion']
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==profile['candidate']]
    controls=json.loads((ROOT/'reports/pas_economic_coexist_20261022/reference_results.json').read_text());rows=[]
    for w in windows:
        rows.extend([r for r in controls if r['arm'] in ['incumbent','size_only'] and r['phase']==w['phase'] and r['scenario']==w['scenario']])
        for arm in p['arms'][2:]:
            file=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz"
            if file.exists():
                with gzip.open(file,'rt') as f:result=json.load(f)
            else:
                c=HeldController(d['times'],z['forecast'][ix],z['carry'][ix],lam,mode='carry_only' if arm=='carry_held' else 'unified')
                result=replay(d,states,None,cfg,profile['trading'],profile['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
                result['allocation_trace']=c.trace
                assert not any(f['asset']==1 for f in result['fills'])
                file.parent.mkdir(exist_ok=True)
                with gzip.open(file,'wt') as f:json.dump(result,f,default=native,allow_nan=False)
            row=summarize(result,arm,w['phase'],w['scenario']);rows.append(row)
            assert row['timing_pass'] and row['accounting_error']<1e-6
            save(OUT/'reference_results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
