"""Frozen candidate family; full-account netting and costed path comparisons."""
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from experiments.cost_horizon_20261016.research import data,SecondController,paired_interval,ts
from experiments.causal_repair_20261017.research import attribution
from experiments.performance_attribution_20261016.engine import simulate
from experiments.risk_preserving_20261016.policy import RiskCapController
from replay_lab.ema_rebuild.engine import Controller
from .signals import build

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')


def protected():
    old=json.loads((ROOT/'experiments/cost_horizon_20261016/artifacts/protocol_frozen.json').read_text())
    counts={}
    for group in ('active_protected_sources','reserved_dependencies'):
        for path,h in old[group].items():assert sha(ROOT/path)==h,path
        counts[group]=len(old[group])
    return counts


class BlendController(RiskCapController):
    def __init__(self,times,retention,base,weights):
        super().__init__();self.times=times;self.retention=retention;self.base=base;self.weights=weights
    def propose(self,time,*args,**kwargs):
        i=int(np.searchsorted(self.times,time))
        active=(self.retention[i]<1)&(self.base[i]*self.weights[i]>0)
        return super().propose(time,*args,retention=np.where(active,.75,1.),**kwargs)


def path_stats(run):
    return dict(mean_expected_vol=float(np.mean([s['daily_volatility'] for s in run['snapshots']])),
                mean_gross=run['metrics']['mean_gross'],engineering=all(run['gates'].values()))


def complement(candidate,baseline):
    ca={x['time']:x['return_'] for x in candidate['daily']};ba={x['time']:x['return_'] for x in baseline['daily']}
    days=sorted(set(ca)&set(ba));x=np.array([ca[t] for t in days]);y=np.array([ba[t] for t in days])
    loss=y<0
    return dict(days=len(days),daily_return_correlation=float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-14 and np.std(y)>1e-14 else None,
                incumbent_loss_days=int(loss.sum()),mean_candidate_on_incumbent_loss_days=float(x[loss].mean()) if loss.any() else None,
                mean_incumbent_on_its_loss_days=float(y[loss].mean()) if loss.any() else None,
                positive_candidate_on_loss_days=int((x[loss]>0).sum()),
                paired_return_difference=paired_interval(candidate,baseline))


def selection_checks(candidate,baseline):
    c=candidate['metrics'];b=baseline['metrics']
    return dict(return_higher=c['roi']>b['roi'],daily_std_no_worse=c['daily_std_sample']<=b['daily_std_sample'],
                mdd_no_worse=c['max_drawdown_magnitude']<=b['max_drawdown_magnitude'],
                engineering=all(candidate['gates'].values()))


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'frozen.json').exists():raise FileExistsError('This family has already been evaluated')
    counts=protected();cfg=json.loads((HERE/'protocol.json').read_text());inp,a,p,m=data()
    fixed=json.loads((ROOT/'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    second=a['candidate']*fixed
    sources={str(q.relative_to(ROOT)):sha(q) for q in [*HERE.glob('*.py'),HERE/'protocol.json']}
    inputs={str(q.relative_to(ROOT)):sha(q) for q in [ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
            ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz']}
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,inputs=inputs,protected_counts=counts,
         no_forward_outcomes_opened=True,orders_sent=0,historical_independence=False))
    raw,diagnostics=build(inp,cfg)
    # Prefix equivalence is checked on real inputs as well as synthetic tests.
    cut=m.exact(ts('2026-09-01'))+1
    prefix,_=build({k:v[:cut] for k,v in inp.items()},cfg)
    for name in raw:np.testing.assert_array_equal(raw[name][:cut],prefix[name])
    np.savez_compressed(OUT/'raw_targets.npz',times=m.times,**raw)
    diagnostic_arrays={}
    for name,d in diagnostics.items():
        for k in ('z','beta'):
            if k in d:diagnostic_arrays[name+'_'+k]=d.pop(k)
    np.savez_compressed(OUT/'pair_diagnostics.npz',times=m.times,**diagnostic_arrays)
    save(OUT/'signal_events.json',diagnostics)
    (OUT/'runs').mkdir(exist_ok=True);rows=[];runs={};calruns=0
    def replay(weights,control,start,end,scenario):
        return simulate(m,weights,a['sigma'],p,m.exact(ts(start)),m.exact(ts(end)),scenario,control)
    cal=cfg['calibration'];ref=replay(second,SecondController(m.times,a['retention']),cal['start'],cal['end'],'base')
    target=path_stats(ref);calibration={};scaled={}
    for name,w in raw.items():
        probes=[]
        for scale in cal['gross_scales']:
            r=replay(scale*w,Controller(),cal['start'],cal['end'],'base');calruns+=1;st=path_stats(r)
            probes.append(dict(scale=scale,**st,accepted=bool(st['engineering'] and
                st['mean_expected_vol']<=target['mean_expected_vol']*cal['path_risk_tolerance'] and
                st['mean_gross']<=target['mean_gross']*cal['gross_tolerance'])))
        admitted=[v['scale'] for v in probes if v['accepted']]
        scale=max(admitted) if admitted else 0.
        calibration[name]=dict(scale=scale,probes=probes,target=target,selected_without_pnl=True)
        scaled[name]=scale*w
        print('calibration',name,scale,flush=True)
    save(OUT/'calibration.json',calibration)
    weights=dict(first_strategy=a['incumbent'],second_strategy=second,second_75pct_cash=second*.75,**scaled)
    for name,w in scaled.items():weights['blend_'+name]=.75*second+.25*w
    weights['blend_equal_experts']=.75*second+.25*np.mean(list(scaled.values()),axis=0)
    np.savez_compressed(OUT/'targets.npz',times=m.times,**weights)
    def control(name):
        if name.startswith('second_'):return SecondController(m.times,a['retention'])
        if name.startswith('blend_'):return BlendController(m.times,a['retention'],second,weights[name])
        return Controller()
    def record(phase,name,start,end,scenario):
        r=replay(weights[name],control(name),start,end,scenario)
        cost=(p['execution']['fee_bps']+p['scenarios'][scenario]['impact_bps'])/10000
        terminal=sum(abs(np.array(r['metrics']['final_qty']))*m.close[m.exact(ts(end))])*cost
        r['metrics']['estimated_terminal_liquidation_cost']=float(terminal)
        r['metrics']['roi_after_estimated_liquidation']=r['metrics']['roi']-terminal/p['initial']
        r['metrics']['one_leg_only_hours']=sum(bool(abs(s['qty'][0])>0)!=bool(abs(s['qty'][1])>0) for s in r['snapshots']) if name.startswith('pair_') else None
        r.update(phase=phase,policy=name,per_asset=attribution(r,m),risk_path=path_stats(r))
        key=f'{phase}-{name}-{scenario}';runs[key]=r
        with gzip.open(OUT/'runs'/f'{key}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append({k:r[k] for k in ('phase','policy','scenario','metrics','gates','risk_path','per_asset')})
        z=r['metrics'];print(key,'ROI',round(z['roi']*100,6),'STD',round(z['daily_std_sample']*100,6),'MDD',round(z['max_drawdown_magnitude']*100,6),'fills',z['fills'],flush=True)
    for phase,(start,end) in cfg['periods'].items():
        for name in weights:
            for scenario in ('base','stress'):record(phase,name,start,end,scenario)
    comparisons=[]
    for phase in cfg['periods']:
        for scenario in ('base','stress'):
            b=runs[f'{phase}-second_strategy-{scenario}'];cash=runs[f'{phase}-second_75pct_cash-{scenario}']
            for name in weights:
                c=runs[f'{phase}-{name}-{scenario}']
                comparisons.append(dict(phase=phase,scenario=scenario,policy=name,checks=selection_checks(c,b),
                    complement=complement(c,b),return_above_75pct_cash=c['metrics']['roi']-cash['metrics']['roi']))
    # Reproduce the inherited baseline, including fills and every equity point.
    reproduced=[]
    for phase in ('all','september_october'):
        for name in ('first_strategy','second_strategy'):
            for scenario in ('base','stress'):
                with gzip.open(ROOT/'experiments/causal_repair_20261017/artifacts/comparison/runs'/f'{phase}-{name}-{scenario}.json.gz','rt') as f:old=json.load(f)
                now=runs[f'{phase}-{name}-{scenario}']
                assert old['fills']==now['fills']
                np.testing.assert_array_equal([s['equity'] for s in old['snapshots']],[s['equity'] for s in now['snapshots']])
                reproduced.append(f'{phase}-{name}-{scenario}')
    for q,h in sources.items():assert sha(ROOT/q)==h,q
    protected()
    save(OUT/'results.json',dict(rows=rows,comparisons=comparisons,calibration=calibration,
        baseline_reproductions=reproduced,calibration_runs=calruns+1,economic_runs=len(rows),
        causal_prefix_exact=True,protected_sources_unchanged=True,sealed_forward_outcomes_read=False,
        orders_sent=0,historical_independence=False,live_replacement_approved=False))
    print('DONE',len(rows),'comparison runs',calruns+1,'calibration runs',flush=True)


if __name__=='__main__':main()
