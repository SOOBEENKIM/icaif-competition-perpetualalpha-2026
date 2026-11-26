"""One prespecified candidate selection, followed by disclosed reused-history audit."""
import gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from replay_lab.common_comparison.models import Market
from replay_lab.ema_rebuild.engine import Controller
from strategy.agent import TAIL
from experiments.risk_preserving_20261016.policy import RiskCapController
from experiments.performance_attribution_20261016.engine import simulate
from .model import fit,features,predict,labels
from .policy import CostController,optimize

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'
SOURCE=ROOT/'experiments/risk_preserving_20261016/artifacts'


def ts(s):return int(np.datetime64(s,'s').astype('int64'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')


class SecondController(RiskCapController):
    def __init__(self,times,retention):super().__init__();self.times=times;self.retention=retention
    def propose(self,time,*args,**kwargs):
        i=int(np.searchsorted(self.times,time));return super().propose(time,*args,retention=self.retention[i],**kwargs)


def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol_frozen.json').exists():raise FileExistsError('Research protocol already frozen')
    manifest=json.loads((ROOT/'experiments/performance_attribution_20261016/deployment/deployment_manifest.json').read_text())
    for p,h in manifest['sources'].items():
        if sha(p)!=h:raise ValueError('Active strategy source changed: '+p)
    reservation=json.loads((ROOT/'experiments/execution_retry_20261016/forward_reservation.json').read_text())
    for p,h in reservation['plan']['artifacts'].items():
        if sha(p)!=h:raise ValueError('Reserved trial dependency changed: '+p)
    save(OUT/'protocol_frozen.json',{'frozen_at_utc':np.datetime_as_string(np.datetime64(int(time.time()),'s'))+'Z',
         'sources':{str(p.relative_to(ROOT)):sha(p) for p in [*HERE.glob('*.py'),HERE/'protocol.json']},
         'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'inputs.npz',SOURCE/'targets.npz']},
         'active_protected_sources':manifest['sources'],'reserved_dependencies':reservation['plan']['artifacts'],
         'historical_independence':False,'orders_sent':0})


def data():
    with np.load(SOURCE/'inputs.npz') as z:inp={k:z[k].copy() for k in z.files}
    with np.load(SOURCE/'targets.npz') as z:arrays={k:z[k].copy() for k in z.files}
    p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    m=Market(*(inp[k] for k in ['market_times','close','volume','trades','funding','observed']),tuple(p['symbols']))
    assert np.array_equal(m.times,arrays['times']) and np.all(np.diff(m.times)==3600)
    return inp,arrays,p,m


def forecast_variant(preds,name):
    kind,h=name.split('_');h=int(h)
    # Two explicit horizons, each estimated directly; ensemble spans model classes.
    ridge=preds[f'ridge_{h}'];tree=preds[f'tree_{h}']
    if kind=='ridge':return ridge,np.zeros_like(ridge),h
    if kind=='tree':return tree,np.zeros_like(tree),h
    return .5*(ridge+tree),.25*abs(ridge-tree),h


def calibrate(times,mu,amb,horizon,sigma,incumbent):
    indexes=np.flatnonzero((times>=ts('2026-07-01'))&(times<ts('2026-08-01'))&(times%86400==0))
    target=float(np.mean([np.sqrt(24*incumbent[i]@sigma[i]@incumbent[i]) for i in indexes]))
    def risk(gamma):
        return float(np.mean([np.sqrt(24*w@sigma[i,:3,:3]@w) for i in indexes
           for w,_ in [optimize(mu[i],sigma[i,:3,:3],np.zeros(3),gamma,horizon,.0019,amb[i])]]))
    lo,hi=1e-3,1e7
    for _ in range(25):
        mid=np.sqrt(lo*hi)
        if risk(mid)>target:lo=mid
        else:hi=mid
    return hi,{'target_expected_daily_volatility':target,'candidate_expected_daily_volatility':risk(hi),
               'calibration_observations':len(indexes),'start':'2026-07-01','end_exclusive':'2026-08-01'}


def paired_interval(a,b):
    da={d['time']:d['return_'] for d in a['daily']};db={d['time']:d['return_'] for d in b['daily']}
    dates=sorted(set(da)&set(db));d=np.array([da[t]-db[t] for t in dates]);n=len(d)
    if n<6:return {'days':n,'interval95':None}
    rng=np.random.default_rng(20261016);starts=rng.integers(n,size=(5000,(n+2)//3))
    ix=np.stack([(starts+j)%n for j in range(3)],axis=-1).reshape(5000,-1)[:,:n]
    return {'days':n,'mean_daily_return_difference':float(d.mean()),'interval95':np.quantile(d[ix].mean(1),[.025,.975]).tolist(),
            'method':'paired circular moving-block bootstrap, 3-day blocks, 5000 draws; descriptive after historical reuse, not a confirmatory p-value'}


def main():
    freeze();inp,arrays,p,m=data();protocol=json.loads((HERE/'protocol.json').read_text())
    trained=time.monotonic();preds,states=fit(inp,protocol,OUT/'models')
    print('models fitted seconds',round(time.monotonic()-trained,3),flush=True)
    multiplier=json.loads((ROOT/'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    first_weights=arrays['incumbent'];second_weights=arrays['candidate']*multiplier
    configs={}
    for name in protocol['variants']:
        mu,amb,h=forecast_variant(preds,name);gamma,cal=calibrate(m.times,mu,amb,h,arrays['sigma'],first_weights)
        configs[name]={'gamma':gamma,'horizon':h,'calibration':cal};print('calibrated',name,gamma,flush=True)
    save(OUT/'calibration.json',configs)
    np.savez_compressed(OUT/'predictions.npz',times=m.times,**preds)
    (OUT/'runs').mkdir(exist_ok=True)
    rows=[];runs={}
    def run(phase,name,start,end,scenario,*,no_cost=False,scale_second=1.):
        if name=='first_strategy':w=first_weights;control=Controller()
        elif name in ('second_strategy','second_at_matched_exposure'):
            w=second_weights*scale_second;control=SecondController(m.times,arrays['retention'])
        else:
            variant=name if name in configs else selected
            mu,amb,h=forecast_variant(preds,variant);c=configs[variant]
            control=CostController(m.times,mu,amb,c['gamma'],h,cost_bps=0 if no_cost else 19.,partial=protocol['partial_adjustment'])
            w=np.zeros_like(first_weights)
        r=simulate(m,w,arrays['sigma'],p,m.exact(ts(start)),m.exact(ts(end)),scenario,control)
        r.update(phase=phase,policy=name)
        label=f'{phase}-{name}-{scenario}'
        with gzip.open(OUT/'runs'/f'{label}.json.gz','wt') as f:json.dump(r,f,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist(),allow_nan=False)
        rows.append({k:r[k] for k in ('phase','policy','scenario','metrics','gates')});runs[label]=r
        print(label,'ROI',round(r['metrics']['roi']*100,6),'STD',round(r['metrics']['daily_std_sample']*100,6),'MDD',round(r['metrics']['max_drawdown_magnitude']*100,6),'fills',r['metrics']['fills'],flush=True)
        return r
    for name in ['first_strategy','second_strategy',*protocol['variants']]:
        for scenario in ('base','stress'):run('validation',name,protocol['validation_start'],protocol['validation_end'],scenario)
    scores=[]
    for name in protocol['variants']:
        a,b=[runs[f'validation-{name}-{s}'] for s in ('base','stress')];z=a['metrics']
        eligible=all(a['gates'].values()) and all(b['gates'].values()) and z['roi']>0 and b['metrics']['roi']>0 and z['fills']>0
        scores.append({'variant':name,'score':z['roi']-z['daily_std_sample']-z['max_drawdown_magnitude'],'eligible':eligible})
    ranked=sorted(scores,key=lambda r:(-r['eligible'],-r['score'],r['variant']));selected=ranked[0]['variant']
    selection={'selected':selected,'eligible_on_validation':ranked[0]['eligible'],'scores':ranked,'configuration':configs[selected],
               'selected_before_comparison':True,'frozen_at_unix':time.time(),'live_orders_permitted':False}
    save(OUT/'selection.json',selection)
    frozen={str(q.relative_to(ROOT)):sha(q) for q in [*HERE.glob('*.py'),HERE/'protocol.json',OUT/'selection.json',*list((OUT/'models').glob('*'))]}
    save(OUT/'selected_manifest.json',frozen)
    exposure_ratio=runs[f'validation-{selected}-base']['metrics']['mean_gross']/runs['validation-second_strategy-base']['metrics']['mean_gross']
    for phase,start,end in [('comparison',protocol['comparison_start'],protocol['comparison_end']),
         ('first_half','2026-09-01','2026-09-22'),('second_half','2026-09-22',protocol['comparison_end'])]:
        for name in ('first_strategy','second_strategy','selected_third','second_at_matched_exposure','third_without_cost_optimization'):
            for scenario in ('base','stress'):
                r=run(phase,name,start,end,scenario,no_cost=name=='third_without_cost_optimization',scale_second=exposure_ratio if name=='second_at_matched_exposure' else 1.)
                if phase=='comparison' and name in ('first_strategy','second_strategy'):
                    oldname='incumbent' if name=='first_strategy' else 'risk_cap_budget_matched_v1'
                    with gzip.open(ROOT/f'experiments/performance_attribution_20261016/artifacts/runs/budget-historical_evaluation-{oldname}-{scenario}.json.gz','rt') as f:old=json.load(f)
                    assert r['fills']==old['fills'];np.testing.assert_array_equal([s['equity'] for s in r['snapshots']],[s['equity'] for s in old['snapshots']])
    checks=[]
    for scenario in ('base','stress'):
        c=runs[f'comparison-selected_third-{scenario}']
        for name in ('first_strategy','second_strategy'):
            b=runs[f'comparison-{name}-{scenario}'];a,bb=c['metrics'],b['metrics']
            gates={'return_improved':a['roi']>bb['roi'],'daily_std_no_worse':a['daily_std_sample']<=bb['daily_std_sample'],
                   'mdd_no_worse':a['max_drawdown_magnitude']<=bb['max_drawdown_magnitude'],'engineering':all(c['gates'].values()),
                   'positive_base_return':a['roi']>0 if scenario=='base' else True}
            checks.append({'scenario':scenario,'comparator':name,'gates':gates,'passed':all(gates.values()),'paired_difference':paired_interval(c,b)})
    forecasts=[];x,vol,_=features(inp)
    for h in protocol['horizons']:
        y,maturity=labels(inp,vol,h)
        # Non-overlapping within-asset labels; cross-asset observations remain dependent.
        idx=(m.times>=ts(protocol['comparison_start']))&(maturity<=ts(protocol['comparison_end']))&(m.times%(h*3600)==0)
        actual=y[idx]*vol[idx]*np.sqrt(h)
        for kind in ('ridge','tree','blend'):
            pp,_,_=forecast_variant(preds,f'{kind}_{h}');prediction=pp[idx]
            forecasts.append({'model':kind,'horizon':h,'time_blocks':int(idx.sum()),'asset_observations':int(actual.size),
                'mse':float(np.mean((actual-prediction)**2)),'zero_mse':float(np.mean(actual**2)),
                'correlation':float(np.corrcoef(actual.ravel(),prediction.ravel())[0,1])})
    # Re-check every source dependency of the real bot and reserved experiment.
    protected=json.loads((OUT/'protocol_frozen.json').read_text())
    for category in ('active_protected_sources','reserved_dependencies'):
        for path,h in protected[category].items():assert sha(path)==h,path
    result={'selection':selection,'rows':rows,'checks':checks,'forecast_metrics':forecasts,'exposure_matching_scale_from_validation':exposure_ratio,
            'baselines_reproduced_exactly':True,'historical_gates_pass':all(c['passed'] for c in checks),
            'historical_independence':False,'independent_forward_completed':False,'live_replacement_approved':False,
            'active_and_reserved_sources_unchanged':True,'orders_sent':0,'elapsed_seconds':time.monotonic()-trained}
    save(OUT/'results.json',result)
    print('DONE',selected,'historical_gates_pass',result['historical_gates_pass'],flush=True)


if __name__=='__main__':main()
