"""Eight prespecified variants; chronological selection and reused-history audit."""
import copy,gzip,json,time,hashlib
from pathlib import Path
import numpy as np
from experiments.cost_horizon_20261016.research import data,ts,SecondController
from experiments.complementary_alpha_20261017.research import protected
from experiments.performance_attribution_20261016.engine import simulate
from experiments.q_execution_20261017.metrics import official_metrics,paired_uncertainty
from replay_lab.ema_rebuild.engine import Controller
from .models import build_formations
from .policy import PairsController

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
    OUT.mkdir(exist_ok=True);(OUT/'runs').mkdir(exist_ok=True)
    if (OUT/'frozen.json').exists():raise FileExistsError('Do not overwrite a completed or started trial')
    before=protected();cfg=json.loads((HERE/'protocol.json').read_text());inp,a,p,m=data()
    p=copy.deepcopy(p)
    p['scenarios']['execution_stress']=dict(impact_bps=20.,participation=.0025,adverse_funding_bps_hour=0.)
    p['scenarios']['funding_stress']=dict(impact_bps=20.,participation=.0025,adverse_funding_bps_hour=1.)
    mult=json.loads((ROOT/'experiments/performance_attribution_20261016/artifacts/budget_frozen.json').read_text())['multiplier']
    second=a['candidate']*mult
    save(OUT/'frozen.json',dict(at=time.time(),sources={str(f.relative_to(ROOT)):sha(f) for f in [*HERE.glob('*.py'),HERE/'protocol.json']},
         inputs={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz']},
         protected=before,historical_independence=False,live_orders_sent=0))
    start=time.monotonic();formations={}
    for window in cfg['formation_hours']:
        forms=build_formations(m,cfg,window);formations[window]=forms
        save(OUT/f'formations_{window}.json',forms)
        print('FORMATION',window,'count',len(forms),'eligible',sum(len(f['models']) for f in forms),'reference',sum(len(f['copulas']) for f in forms),flush=True)
    rows=[];runs={}
    def replay(name,phase,scenario,scale=1.,blend=None):
        begin,end=cfg['periods'][phase];first,last=m.exact(ts(begin)),m.exact(ts(end))
        if name=='first':w=a['incumbent'];control=Controller()
        elif name in ('second','second_same_selection_exposure'):
            w=second*(scale if name=='second_same_selection_exposure' else 1.);control=SecondController(m.times,a['retention'])
        elif name=='second_plus_pairs_same_gross_budget':
            w=.75*second+.25*blend;control=Controller()
        else:
            family,window=name.rsplit('_',1);w=np.zeros_like(second)
            control=PairsController(m,formations[int(window)],cfg,family,p['scenarios'][scenario],last)
        r=simulate(m,w,a['sigma'],p,first,last,scenario,control)
        r.update(policy=name,phase=phase,official=official_metrics(r))
        r['metrics']['daily_std_sample']=r['official']['daily_sample_std']
        r['metrics']['estimated_exit_cost']=float(abs(np.array(r['metrics']['final_qty']))@m.close[last])*(9+p['scenarios'][scenario]['impact_bps'])/10000
        if isinstance(control,PairsController):
            r['events']=control.events
            r['metrics']['pair_entry_intents']=sum(e['event']=='entry_intent' for e in control.events)
            r['metrics']['partial_entry_aborts']=sum(e['event']=='partial_entry_abort' for e in control.events)
            r['metrics']['both_legs_filled']=sum(e['event']=='both_legs_filled' for e in control.events)
            r['gates']['past_only_formation']=all(e['trained_through']<e['time'] for e in control.events if e['event']=='entry_intent')
        label=f'{phase}-{name}-{scenario}'
        with gzip.open(OUT/'runs'/f'{label}.json.gz','wt') as f:json.dump(r,f,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist(),allow_nan=False)
        row={k:r[k] for k in ('policy','phase','scenario','metrics','official','gates')};rows.append(row);runs[label]=r
        print(label,'PnL',round(r['metrics']['net_pnl'],2),'ROI%',round(r['metrics']['roi']*100,6),'Q',round(r['official']['financial_score'],6),'fills',r['metrics']['fills'],flush=True)
        return r
    candidates=[f'{family}_{window}' for family in cfg['families'] for window in cfg['formation_hours']]
    for name in ['first','second',*candidates]:
        for scenario in ('base','execution_stress'):replay(name,'selection',scenario)
    ranked=[]
    for name in candidates:
        base=runs[f'selection-{name}-base'];stress=runs[f'selection-{name}-execution_stress']
        ok=all(base['gates'].values()) and all(stress['gates'].values()) and base['metrics']['roi']>0 and stress['metrics']['roi']>0
        ranked.append(dict(name=name,eligible=ok,pnl=base['metrics']['net_pnl'],Q=base['official']['financial_score'],stress_pnl=stress['metrics']['net_pnl']))
    ranked.sort(key=lambda r:(-r['eligible'],-r['pnl'],-r['Q'],r['name']))
    selected=ranked[0]['name'];selection=dict(selected=selected,eligible=ranked[0]['eligible'],ranking=ranked,at=time.time(),frozen_before_comparison=True)
    save(OUT/'selection.json',selection)
    scale=(runs[f'selection-{selected}-base']['metrics']['mean_gross']/max(runs['selection-second-base']['metrics']['mean_gross'],1e-12))
    for phase in ('comparison','recent'):
        for scenario in ('base','execution_stress','funding_stress'):
            for name in ('first','second',selected,'second_same_selection_exposure'):replay(name,phase,scenario,scale=scale)
            candidate=runs[f'{phase}-{selected}-{scenario}'];targets=np.zeros_like(second)
            for snap in candidate['snapshots']:
                ix=m.exact(snap['time']);targets[ix]=np.array(snap['qty'])*m.close[ix]/snap['equity']
            replay('second_plus_pairs_same_gross_budget',phase,scenario,blend=targets)
    comparisons=[]
    for phase in ('comparison','recent'):
        for scenario in ('base','execution_stress','funding_stress'):
            b=runs[f'{phase}-second-{scenario}'];c=runs[f'{phase}-{selected}-{scenario}']
            comparisons.append(dict(phase=phase,scenario=scenario,pnl_change=c['metrics']['net_pnl']-b['metrics']['net_pnl'],
                Q_change=c['official']['financial_score']-b['official']['financial_score'],uncertainty=paired_uncertainty(c['official'],b['official']),
                improved=c['metrics']['net_pnl']>b['metrics']['net_pnl'] and c['official']['financial_score']>b['official']['financial_score']))
    passed=selection['eligible'] and all(c['improved'] for c in comparisons if c['phase']=='comparison' and c['scenario']!='funding_stress')
    result=dict(selection=selection,rows=rows,comparisons=comparisons,selection_exposure_scale=scale,
        historical_improvement=passed,historical_independence=False,independent_forward_completed=False,
        live_replacement_approved=False,live_orders_sent=0,protected_after=protected(),elapsed_seconds=time.monotonic()-start)
    save(OUT/'results.json',result)
    print('COMPLETE historical_improvement=',passed,'seconds=',result['elapsed_seconds'],flush=True)

if __name__=='__main__':run()
