import gzip,json,time
from pathlib import Path
import numpy as np
from strategy.agent import TAIL
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts,save,sha
from experiments.complementary_alpha_20261017.research import protected
from .signals import design
from .engine import replay

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'


def main():
    OUT.mkdir(exist_ok=False);cfg=json.loads((HERE/'protocol.json').read_text())
    p=json.loads((TAIL/'baseline.json').read_text())['protocol'];run=ROOT/cfg['reference_run']
    manifest=json.loads((run/'manifest.json').read_text());assert sha(run/'inputs.npz')==manifest['inputs_sha256']
    with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(run/'inputs.npz') as z:ref={k:z[k].copy() for k in z.files}
    assert np.array_equal(ref['market_times'],ref['known'])
    formation=json.loads((ROOT/cfg['formation_source']).read_text())['formations'][-1]
    assert formation['date']==cfg['formation_cutoff'] and formation['top_two']==cfg['pairs']
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',INPUT/'inputs.npz',run/'inputs.npz',ROOT/cfg['formation_source']]
    sources={str(q.relative_to(ROOT)):sha(q) for q in paths}
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),historical_independence=False,orders_sent=0))
    de=design(d,ref,cfg,p['symbols']);np.savez_compressed(OUT/'state.npz',**de)
    save(OUT/'signals.json',dict(formation=formation['top_two'],fdr_qualified=formation['fdr_count'],
        triggers={name:int(np.count_nonzero(de[name][d['times']%3600==0])) for name in ('standard','delayed')}))
    arms=['first_strategy','second_strategy',*cfg['arms']];rows=[];runs={};(OUT/'runs').mkdir()
    def record(phase,arm,scenario,overlay=False):
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        r=replay(d,de,None,cfg,p,arm,scenario,ts(start),ts(end),overlay=overlay)
        label='second_plus_'+arm if overlay else arm;key=f'{phase}-{label}-{scenario}'
        r.update(label=label,phase=phase);runs[key]=r
        with gzip.open(OUT/'runs'/f'{key}.json.gz','wt') as f:
            json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append(dict(phase=phase,arm=label,scenario=scenario,metrics=r['metrics']))
        m=r['metrics'];print(key,'PnL',round(m['net_pnl'],4),'Q',round(m['financial_score'],8),'pairs',m['completed_pairs'],'aborts',m['aborted_pairs'],flush=True)
    for arm in arms:
        for scenario in cfg['scenarios']:record('validation',arm,scenario)
    rank=[]
    for arm in cfg['arms']:
        checks=[];differences=[]
        for scenario in cfg['scenarios']:
            a=runs[f'validation-{arm}-{scenario}']['metrics']
            checks.extend([a['net_pnl']>0,a['completed_pairs']>=3,a['risk_breach_bars']==0])
            for bn in arms[:2]:
                b=runs[f'validation-{bn}-{scenario}']['metrics'];differences.append(a['net_pnl']-b['net_pnl'])
                checks.extend([a['net_pnl']>b['net_pnl'],a['financial_score']>b['financial_score']])
        rank.append(dict(arm=arm,eligible=all(checks),minimum_pnl_advantage=min(differences)))
    rank.sort(key=lambda r:(-r['eligible'],-r['minimum_pnl_advantage'],r['arm']))
    selected=rank[0]['arm'];selection=dict(selected=selected,eligible=rank[0]['eligible'],ranking=rank,frozen_before_comparison_replay=True,historical_independence=False)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    for arm in arms:
        for scenario in cfg['scenarios']:record('comparison',arm,scenario)
    for phase in ('validation','comparison'):
        for scenario in cfg['scenarios']:record(phase,selected,scenario,True)
    # The derivative must retain identical incumbent execution under common inputs.
    baseline=HERE.parent/'reference_anchor_pairs_20261017/artifacts/runs'
    for phase in ('validation','comparison'):
        for arm in arms[:2]:
            for scenario in cfg['scenarios']:
                key=f'{phase}-{arm}-{scenario}'
                with gzip.open(baseline/f'{key}.json.gz','rt') as f:old=json.load(f)
                assert old['fills']==runs[key]['fills'] and old['snapshots']==runs[key]['snapshots']
    assert sha(OUT/'selection.json')==selection_hash
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    protected()
    save(OUT/'results.json',dict(rows=rows,selection=selection,historical_replays=len(rows),historical_independence=False,
        protected_sources_unchanged=True,baseline_execution_unchanged=True,orders_sent=0,automatic_live_replacement=False))
    print('DONE',len(rows),'replays; selected',selected,'eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
