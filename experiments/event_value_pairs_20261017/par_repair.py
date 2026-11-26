"""Versioned re-evaluation after finding mismatched pair-array ordering."""
import gzip,json,time
from pathlib import Path
import numpy as np
from .model import source_design
from .mapping_v2 import map_state
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts,save,sha
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.shared_execution_pairs_20261017.portfolio_v2 import replay
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'par_repair_artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfg=json.loads((HERE/'par_protocol.json').read_text());refcfg=json.loads((HERE/'reference_protocol.json').read_text())
    fits=json.loads((HERE/'artifacts/fits.json').read_text());_,_,p=inputs(refcfg)
    with np.load(ROOT/cfg['reference_run']/'inputs.npz') as z:ref={k:z[k].copy() for k in ['market_times','close']}
    with np.load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz') as z:original={k:z[k].copy() for k in ['market_times','close']}
    mask=original['market_times']<=ts(refcfg['end']);ix=np.searchsorted(ref['market_times'],original['market_times'][mask])
    np.testing.assert_array_equal(ref['market_times'][ix],original['market_times'][mask])
    np.testing.assert_array_equal(ref['close'][ix],original['close'][mask])
    with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(ROOT/'experiments/persistent_intent_pairs_20261017/artifacts/par_immediate_state.npz') as z:state={k:z[k].copy() for k in z.files}
    paths=[HERE/'par_repair.py',HERE/'mapping_v2.py',HERE/'model.py',HERE/'par_protocol.json',HERE/'artifacts/fits.json',
        INPUT/'inputs.npz',ROOT/cfg['reference_run']/'inputs.npz',ROOT/'experiments/persistent_intent_pairs_20261017/artifacts/par_immediate_state.npz',
        ROOT/'experiments/shared_execution_pairs_20261017/portfolio_v2.py']
    hashes={str(q.relative_to(ROOT)):sha(q) for q in paths};save(out/'frozen.json',dict(at_unix=time.time(),sources=hashes,protected=protected()))
    raw=source_design(ref,cfg,p['symbols']);mapped={};audits={}
    for mode in ['confirmed','calibrated']:mapped[mode],audits[mode]=map_state(state,d['times'],ref,raw,fits,mode)
    save(out/'signal_audits.json',audits);rows=[];same=True
    for phase in ['validation','comparison']:
        start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start'];end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        for arm in cfg['controls']+cfg['candidates']:
            for scenario in cfg['scenarios']:
                original_arm=arm;st=state
                if arm in cfg['candidates']:
                    st=mapped['confirmed' if 'confirmed' in arm else 'calibrated'];original_arm=arm.split('_')[0]+'_plus_pairs'
                r=replay(d,st,None,cfg,p,original_arm,scenario,ts(start),ts(end));r['arm']=arm
                label=f'par-{phase}-{arm}-{scenario}'
                with gzip.open(HERE/f'artifacts/runs/{label}.json.gz','rt') as f:old=json.load(f)
                same&=r['metrics']==old['metrics'] and r['fills']==old['fills'] and r['snapshots']==old['snapshots']
                if arm in cfg['controls']:assert r['metrics']==old['metrics'] and r['fills']==old['fills'] and r['snapshots']==old['snapshots']
                with gzip.open(out/'runs'/f'{label}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
                rows.append(dict(venue='par',phase=phase,arm=arm,scenario=scenario,metrics=r['metrics']))
                print(label,round(r['metrics']['net_pnl'],4),flush=True)
    for rel,digest in hashes.items():assert sha(ROOT/rel)==digest,rel
    save(out/'results.json',dict(rows=rows,actual_replays=len(rows),all_old_metrics_fills_snapshots_unchanged=bool(same),
        reference_source_prices_equal_through_evaluation_end=True,protected=protected(),orders_sent=0))
    save(HERE/'artifacts/par_mapping_superseded.json',dict(reason='The old mapper used raw pair column position across reference/par arrays with different pair order. Use par_repair_artifacts for source attribution and par comparisons.',
        replacement='par_repair_artifacts',old_reference_results_valid=True,par_account_results_unchanged=bool(same),orders_sent=0))
    print('DONE par mapping repair',len(rows),'account paths unchanged',same,flush=True)


if __name__=='__main__':main()
