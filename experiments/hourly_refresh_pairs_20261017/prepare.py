"""Build hourly states without changing model weights or consuming future labels."""
import json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as INPUT, save, sha, ts
from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.model import design
from experiments.dense_history_pairs_20261017.research import load
from .model import infer

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'prepared';out.mkdir(exist_ok=False)
    cfgs={v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    spec=json.loads((HERE/'model_protocol.json').read_text());models={}
    paths=[HERE/'prepare.py',HERE/'model.py',HERE/'model_protocol.json',HERE/'test_hourly.py',HERE/'test_results.txt',
           *[HERE/f'{v}_protocol.json' for v in cfgs], INPUT/'inputs.npz',
           ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
           ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz',
           ROOT/cfgs['par']['reference_run']/'inputs.npz',
           ROOT/'experiments/dense_history_pairs_20261017/model.py',
           ROOT/'experiments/pairs_forecast_20261017/model.py',
           ROOT/'experiments/adaptive_residual_pairs_20261017/model.py',
           ROOT/'experiments/adaptive_residual_pairs_20261017/model_v2.py']
    for folder in ['dense_history_pairs_20261017/artifacts','adaptive_residual_pairs_20261017/artifacts_v2']:
        parent=ROOT/'experiments'/folder;paths.append(parent/'frozen.json')
        for rel,digest in json.loads((parent/'frozen.json').read_text())['sources'].items():assert sha(ROOT/rel)==digest,rel
    for pair in spec['pairs']:
        for h in spec['horizons']:
            path=ROOT/spec['model_directory']/('_'.join(pair)+f'_{h}.json');paths.append(path)
            rec=json.loads(path.read_text());assert rec['fit_available']==ts(spec['model_available'])
            models[tuple(pair),h]=rec
    for venue in cfgs:
        for kind in ['ridge','bias','affine','recent']:
            parent=ROOT/'experiments/dense_history_pairs_20261017/artifacts' if kind=='ridge' else ROOT/'experiments/adaptive_residual_pairs_20261017/artifacts_v2'
            for h in spec['horizons']:paths.append(parent/f'{venue}_{kind}_{h}_state.npz')
    digests={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=digests,protected=protected(),orders_sent=0,historical_independence=False))
    rd,_,protocol=inputs(cfgs['reference']);data={'reference':rd,'par':load(INPUT/'inputs.npz')}
    refs={'reference':load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz'),
          'par':load(ROOT/cfgs['par']['reference_run']/'inputs.npz')}
    records=[];checks=[];counts=[]
    for venue,r in refs.items():
        cfg=cfgs[venue]
        features=design(r['market_times'],r['close'],r['volume'],r['trades'],protocol['symbols'],cfg['dense_pairs'])
        for h in spec['horizons']:
            fresh,six,cal=infer(data[venue]['times'],r,features,models,cfg['dense_pairs'],h,spec)
            records.extend(dict(venue=venue,horizon=h,**row) for row in cal)
            for kind,state in fresh.items():
                parent=ROOT/'experiments/dense_history_pairs_20261017/artifacts' if kind=='ridge' else ROOT/'experiments/adaptive_residual_pairs_20261017/artifacts_v2'
                old=load(parent/f'{venue}_{kind}_{h}_state.npz')
                for key in old:np.testing.assert_array_equal(old[key],six[kind][key],err_msg=f'{venue}-{kind}-{h}-{key}')
                checks.append(dict(venue=venue,model=kind,horizon=h,six_hour_state_exact=True))
                np.savez_compressed(out/f'{venue}_{kind}_{h}_state.npz',**state)
                counts.append(dict(venue=venue,model=kind,horizon=h,valid_observations=int(state['directions'].sum()),
                                   old_valid_observations=int(old['directions'].sum())))
            print('PREPARE',venue,h,'calibrations',len(cal),'controls exact',flush=True)
    for rel,digest in digests.items():assert sha(ROOT/rel)==digest,rel
    save(out/'fits.json',records)
    save(out/'manifest.json',dict(checks=checks,counts=counts,model_weights_unchanged=True,
                                 maximum_source_age_hours=1,calibration_targets_strictly_mature=True,
                                 protected=protected(),orders_sent=0))


if __name__=='__main__':main()
