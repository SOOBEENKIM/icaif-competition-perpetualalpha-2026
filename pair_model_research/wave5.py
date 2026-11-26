import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from auto_pair_validation.study import OUT as V2,verify as verify_v2
from .study import ROOT,read,save,sha,verify as verify_wave1
from .opportunities import fit,predict,slope
from . import evaluation

OUT=ROOT/'reports/pair_model_research_20261023/wave5'
ARMS=['opportunity_slope','opportunity_ridge','opportunity_ensemble']
SPEC=dict(lookback_days=90,minimum_rows=200,minimum_sources=50,ridge_penalty=1.)


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=verify_v2();verify_wave1()
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),wave=5,arms=ARMS,spec=SPEC,input=p['input'],input_sha256=p['input_sha256'],
      hypothesis='Shared unconditional return fits, static relationship screens and forecast-parameter uncertainty failed. Learn conditional signed return among actual past published large-forecast opportunities, without knowing their result at selection time.',
      training='Daily90d past OOF raw forecasts,6h sources,abs(raw)>38bp,strict24h target maturity. First-stage raw forecasts were fitted before their source. Predates August OOF archive; SPO is not used in this wave because its pre-August archive was not fitted. Full repeated-history development, not independent.',
      arms_description='Per-pair through-origin shrunk slope (min12 rows;6 pseudo-day zero prior); or pooled linear/linear+64-depth2-tree forecasts of normalized signed24h relative return. Source timestamps have equal total weight across correlated pairs then30d decay. Features include original20, direction-aligned20, magnitude and cost/source-vol. No asset identities.',
      valuation='Forecast magnitude clipped between zero and original magnitude, original sign retained. No intercept alpha for slope. Entry when conditional gross estimate exceeds38bp, then actual planner fees/funding/depth/risk. Proxy future returns are not fills. Keep forecast schema for holding/renewals.',
      fixed='All45 IDs,original source beta,24h horizon,6000gross,50k own+50k reserve,base strategy,risk/fees/funding/partial-fill execution unchanged.',
      selection='August worst Q then PnL; reject risk violations, save choice before later comparison.',history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),Path(__file__).with_name('opportunities.py'),Path(__file__).with_name('evaluation.py'),OUT/'protocol.json']})


def verify():
    verify_v2();verify_wave1()
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    return read(OUT/'protocol.json')


def build():
    p=verify();v=read(V2/'protocol.json');s=dict(np.load(V2/'states.npz'));l=dict(np.load(V2/'labels.npz'));f=dict(np.load(ROOT/'reports/pair_model_research_20261023/features.npz'));t=s['times']
    assert np.array_equal(t,f['times']) and np.array_equal(t,l['times'])
    mu={a:np.full_like(s['raw_mu'],np.nan) for a in ARMS};records=[]
    for day in np.unique(t[t>=v['evaluation_start']]//86400*86400):
        model,tree=fit(t,f['x'],s['raw_mu'],l['unit'],l['actual'],l['maturity'],int(day),p['spec'])
        coeff,counts=slope(t,s['raw_mu'],l['unit'],l['actual'],l['maturity'],int(day));ix=np.flatnonzero((t>=day)&(t<day+86400))
        mu['opportunity_slope'][ix]=s['raw_mu'][ix]*coeff
        if model:
            a,b=predict(f['x'][ix],s['raw_mu'][ix],l['unit'][ix],model,tree)
            mu['opportunity_ridge'][ix]=a;mu['opportunity_ensemble'][ix]=b
            save(OUT/'models'/f'{day}.json',model);tree.save_model(OUT/'models'/f'{day}.ubj')
        records.append(dict(source=int(day),coefficient=coeff.tolist(),counts=counts,model_available=model is not None))
    for arm in ARMS:
        out={k:s[k] for k in ['times','dense','pairs','signal_times','source_prices','beta','vol','forecast_times','directions']};out['mu']=mu[arm]
        available=np.isfinite(out['mu'])&s['directions'].astype(bool)
        out['economic_available']=available;out['economic_action']=np.where(available&(abs(out['mu'])>.0038),np.sign(np.nan_to_num(out['mu'])),0).astype(int)
        np.savez_compressed(OUT/f'{arm}.npz',**out)
    save(OUT/'fit_audit.json',records);verify();print('built',len(records),'daily models',flush=True)


def stage_a():
    p=verify();evaluation.stage_a(OUT,p['arms']);verify()


def stage_b():
    verify();evaluation.stage_b(OUT);verify()


if __name__=='__main__':globals()[sys.argv[1]]()
