"""Replay only signals actually published; never invent missing-hour signals."""
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from .study import ROOT, OUT, OLD, REF, START, END, read, save, sha, verify

def main():
    from competition_live.profile import load_profile
    from competition_live.candidate import build_candidate
    from pas_economic_allocation import model
    from pas_economic_allocation.anchored import RATIO
    from universe_repair.core import calculate, REVISION
    from experiments.published_pair_lifecycle_20261017.signals import FRAME_KEYS
    verify();directory=OUT/'dataset'
    amendment=OUT/'publication_gap_amendment.json'
    if not amendment.exists():
        save(amendment,dict(created_utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha(__file__),
            reason='Original prepare aborted before results: no actual publication forOct24 07UTC. Skip absent publications instead of backdating later complete data. No parameter,arm,boundary,metric or cost change.',
            outcome_results_opened=False))
    assert read(amendment)['code_sha256']==sha(__file__)
    d=dict(np.load(REF/'inputs.npz'));z=dict(np.load(REF/'targets.npz'));times=d['market_times']
    anchor,gate,parts=calculate(d,z['chronos_forecast'],z['frozen_regime_probability'],REVISION)
    fit=read(ROOT/'reports/pas_economic_allocation_20261022/fit.json');x,u,carry=model.features(d)
    price=model.predict(x,u,fit)*fit['attenuation'];publications={}
    for f in (ROOT/'practice_live/updates').glob('*/signals.json'):
        r=read(f);source=int(datetime.fromisoformat(r['input_completed_bar_utc']).timestamp());at=datetime.fromisoformat(r['created_at_utc']).timestamp()
        if START-43200<=source<END and (source not in publications or at<publications[source][0]):publications[source]=(at,f)
    adapter=build_candidate(load_profile(),lambda *a:None)['signals'];records=[];pairs=[];skipped=[];asof_checks=[]
    for i,source in enumerate(times):
        source=int(source)
        if not START-43200<=source<END or (source<START and source%86400):continue
        if source not in publications:
            skipped.append(source);continue
        at,file=publications[source];at=max(at,source+120)
        saved_input=dict(np.load(file.parent/'inputs.npz'))
        for field in ['close','funding','volume','trades']:
            assert np.array_equal(saved_input[field],d[field][:i+1]),f'As-of input changed:{source}/{field}'
        asof_checks.append(source)
        historical=dict(np.load(file.parent/'targets.npz'))
        frame=dict(source_time=source,published_time=at,weights=historical['default'][-1].tolist(),retention=[1.]*10,
            sigma=parts['sigma'][i].tolist(),volume=d['volume'][i].tolist(),trades=d['trades'][i].tolist(),source_hash=sha(file.parent/'inputs.npz'))
        event=dict(channel='strategySignal',received_ms=int(at*1000)+1,data=dict(frame=frame,incumbent=frame['weights'],candidate=frame['weights'],
            publication_sha256=hashlib.sha256(json.dumps({k:frame[k] for k in FRAME_KEYS},sort_keys=True,allow_nan=False).encode()).hexdigest()))
        fs=adapter.ingest(event,{k:v[:i+1].copy() for k,v in d.items()})
        if source<START:continue
        pairs.append(dict(source=source,forecasts=fs))
        records.append(dict(source_time=source,time=int(at*1000)+1,weights=frame['weights'],sigma=frame['sigma'],volume=frame['volume'],trades=frame['trades'],
            anchor=(anchor[i]*RATIO*2).tolist(),protected=gate['protected'][i].tolist(),forecast=price[i].tolist(),carry=carry[i].tolist()))
    save(directory/'events.json',records);save(directory/'pair_forecasts.json',pairs)
    save(directory/'publication_audit.json',dict(skipped_sources=skipped,asof_prefix_verified=asof_checks))
    np.savez_compressed(directory/'candidate_inputs.npz',times=[r['source_time'] for r in records],forecast=[r['forecast'] for r in records],carry=[r['carry'] for r in records],anchor=[r['anchor'] for r in records],protected=np.array([r['protected'] for r in records],bool))
    nonzero=[dict(source=r['source'],pair=f['pair'],action=f.get('economic_action')) for r in pairs for f in r['forecasts'] if f['model']=='decision_gate' and (not f['economic_available'] or f['economic_action']!=0)]
    save(directory/'pair_audit.json',dict(nonzero=nonzero,can_use_base_only=not nonzero,events=len(records)))
    assert not nonzero,'Active pair must be replayed'
    new=dict(np.load(directory/'candidate_inputs.npz'));old=dict(np.load(OLD/'dataset/candidate_inputs.npz'));ix=np.searchsorted(new['times'],old['times'])
    checks={k:bool(np.allclose(new[k][ix],v,equal_nan=True,rtol=1e-11,atol=1e-12)) for k,v in old.items()}
    save(OUT/'input_prefix_checks.json',checks);assert all(checks.values()),checks
    save(OUT/'input_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in [*directory.glob('*.json'),directory/'candidate_inputs.npz',*(OUT/'raw').glob('*.json')]})
    print('Prepared',len(records),'actual publications; absent hours',skipped,'pair gates',len(nonzero),flush=True)
    verify()

if __name__=='__main__':main()
