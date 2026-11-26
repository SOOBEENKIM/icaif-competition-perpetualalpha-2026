"""Freeze source inputs and rebuild controls; never publish to the live worker."""
import json,hashlib
from datetime import datetime
import numpy as np
from all_market_strategy.signals import build
from .data import ROOT,OUT

def main():
    pointer=json.loads((ROOT/'practice_live/latest.json').read_text());run=ROOT/pointer['run']
    assert hashlib.sha256((run/'inputs.npz').read_bytes()).hexdigest()==pointer['inputs_sha256']
    (OUT/'reference_pointer.json').write_text(json.dumps(pointer,indent=2))
    with np.load(run/'inputs.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    with np.load(run/'targets.npz',allow_pickle=False) as z:
        old=z['default'].copy();forecast=z['chronos_forecast'].copy();prob=z['frozen_regime_probability'].copy()
    legacy,_=build(d,forecast,prob,scope='legacy')
    np.testing.assert_allclose(old,legacy,rtol=1e-11,atol=1e-13)
    allw,allparts=build(d,forecast,prob);sym,symparts=build(d,forecast,prob,symmetric=True)
    n=len(d['market_times'])-24
    prefix,_=build({k:v[:n] for k,v in d.items()},forecast[:n],prob[:n],symmetric=True)
    np.testing.assert_allclose(prefix,sym[:n],rtol=1e-11,atol=1e-13)
    np.savez_compressed(OUT/'reference_targets.npz',times=d['market_times'],incumbent_published=old,all_markets_long_only=allw,all_markets_symmetric_daily=sym)
    cfg=json.loads((OUT/'dataset.json').read_text());publications={}
    for path in sorted((ROOT/'practice_live/updates').glob('*/signals.json')):
        s=json.loads(path.read_text());t=int(datetime.fromisoformat(s['input_completed_bar_utc']).timestamp()*1000)
        if cfg['start_ms']<=t<=cfg['cutoff_ms']:
            at=int(datetime.fromisoformat(s['created_at_utc']).timestamp()*1000)
            if t not in publications or at<publications[t]:publications[t]=at
    (OUT/'publication_times.json').write_text(json.dumps(publications,indent=2))
    result={'legacy_max_absolute_error':float(np.max(abs(old-legacy))),'symmetric_prefix_error':float(np.max(abs(prefix-sym[:n]))),'publications':len(publications),'last_source':int(d['market_times'][-1]),'assets':[]}
    for j,s in enumerate(cfg['symbols']):
        result['assets'].append({'symbol':s,'old_weight':float(old[-1,j]),'all_weight':float(allw[-1,j]),'symmetric_weight':float(sym[-1,j]),'symmetric_ema_state':float(symparts['ema'][-1,j]),'symmetric_reversion_state':float(symparts['reversion'][-1,j])})
    (OUT/'target_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
