"""Reproduce the incumbent from verified cached forecasts, then add one ceiling."""
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import numpy as np
from strategy.agent import load_agent,TAIL
from practice_live.infer import protected_hashes
from replay_lab.common_comparison.models import Market
from .policy import target_path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'artifacts'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def json_default(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=json_default))

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'prepared.json').exists():raise ValueError('Prepared comparison inputs already exist')
    preserved=protected_hashes()
    additional={str(p.relative_to(ROOT)):sha(p) for folder in ('practice_live','bot') for p in (ROOT/folder).glob('*.py')}
    save(OUT/'protected_before.json',{**preserved,**additional})
    pointer=json.loads((ROOT/'practice_live/latest.json').read_text())
    run=ROOT/pointer['run'];verification=json.loads((run/'verification.json').read_text())
    for f,digest in [('inputs.npz',pointer['inputs_sha256']),('targets.npz',verification['targets_sha256'])]:
        if sha(run/f)!=digest:raise ValueError('Published input/target hash mismatch')
    with np.load(run/'inputs.npz') as z:inputs={k:z[k].copy() for k in z.files}
    with np.load(run/'targets.npz') as z:cached={k:z[k].copy() for k in z.files}
    if not np.array_equal(inputs['market_times'],cached['times']):raise ValueError('Forecast timestamps mismatch')
    baseline=json.loads((TAIL/'baseline.json').read_text());symbols=tuple(baseline['protocol']['symbols'])
    m=Market(inputs['market_times'],inputs['close'],inputs['volume'],inputs['trades'],inputs['funding'],inputs['observed'],symbols)
    original={k:v for k,v in inputs.items() if k not in ('market_times','funding','observed')}
    agent=load_agent('default')
    for i in np.flatnonzero((np.arange(len(m.times))>=1008)&(m.times%21600==0)):
        context=np.log(m.close[i-167:i+1].T).astype('float32')
        agent.agent.signal.cached[(int(m.times[i]),context.tobytes())]=cached['chronos_forecast'][i].copy()
    print('Reproducing fixed incumbent with cached identical Chronos forecasts',flush=True)
    agent.prepare_history(m,original,funding_source='mainnet_reference_settled')
    np.testing.assert_array_equal(agent.weights,cached['default'])
    print('Incumbent target reproduction exact',flush=True)
    parts=target_path(m,agent.weights,agent.agent.base_weights,agent.agent.frozen_probability,agent.agent.signal.forecast)
    a=int(np.datetime64('2026-08-01T00:00:00','s').astype('int64'))
    b=int(np.datetime64('2026-09-01T00:00:00','s').astype('int64'))
    fit=(m.times>=a)&(m.times<b)
    scale=float(abs(parts['candidate'][fit]).sum()/abs(parts['incumbent'][fit]).sum())
    np.savez_compressed(OUT/'inputs.npz',**inputs)
    np.savez_compressed(OUT/'targets.npz',times=m.times,sigma=agent.agent.sigma,constant_scale_control=agent.weights*scale,**parts)
    np.savez_compressed(OUT/'components.npz',trend_after_original_allocation=agent.agent.frozen_baseline_weights,
                        original_blend_scaled=agent.weights,chronos_forecast=agent.agent.signal.forecast,
                        regime_probability=agent.agent.frozen_probability)
    if protected_hashes()!=preserved:raise ValueError('Incumbent files changed')
    # Prefix consistency checks the new overlay, never future outcomes.
    end=m.exact(int(np.datetime64('2026-09-22T00:00:00','s').astype('int64')))+1
    prefix=Market(*(getattr(m,k)[:end] for k in ('times','close','volume','trades','funding','observed')),symbols)
    check=target_path(prefix,agent.weights[:end],agent.agent.base_weights[:end],agent.agent.frozen_probability[:end],agent.agent.signal.forecast[:end])
    np.testing.assert_array_equal(check['candidate'],parts['candidate'][:end])
    info={'created_at_utc':datetime.now(timezone.utc).isoformat(),'published_input_run':str(run.relative_to(ROOT)),
          'input_sha256':sha(OUT/'inputs.npz'),'targets_sha256':sha(OUT/'targets.npz'),
          'incumbent_targets_exact':True,'new_overlay_prefix_invariant':True,'constant_scale_control':scale,
          'frozen_files_checked':len(preserved),'hours':len(m.times),'source':'mainnet_reference_settled',
          'all_evaluation_history_previously_exposed':True,'live_orders_sent':0}
    save(OUT/'prepared.json',info);print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__':main()
