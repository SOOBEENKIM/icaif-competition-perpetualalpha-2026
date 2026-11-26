"""Compute and atomically publish a candidate signal outside the risk loop.

Uses one explicitly supplied, verified incumbent input publication. It never
changes practice_live/latest.json and never fits/selects new parameters.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from replay_lab.common_comparison.models import Market
from strategy.agent import TAIL
from .agent import RiskPreservingAgent
from .runtime import SignalFile,SignalFrame
from .prepare import ROOT,sha,save


def produce(run,destination):
    run=Path(run).resolve();destination=Path(destination).resolve()
    if not run.is_relative_to(ROOT/'practice_live/updates'):raise ValueError('Require an existing validated input publication')
    if destination.is_relative_to(ROOT/'practice_live'):raise ValueError('Never write incumbent live data')
    v=json.loads((run/'verification.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    if manifest['source']!='mainnet_reference_settled':raise ValueError('Unexpected data source')
    if sha(run/'inputs.npz')!=manifest['inputs_sha256'] or sha(run/'targets.npz')!=v['targets_sha256']:
        raise ValueError('Published inputs changed')
    with np.load(run/'inputs.npz') as z:inputs={k:z[k].copy() for k in z.files}
    with np.load(run/'targets.npz') as z:old={k:z[k].copy() for k in z.files}
    p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    m=Market(*(inputs[k] for k in ('market_times','close','volume','trades','funding','observed')),tuple(p['symbols']))
    if not np.array_equal(m.times,old['times']):raise ValueError('Cached forecast clock mismatch')
    agent=RiskPreservingAgent()
    for i in np.flatnonzero((np.arange(len(m.times))>=1008)&(m.times%21600==0)):
        context=np.log(m.close[i-167:i+1].T).astype('float32')
        agent.incumbent.agent.signal.cached[(int(m.times[i]),context.tobytes())]=old['chronos_forecast'][i].copy()
    agent.prepare_history(m,{k:v for k,v in inputs.items() if k not in ('market_times','funding','observed')},funding_source=manifest['source'])
    np.testing.assert_array_equal(agent.incumbent.weights,old['default'])
    from .prepare import OUT
    prepared=json.loads((OUT/'prepared.json').read_text())
    if str(run.relative_to(ROOT))==prepared['published_input_run']:
        with np.load(OUT/'targets.npz') as z:np.testing.assert_array_equal(agent.weights,z['candidate'])
    f=SignalFrame(int(m.times[-1]),time.time(),agent.weights[-1].tolist(),agent.retention[-1].tolist(),agent.sigma[-1].tolist(),
                  m.volume[-1].tolist(),m.trades[-1].tolist(),manifest['inputs_sha256'])
    SignalFile(destination).publish(f)
    # Last scheduled intent is an offline reproduction, not a current/live order.
    i=int(np.flatnonzero(m.times%21600==0)[-1]);intent=agent.decide(int(m.times[i]),np.zeros(10),1e6)
    saved=agent.snapshot();restored=RiskPreservingAgent(saved)
    assert restored.snapshot()==saved
    save(destination.with_suffix('.audit.json'),{'baseline_targets_exact':True,'restart_state_exact':True,
         'candidate_matches_compared_path':str(run.relative_to(ROOT))==prepared['published_input_run'],
         'latest_source_hour':int(m.times[-1]),'last_scheduled_reference_intent':intent,
         'source_run':str(run.relative_to(ROOT)),'orders_sent':0,'paper_only':True})
    return f

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();f=produce(a.run,a.output);print(json.dumps({'source_time':f.source_time,'orders_sent':0}))
