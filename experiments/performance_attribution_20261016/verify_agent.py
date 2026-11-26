"""Exercise the candidate against the same verified complete history as replay."""
import json,time
import numpy as np
from replay_lab.common_comparison.models import Market
from strategy.agent import TAIL
from experiments.risk_preserving_20261016.runtime import SignalFile,SignalFrame
from experiments.risk_preserving_20261016.prepare import sha,save
from .collect import ROOT,OUT
from .agent import BudgetMatchedAgent

def main():
    source=ROOT/'experiments/risk_preserving_20261016/artifacts';prepared=json.loads((source/'prepared.json').read_text())
    run=ROOT/prepared['published_input_run'];manifest=json.loads((run/'manifest.json').read_text());v=json.loads((run/'verification.json').read_text())
    assert sha(run/'inputs.npz')==manifest['inputs_sha256'] and sha(run/'targets.npz')==v['targets_sha256']
    with np.load(run/'inputs.npz') as z:inputs={k:z[k].copy() for k in z.files}
    with np.load(run/'targets.npz') as z:old={k:z[k].copy() for k in z.files}
    p=json.loads((TAIL/'baseline.json').read_text())['protocol'];m=Market(*(inputs[k] for k in ['market_times','close','volume','trades','funding','observed']),tuple(p['symbols']))
    agent=BudgetMatchedAgent()
    for i in np.flatnonzero((np.arange(len(m.times))>=1008)&(m.times%21600==0)):
        context=np.log(m.close[i-167:i+1].T).astype('float32')
        agent.incumbent.agent.signal.cached[(int(m.times[i]),context.tobytes())]=old['chronos_forecast'][i].copy()
    print('Rebuilding candidate with identical cached forecasts',flush=True)
    agent.prepare_history(m,{k:v for k,v in inputs.items() if k not in ('market_times','funding','observed')},funding_source=manifest['source'])
    np.testing.assert_array_equal(agent.incumbent.weights,old['default'])
    with np.load(source/'targets.npz') as z:np.testing.assert_array_equal(agent.weights,z['candidate']*agent.multiplier)
    i=int(np.flatnonzero(m.times%21600==0)[-1]);action=agent.decide(int(m.times[i]),np.zeros(10),1e6)
    state=agent.snapshot();assert BudgetMatchedAgent(state).snapshot()==state
    frame=SignalFrame(int(m.times[-1]),time.time(),agent.weights[-1].tolist(),agent.retention[-1].tolist(),agent.sigma[-1].tolist(),m.volume[-1].tolist(),m.trades[-1].tolist(),manifest['inputs_sha256'])
    SignalFile(OUT/'budget_signal.json').publish(frame)
    save(OUT/'agent_verification.json',{'compared_weights_exact':True,'incumbent_reproduction_exact':True,'restart_exact':True,
       'source_run':prepared['published_input_run'],'latest_source_hour':int(m.times[-1]),'rows_checked':len(m.times),
       'last_scheduled_offline_intent':action,'live_orders_sent':0,'actual_execution_connection':False})
    print('Agent weights match all replay inputs; restart and attribution verified',flush=True)

if __name__=='__main__':main()
