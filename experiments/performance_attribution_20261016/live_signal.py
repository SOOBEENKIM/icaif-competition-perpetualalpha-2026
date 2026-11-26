"""Independent, non-signing signal producer for the frozen budget candidate."""
import hashlib,json,os,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from practice_live.refresh import ROOT,OUT as LIVE,load_inputs,refresh,sha,save_json
from practice_live.infer import infer
from replay_lab.common_comparison.models import Market
from strategy.agent import TAIL
from experiments.risk_preserving_20261016.runtime import SignalFrame
from .agent import BudgetMatchedAgent

PROFILE=BudgetMatchedAgent.PROFILE
DIRECTORY=ROOT/'experiments/performance_attribution_20261016/deployment'
PUBLICATION=DIRECTORY/'signal.json'

def verified_run(now):
    pointer=json.loads((LIVE/'latest.json').read_text());run=ROOT/pointer['run']
    manifest=json.loads((run/'manifest.json').read_text());v=json.loads((run/'verification.json').read_text())
    if sha(run/'inputs.npz')!=pointer['inputs_sha256'] or sha(run/'targets.npz')!=v['targets_sha256']:
        raise ValueError('Incumbent publication digest mismatch')
    inputs=load_inputs(run/'inputs.npz')
    if int(inputs['market_times'][-1])!=int(now//3600)*3600:raise ValueError('Reference publication needs refresh')
    return run,inputs,manifest

def read_publication(path=PUBLICATION):
    obj=json.loads(Path(path).read_text());frame=SignalFrame(**obj['frame']);frame.validate()
    digest=hashlib.sha256(json.dumps(obj['frame'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    fixed=json.loads((Path(__file__).parent/'artifacts/budget_frozen.json').read_text())['multiplier']
    if obj['sha256']!=digest or obj['profile']!=PROFILE or obj['multiplier']!=fixed:raise ValueError('Invalid candidate publication')
    return frame,obj

def produce():
    DIRECTORY.mkdir(exist_ok=True)
    try:run,inputs,manifest=verified_run(time.time())
    except (FileNotFoundError,ValueError):
        infer(refresh());run,inputs,manifest=verified_run(time.time())
    with np.load(run/'targets.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    m=Market(*(inputs[k] for k in ['market_times','close','volume','trades','funding','observed']),tuple(p['symbols']))
    if not np.array_equal(m.times,old['times']):raise ValueError('Clock mismatch')
    agent=BudgetMatchedAgent()
    for i in np.flatnonzero((np.arange(len(m.times))>=1008)&(m.times%21600==0)):
        context=np.log(m.close[i-167:i+1].T).astype('float32')
        agent.incumbent.agent.signal.cached[(int(m.times[i]),context.tobytes())]=old['chronos_forecast'][i].copy()
    agent.prepare_history(m,{k:v for k,v in inputs.items() if k not in ('market_times','funding','observed')},funding_source=manifest['source'])
    np.testing.assert_array_equal(agent.incumbent.weights,old['default'])
    f=SignalFrame(int(m.times[-1]),time.time(),agent.weights[-1].tolist(),agent.retention[-1].tolist(),agent.sigma[-1].tolist(),
        m.volume[-1].tolist(),m.trades[-1].tolist(),manifest['inputs_sha256']);f.validate()
    frame=asdict(f);digest=hashlib.sha256(json.dumps(frame,sort_keys=True,allow_nan=False).encode()).hexdigest()
    obj={'profile':PROFILE,'multiplier':agent.multiplier,'frame':frame,'sha256':digest,'source_run':str(run.relative_to(ROOT)),
        'attribution':{k:v[-1].tolist() for k,v in agent.components.items()},'orders_sent':0,'signing_client_loaded':False}
    # All metadata and the frame change together, never as two publications.
    temporary=PUBLICATION.with_suffix('.tmp')
    with temporary.open('w') as out:json.dump(obj,out,allow_nan=False);out.flush();os.fsync(out.fileno())
    temporary.replace(PUBLICATION)
    save_json(DIRECTORY/f'signal_{f.source_time}.json',obj)
    print(json.dumps({'candidate':PROFILE,'source_time':f.source_time,'published_time':f.published_time,'orders_sent':0}),flush=True)

if __name__=='__main__':produce()
