"""Bounded read-only supervision test during a 100-second injected inference delay."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json,multiprocessing,time,urllib.request
import numpy as np
from practice_live.live_logic import accounting
from strategy.agent import TAIL
from .prepare import HERE,OUT,ROOT,save,sha
from .runtime import SignalFrame,SignalFile,Supervisor

WALLET='0x125778065a578d598D88bddaA3FdD635280d0E47'

def info(payload):
    req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=5) as r:return json.load(r)

def account():
    end=int(time.time()*1000);start=1792065731086
    payloads={'meta':{'type':'metaAndAssetCtxs','dex':'par'},
              'positions':{'type':'clearinghouseState','user':WALLET,'dex':'par'}}
    for key,kind in [('fills','userFillsByTime'),('funding','userFunding'),('transfers','userNonFundingLedgerUpdates')]:
        payloads[key]={'type':kind,'user':WALLET,'startTime':start,'endTime':end}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures={k:pool.submit(info,p) for k,p in payloads.items()}
        r={k:f.result() for k,f in futures.items()}
    if len(r['fills'])>=2000 or len(r['funding'])>=500 or len(r['transfers'])>=500:
        raise ValueError('Bounded observer history requires pagination; refusing incomplete accounting')
    marks={u['name']:float(c['markPx']) for u,c in zip(r['meta'][0]['universe'],r['meta'][1])}
    ledger=accounting({'initial_equity':1e6},r['fills'],r['funding'],r['transfers'],r['positions']['assetPositions'],marks,WALLET)
    if ledger['external_capital']!=0:raise ValueError('New capital requires an explicit performance re-anchor')
    symbols=json.loads((TAIL/'baseline.json').read_text())['protocol']['symbols']
    return {'time':r['positions']['time']/1000,'equity':ledger['equity'],
            'qty':[ledger['qty']['par:'+s] for s in symbols],'prices':[marks['par:'+s] for s in symbols],
            'raw':r,'ledger':ledger}

def delayed_producer(path,frame,delay):
    # Fault injection, explicitly not a claim to have rerun inference here.
    time.sleep(delay);frame.published_time=time.time();SignalFile(path).publish(frame)

def main():
    directory=OUT/'read_only_supervision';directory.mkdir(exist_ok=False)
    prepared=json.loads((OUT/'prepared.json').read_text())
    with np.load(OUT/'targets.npz') as z:
        frame=SignalFrame(int(z['times'][-1]),time.time(),z['candidate'][-1].tolist(),z['retention'][-1].tolist(),
                          z['sigma'][-1].tolist(),[0.]*10,[0.]*10,prepared['input_sha256'])
    with np.load(OUT/'inputs.npz') as z:frame.volume=z['volume'][-1].tolist();frame.trades=z['trades'][-1].tolist()
    f=SignalFile(directory/'signal.json');f.publish(frame)
    p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    s=Supervisor(p,f,directory/'decisions.jsonl')
    context=multiprocessing.get_context('spawn')
    worker=context.Process(target=delayed_producer,args=(f.path,frame,100));worker.start()
    started=time.monotonic();events=[];errors=[]
    try:
        for tick in range(5):
            due=started+tick*30
            time.sleep(max(0,due-time.monotonic()))
            try:
                a=account();e=s.observe(a);e['inference_worker_running']=worker.is_alive()
                events.append(e);save(directory/f'account_{tick}.json',a)
                print(json.dumps({'tick':tick,'equity':e['equity'],'model_pending':worker.is_alive(),'orders_sent':0}),flush=True)
            except Exception as exc:
                errors.append({'tick':tick,'error':str(exc)});print(json.dumps(errors[-1]),flush=True)
        worker.join(timeout=5)
    finally:
        if worker.is_alive():worker.terminate();worker.join()
    gaps=[b['at']-a['at'] for a,b in zip(events,events[1:])]
    result={'test':'Real public-account observation while separate process has injected 100-second signal-production delay',
        'observations':len(events),'observations_during_injected_delay':sum(e['inference_worker_running'] for e in events),
        'max_observation_gap_seconds':max(gaps) if gaps else None,'errors':errors,
        'passed':len(events)==5 and not errors and max(gaps)<45 and sum(e['inference_worker_running'] for e in events)>=3,
        'orders_sent':0,'actual_model_inference_in_this_test':False,'state_handoff':s.snapshot()}
    save(directory/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise ValueError('Independent supervision integration failed')

if __name__=='__main__':main()
