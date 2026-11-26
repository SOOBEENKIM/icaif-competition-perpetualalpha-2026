"""Fresh public-book capture and later sealed three-arm paper comparison.

No wallet configuration, private key, signing client, or exchange endpoint.
The live strategy and the previously reserved trial remain untouched.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip,hashlib,json,os,shutil,threading,time,urllib.request
from pathlib import Path
import numpy as np
import websocket
from experiments.execution_retry_20261016.engine import Paper
from experiments.performance_attribution_20261016.live_signal import read_publication
from replay_lab.ema_rebuild.engine import Controller
from replay_lab.common_comparison.engine import risk_breaches
from .research import HERE,ROOT,OUT,sha,save,SecondController
from .band import InwardBandController

DEST=OUT/'forward';NAMES=('first_strategy','second_strategy','inward_band')
END=1792454400  # 2026-10-20 00:00 UTC / 09:00 KST


def info(payload):
    if payload['type'] not in ('metaAndAssetCtxs','fundingHistory'):raise ValueError('Read-only allowlist')
    req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)


def atomic(path,value):
    temp=Path(str(path)+'.tmp');save(temp,value);os.replace(temp,path)


def prepare():
    DEST.mkdir(exist_ok=True)
    if (DEST/'plan.json').exists():raise FileExistsError('Forward plan already exists')
    metadata=info({'type':'metaAndAssetCtxs','dex':'par'})[0]
    meta={x['name']:x for x in metadata['universe']}
    frame,publication=read_publication()
    from strategy.agent import TAIL
    trading=json.loads((TAIL/'baseline.json').read_text())['protocol']
    coins=['par:'+s for s in trading['symbols']]
    assert set(coins)<=set(meta)
    protocol={'candidate':{'attempts_per_coin_phase':1,'retry_seconds':30,'initial_quote_drift_bps':10},
              'common':{'quote_limit_bps':10,'asset_notional_limit':150000,'minimum_order_notional':10},
              'scenarios':{'base':{'latency_seconds':1,'visible_depth_fraction':1.,'fee_bps':9},
                           'stress':{'latency_seconds':3,'visible_depth_fraction':.5,'fee_bps':12}}}
    sources={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}
    protected=json.loads((ROOT/'experiments/performance_attribution_20261016/deployment/deployment_manifest.json').read_text())['sources']
    sources.update(protected)
    for p in (ROOT/'experiments/execution_retry_20261016').glob('*.py'):sources[str(p.relative_to(ROOT))]=sha(p)
    plan={'frozen_at_unix':time.time(),'capture_start_unix':time.time(),'capture_end_unix':END,
          'paper_start_unix':(int(time.time())//21600+1)*21600,'arms':NAMES,'initial_equity_each':1000000,
          'metadata':meta,'coins':coins,'trading_protocol':trading,'execution_protocol':protocol,'sources':sources,
          'source':str(HERE.relative_to(ROOT)),'historical_candidates_not_proven_superior':True,
          'candidate_choice':'Fixed inward-band execution candidate; new forecast and diversification families failed historical return gates. Selection is known and disclosed before new data.',
          'success_rule':{'positive_base_roi':True,'higher_roi_than_both_baselines_base_and_stress':True,
                          'daily_std_and_mdd_no_worse':True,'daily_return_samples_minimum':3,'completed_lifecycles_minimum':4,
                          'zero_fatal_errors':True,'no_automatic_promotion':True},
          'sampling':'Per coin/channel at most one archived book/context event per second; receipt-time ordering. No hypothetical maker fills.',
          'limitations':'Counterfactual depth fills, not actual executions. No self-impact or liquidation engine. Practice DEX may stop or become inactive before end: missing/stale data invalidate the comparison rather than being filled forward.',
          'orders_sent':0,'signing_client_loaded':False}
    save(DEST/'plan.json',plan);return plan


def capture(plan):
    counters=Counter();last_emit={};last_received={};errors=[];lock=threading.Lock();stop=threading.Event();writers={}
    def emit(channel,data):
        now=time.time()
        if now>=plan['capture_end_unix']:return
        coin=data.get('coin','signal');key=channel+':'+coin
        if channel!='strategySignal' and now-last_emit.get(key,0)<1:return
        last_emit[key]=now
        row={'received_ms':int(now*1000),'channel':channel,'data':data}
        bucket=datetime.fromtimestamp(now,timezone.utc).strftime('%Y%m%d_%H')
        with lock:
            for old in list(writers):
                if old!=bucket:writers.pop(old).close()
            if bucket not in writers:writers[bucket]=gzip.open(DEST/(bucket+'.jsonl.gz'),'at')
            writers[bucket].write(json.dumps(row,separators=(',',':'))+'\n');counters[channel]+=1;last_received[key]=now
    def signals():
        previous=-1
        while not stop.is_set() and time.time()<plan['capture_end_unix']:
            try:
                frame,obj=read_publication()
                if frame.source_time!=previous:
                    emit('strategySignal',{'frame':obj['frame'],'incumbent':obj['attribution']['incumbent'],'candidate':obj['attribution']['candidate'],
                                          'source_run':obj['source_run'],'publication_sha256':obj['sha256']})
                    previous=frame.source_time
            except Exception as exc:errors.append({'source':'signal','error':str(exc),'at':time.time()})
            stop.wait(5)
    worker=threading.Thread(target=signals,daemon=True);worker.start();socket=None;last_ping=last_health=0
    try:
        while time.time()<plan['capture_end_unix']:
            if shutil.disk_usage(DEST).free<5*1024**3:raise RuntimeError('Disk reserve reached')
            try:
                if socket is None:
                    socket=websocket.create_connection('wss://api.hyperliquid-testnet.xyz/ws',timeout=5)
                    for coin in plan['coins']:
                        for channel in ('l2Book','activeAssetCtx'):
                            socket.send(json.dumps({'method':'subscribe','subscription':{'type':channel,'coin':coin}}))
                if time.time()-last_ping>=20:socket.send(json.dumps({'method':'ping'}));last_ping=time.time()
                message=json.loads(socket.recv())
                if message.get('channel') in ('l2Book','activeAssetCtx'):emit(message['channel'],message['data'])
                elif message.get('channel')=='error':raise ValueError(str(message.get('data')))
            except websocket.WebSocketTimeoutException:pass
            except Exception as exc:
                errors.append({'source':'websocket','error':str(exc),'at':time.time()})
                if socket:socket.close()
                socket=None;stop.wait(2)
            if time.time()-last_health>=15:
                with lock:
                    for writer in writers.values():writer.flush()
                    atomic(DEST/'health.json',{'state':'capturing_raw_only','at_unix':time.time(),'counts':dict(counters),
                        'last_received':last_received,'recent_errors':errors[-10:],'paper_start_unix':plan['paper_start_unix'],
                        'end_unix':plan['capture_end_unix'],'orders_sent':0,'partial_performance_inspected':False})
                last_health=time.time()
    finally:
        stop.set();worker.join(timeout=6)
        if socket:socket.close()
        for writer in writers.values():writer.close()


def events():
    for path in sorted(DEST.glob('*.jsonl.gz')):
        with gzip.open(path,'rt') as f:rows=[json.loads(x) for x in f]
        yield from sorted(rows,key=lambda x:x['received_ms'])


def evaluate(plan):
    if time.time()<plan['capture_end_unix']:raise ValueError('Forward outcomes remain sealed until the declared end')
    for path,h in plan['sources'].items():
        if sha(ROOT/path)!=h:raise ValueError('Frozen evaluator or strategy source changed: '+path)
    start=int(plan['paper_start_unix']*1000);end=int(plan['capture_end_unix']*1000)
    def funding(coin):return coin,info({'type':'fundingHistory','coin':coin,'startTime':start,'endTime':end})
    rates=defaultdict(dict)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for coin,rows in pool.map(funding,plan['coins']):
            if not isinstance(rows,list) or len(rows)>=500:raise ValueError('Incomplete settled funding')
            save(DEST/('funding_'+coin.replace(':','_')+'.json'),rows)
            for row in rows:
                stamp=int(row['time']);bucket=stamp//3600000*3600000
                if stamp-bucket>=1000:raise ValueError('Funding timestamp outside subsecond tolerance')
                rates[bucket][coin]=float(row['fundingRate'])
    # Every expected hour is present in the processing clock, including hours
    # with missing API records; a held position then records a fatal data gap.
    funding_times=list(range((start//3600000+1)*3600000,end+1,3600000));cursor=0;coins=plan['coins'];p=plan['trading_protocol']
    arms={(name,scenario):Paper('baseline',plan['execution_protocol'],sc,plan['metadata']) for name in NAMES for scenario,sc in plan['execution_protocol']['scenarios'].items()}
    books={};contexts={};signal=None;decisions=set();controller_state={};peak={k:1e6 for k in arms};halted={k:False for k in arms}
    last_risk=-1;last_snapshot=-1
    for event in events():
        now=event['received_ms']
        if now>end:break
        while cursor<len(funding_times) and funding_times[cursor]<=now:
            when=funding_times[cursor]
            if when>=start:
                for arm in arms.values():arm.charge_funding(when,rates[when],contexts)
            cursor+=1
        channel,d=event['channel'],event['data']
        if channel=='l2Book':
            books[d['coin']]={'received_ms':now,'data':d}
            for arm in arms.values():arm.depth.update(d['coin'],d)
        elif channel=='activeAssetCtx':contexts[d['coin']]={'received_ms':now,'ctx':d['ctx']}
        elif channel=='strategySignal':signal=d
        if now<start:continue
        if now-last_risk>=30000 and signal and all(c in contexts and now-contexts[c]['received_ms']<=15000 for c in coins):
            frame=signal['frame'];cov=np.array(frame['sigma']);prices=np.array([float(contexts[c]['ctx']['markPx']) for c in coins]);hour=now//3600000*3600
            if 0<=hour-frame['source_time']<=6*3600:
                for key,arm in arms.items():
                    name,scenario=key;eq=arm.equity(contexts);q=np.array([arm.qty.get(c,0.) for c in coins])
                    peak[key]=max(peak[key],eq);halted[key]|=eq/peak[key]-1<=-p['risk']['drawdown_halt']
                    forced=halted[key] or bool(risk_breaches(q,eq,prices,cov,p['risk']))
                    boundary=now//21600000*21600000
                    normal=now-boundary<=900000 and frame['source_time']==hour and (key,boundary) not in decisions
                    if (not normal and not forced) or arm.pending:continue
                    when=now//1000 if forced else boundary//1000
                    if name=='first_strategy':control=Controller(**controller_state.get(key,{}));raw=signal['incumbent']
                    else:
                        cls=SecondController if name=='second_strategy' else InwardBandController
                        control=cls([when],[frame['retention']]);raw=signal['candidate']
                        if key in controller_state:control.controller.last_decision=controller_state[key]['last_decision']
                    action=control.propose(when,prices,q,eq,np.array(raw),cov,np.array(frame['volume']),np.array(frame['trades']),p,buffer=.2,halted=halted[key])
                    if action is not None:
                        last=control.last_decision;controller_state[key]={'last_decision':last}
                        arm.request_target(dict(zip(coins,action['target_qty'])),now,now+900000,f'{name}:{when}')
                        if normal:decisions.add((key,boundary))
            last_risk=now
        for arm in arms.values():arm.tick(now,books,contexts)
        if now-last_snapshot>=10000:
            for arm in arms.values():arm.snapshot(now,contexts)
            last_snapshot=now
    results=[]
    for (name,scenario),arm in arms.items():
        if not arm.equity_series:raise ValueError('No complete equity path')
        if end-arm.equity_series[-1]['time']>15000:arm.fatal.append('No fresh terminal valuation')
        gaps=[b['time']-a['time'] for a,b in zip(arm.equity_series,arm.equity_series[1:])]
        if gaps and max(gaps)>60000:arm.fatal.append('Valuation path gap exceeds 60 seconds')
        r=arm.result(contexts);r.update(arm=name,scenario=scenario)
        series=arm.equity_series;daily={};i=0
        for t in range((start//86400000+1)*86400000,end+1,86400000):
            while i+1<len(series) and series[i+1]['time']<=t:i+=1
            if 0<=t-series[i]['time']<=15000:daily[t]=series[i]['oracle_equity']
        dr=[daily[t]/daily[t-86400000]-1 for t in sorted(daily) if t-86400000 in daily]
        r.update(complete_daily_returns=len(dr),daily_sample_std_percent=float(np.std(dr,ddof=1)*100) if len(dr)>1 else None,
                 completed_lifecycles=sum(abs(f['qty']+sum(g['qty'] for g in arm.fills[:idx] if g['coin']==f['coin']))<1e-10 for idx,f in enumerate(arm.fills)))
        results.append(r)
        with gzip.open(DEST/f'result_{name}_{scenario}.json.gz','wt') as f:json.dump({'summary':r,'equity':series,'fills':arm.fills,'attempts':arm.attempts,'funding':arm.funding_events},f)
    indexed={(r['arm'],r['scenario']):r for r in results};checks=[]
    for scenario in ('base','stress'):
        c=indexed['inward_band',scenario]
        for name in ('first_strategy','second_strategy'):
            b=indexed[name,scenario]
            enough=min(c['complete_daily_returns'],b['complete_daily_returns'])>=3
            gates={'positive_base_roi':c['oracle_net_pnl']>0 if scenario=='base' else True,
                   'higher_roi':c['oracle_net_pnl']>b['oracle_net_pnl'],
                   'enough_daily_observations':enough,
                   'daily_std_no_worse':enough and c['daily_sample_std_percent']<=b['daily_sample_std_percent'],
                   'mdd_no_worse':c['max_drawdown_percent']<=b['max_drawdown_percent'],
                   'completed_lifecycles':c['completed_lifecycles']>=4,
                   'zero_fatal_errors':not c['fatal'] and not b['fatal'],
                   'no_unresolved_orders':c['pending_orders']==b['pending_orders']==0}
            checks.append({'scenario':scenario,'comparator':name,'gates':gates,'passed':all(gates.values())})
    save(DEST/'results.json',{'rows':results,'checks':checks,'all_gates_pass':all(c['passed'] for c in checks),
                             'independent_after_freeze':True,'orders_sent':0,'automatic_promotion':False,
                             'decision':'REVIEW_REQUIRED; no real strategy changes performed'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','capture','evaluate']);args=parser.parse_args()
    if args.action=='prepare':
        p=prepare();print(json.dumps({'prepared':True,'paper_start_unix':p['paper_start_unix'],'end_unix':p['capture_end_unix']}));return
    plan=json.loads((DEST/'plan.json').read_text())
    if args.action=='capture':capture(plan)
    evaluate(plan)


if __name__=='__main__':main()
