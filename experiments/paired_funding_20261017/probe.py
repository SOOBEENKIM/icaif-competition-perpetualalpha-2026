"""One prospectively registered 72h paired execution probe, no signing client."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import time
from pathlib import Path
import numpy as np
from experiments.cost_horizon_20261016 import forward as capture_module
from experiments.cost_horizon_20261016.research import save,sha
from experiments.paired_venue_20261017.forward import published_pairs,current_snapshots
from experiments.paired_venue_20261017.engine import quote_pair,PairExecution
from experiments.paired_venue_20261017.arrival import broker
from replay_lab.common_comparison.engine import risk_breaches
from .quote import request

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DEST=HERE/'forward_probe'
HOUR=3600000


def settlement_clock(start,end):return list(range((start//HOUR+1)*HOUR,end+1,HOUR))


class Probe:
    def __init__(self,plan):
        self.plan=plan;self.cfg=plan['paired_protocol'];self.attempted=False;self.last_risk=-1;self.last_entry_check=-1
        self.arms={s:broker(plan['metadata'],sc) for s,sc in self.cfg['scenarios'].items()}
        self.managers={s:PairExecution(p,self.cfg) for s,p in self.arms.items()}
        self.peaks={s:1e6 for s in self.arms};self.halted={s:False for s in self.arms}

    def step(self,now,books,contexts,signal):
        plan=self.plan;start=int(plan['paper_start_unix']*1000);end=int(plan['capture_end_unix']*1000)
        if now<start:return
        fresh_context=all(c in contexts and 0<=now-contexts[c]['received_ms']<=15000 for c in plan['coins'])
        fresh_signal=signal is not None and 0<=now//HOUR*3600-signal['frame']['source_time']<=21600
        if fresh_context and fresh_signal:
            ids=[plan['symbols'].index(c.split(':')[1]) for c in plan['coins']]
            sigma=np.array(signal['frame']['sigma'])[np.ix_(ids,ids)]
            prices=np.array([float(contexts[c]['ctx']['markPx']) for c in plan['coins']])
        else:sigma=prices=None
        if not self.attempted and now<=start+plan['entry_window_seconds']*1000 and now-self.last_entry_check>=1000:
            self.last_entry_check=now
            snapshots=current_snapshots(now,books,contexts,self.cfg)
            if fresh_context and fresh_signal and all(c in snapshots for c in plan['coins']):
                proposals={s:quote_pair(plan['coins'],snapshots,plan['metadata'],plan['beta'],plan['direction'],72,self.cfg,sc)
                           for s,sc in self.cfg['scenarios'].items()}
                good=all(q and q['predicted_net_bps']>self.cfg['entry_buffer_bps'] and
                    not risk_breaches(np.array(q['target_qty']),1e6,prices,sigma,plan['trading_protocol']['risk']) for q in proposals.values())
                if good:
                    for s,q in proposals.items():self.managers[s].enter(q,now)
                    self.attempted=True
        terminal=now>=end-plan['end_liquidation_seconds']*1000
        if now-self.last_risk>=30000 or terminal:
            for s,manager in self.managers.items():
                arm=self.arms[s]
                if terminal:manager.close(now,'observation_end')
                elif not fresh_context or not fresh_signal:manager.close(now,'missing_or_stale_risk_inputs')
                else:
                    equity=arm.equity(contexts);self.peaks[s]=max(self.peaks[s],equity)
                    self.halted[s]|=equity/self.peaks[s]-1<=-plan['trading_protocol']['risk']['drawdown_halt']
                    qty=np.array([arm.qty.get(c,0.) for c in plan['coins']])
                    if self.halted[s] or risk_breaches(qty,equity,prices,sigma,plan['trading_protocol']['risk']):manager.close(now,'account_risk')
            self.last_risk=now
        for manager in self.managers.values():manager.tick(now,books,contexts)


def prepare():
    DEST.mkdir(exist_ok=False)
    quote=json.loads(sorted((HERE/'artifacts').glob('current_*.json'))[-1].read_text())
    example=next(r['proposal'] for r in quote['rows'] if r['model']=='published_current_persistence' and r['scenario']=='stress'
                 and r['proposal']['coins']==['par:EWY','par:SP500'] and r['proposal']['horizon_hours']==72)
    if example['predicted_net_bps']<=5:raise ValueError('Frozen probe motivation no longer matches expected edge')
    existing=json.loads((ROOT/'experiments/paired_venue_20261017/artifacts/forward/plan.json').read_text())
    fresh=request(dict(type='metaAndAssetCtxs',dex='par'));metadata={r['name']:r for r in fresh['response'][0]['universe']}
    start=(int(time.time())//60+3)*60;end=start+72*3600+1200
    sources=dict(existing['sources'])
    sources.update(json.loads((HERE/'artifacts/frozen.json').read_text())['sources'])
    sources.update(json.loads((HERE/'artifacts/quote_frozen.json').read_text())['sources'])
    for f in (HERE/'probe.py',HERE/'test_probe.py'):sources[str(f.relative_to(ROOT))]=sha(f)
    p=dict(created_at=time.time(),capture_start_unix=time.time(),paper_start_unix=start,capture_end_unix=end,
        outcome_release_unix=end,coins=example['coins'],beta=example['beta'],direction=example['direction'],
        metadata=metadata,symbols=existing['trading_protocol']['symbols'],trading_protocol=existing['trading_protocol'],
        paired_protocol=existing['paired_protocol'],entry_window_seconds=600,end_liquidation_seconds=600,
        sources=sources,initial_each=1e6,selection='One fixed EWY-short/SP500-long opportunity, motivated by positive base and stress expected edge in the timestamped public snapshot. No other pairs or directions searched during the probe.',
        entry_rule='One simultaneous decision only when both scenario forecasts pass 5bp after costs and portfolio risk caps. Both legs fill independently from subsequent received books; missing depth is unfilled. No retry after an unfilled or completed pair.',
        success_rule='At least one balanced opening and completed closure in each scenario, positive fully closed net PnL, no fatal errors or residual inventory, complete observation coverage.',
        scope='Economic-mechanism probe, not a full strategy comparison or leaderboard Q evaluation. The separate seven-day first/second/pair trial remains the primary comparative test.',
        independence='All entry and exit observations occur after this registration. Calendar overlaps the other registered trial: this is not an independent second replication and does not permit changing its frozen strategy.',
        raw_outcomes_read=False,orders_sent=0,automatic_live_promotion=False)
    save(DEST/'metadata_receipt.json',fresh);save(DEST/'plan.json',p)
    print(json.dumps(dict(start=start,end=end,coins=p['coins'],sources=len(sources),orders_sent=0)))


def evaluate(plan):
    if time.time()<plan['outcome_release_unix']:raise ValueError('Probe outcomes sealed until registered end')
    if (DEST/'results.json').exists():return
    for path,digest in plan['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen dependency changed: '+path)
    start,end=[int(plan[k]*1000) for k in ('paper_start_unix','capture_end_unix')]
    def funding(coin):return request(dict(type='fundingHistory',coin=coin,startTime=start,endTime=end))
    rates=defaultdict(dict)
    with ThreadPoolExecutor(max_workers=2) as pool:receipts=list(pool.map(funding,plan['coins']))
    save(DEST/'funding_receipts.json',receipts)
    for coin,receipt in zip(plan['coins'],receipts):
        rows=receipt['response']
        if not isinstance(rows,list) or len(rows)>=500:raise ValueError('Incomplete funding')
        for r in rows:
            if int(r['time'])%HOUR>=1000:raise ValueError('Unexpected funding timestamp')
            rates[int(r['time'])//HOUR*HOUR][coin]=float(r['fundingRate'])
    probe=Probe(plan);books={};contexts={};signal=None;last_snapshot=-1;cursor=0
    settlements=settlement_clock(start,end);capture_module.DEST=DEST
    for event in capture_module.events():
        now=event['received_ms']
        if now>end:break
        while cursor<len(settlements) and settlements[cursor]<=now:
            at=settlements[cursor]
            for arm in probe.arms.values():arm.charge_funding(at,rates[at],contexts)
            cursor+=1
        channel,d=event['channel'],event['data']
        if channel=='l2Book':
            books[d['coin']]=dict(received_ms=now,data=d)
            for arm in probe.arms.values():arm.depth.update(d['coin'],d)
        elif channel=='activeAssetCtx':contexts[d['coin']]=dict(received_ms=now,ctx=d['ctx'])
        elif channel=='strategySignal':signal=d
        probe.step(now,books,contexts,signal)
        if now>=start and now-last_snapshot>=10000:
            for arm in probe.arms.values():arm.snapshot(now,contexts)
            last_snapshot=now
    output=[]
    for name,arm in probe.arms.items():
        series=arm.equity_series;manager=probe.managers[name]
        if not series:raise ValueError('No equity observations')
        if series[0]['time']-start>15000 or end-series[-1]['time']>15000:arm.fatal.append('Missing initial or terminal equity')
        if any(b['time']-a['time']>60000 for a,b in zip(series,series[1:])):arm.fatal.append('Equity observation gap above 60 seconds')
        r=arm.result(contexts);r.update(scenario=name,balanced_opens=sum(e['event']=='balanced_open' for e in manager.events),
             completed_pairs=sum(e['event']=='flat_confirmed' for e in manager.events),financial_score=None,leaderboard_rank=None)
        r['mechanism_gate_pass']=r['balanced_opens']>=1 and r['completed_pairs']>=1 and r['flat'] and not r['pending_orders'] and not r['fatal'] and r['mark_net_pnl']>0
        with gzip.open(DEST/f'result_{name}.json.gz','wt') as stream:
            json.dump(dict(summary=r,equity=series,fills=arm.fills,funding=arm.funding_events,attempts=arm.attempts,events=manager.events),stream,
                      allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
        output.append(r)
    save(DEST/'results.json',dict(rows=output,mechanism_supported=all(r['mechanism_gate_pass'] for r in output),
        existing_strategy_superiority_proven=False,orders_sent=0,automatic_live_promotion=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','capture','evaluate']);args=parser.parse_args()
    if args.action=='prepare':prepare();return
    plan=json.loads((DEST/'plan.json').read_text())
    if args.action=='capture':
        capture_module.DEST=DEST;capture_module.read_publication=published_pairs
        capture_module.capture(plan)
    evaluate(plan)


if __name__=='__main__':main()
