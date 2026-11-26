"""Prespecified three-arm public-book comparison, sealed until Oct20 09:00 KST.

Uses the existing read-only capture without editing it. No signing client or
real order interface. Imported Paper only constructs counterfactual orders.
"""
import argparse,gzip,json,time
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
from .research import HERE,ROOT,OUT,sha,save,protected
from .policy import AdjustmentController
from .metrics import official_metrics
from experiments.execution_retry_20261016.engine import Paper
from experiments.cost_horizon_20261016.forward import info
from experiments.risk_preserving_20261016.runtime import SignalFrame
from replay_lab.common_comparison.engine import risk_breaches

SOURCE=ROOT/'experiments/cost_horizon_20261016/artifacts/forward'
DEST=OUT/'prospective'
START=int(datetime(2026,10,17,12,tzinfo=timezone.utc).timestamp())
END=int(datetime(2026,10,19,12,tzinfo=timezone.utc).timestamp())
RELEASE=int(datetime(2026,10,20,0,tzinfo=timezone.utc).timestamp())
ARMS={'incumbent':('point',1.),'scale_only_control':('point',4.),'staged_candidate':('partial_6h',4.)}


class StrictPaper(Paper):
    def apply_pending(self,now,books):
        ready=[];waiting=[]
        for order in self.pending:
            book=books.get(order['coin'])
            if now<order['arrives']:
                waiting.append(order)
            elif book and book['received_ms']>=order['arrives']:
                ready.append(order)
            elif now<=order['arrives']+5000:
                waiting.append(order)
            else:
                self.attempts.append({**order,'status':'unfilled_no_new_arrival_book'})
        self.pending=ready
        super().apply_pending(now,books)
        self.pending.extend(waiting)


def prepare():
    DEST.mkdir(exist_ok=True)
    if (DEST/'plan.json').exists():raise FileExistsError('Prospective plan already frozen')
    if time.time()>=START:raise ValueError('Cannot register this comparison after observation starts')
    # Plans and capture health are permitted; raw observations stay sealed.
    source_plan=json.loads((SOURCE/'plan.json').read_text())
    health=json.loads((SOURCE/'health.json').read_text())
    if time.time()-health['at_unix']>90:raise ValueError('Public collector is stale')
    selection=json.loads((OUT/'selection.json').read_text())
    assert selection['selected']=='partial_6h_x4'
    protected()
    sources={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}
    sources.update({str(ROOT/'experiments/execution_retry_20261016/engine.py'):sha(ROOT/'experiments/execution_retry_20261016/engine.py')})
    trading=json.loads(json.dumps(source_plan['trading_protocol']))
    trading['execution']['decimals']=[source_plan['metadata'][c]['szDecimals'] for c in source_plan['coins']]
    plan=dict(created_at_unix=time.time(),start_unix=START,end_unix=END,release_unix=RELEASE,
        source_capture_plan_sha256=sha(SOURCE/'plan.json'),source_capture=str(SOURCE.relative_to(ROOT)),
        selection_sha256=sha(OUT/'selection.json'),arms=ARMS,initial_equity_each=1e6,
        metadata=source_plan['metadata'],coins=source_plan['coins'],trading_protocol=trading,
        execution_protocol=source_plan['execution_protocol'],sources=sources,
        accounting='Mark equity at 12:00 UTC, actual settled oracle-based funding, fees, visible book depth, next-received-book latency; no mid-price fills',
        comparison='Incumbent vs scale-only vs staged execution. Report Q and return, sigma, MDD, fill capacity and unfilled targets; same initial capital and hard limits.',
        limits='Two completed daily returns are enough only to compute the published statistic, not establish stable profitability. Displayed-depth counterfactual fills do not establish actual execution or impact.',
        state='registered_using_running_public_collector',raw_outcomes_read=False,orders_sent=0,automatic_promotion=False)
    save(DEST/'plan.json',plan)
    print('Registered future comparison; evaluation remains sealed',flush=True)


def events(plan):
    if time.time()<plan['release_unix']:raise ValueError('Future observations remain sealed')
    for path in sorted(SOURCE.glob('*.jsonl.gz')):
        bucket=int(datetime.strptime(path.name[:11],'%Y%m%d_%H').replace(tzinfo=timezone.utc).timestamp())
        if bucket+3600<plan['start_unix'] or bucket>plan['end_unix']:continue
        with gzip.open(path,'rt') as f:rows=[json.loads(line) for line in f]
        yield from sorted(rows,key=lambda row:row['received_ms'])


def evaluate(plan):
    if time.time()<plan['release_unix']:raise ValueError('Evaluation is sealed until 2026-10-20 09:00 KST')
    if (DEST/'results.json').exists():raise FileExistsError('Prospective results already exist')
    if sha(SOURCE/'plan.json')!=plan['source_capture_plan_sha256']:raise ValueError('Source capture changed')
    for path,h in plan['sources'].items():
        if sha(ROOT/path)!=h:raise ValueError('Frozen evaluator changed: '+path)
    start=plan['start_unix']*1000;end=plan['end_unix']*1000;coins=plan['coins'];p=plan['trading_protocol']
    settlements=[]
    for coin in coins:
        rows=info({'type':'fundingHistory','coin':coin,'startTime':start,'endTime':end})
        if not isinstance(rows,list) or len(rows)>=500:raise ValueError('Incomplete funding response')
        save(DEST/('funding_'+coin.replace(':','_')+'.json'),rows)
        expected=set(range(start//3600000+1,end//3600000+1))
        found={r['time']//3600000 for r in rows if start<r['time']<=end}
        # A subsecond settlement after the terminal instant is not owned.
        expected.discard(end//3600000)
        if not expected.issubset(found):raise ValueError('Missing hourly settlement: '+coin)
        settlements.extend((r['time'],coin,float(r['fundingRate'])) for r in rows if start<r['time']<=end)
    settlements.sort();cursor=0
    arms={(name,scenario):StrictPaper('baseline',plan['execution_protocol'],sc,plan['metadata'])
          for name in plan['arms'] for scenario,sc in plan['execution_protocol']['scenarios'].items()}
    controls={key:AdjustmentController(plan['arms'][key[0]][0]) for key in arms}
    peak={key:1e6 for key in arms};halted={key:False for key in arms};books={};contexts={};signal=None
    decisions=set();last_risk=-1;last_snapshot=-1
    for event in events(plan):
        now=event['received_ms']
        if now>end:break
        # Apply funding before the new event: only previously received oracle
        # observations may price a settlement.
        while cursor<len(settlements) and settlements[cursor][0]<=now:
            when,coin,rate=settlements[cursor]
            for arm in arms.values():
                qty=arm.qty.get(coin,0.)
                if abs(qty)>1e-12:
                    ctx=contexts.get(coin)
                    if not ctx or not 0<=when-ctx['received_ms']<=15000:
                        arm.fatal.append(f'Missing causal settlement oracle {when}:{coin}')
                    else:
                        amount=-qty*float(ctx['ctx']['oraclePx'])*rate;arm.cash+=amount;arm.funding+=amount
                        arm.funding_events.append(dict(time=when,coin=coin,qty=qty,payment=amount))
            cursor+=1
        channel,d=event['channel'],event['data']
        if channel=='l2Book':
            books[d['coin']]={'received_ms':now,'data':d}
            for arm in arms.values():arm.depth.update(d['coin'],d)
        elif channel=='activeAssetCtx':contexts[d['coin']]={'received_ms':now,'ctx':d['ctx']}
        elif channel=='strategySignal':
            frame=SignalFrame(**d['frame']);frame.validate()
            if frame.published_time*1000>now+1:raise ValueError('Future publication')
            signal=d
        if now<start:continue
        if now-last_risk>=30000 and signal and all(c in contexts and 0<=now-contexts[c]['received_ms']<=15000 for c in coins):
            frame=signal['frame'];cov=np.array(frame['sigma']);prices=np.array([float(contexts[c]['ctx']['markPx']) for c in coins]);hour=now//3600000*3600
            usable=0<=now/1000-frame['source_time']<=6*3600
            if usable:
                for key,arm in arms.items():
                    eq=arm.equity(contexts);qty=np.array([arm.qty.get(c,0.) for c in coins]);peak[key]=max(peak[key],eq)
                    halted[key]|=eq/peak[key]-1<=-p['risk']['drawdown_halt']
                    forced=halted[key] or bool(risk_breaches(qty,eq,prices,cov,p['risk']))
                    boundary=now//21600000*21600000
                    normal=now-boundary<=900000 and frame['source_time']==hour and (key,boundary) not in decisions
                    if (not normal and not forced) or arm.pending:continue
                    when=now//1000 if forced else boundary//1000
                    raw=np.array(frame['weights'])*plan['arms'][key[0]][1]
                    action=controls[key].propose(when,prices,qty,eq,raw,cov,np.array(frame['volume']),np.array(frame['trades']),p,
                        buffer=.2,halted=halted[key],retention=np.array(frame['retention']))
                    if action is not None:
                        arm.request_target(dict(zip(coins,action['target_qty'])),now,now+900000,f'{key[0]}:{when}')
                        if normal:decisions.add((key,boundary))
            last_risk=now
        for arm in arms.values():arm.tick(now,books,contexts)
        if now-last_snapshot>=10000:
            for arm in arms.values():arm.snapshot(now,contexts)
            last_snapshot=now
    results=[]
    for key,arm in arms.items():
        if not arm.equity_series:raise ValueError('Missing equity series')
        if end-arm.equity_series[-1]['time']>15000:arm.fatal.append('Stale terminal valuation')
        gaps=np.diff([s['time'] for s in arm.equity_series])
        if len(gaps) and max(gaps)>60000:arm.fatal.append('Valuation gap above 60 seconds')
        # Initial cash is exact. Subsequent daily snapshots use an as-of mark
        # at the boundary, with at most 15 seconds of staleness.
        daily=[dict(time=start//1000,equity=1e6)]
        for boundary in range(start+86400000,end+1,86400000):
            rows=[s for s in arm.equity_series if 0<=boundary-s['time']<=15000]
            if not rows:arm.fatal.append(f'Missing daily equity {boundary}')
            else:daily.append(dict(time=boundary//1000,equity=rows[-1]['mark_equity']))
        result=arm.result(contexts);result.update(arm=key[0],scenario=key[1])
        result['official_formula']=official_metrics(dict(snapshots=daily,fills=[dict(time=f['time']//1000) for f in arm.fills])) if len(daily)==3 else None
        results.append(result)
        with gzip.open(DEST/f'{key[0]}-{key[1]}.json.gz','wt') as f:json.dump(dict(summary=result,equity=arm.equity_series,fills=arm.fills,attempts=arm.attempts,funding=arm.funding_events),f)
    save(DEST/'results.json',dict(rows=results,orders_sent=0,automatic_promotion=False,
        independent_after_registration=True,official_submitted_score=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','evaluate']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:evaluate(json.loads((DEST/'plan.json').read_text()))


if __name__=='__main__':main()
