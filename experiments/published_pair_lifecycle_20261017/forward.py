"""Additive, sealed comparison using the existing public-only capture.

prepare reads metadata and source hashes only. evaluate cannot read the reserved
raw tapes before their original release. No capture, signer, wallet, or trading
service is started, replaced, or modified by this module.
"""
import argparse
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments.basis_carry_20261017.probe import asof
from experiments.basis_carry_20261017.score import score
from experiments.complementary_alpha_20261017.research import protected
from .portfolio import Portfolio
from .signals import PublishedSignals

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAPTURE=ROOT/'experiments/paired_venue_20261017/artifacts/forward'
OUT=HERE/'artifacts'
ARMS=['first_strategy','second_strategy','first_plus_hourly_bias','first_plus_direct']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')


class TapeWindow:
    """Bounded lookahead for arrival simulation; all decisions use as-of reads."""
    def __init__(self, events):
        self.events=iter(events);self.next=next(self.events,None)
        self.data={};self.signals=deque();self.streams={};self.last_seen=-1

    def ensure(self, now_ms, lookahead_ms=10000):
        until=now_ms+lookahead_ms
        while self.next is not None and self.next['received_ms']<=until:
            event=self.next;at=event['received_ms']
            if at<self.last_seen:raise ValueError('Out-of-order capture event')
            self.last_seen=at;channel=event['channel'];data=event['data']
            if channel=='strategySignal':self.signals.append(event)
            elif channel in ('l2Book','activeAssetCtx'):
                symbol=data['coin'].split(':')[-1]
                stream=self.data.setdefault(symbol,{'book':deque(),'ctx':deque()})
                field='book' if channel=='l2Book' else 'ctx'
                stream[field].append((at,0,data if field=='book' else data['ctx']))
                # The initial warm-up may traverse many hours; prune while
                # reading rather than materializing the whole tape in RAM.
                for rows in stream.values():
                    while len(rows)>1 and rows[1][0]<at-30000:rows.popleft()
            self.next=next(self.events,None)
        self.streams={}
        for symbol,fields in self.data.items():
            output={'resets':[]}
            for field,rows in fields.items():
                while len(rows)>1 and rows[1][0]<now_ms-20000:rows.popleft()
                output[field]=list(rows);output[field+'_times']=[r[0] for r in rows]
            self.streams[symbol]=output
        return self.streams

    def publications(self, now_ms):
        while self.signals and self.signals[0]['received_ms']<=now_ms:
            yield self.signals.popleft()


class FundingClock:
    def __init__(self, start, tape, rates, symbols, cfg, receipt_multiplier):
        self.last=start*1000;self.next=(start//3600+1)*3600000
        self.tape,self.rates,self.symbols,self.cfg=tape,rates,symbols,cfg
        self.receipt_multiplier=receipt_multiplier

    def __call__(self, at, account):
        if at<self.last:raise ValueError('Settlement clock moved backwards')
        while self.next<=at:
            bucket=self.next;prices=np.ones(len(self.symbols));rates=np.zeros(len(prices));observed=np.zeros(len(prices),bool)
            for j,symbol in enumerate(self.symbols):
                stream=self.tape.streams.get(symbol)
                prior=asof(stream,bucket-1,self.cfg) if stream else None
                rate=self.rates.get(bucket,{}).get(symbol)
                if prior is not None and rate is not None:
                    prices[j]=prior['oracle'];rates[j]=rate;observed[j]=True
                if any(abs(o.qty[j])>1e-12 for o in account.owners.values()) and not observed[j]:
                    raise ValueError(f'Missing settled funding or pre-settlement oracle: {symbol} {bucket}')
            cash=-account.physical.qty*prices*rates
            rates=np.where(cash>0,rates*self.receipt_multiplier,rates)
            account.charge_funding(bucket/1000,prices,rates,observed)
            self.next+=3600000
        self.last=at


def raw_events(directory):
    for path in sorted(Path(directory).glob('*.jsonl.gz')):
        with gzip.open(path,'rt') as handle:rows=[json.loads(line) for line in handle]
        yield from sorted(rows,key=lambda r:r['received_ms'])


def assert_ready(plan, now):
    if now<plan['outcome_release_unix']+60:
        raise ValueError('Reserved raw outcomes remain sealed until the original release plus 60 seconds')
    if sha(CAPTURE/'plan.json')!=plan['capture_plan_sha256']:
        raise ValueError('Original capture plan changed')
    for rel,digest in plan['sources'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen dependency changed: '+rel)


def prepare():
    capture=json.loads((CAPTURE/'plan.json').read_text())
    if time.time()>=capture['paper_start_unix']:
        raise ValueError('Cannot preregister after the common comparison start')
    if (OUT/'plan.json').exists():raise FileExistsError('Comparison already registered')
    direct=ROOT/'experiments/rolling_feature_pairs_20261017'
    hourly=ROOT/'experiments/hourly_refresh_pairs_20261017'
    sources=json.loads((direct/'artifacts/frozen.json').read_text())['sources'].copy()
    sources.update(capture['sources'])
    selection=json.loads((direct/'artifacts/selection.json').read_text())
    cfg=json.loads((ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json').read_text())
    historical=json.loads((hourly/'par_protocol.json').read_text())
    for key in ('dense_pairs','dense_policy','renewal_policy','rehedge_policy'):
        cfg[key]=historical[key]
    cfg['book_intent_lifetime_hours']=1
    cfg['funding_forecast_hours']=24
    cfg['scenarios']['base']['funding_receipt_multiplier']=1.
    cfg['scenarios']['stress']['funding_receipt_multiplier']=.5
    paths=[*HERE.glob('*.py'),HERE/'test_results.txt',direct/'artifacts/frozen.json',direct/'artifacts/selection.json',
           hourly/'par_protocol.json',hourly/'model_protocol.json',direct/'protocol.json',
           ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json',
           OUT/'entry_adapter_audit.json']
    for rel in ['experiments/shared_execution_pairs_20261017/account_v2.py',
                'experiments/shared_execution_pairs_20261017/book_session.py',
                'experiments/forecast_validity_pairs_20261017/book_session.py',
                'experiments/forecast_validity_pairs_20261017/book_plan.py',
                'experiments/basis_carry_20261017/probe.py','experiments/basis_carry_20261017/score.py',
                'experiments/cost_horizon_20261016/forward.py','experiments/cost_horizon_20261016/research.py',
                'replay_lab/common_comparison/engine.py','replay_lab/common_comparison/models.py',
                'replay_lab/ema_rebuild/engine.py','replay_lab/pullback/broker.py']:
        paths.append(ROOT/rel)
    for path in paths:sources[str(path.relative_to(ROOT))]=sha(path)
    for rel,digest in sources.items():
        if sha(ROOT/rel)!=digest:raise ValueError('Changed dependency before registration: '+rel)
    OUT.mkdir(exist_ok=True)
    plan=dict(created_at_unix=time.time(),paper_start_unix=capture['paper_start_unix'],
        capture_end_unix=capture['capture_end_unix'],outcome_release_unix=capture['outcome_release_unix'],
        capture_directory=str(CAPTURE.relative_to(ROOT)),capture_plan_sha256=sha(CAPTURE/'plan.json'),
        sources=sources,arms=ARMS,trading_protocol=capture['trading_protocol'],book_protocol=cfg,
        direct_spec=json.loads((direct/'protocol.json').read_text()),bias_spec=json.loads((hourly/'model_protocol.json').read_text()),
        model_latency_seconds=5,end_liquidation_seconds=600,step_seconds=30,protected=protected(),
        historical_selection=selection,candidate_status='diagnostic_only_historical_minimum_completed_pairs_gate_failed',
        forecast_rule='Use each captured source input digest, source ending hour, and receipt time. Fit direct coefficients only from the captured midnight publication. Never backfill a missing midnight fit. Bias refits at each received hourly source. Both add a fixed 5-second inference allowance.',
        execution_rule='One physical/owner-conserving ledger per arm. Common received-book IOC executor, explicit owner limits, per-server-book consumed depth, independent receipts and settlement before fills. Pair and base intents share packets. Partial or unknown receipts do not become atomic pair fills.',
        holding_rule='Source-fixed 24h forecast, original 3-sigma price loss anchor, six-hour renewal opportunity in final six hours, 168h hard holding cap. Rehedging pays actual delta fills; incomplete hedge changes unwind. Renewals cannot extend past common liquidation boundary. Current reverse depth only estimates future exit cost; actual exit must be observed.',
        comparison_rule='Same 1M capital and global limits. Direct versus first, second and hourly-bias companion under base/stress. PnL and Q use full account mark equity at common times; pair attribution is separate. No reselection from future outcomes.',
        success_rule=dict(positive_pnl=True,pnl_and_q_above_both_native_arms_both_scenarios=True,
            minimum_completed_pairs=3,minimum_daily_returns=5,no_fatal_data_or_execution_errors=True,no_residual_inventory=True),
        limitations=['Paper counterfactual depth fills, no actual exchange orders or own impact.',
            'Historical candle metrics and this book evaluation differ in receipt latency, observable depth and risk-clock granularity; historical profit does not carry over.',
            'The old capture does not archive reconnect markers; freshness and receipt/server clocks are enforced, not an invented connectivity history.',
            'Existing capture ends after the announced practice phase. Inactive par markets or coverage gaps invalidate evaluation; the window is not silently shortened.',
            'Historical selection failed its trade-count gate; prospective results cannot retroactively change that status.'],
        orders_sent=0,automatic_live_replacement=False,reserved_raw_read_during_preparation=False)
    save(OUT/'plan.json',plan)
    return plan


def metrics(arm, start, end):
    rows=arm.snapshots;fatal=list(arm.fatal)
    if not rows:return dict(valid=False,fatal=['no_equity_samples'],financial_score=None,
        net_pnl=None,completed_pairs=0,completed_days=0,interpretable_financial_score=None)
    if rows[0]['time']!=start or rows[-1]['time']!=end:fatal.append('missing_initial_or_terminal_equity')
    if any(b['time']-a['time']>60 for a,b in zip(rows,rows[1:])):fatal.append('equity_gap_over_60_seconds')
    by_time={r['time']:r['equity'] for r in rows};daily=[]
    for at in range(start+86400,end+1,86400):
        if at not in by_time or at-86400 not in by_time:fatal.append('missing_daily_sample')
        else:daily.append(dict(time=at,return_=by_time[at]/by_time[at-86400]-1))
    initial=arm.p['initial'];pnl=rows[-1]['equity']-initial
    filled=arm.account.physical.fills
    financial=score(pnl/initial,[r['return_'] for r in daily],
        trading_days=len({int(f['time']-43200)//86400 for f in filled}),start_sample=rows[0]['time']==start)
    sessions=[*arm.sessions,*([arm.pair] if arm.pair else [])]
    complete=sum(s.state=='closed' and s.was_balanced for s in sessions)
    residual=bool(np.any(abs(arm.account.physical.qty)>1e-10) or any(np.any(abs(o.qty)>1e-10) for o in arm.account.owners.values()))
    if residual:fatal.append('residual_inventory')
    if arm.account.unknown:fatal.append('unobserved_order_outcome')
    values=np.array([r['equity'] for r in rows]);mdd=float(np.max(1-values/np.maximum.accumulate(values)))
    return dict(valid=not fatal,fatal=sorted(set(fatal)),net_pnl=pnl,**financial,
        max_drawdown=mdd,fees=arm.account.physical.fees,funding=arm.account.physical.funding,
        completed_pairs=complete,residual_inventory=residual,daily=daily,
        fills=len(filled),maximum_ownership_error=arm.account.maximum_error,
        interpretable_financial_score=financial['financial_score'] if not fatal else None)


def evaluate(plan):
    assert_ready(plan,time.time())
    if (OUT/'results.json').exists():raise FileExistsError('Evaluation already completed')
    # Public settled funding is retrieved only after the sealed outcome release.
    from experiments.cost_horizon_20261016.forward import info
    start,end=plan['paper_start_unix'],plan['capture_end_unix'];rates={}
    for symbol in plan['trading_protocol']['symbols']:
        data=info(dict(type='fundingHistory',coin='par:'+symbol,startTime=start*1000,endTime=end*1000))
        if not isinstance(data,list) or len(data)>=500:raise ValueError('Incomplete settled funding response')
        save(OUT/f'funding_{symbol}.json',data)
        for row in data:
            at=int(row['time']);bucket=at//3600000*3600000
            if not 0<=at-bucket<1000:raise ValueError('Unexpected funding settlement timestamp')
            if symbol in rates.setdefault(bucket,{}):raise ValueError('Duplicate funding settlement')
            rates[bucket][symbol]=float(row['fundingRate'])
    tape=TapeWindow(raw_events(ROOT/plan['capture_directory']))
    bias=plan['bias_spec'];models={}
    for pair in bias['pairs']:
        models[tuple(pair),24]=json.loads((ROOT/bias['model_directory']/('_'.join(pair)+'_24.json')).read_text())
    reconstructor=PublishedSignals(plan['trading_protocol']['symbols'],bias['pairs'],models,
        plan['direct_spec'],bias,plan['model_latency_seconds'])
    arms={}
    for scenario,sc in plan['book_protocol']['scenarios'].items():
        for name in plan['arms']:
            clock=FundingClock(start,tape,rates,plan['trading_protocol']['symbols'],plan['book_protocol'],sc['funding_receipt_multiplier'])
            arms[name,scenario]=Portfolio(name,scenario,plan,clock)
    publication=None;forecasts=[]
    for now in range(start,end+1,plan['step_seconds']):
        streams=tape.ensure(now*1000)
        for event in tape.publications(now*1000):
            forecasts=reconstructor.ingest_file(ROOT,event);publication=event
        for arm in arms.values():arm.step(now,streams,publication,forecasts)
    rows=[]
    for (name,scenario),arm in arms.items():
        result=metrics(arm,start,end);rows.append(dict(arm=name,scenario=scenario,metrics=result))
        sessions=[*arm.sessions,*([arm.pair] if arm.pair else [])]
        payload=dict(metrics=result,snapshots=arm.snapshots,events=arm.events,fills=arm.account.physical.fills,
            owner_fills={k:o.fills for k,o in arm.account.owners.items()},funding=arm.account.physical.funding_events,
            pair_sessions=[dict(intent=s.intent,state=s.state,events=s.events,closed_reason=s.closed_reason) for s in sessions])
        with gzip.open(OUT/f'{name}-{scenario}.json.gz','wt') as handle:
            json.dump(payload,handle,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
    indexed={(r['arm'],r['scenario']):r['metrics'] for r in rows};checks=[]
    for scenario in plan['book_protocol']['scenarios']:
        candidate=indexed['first_plus_direct',scenario]
        checks.extend([candidate['valid'],candidate['net_pnl'] is not None and candidate['net_pnl']>0,candidate['completed_pairs']>=3,candidate['completed_days']>=5])
        for base in ['first_strategy','second_strategy']:
            control=indexed[base,scenario]
            checks.append(candidate['valid'] and control['valid'] and candidate['net_pnl']>control['net_pnl']
                and candidate['interpretable_financial_score'] is not None and control['interpretable_financial_score'] is not None
                and candidate['interpretable_financial_score']>control['interpretable_financial_score'])
    assert_ready(plan,time.time())
    save(OUT/'results.json',dict(rows=rows,success=all(checks),checks=checks,protected=protected(),
        actual_exchange_executions=False,orders_sent=0,automatic_live_replacement=False))
    save(OUT/'fit_audit.json',reconstructor.audit)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','evaluate']);args=parser.parse_args()
    if args.command=='prepare':
        p=prepare();print(json.dumps(dict(prepared=True,start=p['paper_start_unix'],end=p['capture_end_unix'],sources=len(p['sources']))))
    else:evaluate(json.loads((OUT/'plan.json').read_text()))
