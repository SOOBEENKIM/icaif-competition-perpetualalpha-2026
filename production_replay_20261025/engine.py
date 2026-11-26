"""Offline venue around unchanged production code, with explicit OHLC proxies.

All generated receipt records are SYNTHETIC. No exchange client is constructed.
This does not reconstruct queue priority, historical marks or official equity.
"""
import bisect,copy,hashlib,json,math
from collections import Counter
from contextlib import ExitStack
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from competition_live.profile import load_profile
from competition_live.live_account import LiveAccount
from competition_live.runtime import fingerprint
from candidate_handoff.portfolio import CandidatePortfolio
from candidate_handoff.entry_policy import NAME,summarize_candles
from execution_revision_20261024.portfolio import ExecutionPortfolio
from execution_revision_20261024.guarded_portfolio import GuardedPortfolio,GuardedAccount
from experiments.published_pair_lifecycle_20261017.signals import FRAME_KEYS
from replay_lab.common_comparison.engine import rounded,risk_values,risk_breaches
from .protocol import ROOT,DATA,OUT,save,sha


def packets(directory,profile):
    rows=json.loads((directory/'events.json').read_text())
    audit=json.loads((directory/'pair_audit.json').read_text())
    if not audit['can_use_base_only']:raise ValueError('Nonzero pair gate requires pair forecasts')
    raw={s:json.loads((directory/'raw'/f'pas_candles_{s}.json').read_text())['data'] for s in profile['trading']['symbols']}
    result=[]
    for row in rows:
        source=row['source_time'];available=row['time']/1000
        # Envelope reconstructed from previously audited immutable inputs.
        digest=fingerprint(dict(source=source,row=row))
        original_frame=dict(source_time=source,published_time=available,weights=row['weights'],retention=[1.]*10,
            sigma=row['sigma'],volume=row['volume'],trades=row['trades'],source_hash=digest)
        envelope=dict(channel='strategySignal',received_ms=row['time'],data=dict(frame=original_frame,
            incumbent=row['weights'],candidate=row['weights'],publication_sha256=hashlib.sha256(json.dumps({k:original_frame[k] for k in FRAME_KEYS},sort_keys=True,allow_nan=False).encode()).hexdigest()))
        history=summarize_candles(raw,profile['trading']['symbols'],source)
        history['received_at']=available
        frame=dict(source_time=source,source_hash=digest,symbols=profile['trading']['symbols'],
            anchor_weights=row['anchor'],protected=row['protected'],price_forecast_168h=row['forecast'],
            reference_funding_forecast_168h=row['carry'],sigma=row['sigma'],volume=row['volume'],trades=row['trades'],
            risk_aversion=5.,execution_policy=NAME,venue_history=history)
        original=dict(publication=envelope,forecasts=[],profile_hash=fingerprint(profile),candidate=profile['candidate'],published_at=available)
        body=dict(candidate='budget_lifecycle',profile_hash=fingerprint(profile),published_at=available,frame=frame,
            original_signal=original,core_revision='all_market_signed_core_v1',execution_policy=NAME,
            universe_audit=[dict(symbol=s,considered=True) for s in profile['trading']['symbols']])
        result.append(dict(body=body,sha256=fingerprint(body)))
    return result


class SyntheticVenue:
    """Causal observations and pessimistic future-minute fill assumption."""
    def __init__(self,d,profile,scenario,funding,start,end):
        self.d=d;self.profile=profile;self.sc=scenario;self.symbols=profile['trading']['symbols']
        self.now=float(start);self.end=end;self.gateway=self;self.rows={};self.orders=[];self.oid=0
        self.used={};self.funding=sorted(funding,key=lambda x:x['time']);self.fidx=bisect.bisect_right([f['time'] for f in self.funding],start*1000)
        self.account=None;self.audit=Counter();self.receipts=[];self.checkpoints=Counter()
        self.observe=None

    def index(self,at=None):return bisect.bisect_right(self.d['times'],(self.now if at is None else at)*1000)-1

    def marks(self,at=None):
        i=self.index(at)
        if i<0:raise ValueError('No completed valuation observation')
        marks=self.d['close'][i].copy();missing=~np.isfinite(marks)|(marks<=0)
        if self.account is not None and any(np.any(missing&(l.qty!=0)) for l in [self.account.physical,*self.account.owners.values()]):
            raise ValueError('Unpriced held asset')
        # Same explicit zero-notional sentinel as production; never executable.
        marks[missing]=0.
        return marks

    def stream(self,symbol):
        j=self.symbols.index(symbol);i=self.index();now=int(self.now*1000)
        price=float(self.d['close'][i,j])
        if not np.isfinite(price) or price<=0:return dict(ctx=[],ctx_times=[],book=[],book_times=[],resets=[])
        # A synthetic carried trade mark is explicit research input, not live data.
        ctx=dict(markPx=str(price),oraclePx=str(price),funding='0',synthetic=True,source_close_ms=int(self.d['times'][i]),source_age_ms=float(self.d['age_ms'][i,j]))
        result=dict(ctx=[(now,0,ctx)],ctx_times=[now],book=[],book_times=[],resets=[])
        if self.d['active'][i,j] and now-self.d['times'][i]<=60000:
            size=float(self.d['volume'][i,j])*self.sc['participation']
            levels=[[dict(px=str(price*.9998),sz=str(size),n=1)],[dict(px=str(price*1.0002),sz=str(size),n=1)]]
            result.update(book=[(now,0,dict(coin='pas:'+symbol,time=now,levels=levels,synthetic=True))],book_times=[now])
        return result

    def streams(self):return {s:self.stream(s) for s in self.symbols}

    def market(self,coin):
        j=self.symbols.index(coin[4:]);return dict(name=coin,szDecimals=self.profile['trading']['execution']['decimals'][j]),dict(markPx=str(self.marks()[j]))

    def set_risk_state(self,qty,sigma):self.risk_qty=np.array(qty);self.risk_sigma=np.array(sigma)

    def submit(self,order):return self.gateway.send(order)

    def send(self,order):
        self.oid+=1;oid=self.oid;sent=self.now;j=self.symbols.index(order.coin[4:]);side=1 if order.is_buy else -1
        previous=self.index();arrival=(int(self.now)//60+1)*60
        if arrival>=self.end:
            from bot.execution import OrderRejected
            raise OrderRejected('Simulation boundary: no new receipt at/after end')
        self.now=float(arrival)
        self.settle(arrival*1000,self.account)
        if self.observe is not None:self.observe(arrival)
        i=self.index();assert self.d['times'][i]/1000==arrival and i>previous
        limit=float(order.price);quantity=0.;px=None;reason='future_volume_unavailable'
        key=(i,j)
        if self.d['active'][i,j] and self.d['active'][previous,j]:
            px=float(self.d['high' if side>0 else 'low'][i,j])*(1+side*self.sc['impact_bps']/10000)
            reason='limit_not_crossed'
            if side*(px-limit)<=1e-10:
                capacity=min(float(self.d['volume'][previous,j]),float(self.d['volume'][i,j]))*self.sc['participation']
                remaining=max(0.,capacity-self.used.get(key,0.))
                quantity=abs(rounded(min(float(order.size),remaining),self.profile['trading']['execution']['decimals'][j]))
                self.used[key]=self.used.get(key,0.)+quantity
                reason='conditional_fill' if quantity else 'volume_budget_consumed'
        rows=[]
        if quantity:
            fee=quantity*px*self.sc['fee_bps']/10000
            rows=[dict(coin=order.coin,tid=f'SIM-{oid}',oid=oid,feeToken='USDP',side='B' if side>0 else 'A',
                sz=str(quantity),px=str(px),fee=str(fee),time=arrival*1000,synthetic=True)]
        self.rows[oid]=rows
        self.orders.append(dict(sent=sent,arrival=arrival,symbol=order.coin,side=side,requested=float(order.size),limit=limit,
            filled=quantity,price=px,reason=reason,reduce_only=order.reduce_only,synthetic=True))
        self.audit[reason]+=1
        return dict(filled=dict(oid=oid,totalSz=str(quantity)))

    def collect(self,oid,started_ms,reported):return self.rows[oid]

    def checkpoint(self,kind,body):self.checkpoints[kind]+=1

    def settle(self,at_ms,account):
        if account is None:raise ValueError('No simulated account attached')
        while self.fidx<len(self.funding) and self.funding[self.fidx]['time']<=at_ms:
            f=self.funding[self.fidx];self.fidx+=1;j=self.symbols.index(f['coin'][4:])
            own=account.physical.qty[j]
            if not own:continue
            unit=f.get('cash_per_unit')
            if unit is None:
                if float(f['rate'])!=0:raise ValueError('Missing settled funding cash '+str(f))
                unit=0.
            amount=float(unit)*own
            if amount>0:amount*=self.sc['receipt_multiplier']
            account.apply_actual_funding([dict(time=f['time'],delta=dict(coin=f['coin'],usdc=str(amount),szi=str(own)),synthetic=True)])
            self.receipts.append(dict(time=f['time']/1000,coin=f['coin'],qty=own,cash=amount,synthetic=True))


def build(venue,arm):
    from .hourly_portfolio import HourlyPortfolio
    profile=venue.profile
    cls={'incumbent':CandidatePortfolio,'refresh_entry_1h':ExecutionPortfolio,'fixed_entry_1h':GuardedPortfolio,'hourly_fresh':HourlyPortfolio}[arm]
    account_cls=GuardedAccount if arm in ('fixed_entry_1h','hourly_fresh') else LiveAccount
    account=account_cls(profile,venue,venue.checkpoint,venue.collect,refresh_market=venue.stream)
    venue.account=account
    old=json.loads((ROOT/'competition_live/strategy_plan.json').read_text())
    plan={k:copy.deepcopy(old[k]) for k in ['book_protocol','direct_spec','bias_spec','model_latency_seconds','step_seconds']}
    plan.update(trading_protocol=copy.deepcopy(profile['trading']),paper_start_unix=profile['start_unix'],
        capture_end_unix=profile['end_unix'],end_liquidation_seconds=profile['execution']['end_liquidation_seconds'])
    p=cls('first_plus_decision_gate','base',plan,venue.settle);p.account=account;p.execution_policy=NAME
    # Fees belong to venue receipts; risk previews use the same scenario rate.
    p.sc['fee_bps']=venue.sc['fee_bps'];p.sc['depth_fraction']=1.
    return p


def metrics(snapshots,initial=100000.):
    eq=np.array([r['team_equity'] for r in snapshots]);final=snapshots[-1]
    daily=[r for r in snapshots if r['time']%86400==43200]
    returns=np.diff([r['team_equity'] for r in daily])/np.array([r['team_equity'] for r in daily[:-1]])
    sigma=float(np.std(returns,ddof=1)*np.sqrt(len(returns))) if len(returns)>=2 else None
    R=float(eq[-1]/initial-1);q=(max(R,0)+.005)/(abs(R)+1.5*sigma+.05) if sigma is not None else None
    return dict(net_pnl=float(eq[-1]-initial),R=R,sigma=sigma,Q=q,score=None if q is None else .8*q,
        max_drawdown=float(np.max(1-eq/np.maximum.accumulate(eq))),fees=final['fees'],funding=final['funding'],
        price_pnl=float(eq[-1]-initial+final['fees']-final['funding']),
        mean_notional=float(np.mean([r['notional'] for r in snapshots])),end_notional=final['notional'],
        exit_reserve_pnl=final['exit_reserve_equity']-initial,completed_daily_returns=len(returns))


def run(d,profile,scenario,funding,publications,start,end,arm='incumbent',switch=None,venue_class=SyntheticVenue,packet_transform=None):
    venue=venue_class(d,profile,scenario,funding,start,end)
    portfolio=build(venue,'incumbent' if switch else arm)
    snapshots=[];available=[p['body']['published_at'] for p in publications];lastsnap=None
    adopted=False;risk_audit=[]
    clock=SimpleNamespace(time=lambda:venue.now)
    def snapshot(at):
        nonlocal lastsnap
        marks=venue.marks(at);a=portfolio.account;eq=a.reconcile(marks);notional=float(abs(a.physical.qty*marks).sum())
        snapshots.append(dict(time=int(at),equity=eq,team_equity=eq+profile['reserved_capital'],notional=notional,
            cash=a.physical.cash,fees=a.physical.fees,funding=a.physical.funding,qty=a.physical.qty.tolist(),
            mark_proxy_ages=d['age_ms'][venue.index(at)].tolist(),
            exit_reserve_equity=eq+profile['reserved_capital']-notional*(scenario['fee_bps']+scenario['impact_bps']+2)/10000))
        lastsnap=at
    def observe(at):
        if at%3600==0 and (lastsnap is None or at>lastsnap):snapshot(at)
    venue.observe=observe
    with ExitStack() as stack:
        stack.enter_context(patch('competition_live.live_account.time',clock))
        stack.enter_context(patch('candidate_handoff.risk_priority.time',clock))
        snapshot(start)
        while venue.now<end:
            now=int(venue.now);venue.settle(now*1000,portfolio.account)
            observe(now)
            if switch and now>=switch and not adopted:
                from .hourly_portfolio import HourlyPortfolio
                before=portfolio.account.physical.qty.copy()
                portfolio.__class__={'fixed_entry_1h':GuardedPortfolio,'refresh_entry_1h':ExecutionPortfolio,'hourly_fresh':HourlyPortfolio}[arm]
                if arm in ('fixed_entry_1h','hourly_fresh'):portfolio.account.__class__=GuardedAccount
                portfolio.base_task=None;adopted=True
                assert np.array_equal(before,portfolio.account.physical.qty)
                portfolio.events.append(dict(time=now,kind='simulation_same_inventory_switch',arm=arm))
            i=bisect.bisect_right(available,now)-1
            packet=publications[i] if i>=0 and now-publications[i]['body']['frame']['source_time']<21600 else None
            if packet is not None and packet_transform is not None:packet=packet_transform(packet,now)
            if packet:
                before=len(portfolio.account.physical.fills)
                portfolio.step_candidate(now,venue.streams(),packet,profile)
                if len(portfolio.account.physical.fills)>before:
                    a=portfolio.account;m=venue.marks();eq=a.reconcile(m);s=np.array(packet['body']['frame']['sigma'])
                    violations=risk_breaches(a.physical.qty,eq,m,s,profile['trading']['risk'])
                    if violations:risk_audit.append(dict(time=venue.now,violations=violations))
            else:portfolio.step_candidate(now,venue.streams(),None,profile)
            # Snapshot at actual final receipt time; never backdate post-fill cash.
            venue.now=min(end,max(now+60,(math.floor(venue.now/60)+1)*60))
        venue.settle(end*1000,portfolio.account)
        if snapshots[-1]['time']!=end:snapshot(end)
    # Daily12UTC points must not be skipped by sequential IOC latency.
    if any(not any(x['time']==t for x in snapshots) for t in range(start,end+1,86400)):
        raise ValueError('Simulation skipped a daily accounting boundary')
    return dict(arm=arm,switch=switch,metrics=metrics(snapshots),snapshots=snapshots,
        fills=portfolio.account.physical.fills,events=portfolio.events,orders=venue.orders,funding=venue.receipts,
        audit=dict(synthetic=True,actual_orders_sent=0,risk_violations=risk_audit,unknown=list(portfolio.account.unknown),
            pending=len(portfolio.account.pending_orders),status=dict(venue.audit),checkpoints=dict(venue.checkpoints)),
        terminal_owners={k:dict(qty=v.qty.tolist(),cash=v.cash) for k,v in portfolio.account.owners.items()})
