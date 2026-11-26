"""Received-book evaluator with explicitly stressed worker cadence.

Copied from the paced evaluator; unavailable public input pauses decisions.
The original result remains available. Each pause has an explicit bounded-gap
audit and no invented prices; unavailable input never becomes an order. Actual
status publications observed 87.9/88.0seconds apart. This does not claim exact
past evaluation times. Registered90second cadence and0/45second phases are
both reported, with unchanged signals, positions, order latency and costs.
"""
import bisect,copy,math
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from . import book_replay as original
from experiments.basis_carry_20261017.probe import asof
from execution_revision_20261024.guarded_portfolio import adopt



def gap_audit(streams,symbols,cfg,pauses,start,end):
    """Post-run coverage only; future observations never enter a decision.

    A pause must be surrounded by received valid observations within one normal
    90-second worker interval. No intragap mark or price is interpolated.
    """
    def available(at):return all(asof(streams[s],int(at*1000),cfg) is not None for s in symbols)
    rows=[]
    for at in pauses:
        left=right=None
        for distance in range(1,91):
            if left is None and at-distance>=start and available(at-distance):left=at-distance
            if right is None and at+distance<=end and available(at+distance):right=at+distance
            if left is not None and right is not None:break
        span=right-left if left is not None and right is not None else None
        rows.append(dict(at=at,prior_received_valid_time=left,next_received_valid_time=right,
                         surrounding_span_seconds=span,bounded=span is not None and span<=90))
    return rows

LOOP_SECONDS=90
PHASE_SECONDS=0

def run_book_data_aware(streams,source,profile,scenario,start,end,seed,arm,funding):
    p=copy.deepcopy(seed)
    transform=None
    if arm=='fixed_entry_1h':p=adopt(p)
    elif arm=='size_only_2':
        from .size_study import ScaledPacket
        transform=ScaledPacket(2.,start)
    elif arm=='hourly_fresh':
        from .hourly_portfolio import adopt as adopt_hourly
        p=adopt_hourly(p)
    elif arm!='incumbent':raise ValueError(arm)
    v=original.BookVenue(streams,profile,scenario,start,end,funding);v.account=p.account
    a=p.account;a.executor=v;a.checkpoint=v.checkpoint;a.collect_fills=v.collect;a.refresh_market=v.refresh
    p.settlement=v.settle;p.sc['fee_bps']=scenario['fee_bps'];p.sc['depth_fraction']=scenario['depth_fraction']
    if p.pair is not None:raise ValueError('Active specialist needs its own forward renewal audit')
    for pair in p.sessions:pair.on_time=v.settle;pair.account=a;pair.owner=a.owners['pair']
    initial_fills=len(a.physical.fills);initial_events=len(p.events);initial_fees=a.physical.fees;initial_funding=a.physical.funding
    snapshots=[];missing=[];pauses=[];known=[r['received_at'] for r in source];risk_audit=[]
    clock=SimpleNamespace(time=lambda:v.now)
    with ExitStack() as stack:
        stack.enter_context(patch('competition_live.live_account.time',clock))
        stack.enter_context(patch('candidate_handoff.risk_priority.time',clock))
        while v.now<=end:
            now=v.now;v.settle(int(now*1000),a);marks=v.marks()
            i=bisect.bisect_right(known,now)-1;packet=source[i]['packet'] if i>=0 else None
            if packet is not None and now-packet['body']['frame']['source_time']>=21600:packet=None
            if packet is not None and transform is not None:packet=transform(packet,now)
            if marks is not None:
                eq=a.reconcile(marks);notional=float(abs(a.physical.qty*marks).sum())
                if packet is not None:
                    from replay_lab.common_comparison.engine import risk_breaches
                    breaches=risk_breaches(a.physical.qty,eq,marks,np.asarray(packet['body']['frame']['sigma']),profile['trading']['risk'])
                    if breaches:risk_audit.append(dict(time=now,limits=breaches))
                # Same idle-reserve convention; actual teammate returns excluded.
                snapshots.append(dict(time=now,equity=eq,qty=a.physical.qty.tolist(),marks=marks.tolist(),notional=notional,
                    fees=a.physical.fees-initial_fees,funding=a.physical.funding-initial_funding))
            else:missing.append(now)
            if now>=end:break
            if now>=start+PHASE_SECONDS:
                if all(asof(streams[s],int(now*1000),v.cfg) is not None for s in profile['trading']['symbols']):
                    p.step_candidate(now,streams,packet,profile)
                else:
                    pauses.append(now)
                # Reconciliation, data collection and the runner's sleep still
                # occur after a slow sequence of submissions. Do not retry an
                # incomplete rebalance one second after its final receipt.
                v.now=min(end,v.now+LOOP_SECONDS)
            else:
                v.now=min(end,start+PHASE_SECONDS)
    if not snapshots:raise ValueError('No valid mark snapshots')
    gap_checks=gap_audit(streams,profile['trading']['symbols'],v.cfg,pauses,start,end)
    v.failures.extend(dict(kind='extended_unobserved_input_gap',**g) for g in gap_checks if not g['bounded'])
    first,last=snapshots[0],snapshots[-1];pnl=last['equity']-first['equity']
    return dict(arm=arm,net_pnl=pnl,fee_increment=last['fees'],funding_increment=last['funding'],
        price_pnl=pnl+last['fees']-last['funding'],start_equity=first['equity'],end_equity=last['equity'],
        exposure_times=[first['time'],last['time']],snapshots=snapshots,orders=v.orders,
        fills=a.physical.fills[initial_fills:],events=p.events[initial_events:],funding=v.cash_records,
        missing_mark_times=missing,input_pauses=pauses,input_gap_audit=gap_checks,failures=v.failures,
        risk_violations=risk_audit,accounting_error=max(a.maximum_error,a.physical.max_error),
        pending_orders=len(a.pending_orders),unknown=list(a.unknown),fatal=p.fatal,
        conditional=True,official_Q=None,actual_orders_sent=0)
