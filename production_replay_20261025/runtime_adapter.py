"""Unsigned staging adapter. Importing it never changes the running worker.

The old migration.wire resets the account class to LiveAccount. Candidate
guards therefore require an explicit, tested wire hook on EVERY reconnect.
This file prepares that hook; it does not install it or open a signer.
"""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from candidate_handoff.portfolio import CandidatePortfolio
from candidate_handoff.migration import wire as original_wire
from competition_live.live_account import LiveAccount
from execution_revision_20261024.guarded_portfolio import GuardedAccount,GuardedPortfolio,adopt as guard_adopt
from .catchup_portfolio import CatchupPortfolio,adopt as catchup_adopt
from .quote_guard import QuoteAccount,QuotePortfolio,adopt as quote_adopt
from .quote_history import QuoteHistoryPortfolio
from .quote_archive import merge_features
from .size_study import ScaledPacket
from .protocol import ROOT,sha


class BudgetPortfolio(CandidatePortfolio):
    execution_revision='same_hard_limits_anchor_budget_2_v1'
    def step_candidate(self,now,streams,packet,profile):
        if packet is not None:packet=self.budget_transform(packet,now)
        return super().step_candidate(now,streams,packet,profile)


class QuoteHistoryRuntime(QuoteHistoryPortfolio):
    execution_revision='received_pas_quote_history_one_hour_entry_v1'
    def step_candidate(self,now,streams,packet,profile):
        if packet is not None:
            source=packet['body']['frame']['source_time'];records=[];hashes={}
            directory=Path(self.quote_archive_directory).resolve()
            if not directory.is_relative_to(ROOT):raise ValueError('Quote archive must be in the project')
            for point in [source-86400,source-10800,source]:
                path=directory/f'{point}.json'
                if not path.exists():continue
                data=path.read_bytes();record=json.loads(data)
                if record['available_at']>now:continue
                if record['point']!=point:raise ValueError('Quote archive filename/cutoff mismatch')
                records.append(record);hashes[str(point)]=hashlib.sha256(data).hexdigest()
            streams=merge_features(streams,records,self.p['symbols'],source,self.cfg,now)
            self.last_quote_archive_evidence=dict(source=source,at=now,hashes=hashes)
        return super().step_candidate(now,streams,packet,profile)


def account_class(portfolio):
    # Exact known classes prevent a future unknown execution behavior from
    # being silently removed by a broad isinstance check.
    cls=type(portfolio)
    if cls in (QuotePortfolio,QuoteHistoryRuntime):return QuoteAccount
    if cls in (GuardedPortfolio,CatchupPortfolio):return GuardedAccount
    if cls in (CandidatePortfolio,BudgetPortfolio):return LiveAccount
    raise TypeError('Unknown staged portfolio class: '+cls.__name__)


def wire_candidate(portfolio,bridge,executor,checkpoint,collect_fills,refresh_market):
    expected=account_class(portfolio)
    original_wire(portfolio,bridge,executor,checkpoint,collect_fills,refresh_market)
    portfolio.account.__class__=expected
    if expected is QuoteAccount:portfolio.account.frozen_side_prices=None
    if expected is GuardedAccount:portfolio.account.base_entry_anchor=None
    if type(portfolio.account) is not expected:raise AssertionError('Candidate account guard lost')
    return portfolio


def state_vector(p):
    ledgers={}
    for name,a in [('physical',p.account.physical),('reserved',p.account.reserved),*p.account.owners.items()]:
        ledgers[name]={f:copy.deepcopy(getattr(a,f)) for f in ['qty','entry','cash','realized','fees','funding']}
    return dict(ledgers=ledgers,last_decision=p.last_decision,counter=p.counter,peak=p.peak,
                halted=p.halted,used=copy.deepcopy(p.used),fatal=copy.deepcopy(p.fatal))


def stage(portfolio,arm,*,quote_archive_directory=None):
    a=portfolio.account
    if a.pending_orders or a.unknown or portfolio.fatal or portfolio.halted:
        raise ValueError('Reconcile outstanding orders or halt before staging')
    if type(portfolio) is not CandidatePortfolio or type(a) is not LiveAccount:
        raise TypeError('Stage only the audited incumbent; do not override an unknown newer worker')
    before=state_vector(portfolio)
    if arm=='fixed_entry_1h':p=guard_adopt(portfolio)
    elif arm=='fresh_block_catchup':p=catchup_adopt(portfolio)
    elif arm in ('frozen_side_quote','quote_inputs'):
        p=quote_adopt(portfolio)
        if arm=='quote_inputs':
            if quote_archive_directory is None:raise ValueError('Quote-history candidate requires its archived input')
            p.__class__=QuoteHistoryRuntime;p.quote_mode='quote_inputs'
            p.quote_archive_directory=str(Path(quote_archive_directory).resolve())
    elif arm=='size_only_2':
        p=copy.deepcopy(portfolio);p.__class__=BudgetPortfolio;p.base_task=None
        p.budget_transform=ScaledPacket(2.,0)
    else:raise ValueError('Arm is not a prespecified primary')
    after=state_vector(p)
    for name,values in before['ledgers'].items():
        for field,value in values.items():np.testing.assert_array_equal(value,after['ledgers'][name][field])
    for field in before:
        if field!='ledgers':assert before[field]==after[field],field
    return p
