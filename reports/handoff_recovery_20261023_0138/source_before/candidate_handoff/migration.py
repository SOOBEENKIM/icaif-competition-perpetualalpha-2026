"""Copy and reconcile an existing coordinator; never flatten or reset capital."""
import copy,hashlib,time
import numpy as np
from competition_live.runtime import private_load,fingerprint,save_coordinator,json_save
from competition_live.live_account import LiveAccount
from .portfolio import adopt


def load_existing(path,profile):
    for _ in range(3):
        before=hashlib.sha256(path.read_bytes()).hexdigest();saved=private_load(path)
        after=hashlib.sha256(path.read_bytes()).hexdigest()
        if before==after:break
    else:raise ValueError('Checkpoint changed repeatedly; retry after the current tick')
    if saved['fingerprint']!=fingerprint(profile):raise ValueError('Old profile mismatch')
    p=saved['portfolio'];a=p.account
    if a.pending_orders or a.unknown:raise ValueError('Unresolved legacy order cannot be handed off')
    if p.fatal or p.halted:raise ValueError('Existing fatal/halted state requires resolution, not a reset')
    return p,after


def wire(portfolio,bridge,executor,checkpoint,collect_fills,refresh_market):
    a=portfolio.account;a.__class__=LiveAccount
    a.executor=executor;a.checkpoint=checkpoint;a.collect_fills=collect_fills;a.refresh_market=refresh_market
    portfolio.settlement=bridge.settlement
    for pair in [portfolio.pair,*portfolio.sessions]:
        if pair is not None:
            pair.account=a;pair.owner=a.owners['pair'];pair.on_time=bridge.settlement
    return portfolio


def migrate(old,bridge,book,marks):
    p=adopt(old)
    # Synchronize delayed funding on the copy, never original runtime state.
    state=bridge.sync(p.account,book,marks)
    p.strategy_id='budget_lifecycle';p.handoff_at=time.time()
    for k in ('last_decision','counter','peak','halted','used','attempts','pair_bars'):
        assert getattr(p,k)==getattr(old,k),k
    for name,ledger in old.account.owners.items():
        np.testing.assert_array_equal(ledger.qty,p.account.owners[name].qty)
        np.testing.assert_array_equal(ledger.entry,p.account.owners[name].entry)
        assert ledger.realized==p.account.owners[name].realized
    assert p.account.physical.fees==old.account.physical.fees
    return p,state
