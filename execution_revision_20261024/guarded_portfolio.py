"""Staged execution-only candidate with a fixed decision-price entry limit.

The existing price limit was re-anchored at each execution attempt. Extending
its lifetime alone could chase later prices. Here only new BASE exposure uses
the saved decision mark +/-10bp. Explicit pair limits and exits are preserved.
"""
import copy
import numpy as np
from competition_live.live_account import LiveAccount
from experiments.shared_execution_pairs_20261017.account_v2 import difference
from .portfolio import ExecutionPortfolio,adopt as adopt_lifetime

REVISION='one_hour_fixed_entry_guard_v1'

class GuardedAccount(LiveAccount):
    def seal(self,intent,p,sc):
        raw=copy.deepcopy(intent)
        if str(raw.get('packet_id','')).startswith('base:'):
            old=self.owners['base'].qty
            limits=raw.setdefault('owner_limits',{}).setdefault('base',{})
            for j,new in enumerate(raw['owner_targets']['base']):
                delta=difference(new,old[j])
                if not delta or old[j]*delta<0:continue
                anchor=getattr(self,'base_entry_anchor',None)
                if anchor is None or len(anchor)!=len(old) or not np.isfinite(anchor).all() or np.any(np.asarray(anchor)<=0):
                    raise ValueError('New base exposure requires valid saved decision prices')
                side=1 if delta>0 else -1
                cap=float(anchor[j])*(1+side*.001)
                # If another owner-supplied limit is tighter, never loosen it.
                previous=limits.get(str(j))
                limits[str(j)]=cap if previous is None else min(cap,previous) if side>0 else max(cap,previous)
        return super().seal(raw,p,sc)

class GuardedPortfolio(ExecutionPortfolio):
    execution_revision=REVISION

    def submit_baseline(self,*args,**kwargs):
        self.account.base_entry_anchor=(self.base_task.get('decision_prices') if self.base_task else None)
        try:return super().submit_baseline(*args,**kwargs)
        finally:self.account.base_entry_anchor=None

def adopt(portfolio):
    if type(portfolio.account) not in (LiveAccount,GuardedAccount):
        raise TypeError('Review additional account execution behavior before adoption')
    candidate=adopt_lifetime(portfolio)
    candidate.__class__=GuardedPortfolio
    candidate.account.__class__=GuardedAccount
    candidate.account.base_entry_anchor=None
    return candidate
