"""Opt-in operational adapter; unchanged alpha and 50,000 USDP user budget.

Default --audit reads public history and a private checkpoint without signing,
writing the checkpoint, changing services, or sending orders.
"""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from bot.execution import OutcomeUnknown
from competition_live import runner as original
from competition_live.execution import PASExecutor
from competition_live.profile import ROOT,load_profile,verify_deployment,verify_bundle
from competition_live.runtime import json_save
from competition_live.venue import PublicPAS
from .account import SharedWalletAccount,external_ledger,reconcile_shared,isolated_snapshot,PROTECTED


class RiskView:
    def __init__(self,public,executor):self.public=public;self.executor=executor
    def __getattr__(self,name):return getattr(self.public,name)
    def account(self):
        if self.executor.sleeve_account is None:raise OutcomeUnknown('External ownership not initialized')
        return isolated_snapshot(self.public.account(),self.executor.profile,self.executor.sleeve_account.reserved)


class ProtectedExecutor(PASExecutor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.sleeve_account=None;self.public=RiskView(self.public,self)
    def submit(self,order):
        if order.coin==PROTECTED:raise OutcomeUnknown('Teammate ETH orders are prohibited in this adapter')
        return super().submit(order)
    def cancel(self,coin,cloid):
        if coin==PROTECTED:raise OutcomeUnknown('Teammate ETH cancellation is prohibited')
        return super().cancel(coin,cloid)


class SharedRunner(original.LiveRunner):
    def __init__(self,*args,audit=False,**kwargs):
        self.audit=audit
        super().__init__(*args,**kwargs)
        account=self.portfolio.account
        j=account.physical.symbols.index('ETH')
        if account.physical.qty[j] or any(o.qty[j] for o in account.owners.values()):
            raise OutcomeUnknown('Cannot adopt disjoint ETH reservation with user-owned ETH')
        # No ledger quantities, alpha state or previous order identities reset.
        account.__class__=SharedWalletAccount
        if self.executor is not None:self.executor.sleeve_account=account

    def checkpoint(self,reason,detail):
        if not self.audit:super().checkpoint(reason,detail)

    def settlement(self,at_ms,account):
        if time.time()-self.last_funding<15:return
        end=int(time.time()*1000);start=self.profile['start_unix']*1000
        rows=original.history(self.public,'userFillsByTime',self.profile,start,end)
        funding=original.history(self.public,'userFunding',self.profile,start,end)
        journal=self.directory/'orders.jsonl'
        events=[json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
        own_cloids={e['order']['cloid'] for e in events if e.get('event')=='submit_intent'}
        for f in rows:
            if not f['coin'].startswith('pas:') or f['coin']==PROTECTED:continue
            if (f['coin'],f['tid']) not in account.applied_fills:
                raise OutcomeUnknown('New unrecorded fill outside teammate ETH reservation')
        reserve=external_ledger(self.profile,rows,funding,own_cloids,end)
        account.reserved=reserve
        account.apply_actual_funding(funding)
        self.last_funding=time.time();self.funding_cursor=start
        self.checkpoint('teammate_ETH_history_reconciled',{'cutoff_ms':end})

    def tick(self):
        report=super().tick()
        account=self.portfolio.account
        report.update(shared_wallet_mode='disjoint_ETH_teammate',teammate_market='pas:ETH',
                      teammate_qty=account.reserved.qty.tolist(),teammate_funding=account.reserved.funding,
                      strategy_changed=False,teammate_orders_sent=0)
        json_save(self.directory/'status.json',report)
        return report


def audit():
    profile=load_profile();verify_bundle();verify_deployment()
    public=PublicPAS(profile);directory=ROOT/'.runtime/pas_competition_live'
    if not (directory/'coordinator.pkl').exists():raise ValueError('Audit requires existing checkpoint')
    instance=SharedRunner(profile,directory,public,None,audit=True)
    instance.settlement(int(time.time()*1000),instance.portfolio.account)
    value=reconcile_shared(instance.portfolio.account,public.account(),profile)
    account=instance.portfolio.account
    return dict(at=time.time(),inspection_mode='read_only_recovery_audit',**value,
        own_qty=account.physical.qty.tolist(),teammate_qty=account.reserved.qty.tolist(),
        own_funding=account.physical.funding,teammate_funding=account.reserved.funding,
        orders_sent=0,signer_loaded=False,checkpoint_written=False,alpha_changed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--once',action='store_true')
    args=p.parse_args()
    if not args.execute:
        print(json.dumps(audit()));return
    # Guard the operational adapter separately from the immutable original alpha.
    import hashlib
    manifest=json.loads((Path(__file__).parent/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Operational adapter changed: '+name)
    with patch.object(original,'LiveRunner',SharedRunner),patch.object(original,'PASExecutor',ProtectedExecutor), \
         patch.object(original,'LiveAccount',SharedWalletAccount),patch.object(original,'reconcile_account',reconcile_shared):
        original.main(execute=True,once=args.once)


if __name__=='__main__':main()
