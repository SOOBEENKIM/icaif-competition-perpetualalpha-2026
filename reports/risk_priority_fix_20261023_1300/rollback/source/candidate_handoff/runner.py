"""Staged execution wiring. Default is unsigned and cannot acquire a signer.

The old reference/pair producer continues; only the old order worker is replaced.
Owner deltas can use a common inbox or explicitly authorized external-fill
attribution. Only the former serializes the teammate's submissions as well.
"""
import argparse,json,time
from pathlib import Path
from . import producer  # import research modules before frozen profile path
from .portfolio import CandidatePortfolio
from .migration import load_existing,migrate,wire
from .bridge import HistoryBridge,JointExecutor
from .locks import WriterLease
from competition_live.profile import ROOT,load_profile,verify_deployment
from competition_live.runtime import json_save,save_coordinator,load_coordinator
from competition_live.runner import history,fresh_preflight,explicit_signer
from competition_live.execution import build_order_client
from competition_live.venue import phase_state
from .public import PublicPAS,PublicQuoteProvider,collect_books,PublicReadUnavailable
from .recovery import recover_confirmed
from .entry_policy import configured_policy


def require_contract(directory,profile):
    path=directory/'team_execution_contract.json'
    if not path.exists():raise ValueError('Execution ownership policy is missing')
    c=json.loads(path.read_text())
    if (c.get('schema')!='owner_delta_v1' or c.get('team_wallet','').lower()!=profile['team_wallet'].lower()
            or c.get('owner_capital')!={'ours':50000.,'teammate':50000.}
            or c.get('global_lock')!=str(ROOT/'.runtime/wallet_writer.lock')):
        raise ValueError('Invalid execution ownership policy')
    if c.get('transport')=='shared_inbox':
        if c.get('teammate_direct_orders_disabled') is not True or not c.get('verified_teammate_adapter'):
            raise ValueError('Unverified cooperative teammate execution protocol')
    elif c.get('transport')=='external_fill_reconciliation':
        if (c.get('authorized_without_teammate_adapter') is not True
                or c.get('unattributed_fills_owner')!='teammate'
                or c.get('teammate_direct_orders_disabled') is not False
                or c.get('verified_teammate_adapter') is not False):
            raise ValueError('External fill attribution requires explicit authorization')
    else:raise ValueError('Unknown execution ownership mode')
    return c


def connect(root,directory,profile,public,executor_factory):
    bridge=HistoryBridge(root,directory,profile,public)
    for attempt in range(3):
        state_path=directory/'coordinator.pkl'
        resumed=state_path.exists()
        old,digest=load_existing(state_path if resumed else root/'.runtime/pas_competition_live/coordinator.pkl',profile,
            allow_pending=resumed)
        try:
            book=bridge.refresh();checked=bridge.reconcile_exchange(book)
            if resumed:
                if getattr(old,'strategy_id',None)!='budget_lifecycle':raise ValueError('Wrong candidate checkpoint')
                recovery=None
                if old.account.pending_orders or old.account.unknown:
                    dispatch=json.loads((directory/'dispatch.json').read_text())
                    old,recovery=recover_confirmed(old,dispatch,bridge.rows,checked['marks'],now=time.time())
                portfolio=old;state=bridge.sync(portfolio.account,book,checked['marks'])
                if recovery is not None:state['recovery']=recovery
            else:portfolio,state=migrate(old,bridge,book,checked['marks'])
            break
        except ValueError:
            if attempt==2:raise
    def checkpoint(reason,detail):save_coordinator(directory/'coordinator.pkl',portfolio,profile,reason)
    def fills(oid,start,expected):
        cached=getattr(executor,'confirmed_rows',{}).get(oid)
        if cached is not None:return cached
        for _ in range(6):
            rows=history(public,'userFillsByTime',profile,start-2000,int(time.time()*1000))
            matching={(r['coin'],r['tid']):r for r in rows if r['oid']==oid and r['coin'].startswith('pas:')}
            if abs(sum(float(r['sz']) for r in matching.values())-float(expected))<1e-10:return list(matching.values())
            time.sleep(1)
        return list(matching.values())
    portfolio.execution_policy=configured_policy(directory)
    executor=executor_factory(bridge)
    wire(portfolio,bridge,executor,checkpoint,fills,PublicQuoteProvider(profile,public))
    return portfolio,bridge,checked,dict(old_checkpoint_sha256=digest,**state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--once',action='store_true');a=p.parse_args()
    if not a.execute:
        from .rehearsal import main as preview
        return preview(Path(a.directory))
    directory=Path(a.directory);directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    profile=load_profile();verify_deployment();contract=require_contract(directory,profile)
    revision=producer.configured_revision(directory)
    policy=configured_policy(directory)
    public=PublicPAS(profile)
    with WriterLease(ROOT) as lease:
        packet=json.loads((directory/'signal.json').read_text())
        producer.validate(packet,profile,time.time(),expected_revision=revision,expected_policy=policy)
        if packet['body']['frame']['source_time']!=int(time.time()//3600)*3600:
            raise ValueError('Handoff requires the current completed reference hour')
        # No key is loaded until all old-writer locks have been acquired.
        snapshot=fresh_preflight(public,profile)
        signer=explicit_signer(ROOT,profile,snapshot);client=build_order_client(profile,signer,enable_orders=True)
        portfolio,bridge,checked,adoption=connect(ROOT,directory,profile,public,
            lambda b:JointExecutor(profile,directory,b,lease=lease,client=client))
        def checkpoint(reason,detail):save_coordinator(directory/'coordinator.pkl',portfolio,profile,reason)
        portfolio.account.checkpoint=checkpoint;checkpoint('handoff_connected',adoption)
        reconnect=False
        while not (directory/'STOP').exists():
            try:
                lease.require();now=time.time()
                if phase_state(now,profile)=='ended':break
                if reconnect:
                    # Reload the durable checkpoint, then reconcile terminal
                    # receipts. Never repeat the interrupted send/packet.
                    portfolio,bridge,checked,adoption=connect(ROOT,directory,profile,public,
                        lambda b:JointExecutor(profile,directory,b,lease=lease,client=client))
                    portfolio.account.checkpoint=checkpoint
                    checkpoint('public_read_reconciled',adoption);reconnect=False
                bridge.settlement(now*1000,portfolio.account)
                streams=collect_books(profile,public);now=time.time();error=None
                try:
                    packet=json.loads((directory/'signal.json').read_text());producer.validate(packet,profile,now,expected_revision=revision,expected_policy=policy)
                except (OSError,ValueError,KeyError) as e:packet=None;error=str(e)
                portfolio.step_candidate(now,streams,packet,profile)
                if packet is not None and contract['transport']=='shared_inbox':
                    from .inbox import process
                    import numpy as np
                    process(directory,portfolio.account.executor,profile,np.asarray(packet['body']['frame']['sigma']))
                bridge.settlement(time.time()*1000,portfolio.account)
                json_save(directory/'ownership.json',bridge.latest.serialize())
                checkpoint('candidate_tick',{})
                json_save(directory/'status.json',dict(at=time.time(),state='running',candidate='budget_lifecycle',orders_enabled=True,
                    core_revision=revision,execution_policy=policy,ownership_mode=contract['transport'],teammate_adapter_integrated=contract['transport']=='shared_inbox',
                    own_qty=portfolio.account.physical.qty.tolist(),teammate_qty=portfolio.account.reserved.qty.tolist(),
                    last_decision=portfolio.last_decision,signal_error=error,fatal=portfolio.fatal[-3:],
                    signal_source=packet['body']['frame']['source_time'] if packet else None,
                    own_fills=len(portfolio.account.physical.fills),own_fees=portfolio.account.physical.fees,
                    own_funding=portfolio.account.physical.funding,pair_state=portfolio.pair.state if portfolio.pair else None,
                    recent_decisions=portfolio.events[-12:]))
                if a.once:return
                time.sleep(15)
            except PublicReadUnavailable as e:
                reconnect=True
                json_save(directory/'status.json',dict(at=time.time(),state='paused_public_api',
                    error=str(e),orders_enabled=False,reconcile_before_resume=True,retry_after=e.retry_after))
                # Public reads may be retried after the shared cooldown; order
                # submissions remain governed by the durable dispatch journal.
                time.sleep(min(e.retry_after,60))
            except Exception as e:
                json_save(directory/'status.json',dict(at=time.time(),state='paused_requires_reconciliation',error=str(e),orders_enabled=False));raise


if __name__=='__main__':main()
