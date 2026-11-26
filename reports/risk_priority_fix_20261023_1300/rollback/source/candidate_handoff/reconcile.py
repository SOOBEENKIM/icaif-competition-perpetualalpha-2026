"""Public-only bookkeeping recovery. --apply writes local state, never orders.

Usage: python -m candidate_handoff.reconcile --directory .runtime/pas_budget_lifecycle
       --output reports/... [--apply]
The live service is neither started nor stopped; the common writer lease must
be free. Unknown order outcomes remain blocked rather than erased.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from . import producer
from .public import PublicPAS
from .recovery import recover_confirmed
from .migration import load_existing
from .bridge import HistoryBridge
from .locks import WriterLease
from .runner import require_contract
from competition_live.profile import ROOT,load_profile,verify_deployment
from competition_live.runtime import private_load,save_coordinator,json_save,fingerprint
from joint_wallet.dispatch import Dispatcher
from joint_wallet.book import ReconciliationError

class AuditPublic(PublicPAS):
    def __init__(self,profile,path):super().__init__(profile);self.path=path;self.index=0
    def post(self,request):
        start=time.time();result=super().post(request);self.index+=1
        json_save(self.path/f'{self.index:03d}_{request["type"]}.json',
            dict(request=request,started_at=start,received_at=time.time(),response=result))
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    directory=args.directory.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True,mode=0o700)
    if directory==out:raise ValueError('Evidence must be separate from runtime')
    profile=load_profile();verify_deployment();require_contract(directory,profile)
    public=AuditPublic(profile,out/'public');state_path=directory/'coordinator.pkl'
    with WriterLease(ROOT),Dispatcher(directory,profile).locked():
        original,digest=load_existing(state_path,profile,allow_pending=True)
        dispatch_path=directory/'dispatch.json';dispatch_bytes=dispatch_path.read_bytes();dispatch=json.loads(dispatch_bytes)
        original_revision=(directory/'strategy_revision.json').read_bytes()
        statuses={}
        for cloid,pending in original.account.pending_orders.items():
            status=public.post(dict(type='orderStatus',user=profile['team_wallet'],oid=cloid));statuses[cloid]=status
            record=status.get('order',{});order=record.get('order',{})
            receipt=dispatch.get('completed',{}).get(cloid,{})
            if (status.get('status')!='order' or record.get('status') not in ('filled','canceled','marginCanceled','selfTradeCanceled','reduceOnlyCanceled','scheduledCancel')
                    or order.get('oid')!=receipt.get('oid') or order.get('coin')!=pending['order']['coin']):
                raise ReconciliationError('Venue has not confirmed a terminal matching order')
        bridge=HistoryBridge(ROOT,directory,profile,public)
        for attempt in range(3):
            book=bridge.refresh()
            try:
                checked=bridge.reconcile_exchange(book)
                recovered,changes=recover_confirmed(original,dispatch,bridge.rows,checked['marks'],now=time.time())
                synced=bridge.sync(recovered.account,book,checked['marks'])
                break
            except ReconciliationError:
                if attempt==2:raise
        own_ids={(e['row']['coin'],e['row']['tid']) for e in book.events if e['kind']=='fill' and e['owner']=='ours'}
        if own_ids!=recovered.account.applied_fills:raise ReconciliationError('Own fill identity set differs from public owner book')
        recovered.account.reconcile(checked['marks'])
        for k in ('last_decision','counter','peak','halted','used','attempts','pair_bars','strategy_id'):
            if getattr(recovered,k)!=getattr(original,k):raise ReconciliationError('Unexpected strategy-state change: '+k)
        if recovered.pair is not None:raise ReconciliationError('Active pair needs lifecycle-specific recovery review')
        result=dict(at=time.time(),applied=args.apply,signer_loaded=False,orders_sent=0,service_started=False,
            previous_checkpoint_sha256=digest,strategy_revision=json.loads(original_revision),
            changes=changes,conservation=synced,physical_qty=book.physical.qty.tolist(),
            own_qty=recovered.account.physical.qty.tolist(),teammate_qty=recovered.account.reserved.qty.tolist(),
            own_fills=len(recovered.account.physical.fills),own_fees=recovered.account.physical.fees,
            own_funding=recovered.account.physical.funding,pending_orders=len(recovered.account.pending_orders),
            unknown_symbols=sorted(recovered.account.unknown),open_orders=checked['snapshot']['orders'],
            venue_order_status=statuses,public_read_count=public.read_count,
            observed_equity=checked['snapshot']['spot']['balances'],
            observed_valuation_interval=checked['observed_valuation_interval'])
        json_save(out/'public_fills.json',bridge.rows);json_save(out/'public_funding.json',bridge.funding)
        json_save(out/'joint_book.json',book.serialize());json_save(out/'account_snapshot.json',checked['snapshot'])
        save_coordinator(out/'recovered_coordinator.pkl',recovered,profile,'confirmed_fill_recovery_preview')
        load_existing(out/'recovered_coordinator.pkl',profile) # default fail-closed loader must now pass
        if args.apply:
            if (hashlib.sha256(state_path.read_bytes()).hexdigest()!=digest or dispatch_path.read_bytes()!=dispatch_bytes
                    or (directory/'strategy_revision.json').read_bytes()!=original_revision):
                raise ReconciliationError('Runtime changed during recovery')
            backup=out/'before_apply';backup.mkdir(exist_ok=True,mode=0o700)
            for name in ('coordinator.pkl','ownership.json','status.json','dispatch.json','strategy_revision.json'):
                shutil.copy2(directory/name,backup/name);(backup/name).chmod(0o600)
            save_coordinator(state_path,recovered,profile,'confirmed_fill_recovery_orders_disabled')
            json_save(directory/'ownership.json',book.serialize())
            json_save(directory/'status.json',dict(at=time.time(),state='reconciled_waiting_activation',orders_enabled=False,
                candidate='budget_lifecycle',core_revision=json.loads(original_revision)['revision'],
                own_qty=result['own_qty'],teammate_qty=result['teammate_qty'],pending_orders=0,
                evidence=str(out/'result.json'),last_decision=recovered.last_decision,
                note='Local bookkeeping recovered; order service remains stopped.'))
            load_existing(state_path,profile)
            result['checkpoint_sha256']=hashlib.sha256(state_path.read_bytes()).hexdigest()
        json_save(out/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('observed_equity','venue_order_status')},ensure_ascii=False))
    public.session.close()
if __name__=='__main__':main()
