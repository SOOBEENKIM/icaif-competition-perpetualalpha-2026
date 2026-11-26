"""Unsigned live-ownership adoption and one-order public preflight.

No signer, no wallet writer lock, no client supplied to the gateway, and no
runtime checkpoint/configuration replacement. Only this report directory is
written (aside from the existing shared public-read rate limiter).
"""
import copy,time
from datetime import datetime,timezone
import numpy as np
from .study import ROOT,OUT,read,save,sha,verify
from .portfolio import adopt,REVISION

def main():
    from competition_live.profile import load_profile
    from competition_live.runtime import save_coordinator
    from competition_live.execution import prepare_pas_order
    from candidate_handoff.public import PublicPAS
    from candidate_handoff.bridge import HistoryBridge
    from candidate_handoff.migration import load_existing
    from candidate_handoff.producer import validate
    from candidate_handoff.entry_policy import configured_policy
    from candidate_handoff.portfolio import target_from_frame
    from joint_wallet.pas_gateway import PASGateway
    from joint_wallet.dispatch import Dispatcher
    verify();directory=ROOT/'.runtime/pas_budget_lifecycle';release=OUT/'release';release.mkdir(exist_ok=True)
    profile=load_profile();api=PublicPAS(profile)
    old,digest=load_existing(directory/'coordinator.pkl',profile)
    baseline_config=read(directory/'execution_policy.json')
    current_policy=configured_policy(directory)
    bridge=HistoryBridge(ROOT,directory,profile,api)
    try:
        book=bridge.refresh();checked=bridge.reconcile_exchange(book)
        candidate=adopt(old);candidate.account.checkpoint=lambda *a:None
        state=bridge.sync(candidate.account,book,checked['marks'])
        for name in ['base','pair']:
            for key in ['qty','entry','fees','realized']:
                np.testing.assert_equal(getattr(candidate.account.owners[name],key),getattr(old.account.owners[name],key))
        packet=read(directory/'signal.json');now=time.time();body=validate(packet,profile,now,expected_policy=current_policy)
        frame=body['frame'];source=frame['source_time'];marks=checked['marks']
        action,trace=target_from_frame(frame,marks,candidate.account.owners['base'].qty,candidate.account.reconcile(marks),profile['trading'])
        first=None;preflight=None;preflight_error=None
        if action:
            gateway=PASGateway(profile,api,client=None);meta,ctx=api.market()
            for j,s in enumerate(profile['trading']['symbols']):
                own=candidate.account.owners['base'].qty[j];desired=action['target_qty'][j];delta=desired-own
                if abs(delta)*marks[j]<10:continue
                # Preview only the nearest executable side with the existing
                # fixed10bp IOC allowance; all rounding uses production rules.
                quote=api.post(dict(type='l2Book',coin='pas:'+s));side=bool(delta>0)
                levels=quote['levels'][1 if side else 0]
                if not levels:continue
                px=float(levels[0]['px'])*(1.001 if side else .999)
                net=book.physical.qty[j]
                reduce_only=bool(net*delta<0 and abs(delta)<=abs(net))
                order=prepare_pas_order(profile,'pas:'+s,side,min(abs(delta),7500/px),px,meta['universe'][j],ctx[j],current_qty=net,reduce_only=reduce_only,slot='revision-unsigned-preview')
                rounded_delta=float(order.size)*(1 if side else -1)
                plan=book.plan('ours',s,float(book.owners['ours'].qty[j]+rounded_delta),float(order.price),profile['trading']['execution']['decimals'][j],max_notional=7500,slot='revision-unsigned-preview')
                if plan is None:continue
                first=plan
                try:
                    Dispatcher(release/'preview_dispatch',profile).validate_request(plan,book,marks,np.array(frame['sigma']))
                    gateway.preflight(plan,book);preflight=True
                except Exception as e:preflight=False;preflight_error=type(e).__name__+': '+str(e)
                break
        save_coordinator(release/'adopted_coordinator.pkl',candidate,profile,'unsigned_execution_revision_adoption')
        save(release/'incumbent_execution_policy.json',baseline_config)
        new_config=copy.deepcopy(baseline_config);new_config['execution_revision']=REVISION
        new_config['source_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'execution_revision_20261024').glob('*.py')})
        new_config['staged_only']=True
        save(release/'candidate_execution_policy.json',new_config)
        result=dict(at_utc=datetime.now(timezone.utc).isoformat(),old_checkpoint_sha256=digest,revision=REVISION,
            signal_source=source,current_completed_hour=int(now//3600)*3600,current_hour_ready=source==int(now//3600)*3600,
            signal_within_six_hour_validity=0<=now-source<21600,
            own_qty=candidate.account.physical.qty.tolist(),teammate_qty=candidate.account.reserved.qty.tolist(),
            own_equity=state['own_equity'],teammate_equity=state['teammate_equity'],account_conservation=state['conservation'],
            last_decision=candidate.last_decision,next_decision=(int(now)//21600+1)*21600,
            scheduled_first_order=None,scheduled_reason='Between six-hour decision rounds; no old pending intent revived by adoption.',
            hypothetical_first_order_if_rebalanced_now=first,first_public_preflight_pass=preflight,first_public_preflight_error=preflight_error,
            hypothetical_targets=action,allocation_trace=trace,
            inventory_reset=False,teammate_inventory_changed=False,signer_loaded=False,orders_sent=0,live_changed=False,
            activation_instructions='Stop only incumbent order worker; acquire shared writer lease; re-read and reconcile current checkpoint; adopt with execution_revision_20261024.portfolio.adopt; preserve profile/owner contract; update hash-protected execution config; atomically replace checkpoint and restart existing runner. Never activate this stale preview checkpoint or use saved prices. Rollback also adopts CURRENT holdings, never restores obsolete holdings.',
            readiness='Code and public preflight only; profitability decision must include matched-inventory switch experiment.')
        save(OUT/'stage.json',result);print({k:result[k] for k in ['current_hour_ready','own_equity','teammate_equity','first_public_preflight_pass','first_public_preflight_error','orders_sent','live_changed']},flush=True)
    finally:api.session.close()
    verify()

if __name__=='__main__':main()
