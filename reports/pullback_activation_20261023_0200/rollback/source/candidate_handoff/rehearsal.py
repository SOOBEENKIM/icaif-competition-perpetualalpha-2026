"""Current account -> migrated model -> current quote -> stop before send."""
import copy,hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
from . import producer
from .portfolio import target_from_frame
from .bridge import JointExecutor,PreviewOnly
from .runner import connect,require_contract
from .locks import WriterLease,WriterBusy
from competition_live.profile import ROOT,load_profile,verify_deployment
from competition_live.runtime import json_save,save_coordinator
from .public import collect_books,PublicPAS


def main(directory=None):
    out=Path(directory or ROOT/'reports/candidate_handoff_20261022');out.mkdir(parents=True,exist_ok=True,mode=0o700)
    profile=load_profile();original=verify_deployment()['files'];public=PublicPAS(profile)
    revision=producer.configured_revision(out)
    packet_path=out/'signal.json'
    try:
        packet=json.loads(packet_path.read_text());producer.validate(packet,profile,time.time(),expected_revision=revision)
        if packet['body']['frame']['source_time']!=int(time.time()//3600)*3600:raise ValueError('Refresh current hour')
    except (ValueError,OSError,KeyError):packet=producer.generate(ROOT,profile,revision);json_save(packet_path,packet)
    portfolio,bridge,checked,adoption=connect(ROOT,out,profile,public,
        lambda b:JointExecutor(profile,out,b,preview=True))
    # No callbacks in any preview may write to the running coordinator.
    portfolio.account.checkpoint=lambda *args:None
    json_save(out/'joint_book.json',bridge.latest.serialize())
    json_save(out/'public_fills.json',bridge.rows);json_save(out/'public_funding.json',bridge.funding)
    json_save(out/'account_snapshot.json',checked['snapshot'])
    save_coordinator(out/'adopted_coordinator.pkl',portfolio,profile,'unsigned_handoff_copy')
    streams=collect_books(profile,public);now=time.time();marks,_=portfolio.prices(now,streams)
    if marks is None or portfolio.unpriced_flat_assets:raise ValueError('Current marks incomplete')
    body=producer.validate(packet,profile,now);frame=body['frame']
    qty=portfolio.account.physical.qty.copy();teammate=portfolio.account.reserved.qty.copy()
    action,trace=target_from_frame(frame,marks,portfolio.account.owners['base'].qty,portfolio.account.reconcile(marks),profile['trading'])
    # Run the exact scheduler separately from the explicitly off-cadence sizing
    # preview. Never label a hypothetical target as an already scheduled order.
    scheduled=copy.deepcopy(portfolio);scheduled.account.executor=portfolio.account.executor
    scheduled.account.checkpoint=lambda *args:None
    scheduled_order=None
    try:scheduled.step_candidate(now,streams,packet,profile)
    except PreviewOnly as e:scheduled_order=e.plan
    dry=copy.deepcopy(portfolio);dry.account.executor=portfolio.account.executor
    dry.account.checkpoint=lambda *args:None
    hypothetical=None
    try:dry.submit_baseline(time.time(),np.array(action['target_qty']),streams,marks,np.array(frame['sigma']),False)
    except PreviewOnly as e:hypothetical=e.plan
    asset_previews=[]
    for j,s in enumerate(profile['trading']['symbols']):
        if abs(action['target_qty'][j]-portfolio.account.owners['base'].qty[j])<1e-10:continue
        # Re-reading the complete wallet for every hypothetical asset exhausts
        # the API quota shared with the live writer. Only the first candidate
        # receives public preflight; later rows are explicitly target evidence.
        first=hypothetical is not None and hypothetical['coin']=='pas:'+s
        asset_previews.append(dict(symbol=s,plan=hypothetical if first else None,
            preflight_pass=True if first else None,
            note='First order checked against public state' if first else
                 'Target only; requires fresh preflight and leverage/margin checks before any send'))
    locks={}
    try:
        with WriterLease(ROOT):locks=dict(old_writer_excluded=True,current_writer_holds_lock=False)
    except WriterBusy:locks=dict(old_writer_excluded=True,current_writer_holds_lock=True)
    np.testing.assert_array_equal(qty,portfolio.account.physical.qty)
    np.testing.assert_array_equal(teammate,portfolio.account.reserved.qty)
    assert original==verify_deployment()['files']
    stamp=time.time();kst=timezone(timedelta(hours=9));next_round=(int(stamp)//21600+1)*21600
    try:
        policy=require_contract(ROOT/'.runtime/pas_budget_lifecycle',profile)
        mode=policy['transport'];authorized=True
    except ValueError:mode=None;authorized=False
    plans=[dict(symbol=s,current=float(portfolio.account.owners['base'].qty[j]),
        target=float(action['target_qty'][j]),delta=float(action['target_qty'][j]-portfolio.account.owners['base'].qty[j]),mark=float(marks[j]))
        for j,s in enumerate(profile['trading']['symbols'])]
    result=dict(at=stamp,at_kst=datetime.fromtimestamp(stamp,kst).isoformat(),candidate='budget_lifecycle',
        core_revision=revision,universe_audit=body.get('universe_audit',[]),
        adoption=adoption,marks=marks.tolist(),own_qty=qty.tolist(),teammate_qty=teammate.tolist(),
        own_entry=portfolio.account.physical.entry.tolist(),own_fees=portfolio.account.physical.fees,
        own_realized=portfolio.account.physical.realized,own_funding=portfolio.account.physical.funding,
        pair_state=portfolio.pair.state if portfolio.pair else None,last_decision=portfolio.last_decision,
        position_and_pnl_reset=False,scheduled_first_order=scheduled_order,
        hypothetical_first_order=hypothetical,asset_previews=asset_previews,
        hypothetical_scope='If normal rebalance were now; current actual quotes. Not a scheduled future order or future fill. Only the first candidate receives public preflight; later rows are target calculations, not approved executable orders.',
        targets_if_rebalanced_now=plans,allocation_trace=trace,signal_source=frame['source_time'],
        signal_sha256=packet['sha256'],next_normal_round_kst=datetime.fromtimestamp(next_round,kst).isoformat(),
        **locks,old_live_files_unchanged=True,live_changed=False,orders_sent=0,signer_loaded=False,
        execution_integration='Candidate policy -> retained pair coordinator -> LiveAccount -> JointExecutor -> Dispatcher -> PASGateway',
        ready_for_activation=authorized and all(r.get('preflight_pass',False) for r in asset_previews),ownership_mode=mode,
        remaining_requirement='Stop only the old order worker, acquire its locks, refresh/adopt again and quote again.' if authorized else 'Execution ownership policy is missing.',
        caveats=['Current positive backtest is not proof of future profit.',
            'The168h predictor/carry input remains a reference-market forecast, not a claim of PAS-only training.',
            'Offsetting zero-net funding requires observed per-unit settlement; missing settlement stops execution rather than fabricating owner funding.',
            'Stored preview prices are never reused to send a later order.'])
    json_save(out/'rehearsal.json',result)
    json_save(out/'source_manifest.json',{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [*(ROOT/'candidate_handoff').glob('*.py'),*(ROOT/'joint_wallet').glob('*.py'),ROOT/'reports/pas_economic_allocation_20261022/fit.json']})
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
    return result


if __name__=='__main__':main()
