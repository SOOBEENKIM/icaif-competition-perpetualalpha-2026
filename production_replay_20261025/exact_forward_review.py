"""Pre-outcome accounting amendment; original results are retained separately."""
import json,time,sys
from datetime import datetime,timezone
from unittest.mock import patch
from .protocol import ROOT,OUT,save,sha

DEST=OUT/'exact_forward_review';END=1792915500


def freeze():
    assert time.time()<END and not (DEST/'protocol.json').exists()
    parent=OUT/'complete_forward_review/protocol.json'
    sources=['exact_forward_review.py','exact_settlement.py','full_forward_review.py','cash_replay.py','liquidation_audit.py','initial_funding.py']
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),parent_sha=sha(parent),
        amendment='Seed saved03:00:17 has last credited funding02:00:00.012. Replaying from03:05 must credit intervening settlement before initial valuation. Also retain actual public settlement milliseconds, rather than treating a payment at06:00:00.012 as pre06:00 cash.',
        rules='Same frozen candidates, scenarios, periods, inventory, decision/execution code and acceptance. Apply missing pre-start verified cash once to a COPY of seed using its settled inventory, at full observed cash; stress haircut applies only after replay start. Team-specific timestamps preferred. For an unheld team market retain public timestamp bounds and invalidate any fill inside that ambiguity interval.',
        outcome_status='No prospective performance result opened. Original approximate and earlier verified-cash runs retained, not overwritten. This is an accounting repair, not strategy selection or retuning.',
        hashes={f'production_replay_20261025/{f}':sha(ROOT/'production_replay_20261025'/f) for f in sources},orders_sent=0))


def check():
    p=json.loads((DEST/'protocol.json').read_text())
    assert sha(OUT/'complete_forward_review/protocol.json')==p['parent_sha']
    for f,h in p['hashes'].items():assert sha(ROOT/f)==h,f
    # Validate all original source/experiment freezes as well.
    from .full_forward_review import check as check_parent
    original_dest=check_parent.__globals__['DEST']
    check_parent.__globals__['DEST']=OUT/'complete_forward_review'
    try:check_parent()
    finally:check_parent.__globals__['DEST']=original_dest
    return p


def run():
    check()
    while time.time()<END+1620:
        save(DEST/'status.json',dict(at=time.time(),state='waiting_for_fixed_end_and_settlement_records',orders_sent=0))
        time.sleep(min(30,max(1,END+1620-time.time())))
    check()
    from . import full_forward_review,cash_replay
    from .exact_settlement import run_exact,read_records
    records=read_records(OUT/'settlement_check/raw')
    initial=[]
    for f in sorted((OUT/'initial_funding').glob('*.json')):
        r=json.loads(f.read_text())
        if 'request' in r:initial.extend(dict(x,coin=r['request']['coin']) for x in r['data'])
    if not initial:raise ValueError('Missing initial settlement schedule')
    def adapter(*args,**kwargs):return run_exact(*args,**kwargs,records=records,initial_settlements=initial)
    # The parent has its own check() global. Avoid recursively replacing that
    # call: all hashes are checked above, then supply this checked amendment.
    amendment=json.loads((DEST/'protocol.json').read_text())
    with patch.object(full_forward_review,'DEST',DEST),patch.object(full_forward_review,'check',return_value=amendment),patch.object(cash_replay,'run_cash',side_effect=adapter):
        full_forward_review.run()


if __name__=='__main__':globals()[sys.argv[1]]()
