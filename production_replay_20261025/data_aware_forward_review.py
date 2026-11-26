"""Pre-outcome correction for actual public-read pause behavior and execution delay.

Delay occurs before the gateway's final deadline check, not after its send.
This represents serial reconciliation/preflight work. The normal order limit
stays fixed during that work, as in the live caller.
"""
import json
import sys
import time
from unittest.mock import patch

from bot.execution import OrderRejected
from .protocol import OUT, ROOT, save, sha

DEST = OUT / 'data_aware_forward_review'
END = 1792915500


from .delayed_forward_review import delayed_submit


def freeze():
    assert time.time() < END and not (DEST/'protocol.json').exists()
    save(DEST/'protocol.json',dict(at=time.time(),
        evidence=sha(OUT/'historical_execution_delay.json'),input_availability_audit=sha(OUT/'input_coverage_audit.json'),
        evidence_description='Pre-test retained actual fills. Under the current joint execution path, first fill delays79.815..112.395seconds; multi-asset completion up to355.148seconds. Earlier1..3second fills predate that path.',
        pre_send_work_seconds=dict(base=90,stress=120),loop_seconds=90,phases_seconds=[0,45],
        input_policy='Pause all policy/order code when any required received book/context is unavailable. Settle cash, preserve holdings, never interpolate prices. Each pause must have valid observations bracketing it within90seconds (the normal worker interval), otherwise still fail. Original unavailable-input fatal results retained. This models the actual runner public-read exception branch. No intragap risk observation is claimed.',
        mechanism='Unchanged five primaries. Advance simulated clock and settle any intervening cash before the final gateway deadline check; retain the previously prepared price/size. Then existing1s/3s network+book-arrival latency. An expired task or reached fixed end cannot submit.',
        interpretation='Execution realism sensitivity, not new alpha or independent replication. Require both phases and both scenarios; no favorable-phase selection. Actual durations vary; these constants do not reproduce all live API races.',
        parent=sha(OUT/'exact_forward_review/protocol.json'),
        hashes={f'production_replay_20261025/{f}':sha(ROOT/'production_replay_20261025'/f) for f in ['data_aware_forward_review.py','data_aware_book_replay.py','delayed_forward_review.py','exact_settlement.py','cash_replay.py','full_forward_review.py']},
        orders_sent=0))


def check():
    p=json.loads((DEST/'protocol.json').read_text())
    assert sha(OUT/'exact_forward_review/protocol.json')==p['parent']
    assert sha(OUT/'historical_execution_delay.json')==p['evidence']
    assert sha(OUT/'input_coverage_audit.json')==p['input_availability_audit']
    for f,h in p['hashes'].items():assert sha(ROOT/f)==h,f
    return p


def run():
    c=check()
    while time.time()<END+1980:
        save(DEST/'status.json',dict(at=time.time(),state='waiting_for_fixed_end_and_settlements'))
        time.sleep(min(30,max(1,END+1980-time.time())))
    check()
    from . import full_forward_review,cash_replay,book_replay,data_aware_book_replay
    from .exact_settlement import run_exact,read_records
    records=read_records(OUT/'settlement_check/raw');initial=[]
    for f in sorted((OUT/'initial_funding').glob('*.json')):
        r=json.loads(f.read_text())
        if 'request' in r:initial.extend(dict(x,coin=r['request']['coin']) for x in r['data'])
    if not initial:raise ValueError('Initial funding schedule missing')
    def adapter(*args,**kwargs):return run_exact(*args,**kwargs,records=records,initial_settlements=initial)
    for phase in c['phases_seconds']:
        folder=DEST/f'phase_{phase}';folder.mkdir(exist_ok=True)
        save(folder/'protocol.json',dict(c,this_phase_seconds=phase))
        with patch.object(full_forward_review,'DEST',folder),patch.object(full_forward_review,'check',return_value=c),patch.object(cash_replay,'run_cash',side_effect=adapter),patch.object(book_replay,'run_book',data_aware_book_replay.run_book_data_aware),patch.object(data_aware_book_replay,'PHASE_SECONDS',phase),patch.object(cash_replay.CashVenue,'submit',delayed_submit):
            full_forward_review.run()
    save(DEST/'status.json',dict(at=time.time(),state='complete',orders_sent=0))


if __name__=='__main__':globals()[sys.argv[1]]()
