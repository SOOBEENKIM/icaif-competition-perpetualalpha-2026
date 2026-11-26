"""Predeclared clock-sensitivity checks; no timing phase winner selection."""
import json,time,sys
from unittest.mock import patch
from .protocol import ROOT,OUT,save,sha

DEST=OUT/'paced_forward_review';END=1792915500


def freeze():
    assert time.time()<END and not (DEST/'protocol.json').exists()
    files=['paced_forward_review.py','paced_book_replay.py','exact_settlement.py','cash_replay.py','full_forward_review.py']
    save(DEST/'protocol.json',dict(at=time.time(),loop_seconds=90,phases_seconds=[0,45],
        reason='Read-only actual worker heartbeat publications observed87.9/88.0seconds apart onOct25 before outcomes. Original evaluator ticks15seconds. Entry-check event intervals were NOT used as tick cadence.',
        rule='Same frozen candidates, five comparison periods and identical starting inventory. Verify both cost scenarios under BOTH clock phases; no best-phase selection. Only outer evaluation timing changes; per-order delay/depth/fees and exact funding remain unchanged. These are timing sensitivities, not independent tests.',
        parent=sha(OUT/'exact_forward_review/protocol.json'),hashes={f'production_replay_20261025/{f}':sha(ROOT/'production_replay_20261025'/f) for f in files},orders_sent=0))


def check():
    p=json.loads((DEST/'protocol.json').read_text())
    assert sha(OUT/'exact_forward_review/protocol.json')==p['parent']
    for f,h in p['hashes'].items():assert sha(ROOT/f)==h,f
    return p


def run():
    c=check()
    while time.time()<END+1740:
        save(DEST/'status.json',dict(at=time.time(),state='waiting_for_fixed_end_and_settlements',orders_sent=0))
        time.sleep(min(30,max(1,END+1740-time.time())))
    check()
    from . import full_forward_review,cash_replay,book_replay,paced_book_replay
    from .exact_settlement import run_exact,read_records
    records=read_records(OUT/'settlement_check/raw');initial=[]
    for f in sorted((OUT/'initial_funding').glob('*.json')):
        r=json.loads(f.read_text())
        if 'request' in r:initial.extend(dict(x,coin=r['request']['coin']) for x in r['data'])
    if not initial:raise ValueError('Initial funding schedule missing')
    def adapter(*args,**kwargs):return run_exact(*args,**kwargs,records=records,initial_settlements=initial)
    for phase in c['phases_seconds']:
        folder=DEST/f'phase_{phase}';folder.mkdir(exist_ok=True)
        with patch.object(full_forward_review,'DEST',folder),patch.object(full_forward_review,'check',return_value=c),patch.object(cash_replay,'run_cash',side_effect=adapter),patch.object(book_replay,'run_book',paced_book_replay.run_book_paced),patch.object(paced_book_replay,'PHASE_SECONDS',phase):
            full_forward_review.run()
    save(DEST/'status.json',dict(at=time.time(),state='complete',orders_sent=0))


if __name__=='__main__':globals()[sys.argv[1]]()
