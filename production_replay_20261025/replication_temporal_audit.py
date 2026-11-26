"""Additional conservative timestamp audit, fixed before outcomes are read.

Keeps the original frozen strategy evaluation intact. Any ambiguity can only
remove qualification, never turn a losing or failed trial into a passing one.
"""
import gzip
import json
import sys
import time

from .replication import OUT, START, END, check
from .protocol import save,sha


def audit_path(result):
    ambiguous=[]
    for order in result['orders']:
        if not order['filled']:continue
        for funding in result['funding']:
            if order['symbol']!=funding['delta']['coin'][4:]:continue
            when=funding['time']/1000
            if int(order['arrival'])==int(when):
                ambiguous.append(dict(symbol=order['symbol'],arrival=order['arrival'],
                    settlement_time=when,reason='legacy_fill_ledger_discards_milliseconds'))
    return ambiguous


def audit_actual(ledger):
    ambiguous=[]
    for f in ledger['fills']:
        if 'time_ms' in f:continue
        for u in ledger['funding_events']:
            if f['symbol']==u['coin'][4:] and int(f['time'])==int(u['time']) and START<=u['time']<=END:
                ambiguous.append(dict(symbol=f['symbol'],recorded_fill_second=f['time'],
                    settlement_time=u['time'],reason='original_millisecond_order_unrecoverable_from_owner_ledger'))
    return ambiguous


def freeze():
    check()
    if time.time()>=END or (OUT/'temporal_audit_contract.json').exists():
        raise ValueError('Register one audit before fixed outcome opening')
    save(OUT/'temporal_audit_contract.json',dict(at=time.time(),source_sha=sha(__file__),
        parent_sha=sha(OUT/'protocol.json'),outcomes_opened=False,
        rule='Retain frozen original results. Flag every same-second simulated fill/funding pair and every actual owner record with ambiguous original milliseconds. Additional eligibility is original eligibility AND zero temporal ambiguities. No favorable-run selection or repaired-profit substitution.',
        correction_file='production_replay_20261025/timestamp_ledger.py',
        correction_deployed=False,orders_sent=0))


def run():
    c=json.loads((OUT/'temporal_audit_contract.json').read_text())
    if c['source_sha']!=sha(__file__):raise ValueError('Audit source changed')
    while time.time()<END+65:
        time.sleep(min(30,max(1,END+65-time.time())))
    while not (OUT/'results.json').exists():
        if (OUT/'failure.json').exists():
            save(OUT/'temporal_audit.json',dict(at=time.time(),complete=False,
                eligible=False,reason='Original fixed evaluation failed',orders_sent=0))
            return
        if time.time()>END+3600:raise ValueError('Main evaluation did not complete')
        time.sleep(15)
    check()
    main=json.loads((OUT/'results.json').read_text());paths=[]
    for f in sorted(OUT.glob('realistic_phase*/*.json.gz')):
        with gzip.open(f,'rt') as fp:r=json.load(fp)
        paths.append(dict(path=str(f.relative_to(OUT)),sha256=sha(f),ambiguities=audit_path(r)))
    if len(paths)!=12:raise ValueError('Missing a primary/control/phase/cost path')
    actual=audit_actual(json.loads((OUT/'actual_owner_ledger.json').read_text()))
    passed=not actual and all(not p['ambiguities'] for p in paths)
    save(OUT/'temporal_audit.json',dict(at=time.time(),complete=True,paths=paths,
        actual_ambiguities=actual,temporal_pass=passed,
        eligible=bool(passed and main['verdict']['short_conditional_trial_eligible']),
        original_result_sha=sha(OUT/'results.json'),orders_sent=0,live_changed=False))


if __name__=='__main__':globals()[sys.argv[1]]()
