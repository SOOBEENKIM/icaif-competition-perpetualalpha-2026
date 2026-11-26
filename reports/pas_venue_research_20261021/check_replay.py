"""Exercise the simulator through a partial exit, then replay the old diagnostic.

No network or order calls. Synthetic and retrospective results are not a new
out-of-sample trial.
"""
import contextlib
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from pas_venue_research import shadow

original=json.loads((OUT/'protocol.json').read_text())


def plan():
    result=copy.deepcopy(original)
    result['created_at']=time.time()
    result['prospective_policy_frozen_before_start']=False
    result['code_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (ROOT/'pas_venue_research').glob('*.py')}
    return result


with tempfile.TemporaryDirectory() as temporary:
    path=Path(temporary);(path/'raw').mkdir()
    protocol=plan();protocol.update(start=100005,end=100150)
    (path/'plan.json').write_text(json.dumps(protocol))
    (path/'completion.json').write_text(json.dumps(dict(at=100150,errors=[])))
    events=[]
    def book(at,bid=99.9,ask=100.1,size=100):
        events.append(dict(received=at,network='pas',channel='l2Book',
            data=dict(coin='pas:BTC',time=int(at*1000),levels=[
                [dict(px=str(bid),sz=str(size))],[dict(px=str(ask),sz='100')]])))
    def context(at):
        events.append(dict(received=at,network='pas',channel='activeAssetCtx',
            data=dict(coin='pas:BTC',ctx=dict(oraclePx='105',funding='0'))))
    book(100004);context(100005);book(100006)
    events.append(dict(received=100007.1,network='pas',channel='trades',
        data=[dict(coin='pas:BTC',tid=1,side='A',px='99.9',sz='104',time=100007000)]))
    book(100066,110,110.1,4);context(100066.1)
    context(100067.2);context(100068);book(100069,110,110.1,4)
    context(100069.1);book(100070,110,110.1,12);context(100150)
    with gzip.open(path/'raw/events.jsonl.gz','wt') as f:
        for event in events:f.write(json.dumps(event)+'\n')
    with patch.object(shadow,'SYMBOLS',['BTC']),contextlib.redirect_stdout(io.StringIO()):
        shadow.main(path)
    result=json.loads((path/'shadow_results.json').read_text())
    for mode in protocol['models']:
        exits=[r for r in result['fills'] if r['mode']==mode and r['kind']=='exit']
        assert len(exits)==2,exits
        assert abs(exits[0]['qty']-1)<1e-9,exits
        assert exits[1]['time']==100070,exits
        row=next(r for r in result['models'] if r['mode']==mode)
        assert row['completed_trades']==1 and not row['residual'],row
        assert not row['activate'] and not row['component_evidence_gate_pass']
    assert not result['policy_frozen_before_start']
    (OUT/'synthetic_partial_exit_check.json').write_text(json.dumps(result,indent=2))

path=OUT/'replay_corrected';path.mkdir(exist_ok=True);(path/'raw').mkdir(exist_ok=True)
raw=path/'raw/events.jsonl.gz'
if not raw.exists():raw.symlink_to(OUT/'synchronized_public.jsonl.gz')
(path/'plan.json').write_text(json.dumps(plan(),indent=2))
(path/'completion.json').write_text((OUT/'capture_manifest.json').read_text())
with contextlib.redirect_stdout(io.StringIO()):
    shadow.main(path)
result=json.loads((path/'shadow_results.json').read_text())
assert not result['policy_frozen_before_start']
assert all(r['accepted_proposals']==0 for r in result['models'])
print(json.dumps(dict(synthetic_partial_exit_check='PASS',
    corrected_retrospective_replay=result['models'],capture_errors=result['capture_errors'],
    prospective_validation=False),ensure_ascii=False))
