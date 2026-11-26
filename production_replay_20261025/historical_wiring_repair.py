"""Correct research-only deepcopy callback ownership; preserve original runs.

The frozen prospective book runner already rebinds these dependencies after
copying. Only the new catchup/quote historical build wrappers omitted this.
No live source, frozen candidate or future evaluator is edited.
"""
import importlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from unittest.mock import patch
from .protocol import ROOT,OUT,DATA,save,sha


def rewire(candidate, original):
    venue=original.account.executor
    if candidate.pair is not None or candidate.sessions:
        raise ValueError('This repair applies only to initially flat historical builds')
    candidate.account.executor=venue
    candidate.account.checkpoint=venue.checkpoint
    candidate.account.collect_fills=venue.collect
    candidate.account.refresh_market=venue.stream
    candidate.settlement=venue.settle
    venue.account=candidate.account
    assert candidate.account.executor is venue
    assert candidate.account.refresh_market.__self__ is venue
    assert candidate.settlement.__self__ is venue
    return candidate


def job(kind,args):
    m=importlib.import_module('production_replay_20261025.'+kind+'_study')
    protocol=m.check();destination=m.DEST/'historical_corrected';original_adopt=m.adopt
    destination.mkdir(parents=True,exist_ok=True)
    def adopter(p):return rewire(original_adopt(p),p)
    with patch.object(m,'DEST',destination),patch.object(m,'check',return_value=protocol),patch.object(m,'adopt',adopter):
        row=m.historical_job(*args)
    row['research_wiring_repaired']=True
    row['original_historical_result_invalidated']=True
    return kind,row


def run():
    items=[]
    for kind in ['catchup','quote']:
        m=importlib.import_module('production_replay_20261025.'+kind+'_study');m.check()
        save(m.DEST/'historical_INVALIDATED.json',dict(at=datetime.now(timezone.utc).isoformat(),
            reason='Historical adopter deep-copied executor, refresh/settlement callbacks and venue clock. Candidate did not use the advancing outer venue. Controls remain valid. Replace only with historical_corrected results.',
            scope='catchup historical candidate full runs; quote historical full runs. Same-inventory in-place quote switches unaffected but rerun for a matched check. Prospective book runner explicitly rebinds and is unaffected.',
            repair_sha256=sha(__file__),orders_sent=0))
        if kind=='catchup':
            items += [(kind,(str(d.relative_to(ROOT)),'fresh_block_catchup',s)) for d in [DATA,OUT/'reserved_eight_hours'] for s in ['base','stress']]
        else:
            items += [(kind,(s,w)) for s in ['base','stress'] for w in [False,True]]
    collected={k:[] for k in ['catchup','quote']}
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,*item) for item in items]):
            kind,row=f.result();collected[kind].append(row)
            dest=OUT/('quote_anchor' if kind=='quote' else kind)/'historical_corrected'
            save(dest/'results.json',collected[kind]);print(kind,row,flush=True)


if __name__=='__main__':run()
