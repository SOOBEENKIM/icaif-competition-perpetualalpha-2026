import sys,difflib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from auto_pair_validation.study import verify as verify_v2
from .study import ROOT,read,save,sha,verify as verify_wave1
from . import ranking_evaluation as evaluation

OUT=ROOT/'reports/pair_model_research_20261023/wave8'
ARMS=['ar_riskrank','consensus_riskrank']
SOURCES={'ar_riskrank':'wave2/reversion_72.npz','consensus_riskrank':'wave7/consensus_72.npz'}


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=verify_v2();verify_wave1()
    old=(ROOT/'pas_economic_allocation/replay.py').read_text();new=Path(__file__).with_name('ranking_replay.py').read_text()
    assert new.replace('from .ranking import proposals','from experiments.decision_gate_pairs_20261017.entry import proposals')==old
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),wave=8,arms=ARMS,input=p['input'],input_sha256=p['input_sha256'],
      hypothesis='Across45 candidates, ranking point expected cash can prefer the noisiest spreads. Rank positive-net executable proposals by expected net cash divided by source-estimated72h cash standard deviation. Do not fabricate a lower price forecast or change fills/sizing to manufacture gains.',
      ranking='cash risk = actual proposed gross * source pair volatility adjusted to current marks * sqrt72. All existing positive cash, costs,funding,depth,min-size,balance,risk checks occur first. Ranking changes only their order. No statistical-significance or exact-score-optimization claim.',
      arms_description='All45 AR72 risk-ranked; all45 individual-ridge/AR72 consensus risk-ranked. Source model states copied byte-exact, no new model fits and no named pairs.',
      fixed='Same6000gross,50k own+50k reserve,base,one active pair,original lifecycle and shared accounting. Only proposal ranking changes.',
      selection='August worst Q then PnL with zero risk violations; choose before later comparison.',history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    paths=[Path(__file__),*[Path(__file__).with_name(x) for x in ['ranking.py','ranking_replay.py','ranking_evaluation.py']],OUT/'protocol.json']
    paths += [ROOT/'reports/pair_model_research_20261023'/v for v in SOURCES.values()]
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in paths})
    (OUT/'engine_diff.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='retained_replay.py',tofile='ranking_replay.py')))


def verify():
    verify_v2();verify_wave1()
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    return read(OUT/'protocol.json')


def build():
    import shutil
    verify()
    for arm,source in SOURCES.items():shutil.copyfile(ROOT/'reports/pair_model_research_20261023'/source,OUT/f'{arm}.npz')
    verify()


def stage_a():
    p=verify();evaluation.stage_a(OUT,p['arms']);verify()


def stage_b():
    verify();evaluation.stage_b(OUT);verify()


if __name__=='__main__':globals()[sys.argv[1]]()
