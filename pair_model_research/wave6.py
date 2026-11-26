import sys,difflib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from auto_pair_validation.study import OUT as V2,verify as verify_v2
from auto_pair_selection.selection import IDS
from .study import ROOT,read,save,sha,verify as verify_wave1
from .stable_reversion import forecasts
from . import execution_evaluation as evaluation

OUT=ROOT/'reports/pair_model_research_20261023/wave6'
ARMS=['stable_reversion','stable_reversion_exit','reversion_exit']


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=verify_v2();verify_wave1()
    original=(ROOT/'pas_economic_allocation/replay.py').read_text();new=Path(__file__).with_name('early_exit_replay.py').read_text()
    restored=new.replace('from .early_exit import reassess\n','').replace('                    state=reassess(state,pair,pair_ledger,i,d,de,cfg,p,sc)\n','')
    assert restored==original,'Unexpected engine change'
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),wave=6,arms=ARMS,input=p['input'],input_sha256=p['input_sha256'],
      hypothesis='All45 equal-dollar AR72 improves continuous base PnL/Q but fails stress. Require disjoint30d relationship forecasts to agree; reassess losing holding value from fresh forecasts instead of only near maturity.',
      stability='Current past30d AR and preceding disjoint30d AR evaluated at current spread; both signed forecasts must agree and both exceed43bp. Current mean remains available for held inventory. This heuristic does not prove stationarity. Same current AR fit and72h horizon as wave2.',
      early_exit='After3h actual holding, inspect each new completed-hour current forecast; if predicted remaining cash after funding/close cost is worse than immediate liquidation by5bp gross, for two consecutive distinct hourly sources, request paid liquidation through the same delayed partial-fill engine. Existing stops take priority; never bypass completion/repair/rehedge handling.',
      arms_description='stable_reversion isolates relationship agreement; stable_reversion_exit adds value reassessment; reversion_exit isolates reassessment on wave2 all45 AR.',
      fixed='6000gross,one active pair,50k own+50k reserve,base signal and hard risk unchanged. Early exit changes order timing and incurs actual modeled costs. Engine diff is only one state reassessment call.',
      selection='August worst Q then PnL, reject risk violations, save selection before later comparison; all outcomes development.',history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    paths=[Path(__file__),*[Path(__file__).with_name(x) for x in ['stable_reversion.py','early_exit.py','early_exit_replay.py','execution_evaluation.py']],OUT/'protocol.json']
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in paths})
    (OUT/'engine_diff.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='retained_replay.py',tofile='early_exit_replay.py')))


def verify():
    verify_v2();verify_wave1()
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    return read(OUT/'protocol.json')


def build():
    p=verify();d=dict(np.load(ROOT/p['input']));t=d['market_times'];mu,f,entry,records=forecasts(d)
    common=dict(times=t,dense=True,pairs=np.array(IDS),signal_times=np.broadcast_to(t[:,None],mu.shape),source_prices=d['close'],beta=np.ones_like(mu),vol=f['vol'],forecast_times=t,mu=mu,directions=np.isfinite(mu).astype(int))
    available=np.isfinite(mu)&f['valid'];actions=np.where(available&entry&(abs(mu)>.0038),np.sign(np.nan_to_num(mu)),0).astype(int)
    for arm in ARMS:
        if arm=='reversion_exit':
            out=dict(np.load(ROOT/'reports/pair_model_research_20261023/wave2/reversion_72.npz'))
        else:out=dict(common,economic_available=available,economic_action=actions)
        np.savez_compressed(OUT/f'{arm}.npz',**out)
    save(OUT/'model_audit.json',records);verify();print('built',len(records),'model records',flush=True)


def stage_a():
    p=verify();evaluation.stage_a(OUT,p['arms']);verify()


def stage_b():
    verify();evaluation.stage_b(OUT);verify()


if __name__=='__main__':globals()[sys.argv[1]]()
