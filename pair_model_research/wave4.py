import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from auto_pair_validation.study import OUT as V2,verify as verify_v2
from .study import ROOT,read,save,sha,verify as verify_wave1
from .relationships import evaluate
from . import evaluation

OUT=ROOT/'reports/pair_model_research_20261023/wave4'
ARMS=['stable_relationships','mutual_peers']


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=verify_v2();verify_wave1()
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),wave=4,arms=ARMS,
      input=p['input'],input_sha256=p['input_sha256'],
      hypothesis='All45 raw expansion admitted unstable cross-market relative predictions. Learn a dynamic peer graph from past synchronous returns, then compare original return/SPO signals. Raw-beta bounds in the first formation rule are not a valid test of the actual clipped hedge; remove that particular screen, retain source-beta execution/risk unchanged.',
      formation='Daily90d,three30d blocks,at least80% synchronous active hours,positive correlation>=.2 each block,median each-leg notional>=4000 as in initial rule. No beta-level or beta-ratio screen. These are relationship diagnostics, NOT cointegration proof.',
      arms_description='stable_relationships considers all relationships passing formation; mutual_peers additionally requires mutual top2 peers per asset ranked by minimum three-block correlation. Exact ties included. Number and names of eligible pairs are dynamic; no requirement to trade.',
      action='Raw all45 causal per-pair ridge + SPO gate; new-entry eligibility only. Held forecasts and original paid renewal/exit stay available if a pair leaves graph.',
      fixed='Same6000 pair gross,50k own+50k reserve,24h horizon,fees,funding,base model,delays,partial fills and risk limits.',
      selection='August worst Q advantage then PnL; risk violations rejected. Lock before later development comparison. Keep every failure.',
      history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),Path(__file__).with_name('relationships.py'),Path(__file__).with_name('evaluation.py'),OUT/'protocol.json']})


def verify():
    verify_v2();verify_wave1()
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    return read(OUT/'protocol.json')


def build():
    p=verify();v=read(V2/'protocol.json');d=dict(np.load(ROOT/p['input'],allow_pickle=False));t=d['market_times']
    s=dict(np.load(V2/'states.npz',allow_pickle=False));masks={a:np.zeros_like(s['gate_available']) for a in ARMS};records=[]
    for day in np.unique(t[t>=v['evaluation_start']]//86400*86400):
        r=evaluate(d,int(day));records.append(r);ix=(t>=day)&(t<day+86400)
        masks['stable_relationships'][ix]=r['stable'];masks['mutual_peers'][ix]=r['mutual']
    for arm in ARMS:
        out={k:s[k] for k in ['times','dense','pairs','signal_times','source_prices','beta','vol','forecast_times','directions']}
        out['mu']=s['raw_mu'];available=s['gate_available']&masks[arm]
        out['economic_available']=available;out['economic_action']=np.where(available,s['gate_action'],0);out['relationship_eligible']=masks[arm]
        np.savez_compressed(OUT/f'{arm}.npz',**out)
    save(OUT/'relationship_history.json',records);verify()
    print('built',len(records),'days x45 relationships',flush=True)


def stage_a():
    p=verify();evaluation.stage_a(OUT,p['arms']);verify()


def stage_b():
    verify();evaluation.stage_b(OUT);verify()


if __name__=='__main__':globals()[sys.argv[1]]()
