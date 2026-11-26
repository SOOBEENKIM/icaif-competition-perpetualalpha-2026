"""New economic hypothesis on exposed history; all results retained."""
import gzip,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
from pas_strategy_audit.data import load
from competition_live.profile import load_profile
from pas_performance_research.selected import controller
from pas_performance_research.funding import forecast
from execution_revision_20261024.carry_engine import simulate
from .neutral_carry import NeutralCarry
from .protocol import ROOT,OUT,DATA,save,sha

DEST=OUT/'neutral_carry'

def job(scenario,switch):
    cfg,d=load(DATA);profile=load_profile();p=profile['trading'];symbols=p['symbols']
    funding=json.loads((DATA/'funding.json').read_text());f=dict(np.load(DATA/'candidate_inputs.npz'))
    carry,counts=forecast(f['times'],[dict(known=r['time']/1000+60,symbol=r['coin'][4:],rate=r['rate']) for r in funding],symbols,horizon=168,decay=False)
    old=controller(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000)
    activation=1792767600 if switch else cfg['start_ms']//1000
    c=NeutralCarry(f['times'],carry,old,activation,d['times']/1000,d['close'],d['age_ms']/1000)
    sc=json.loads((OUT/'protocol.json').read_text())['scenarios'][scenario]
    tape=[]
    for s in symbols:
        for b in json.loads((DATA/'raw'/f'pas_candles_{s}.json').read_text())['data']:
            if b['n'] and float(b['v'])>0 and cfg['start_ms']<=b['t']+60000<=cfg['cutoff_ms']:
                tape.append(dict(time=b['t']+60000,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    # The inherited engine can bind new entries to forecast terminal cash.
    # For the handover prefix its economic caps must remain baseline10bp.
    # This experiment starts both paths under the same old engine; after
    # switch uses new carry alpha and economic budget, explicitly different.
    if switch:
        # source-specific carry=0 would make entry impossible before switch;
        # defer handover to production version after the continuous screen.
        raise ValueError('No handover claim from an engine with different pre-switch limits')
    r=simulate(cfg,d,json.loads((DATA/'events.json').read_text()),tape,funding,p,sc,controller=c,
        entry_lifetime=3600,economic_limits=True,sticky_exits=True)
    r['controller_trace']=c.trace
    row=dict(arm='neutral_carry168',scenario=scenario,**r['metrics'])
    with gzip.open(DEST/f'{scenario}.json.gz','wt') as file:json.dump(r,file,allow_nan=False,default=lambda x:x.tolist())
    return row

def main():
    assert not (DEST/'protocol.json').exists()
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Cross-asset neutral funding carry may earn persistent PAS cash flows while offsetting some directional exposure. Carry72/48hdecay rejected before costs in old study; rate autocorrelation observed in development motivates retained168h horizon with explicit50%forecast receipts, no imposed48hdecay. This is a new hypothesis, not a correction of prior test results.',
        model='Past6h/24h settled funding estimates must agree; at least6observations, confidence24hours,168h horizon. No future-funded labels or winners. All45 pair-coordinate directions evaluated jointly. No cointegration claim.',
        risk='Same50k, gross.6/net.4/asset.15/7500, dailyvol.005, turnover.15. Neutral intended dollars; partial fills create net exposure bounded by shared caps.',
        costs='Sealed new-entry price limit from projected half receipts, full funding liabilities, entry+exit fees, exit impact and5bpbuffer. Paid partial execution; no queue/depth reconstruction. IOCbest10bp upper bound absent in historical trade-volume engine, so screen alone cannot support deployment.',
        status='Exposed-history development only. Must port to actual execution chain and validate new data before promotion.',
        files={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'production_replay_20261025/neutral_carry.py',ROOT/'production_replay_20261025/carry_study.py']},orders_sent=0))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,s,False) for s in ['base','stress','no_receipts']]):
            r=f.result();rows.append(r);save(DEST/'results.json',rows)
            print({k:r[k] for k in ['arm','scenario','net_pnl','Q','fees','funding','mean_notional','risk_breach_events']},flush=True)

if __name__=='__main__':main()
