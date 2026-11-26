"""Corrected equal-band experiment; earlier wide-band results are not promotion evidence."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import gzip,json
import numpy as np
from .study import ROOT,OUT,START,END,BOUNDARY,read,save,sha,verify,SYMBOLS
from .carry_engine import simulate
from pas_strategy_audit.data import load
from pas_cashflow_research.runner import persist
DEST=OUT/'strict_ttl'

def job(scenario,switch):
    from competition_live.profile import load_profile
    from pas_performance_research.selected import controller
    verify();cfg,d=load(OUT/'dataset');f=dict(np.load(OUT/'dataset/candidate_inputs.npz'))
    c=controller(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000)
    sc=read(OUT/'protocol.json')['scenarios'][scenario];tape=[]
    for s in SYMBOLS:
        for b in read(OUT/'dataset/raw'/f'pas_candles_{s}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and cfg['start_ms']<=at<=cfg['cutoff_ms']:
                tape.append(dict(time=at,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    r=simulate(cfg,d,read(OUT/'dataset/events.json'),tape,read(OUT/'dataset/funding.json'),load_profile()['trading'],sc,
        controller=c,entry_lifetime=3600,sticky_exits=True,entry_band_bps=10,switch_ms=BOUNDARY*1000 if switch else None)
    a=next(x for x in r['snapshots'] if x['time']==BOUNDARY*1000);b=r['snapshots'][-1];cost=(sc['fee_bps']+sc['impact_bps'])/10000
    checks={}
    if switch:
        with gzip.open(OUT/'carry_allocation/runs'/f'incumbent_continuous120h_{scenario}.json.gz','rt') as fp:baseline=json.load(fp)
        for field in ['fills','daily','decisions','snapshots']:
            def prefix(run):return [x for x in run[field] if (x['time'] if field in ['fills','daily'] else x['time']/1000)<=BOUNDARY]
            checks[field]=prefix(r)==prefix(baseline)
        assert all(checks.values()),checks
    row=dict(arm='incumbent_ttl1h',scenario=scenario,phase='same_inventory_switch' if switch else 'continuous120h',**r['metrics'],
        tail_pnl=b['equity']-a['equity'],tail_liquidation_adjusted_pnl=b['equity']-a['equity']-(b['notional']-a['notional'])*cost,identical_incumbent_history=checks)
    r['controller_trace']=c.trace;persist(DEST,row,r);return row

def main():
    verify();assert not (DEST/'protocol.json').exists()
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='incumbent_ttl1h',
        amendment='Equal10bpdecisionpricebandforbaselineandcandidatebeforeandafterswitch. Removes5%guardmismatch,doesnotclaimactualrefreshingmark/L2parity. NoTTL,signal,model,riskorcapitalparameterschanged.',
        exposure='Previouslyinspecteddevelopmenthistory;notnewindependenttest.',
        files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('carry_engine.py')]},orders_sent=0))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,s,sw) for s in ['base','stress','no_receipts'] for sw in [False,True]]):
            row=f.result();rows.append(row);save(DEST/'results.json',rows)
            print({k:row[k] for k in ['scenario','phase','net_pnl','Q','tail_pnl','mean_notional']},flush=True)

if __name__=='__main__':main()
