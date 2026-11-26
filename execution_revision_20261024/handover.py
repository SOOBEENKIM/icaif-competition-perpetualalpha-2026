"""Same incumbent holdings before the switch; no retrospective candidate inventory."""
import gzip
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
from .study import ROOT,OUT,BOUNDARY,read,save,sha,verify,SYMBOLS
from .handover_engine import simulate
from pas_strategy_audit.data import load
from pas_cashflow_research.runner import persist

def job(scenario):
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
    r=simulate(cfg,d,read(OUT/'dataset/events.json'),tape,read(OUT/'dataset/funding.json'),load_profile()['trading'],sc,controller=c,entry_lifetime=3600,sticky_exits=True,switch_ms=BOUNDARY*1000)
    with gzip.open(OUT/'evaluation/runs'/f'incumbent_continuous120h_{scenario}.json.gz','rt') as fp:old=__import__('json').load(fp)
    checks={}
    for field in ['fills','daily','decisions','snapshots']:
        def prefix(run):return [x for x in run[field] if (x['time'] if field in ['fills','daily'] else x['time']/1000)<=BOUNDARY]
        checks[field]=prefix(r)==prefix(old)
    assert all(checks.values()),checks
    a=next(s for s in r['snapshots'] if s['time']==BOUNDARY*1000);b=r['snapshots'][-1];cost=(sc['fee_bps']+sc['impact_bps'])/10000
    row=dict(arm='actual_inventory_switch',scenario=scenario,phase='continuous120h',**r['metrics'],
        tail_pnl=b['equity']-a['equity'],tail_liquidation_adjusted_pnl=b['equity']-a['equity']-(b['notional']-a['notional'])*cost,
        tail_start_notional=a['notional'],tail_end_notional=b['notional'],tail_fees=b['fees']-a['fees'],tail_funding=b['funding']-a['funding'],
        identical_incumbent_history=checks,tail_fills=sum(x['time']>BOUNDARY for x in r['fills']))
    persist(OUT/'handover',row,r);return row

def main():
    path=OUT/'handover/protocol.json';assert not path.exists()
    save(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),switch_time=BOUNDARY,
        scope='Post-exposure transferability diagnostic, not independent selection. Same baseline historical holdings,costs,entries and schedule up to switch. Unsubmitted prior intent dropped. No change in selected parameters.',
        files={str(p.relative_to(ROOT)):sha(p) for p in [__import__('pathlib').Path(__file__),__import__('pathlib').Path(__file__).with_name('handover_engine.py')]},orders_sent=0))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,s) for s in read(OUT/'protocol.json')['scenarios']]):
            row=f.result();rows.append(row);save(OUT/'handover/results.json',rows)
            print({k:row[k] for k in ['scenario','net_pnl','Q','tail_pnl','tail_liquidation_adjusted_pnl']},flush=True)

if __name__=='__main__':main()
