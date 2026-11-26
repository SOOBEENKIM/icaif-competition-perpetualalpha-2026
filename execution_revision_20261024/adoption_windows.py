"""All four daily24h handovers; post-exposure robustness, no selection."""
import gzip,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
from .study import ROOT,OUT,START,END,read,save,sha,verify,SYMBOLS
from .handover_engine import simulate
from pas_strategy_audit.data import load

def job(origin,scenario):
    from competition_live.profile import load_profile
    from pas_performance_research.selected import controller
    verify();cfg,d=load(OUT/'dataset');cfg['cutoff_ms']=(origin+86400)*1000
    f=dict(np.load(OUT/'dataset/candidate_inputs.npz'))
    c=controller(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000)
    sc=read(OUT/'protocol.json')['scenarios'][scenario];tape=[]
    for s in SYMBOLS:
        for b in read(OUT/'dataset/raw'/f'pas_candles_{s}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and cfg['start_ms']<=at<=cfg['cutoff_ms']:
                tape.append(dict(time=at,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    r=simulate(cfg,d,read(OUT/'dataset/events.json'),tape,read(OUT/'dataset/funding.json'),load_profile()['trading'],sc,controller=c,entry_lifetime=3600,sticky_exits=True,switch_ms=origin*1000)
    with gzip.open(OUT/'evaluation/runs'/f'incumbent_continuous120h_{scenario}.json.gz','rt') as fp:old=json.load(fp)
    a=next(s for s in old['snapshots'] if s['time']==origin*1000)
    aa=next(s for s in r['snapshots'] if s['time']==origin*1000)
    assert a==aa,'Common starting inventory/cash not preserved'
    b=next(s for s in old['snapshots'] if s['time']==cfg['cutoff_ms']);bb=r['snapshots'][-1]
    cost=(sc['fee_bps']+sc['impact_bps'])/10000
    row=dict(origin=origin,end=origin+86400,scenario=scenario,baseline_pnl=b['equity']-a['equity'],candidate_pnl=bb['equity']-a['equity'],
        pnl_advantage=bb['equity']-b['equity'],liquidation_adjusted_advantage=bb['equity']-b['equity']-(bb['notional']-b['notional'])*cost,
        start_notional=a['notional'],baseline_end_notional=b['notional'],candidate_end_notional=bb['notional'],
        risk_breach_events=r['metrics']['risk_breach_events'],timing_pass=r['metrics']['timing_pass'],common_inventory_verified=True)
    save(OUT/'adoption_windows/rows'/f'{origin}_{scenario}.json',row);return row

def main():
    dest=OUT/'adoption_windows';assert not (dest/'protocol.json').exists()
    origins=list(range(START+86400,END,86400))
    save(dest/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),origins=origins,
        scope='All available four daily24h switches, fixed before these branch runs. Previously exposed market history. No independent-test claim or favourable start-date selection.',code_sha256=sha(__file__),orders_sent=0))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,t,s) for t in origins for s in ['base','stress']]):
            row=f.result();rows.append(row);save(dest/'results.json',rows);print(row,flush=True)

if __name__=='__main__':main()
