"""Recorded final development hypothesis; all arms and failures retained."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,OUT,START,END,BOUNDARY,read,save,sha,verify,SYMBOLS
from .carry_engine import simulate
from .carry_allocation import CarryAllocation
from pas_strategy_audit.data import load
from pas_performance_research.funding import forecast
from pas_cashflow_research.runner import persist

DEST=OUT/'carry_allocation'

def job(arm,scenario,switch):
    from competition_live.profile import load_profile
    from pas_performance_research.selected import controller
    verify();cfg,d=load(OUT/'dataset');f=dict(np.load(OUT/'dataset/candidate_inputs.npz'))
    funding=read(OUT/'dataset/funding.json')
    carry,_=forecast(f['times'],[dict(known=r['time']/1000+60,symbol=r['coin'][4:],rate=r['rate']) for r in funding],SYMBOLS,horizon=168,decay=True)
    old=controller(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000)
    c=old if arm=='incumbent' else CarryAllocation(f['times'],carry,old,BOUNDARY if switch else START,d['times']/1000,d['close'],d['age_ms']/1000,guarded=arm=='carry_adverse_guard')
    sc=read(OUT/'protocol.json')['scenarios'][scenario];tape=[]
    for s in SYMBOLS:
        for b in read(OUT/'dataset/raw'/f'pas_candles_{s}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and cfg['start_ms']<=at<=cfg['cutoff_ms']:
                tape.append(dict(time=at,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    # The strict fixed decision-price band is applied equally to ALL arms.
    # It is not advertised as reconstructed live mark-vs-book execution.
    r=simulate(cfg,d,read(OUT/'dataset/events.json'),tape,funding,load_profile()['trading'],sc,controller=c,
        entry_lifetime=900 if arm=='incumbent' else 3600,sticky_exits=arm!='incumbent',entry_band_bps=10,
        switch_ms=BOUNDARY*1000 if switch else None)
    a=next(x for x in r['snapshots'] if x['time']==BOUNDARY*1000);b=r['snapshots'][-1];cost=(sc['fee_bps']+sc['impact_bps'])/10000
    row=dict(arm=arm,scenario=scenario,phase='same_inventory_switch' if switch else 'continuous120h',**r['metrics'],
        tail_pnl=b['equity']-a['equity'],tail_liquidation_adjusted_pnl=b['equity']-a['equity']-(b['notional']-a['notional'])*cost)
    r['controller_trace']=c.trace;persist(DEST,row,r);return row

def main():
    verify();assert not (DEST/'protocol.json').exists()
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='carry_adverse_guard',diagnostic='carry_only',
        rule='All10markets. PASsettledfunding fast6h/slow24h agreement,min6observations,confidence24h,168hhorizonwith48hhalf-life. Forecastreceipts50%,payments100%. Quadraticriskpenalty5,entryandexitcostplus5bpbuffer. Primaryveto additions against24hprice move exceeding1dailyreferencevolatility. Same50kcapital,unchangedglobalriskandturnovercaps. Priorprotectedreference-modelcapsare replaced withthisnewalpha,not silently relaxedinsideincumbent.',
        caveat='Allhistoryexposed:developmentnotindependenttest. Quotesnotreconstructible:10bpbandisfromcompleteddecisionprice,fixedfortask,notproductionreanchoredmarklimit. Earlier5%resultsnoteligibleforlivepromotion. Noactualorders.',
        files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('carry_allocation.py'),Path(__file__).with_name('carry_engine.py')]},orders_sent=0,live_changed=False))
    rows=[]
    tasks=[(a,s,sw) for a in ['incumbent','carry_adverse_guard','carry_only'] for s in ['base','stress','no_receipts'] for sw in ([False] if a=='incumbent' else [False,True])]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,*args) for args in tasks]):
            row=f.result();rows.append(row);save(DEST/'results.json',rows)
            print({k:row[k] for k in ['arm','scenario','phase','net_pnl','Q','tail_pnl','funding']},flush=True)

if __name__=='__main__':main()
