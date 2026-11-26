"""Predeclared two-arm price-chasing repair, exposed-history development only."""
import gzip,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
from .study import ROOT,OUT,START,END,BOUNDARY,read,save,sha,verify,SYMBOLS
from .price_bound_engine import simulate
from pas_strategy_audit.data import load
from pas_cashflow_research.runner import persist

def job(band,scenario,switch):
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
    r=simulate(cfg,d,read(OUT/'dataset/events.json'),tape,read(OUT/'dataset/funding.json'),load_profile()['trading'],sc,controller=c,entry_lifetime=3600,sticky_exits=True,switch_ms=BOUNDARY*1000 if switch else None,entry_band_bps=band)
    a=next(x for x in r['snapshots'] if x['time']==BOUNDARY*1000);b=r['snapshots'][-1];cost=(sc['fee_bps']+sc['impact_bps'])/10000
    row=dict(arm=f'entry_band{band}',scenario=scenario,phase='same_inventory_switch' if switch else 'continuous120h',**r['metrics'],
        tail_pnl=b['equity']-a['equity'],tail_liquidation_adjusted_pnl=b['equity']-a['equity']-(b['notional']-a['notional'])*cost)
    persist(OUT/'price_bound',row,r);return row

def main():
    dest=OUT/'price_bound';assert not (dest/'protocol.json').exists()
    save(dest/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='entry_band10',diagnostic='entry_band0',
        reason='One-hour continuation allows repricing far beyond initial judgement. Keep initial price limit fixed toone baseline impact allowance10bp. Zero chase allowance as diagnostic. Risk exits bypass this entry-only limit.',
        scope='All outcome prices have been exposed. Development and mechanism study only. No reclassification as untouched test. No horizon/model/symbol/capital/risk fitting.',
        acceptance='Primary improves baseline netPnL,Q and post-switch PnL/terminal-adjustedPnL in base,stress,no_receipts. Diagnostic cannot silently replace failed primary.',
        files={str(p.relative_to(ROOT)):sha(p) for p in [__import__('pathlib').Path(__file__),__import__('pathlib').Path(__file__).with_name('price_bound_engine.py')]},orders_sent=0))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,b,s,sw) for b in [10,0] for s in ['base','stress','no_receipts'] for sw in [False,True]]):
            row=f.result();rows.append(row);save(dest/'results.json',rows)
            print({k:row[k] for k in ['arm','scenario','phase','net_pnl','Q','tail_pnl']},flush=True)

if __name__=='__main__':main()
