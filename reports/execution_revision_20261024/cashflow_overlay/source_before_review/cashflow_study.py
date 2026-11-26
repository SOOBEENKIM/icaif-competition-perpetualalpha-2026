"""Two recorded economic hypotheses, exposed-data development, no tuning loop."""
import gzip,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
from .study import ROOT,OUT,START,END,BOUNDARY,read,save,sha,verify,SYMBOLS
from .handover_engine import simulate
from .cashflow_overlay import CashflowOverlay
from pas_strategy_audit.data import load
from pas_cashflow_research.runner import persist
from pas_performance_research.funding import forecast

def job(decay,scenario,switch):
    from competition_live.profile import load_profile
    verify();cfg,d=load(OUT/'dataset');f=dict(np.load(OUT/'dataset/candidate_inputs.npz'))
    funding=read(OUT/'dataset/funding.json');obs=[dict(known=r['time']/1000+60,symbol=r['coin'][4:],rate=r['rate']) for r in funding]
    carry,counts=forecast(f['times'],obs,SYMBOLS,horizon=168,decay=decay)
    c=CashflowOverlay(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],5.,price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000,entry_rule='pullback_any3',venue_carry=carry,activation=BOUNDARY if switch else START)
    sc=read(OUT/'protocol.json')['scenarios'][scenario];tape=[]
    for s in SYMBOLS:
        for b in read(OUT/'dataset/raw'/f'pas_candles_{s}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and cfg['start_ms']<=at<=cfg['cutoff_ms']:
                tape.append(dict(time=at,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    r=simulate(cfg,d,read(OUT/'dataset/events.json'),tape,funding,load_profile()['trading'],sc,controller=c,entry_lifetime=3600,sticky_exits=True,switch_ms=BOUNDARY*1000 if switch else None)
    a=next(x for x in r['snapshots'] if x['time']==BOUNDARY*1000);b=r['snapshots'][-1];cost=(sc['fee_bps']+sc['impact_bps'])/10000
    row=dict(arm='cashflow_decay' if decay else 'cashflow_persistence',scenario=scenario,phase='same_inventory_switch' if switch else 'continuous120h',**r['metrics'],
        tail_pnl=b['equity']-a['equity'],tail_liquidation_adjusted_pnl=b['equity']-a['equity']-(b['notional']-a['notional'])*cost,
        added_decisions=sum(bool(t.get('cashflow_added_symbols')) for t in c.trace))
    r['controller_trace']=c.trace;persist(OUT/'cashflow_overlay',row,r);return row

def main():
    dest=OUT/'cashflow_overlay';assert not (dest/'protocol.json').exists()
    save(dest/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='cashflow_decay',diagnostic='cashflow_persistence',
        reason='Current allocator reserves nonzero anchor slots even when more same-side risk has positive forecast utility. Test a PAS-funding cash-flow top-up while preserving protected caps, incumbent sides, inventory,6hcadence and entry filter.',
        definitions='Reuse frozen168hprice forecast,6h/24hsettledPASfunding estimates with60sobservationdelay,min6observations/confidenceramp24h. Primary48hhalf-life,diagnosticconstantcarry over168h. Expectedreceipts50%,payments100%. Same capital/risk; no refit,noassetIDs.',
        scope='Exposed development only; all variants/failures retained. No independent-test claim and no replacement of failed primary by diagnostic.',
        files={str(p.relative_to(ROOT)):sha(p) for p in [__import__('pathlib').Path(__file__),__import__('pathlib').Path(__file__).with_name('cashflow_overlay.py')]},orders_sent=0))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,de,s,sw) for de in [True,False] for s in ['base','stress','no_receipts'] for sw in [False,True]]):
            row=f.result();rows.append(row);save(dest/'results.json',rows)
            print({k:row[k] for k in ['arm','scenario','phase','net_pnl','Q','tail_pnl','added_decisions']},flush=True)

if __name__=='__main__':main()
