import copy,gzip,json,sys
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .data import ROOT,OUT,OLD,REF,START,DEV_END,TEST_END,SYMBOLS,read,save,sha,verify
from .policy import CashflowController
from pas_strategy_audit.data import load
from pas_performance_research.funding import forecast as funding_forecast

ARMS=['incumbent','funding_basket72','funding_pairs72','basis_funding_basket72','trend_funding_basket72']

def prepare():
    from pas_performance_research import extension
    verify();directory=OUT/'development'
    save(directory/'protocol.json',dict(input=str((REF/'inputs.npz').relative_to(ROOT)),model_inputs=str((REF/'targets.npz').relative_to(ROOT))))
    extension.OUT=directory;extension.END=DEV_END;extension.verify=verify
    extension.prepare()
    pair_audit=read(directory/'pair_audit.json')
    assert pair_audit['can_use_base_only'],'Cannot ignore nonzero incumbent pair orders'
    cfg,d=load(directory);f=dict(np.load(directory/'candidate_inputs.npz'));funding=read(directory/'funding.json');t=f['times']
    obs=[dict(known=r['time']/1000+60,symbol=r['coin'][4:],rate=r['rate']) for r in funding]
    carry,counts=funding_forecast(t,obs,SYMBOLS,horizon=72,decay=True)
    raw=dict(np.load(REF/'inputs.npz'));ri=np.searchsorted(raw['market_times'],t);assert np.array_equal(raw['market_times'][ri],t)
    basis=np.zeros_like(carry);trend=np.zeros_like(carry);fresh=np.zeros_like(carry,bool);audit=[]
    for i,source in enumerate(t):
        mi=int(np.searchsorted(d['times'],source*1000,side='right')-1)
        assert mi>=0
        fresh[i]=d['age_ms'][mi]<=3600000
        previous=np.searchsorted(raw['market_times'],source-72*3600,side='right')-1
        if previous>=0:trend[i]=.25*np.log(raw['close'][ri[i]]/raw['close'][previous])
        for j,symbol in enumerate(SYMBOLS):
            known=[r for r in funding if r['coin']=='pas:'+symbol and r.get('available',np.inf)<=source*1000 and r.get('oracle')]
            if not known:continue
            r=known[-1]
            if source*1000-r['available']>2*3600000:continue
            refi=np.searchsorted(raw['market_times'],r['time']//3600000*3600,side='right')-1
            oracle=r['oracle']*raw['close'][ri[i],j]/raw['close'][refi,j]
            if not np.isfinite(d['close'][mi,j]) or d['close'][mi,j]<=0:continue
            basis[i,j]=-.5*np.log(d['close'][mi,j]/oracle)
            audit.append(dict(source=int(source),symbol=symbol,oracle_observed=int(r['available']),oracle_projection=oracle,basis_return=basis[i,j]))
    sigma=np.array([r['sigma'] for r in read(directory/'events.json')]);limit=2*np.sqrt(np.maximum(np.diagonal(sigma,axis1=1,axis2=2),1e-12)*72)
    basis=np.clip(basis,-limit,limit);trend=np.clip(trend,-limit,limit)
    np.savez_compressed(directory/'venue_features.npz',times=t,carry=carry,basis=basis,trend=trend,fresh=fresh,counts=counts)
    save(directory/'features_audit.json',audit)
    save(OUT/'wave1_freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in [*list((ROOT/'pas_cashflow_research').glob('*.py')),directory/'venue_features.npz',directory/'candidate_inputs.npz',directory/'events.json',directory/'funding.json',directory/'pair_audit.json']})
    print('Prepared PAS10, oracle projections',len(audit),'retained pair gates all zero',flush=True)

def validate():
    verify()
    for p,h in read(OUT/'wave1_freeze.json').items():assert sha(ROOT/p)==h,p

def job(arm,scenario,phase,start,end):
    from pas_counterfactual_20261022.engine import simulate
    from competition_live.profile import load_profile
    from pas_performance_research.selected import controller as incumbent
    validate();directory=OUT/'development';cfg,d=load(directory);cfg.update(start_ms=start*1000,cutoff_ms=end*1000)
    p=load_profile()['trading'];sc=read(OUT/'protocol.json')['fees'][scenario]
    f=dict(np.load(directory/'candidate_inputs.npz'));z=dict(np.load(directory/'venue_features.npz'))
    events=read(directory/'events.json');funding=read(directory/'funding.json');tape=[]
    for s in SYMBOLS:
        for b in read(directory/'raw'/f'pas_candles_{s}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and start*1000<=at<=end*1000:
                tape.append(dict(time=at,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    if arm=='incumbent':
        c=incumbent(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000)
    else:
        price=z['basis'] if arm.startswith('basis') else z['trend'] if arm.startswith('trend') else np.zeros_like(z['carry'])
        c=CashflowController(z['times'],price,z['carry'],z['fresh'],arm,impact_bps=sc['impact_bps'])
    r=simulate(cfg,d,events,tape,funding,p,sc,controller=c)
    row=dict(arm=arm,scenario=scenario,phase=phase,**r['metrics']);r['controller_trace']=c.trace
    assert row['timing_pass'] and row['maximum_accounting_error']<1e-6
    path=OUT/'wave1'/'runs'/f'{arm}_{phase}_{scenario}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda v:v.tolist())
    save(OUT/'wave1'/'rows'/f'{arm}_{phase}_{scenario}.json',row)
    return row

def run():
    validate();path=OUT/'wave1'/'started.json'
    if path.exists():raise FileExistsError('Do not silently rerun selection')
    save(path,dict(time=datetime.now(timezone.utc).isoformat(),freeze=sha(OUT/'wave1_freeze.json')))
    jobs=[(a,s,ph,lo,hi) for a in ARMS for s in ['base','stress','no_receipts'] for ph,lo,hi in [('selection48h',START,START+48*3600),('later44h',START+48*3600,DEV_END),('continuous92h',START,DEV_END)]]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,*j) for j in jobs]):
            r=f.result();rows.append(r);save(OUT/'wave1'/'results.json',rows)
            print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','fills','risk_breach_events']}),flush=True)
    validate()

if __name__=='__main__':globals()[sys.argv[1]]()
