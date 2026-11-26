import copy,gzip,json,sys
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .data import ROOT,OUT,START,DEV_END,SYMBOLS,read,save,sha
from .study import validate
from .policy_v2 import VenueController
from .model_v2 import fit_predict
from pas_strategy_audit.data import load
from pas_performance_research.funding import forecast

WAVE=OUT/'wave2'
ARMS=['pas_momentum24','pas_ridge6','pas_ridge24','pas_ridge_basis24','funding_receipts72']

def prepare():
    validate()
    if (WAVE/'protocol.json').exists():raise FileExistsError('Preserve wave plan')
    save(WAVE/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,
        reason='Wave1 reference72htrend failed later PAS; carry72h with50% receipts and48hdecay could not cover assumed roundtrip costs. Test venue-price dynamics, not another reference-pair forecast.',
        train='PAS-only, all10 pooled; previous72hours,6hourrefits,strict horizon maturity<fit,sourceandterminalfresh<=1h; source-balanced weights24hdecay; min60rows6distinctsources. Shared7features,noassetidentifier,ridge1,intercept.25,clip4sigma.',
        arms_definition='PASmomentum24=quarter of last24hPAS logreturn. ridge6/ridge24=6h/24h actualPAS price return forecast. ridge_basis24=half ridge24+half contemporaneous oracle basis convergence. funding_receipts72 removes duplicate receipt haircut in prediction only; actualstress still halves receipts.',
        selection='First48h development ranking, all later44handcontinuous92h disclosed. This wave is subsequent research on exposed data. Four-hour uninspectedPAS tail remains unopened.',
        fixed='Same absolute capital/risk,6hdecision,original asymmetric settlementcash and conservative minute execution; no livechanges.'))
    directory=OUT/'development';cfg,d=load(directory);f=dict(np.load(directory/'candidate_inputs.npz'));z=dict(np.load(directory/'venue_features.npz'))
    t=np.arange((START-21600),DEV_END+1,3600);mi=np.searchsorted(d['times'],t*1000,side='right')-1
    good=mi>=0;px=np.full((len(t),10),np.nan);fresh=np.zeros_like(px,bool)
    px[good]=d['close'][mi[good]];fresh[good]=d['age_ms'][mi[good]]<=3600000
    funding=read(directory/'funding.json');obs=[dict(known=r['time']/1000+60,symbol=r['coin'][4:],rate=r['rate']) for r in funding]
    basis=np.zeros_like(px);ii=np.searchsorted(t,z['times']);basis[ii]=z['basis']
    arrays={'times':f['times']};audits={}
    for h in [6,24,72]:
        carry,_=forecast(t,obs,SYMBOLS,horizon=h,decay=True)
        if h!=72:
            pred,audit=fit_predict(t,px,fresh,carry,basis,h);audits[str(h)]=audit
            arrays[f'ridge{h}']=pred[ii];arrays[f'carry{h}']=carry[ii]
        else:arrays['carry72']=carry[ii]
    momentum=np.full_like(px,np.nan);momentum[24:]=.25*np.log(px[24:]/px[:-24])
    arrays['momentum24']=momentum[ii];arrays['basis']=basis[ii];arrays['fresh']=fresh[ii]
    np.savez_compressed(WAVE/'forecasts.npz',**arrays);save(WAVE/'model_audit.json',audits)
    save(WAVE/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'pas_cashflow_research/policy_v2.py',ROOT/'pas_cashflow_research/model_v2.py',ROOT/'pas_cashflow_research/wave2.py',WAVE/'protocol.json',WAVE/'forecasts.npz']})

def verify():
    validate()
    for p,h in read(WAVE/'freeze.json').items():assert sha(ROOT/p)==h,p

def job(arm,scenario,phase,start,end):
    from pas_counterfactual_20261022.engine import simulate
    from competition_live.profile import load_profile
    verify();directory=OUT/'development';cfg,d=load(directory);cfg.update(start_ms=start*1000,cutoff_ms=end*1000)
    sc=read(OUT/'protocol.json')['fees'][scenario];p=load_profile()['trading'];z=dict(np.load(WAVE/'forecasts.npz'))
    h=6 if arm=='pas_ridge6' else 72 if arm=='funding_receipts72' else 24
    pred=z['momentum24'] if arm=='pas_momentum24' else np.zeros_like(z['carry72']) if h==72 else z[f'ridge{h}']
    if arm=='pas_ridge_basis24':pred=.5*pred+.5*z['basis']
    valid=z['fresh']&np.isfinite(pred);pred=np.nan_to_num(pred,nan=0.)
    c=VenueController(z['times'],pred,z[f'carry{h}'],valid,arm,horizon=h,receipt=1. if h==72 else .5,impact_bps=sc['impact_bps'])
    tape=[]
    for s in SYMBOLS:
        for b in read(directory/'raw'/f'pas_candles_{s}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and start*1000<=at<=end*1000:
                tape.append(dict(time=at,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    r=simulate(cfg,d,read(directory/'events.json'),tape,read(directory/'funding.json'),p,sc,controller=c)
    row=dict(arm=arm,phase=phase,scenario=scenario,**r['metrics']);r['controller_trace']=c.trace
    path=WAVE/'runs'/f'{arm}_{phase}_{scenario}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    save(WAVE/'rows'/f'{arm}_{phase}_{scenario}.json',row);return row

def run():
    verify();marker=WAVE/'started.json'
    if marker.exists():raise FileExistsError('Wave already run')
    save(marker,dict(at=datetime.now(timezone.utc).isoformat()))
    jobs=[(a,s,ph,lo,hi) for a in ARMS for s in ['base','stress','no_receipts'] for ph,lo,hi in [('selection48h',START,START+48*3600),('later44h',START+48*3600,DEV_END),('continuous92h',START,DEV_END)]]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,*args) for args in jobs]):
            r=f.result();rows.append(r);save(WAVE/'results.json',rows)
            print(json.dumps({k:r[k] for k in ['arm','scenario','phase','net_pnl','Q','fills','risk_breach_events']}),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
