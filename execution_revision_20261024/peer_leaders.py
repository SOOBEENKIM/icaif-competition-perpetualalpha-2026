"""Daily lagged leader cohort, never today's terminal top10 retroactively."""
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from . import peer_consensus as peers
from .study import ROOT,OUT,START,END,BOUNDARY,read,save,sha,verify,SYMBOLS
from pas_strategy_audit.data import load

DEST=OUT/'peer_leaders'

def freeze():
    assert not (DEST/'protocol.json').exists();peers.check()
    p=read(peers.DEST/'protocol.json').copy()
    p.update(created_utc=datetime.now(timezone.utc).isoformat(),primary='daily_lagged_positive_price_leaders',
        rule='At source t use only last12UTCsnapshot strictly earlier than t-120s. Rank all funded nonself wallets with at least24hof observed activity by gross pricePnL minus fees (funding excluded, explicitly not officialPnL). Choose positive topquartile, minimum3. Average current signed unitgross positions of those leaders, same60%own target and risk caps. No positive eligible cohort -> hold feasible own holdings; no forced reversal without close.',
        scope='Second predeclared peer-information hypothesis, frozen before peer-path results. Development history remains exposed; no untouched-test claim. Selection uses causal daily prices and fills, never final chart ranks.')
    save(DEST/'protocol.json',p);save(DEST/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),DEST/'protocol.json']})
    print('Frozen daily-lagged leader hypothesis',flush=True)

def prepare():
    peers.check();cfg,d=load(OUT/'dataset');z=dict(np.load(peers.DEST/'inputs.npz'));times=z['times']
    cohort=read(DEST/'protocol.json')['cohort'];events=[];first=np.full(len(cohort),np.inf)
    for k,c in enumerate(cohort):
        for r in read(peers.DEST/'raw'/f"{c['address']}.json")['rows']:
            if not r['coin'].startswith('pas:'):continue
            j=SYMBOLS.index(r['coin'][4:]);dq=float(r['sz'])*(1 if r['side']=='B' else -1);known=max(r['time'],c['grant_time'])/1000+120
            events.append((known,k,j,dq,float(r['px']),float(r['fee'])));first[k]=min(first[k],known)
    events.sort();cursor=0;q=np.zeros((len(cohort),10));cash=np.zeros(len(cohort));daily={};audit=[]
    daily_times=list(range(START,END+1,86400))
    for t in daily_times:
        while cursor<len(events) and events[cursor][0]<=t:
            _,k,j,dq,price,fee=events[cursor];q[k,j]+=dq;cash[k]-=dq*price+fee;cursor+=1
        mi=np.searchsorted(d['times'],t*1000,side='right')-1;px=d['close'][mi]
        valid=np.all((q==0)|(np.isfinite(px)&(px>0)),axis=1)
        pnl=cash+q@np.nan_to_num(px,nan=0.);eligible=valid&(first<=t-86400)
        count=int(eligible.sum());number=max(3,int(np.ceil(count/4)))
        ids=[k for k in np.argsort(-pnl,kind='stable') if eligible[k] and pnl[k]>0][:number]
        if len(ids)<3:ids=[]
        daily[t]=ids;audit.append(dict(snapshot=t,eligible=count,chosen=[cohort[k]['address'] for k in ids],chosen_price_pnl=[float(pnl[k]) for k in ids]))
    weights=[];counts=[];used=[]
    for i,t in enumerate(times):
        valid=[s for s in daily if s<t-120]
        at=max(valid) if valid else None;ids=daily[at] if at is not None else []
        units=z['peer_weights'][i,ids] if ids else np.empty((0,10))
        active=abs(units).sum(1)>1e-12;units=units[active]
        weights.append(.6*units.mean(0) if len(units)>=3 else np.zeros(10));counts.append(len(units));used.append(at)
    np.savez_compressed(DEST/'inputs.npz',times=times,weights=weights,active_counts=counts,fresh=z['fresh'])
    save(DEST/'ranking_audit.json',dict(daily=audit,source_times=times,rank_snapshot_used=used))
    save(DEST/'input_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in [DEST/'inputs.npz',DEST/'ranking_audit.json']})
    print('Prepared causal leader inputs; eligible decision points',sum(c>=3 for c in counts),flush=True)

def job(scenario,switch):
    # Reuse the identical risk/turnover/execution engine; change only the
    # precomputed causal vote vector and its source manifest.
    peers.DEST=DEST
    row=peers.job(scenario,switch)
    row['arm']='daily_lagged_positive_price_leaders'
    return row

def run():
    assert not (DEST/'started.json').exists();save(DEST/'started.json',dict(at=datetime.now(timezone.utc).isoformat()))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,s,sw) for s in ['base','stress','no_receipts'] for sw in [False,True]]):
            row=f.result();rows.append(row);save(DEST/'results.json',rows);print({k:row[k] for k in ['scenario','phase','net_pnl','Q','tail_pnl']},flush=True)

if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
