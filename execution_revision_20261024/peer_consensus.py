"""Lagged equal-wallet consensus over the original funded cohort.

No selection using terminal returns/ranks, no wallet names, no training labels.
Public histories are delayed120seconds before being observable to the rule.
This is conditional, exposed-history development, not an untouched test.
"""
import csv,gzip,json,sys,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,OUT,START,END,BOUNDARY,read,save,sha,verify,SYMBOLS
from .handover_engine import simulate
from pas_strategy_audit.data import load
from pas_cashflow_research.runner import persist
from replay_lab.common_comparison.engine import project,rounded,risk_breaches

DEST=OUT/'peer_consensus'
PEERS=ROOT/'reports/top10_live_review_20261024/raw/wallets'
OWN='0x125778065a578d598d88bddaa3fdd635280d0e47'

def freeze():
    verify();assert not (DEST/'protocol.json').exists()
    cohort=[];files={}
    for p in sorted(PEERS.glob('*.json')):
        d=read(p)
        if d['address'].lower()==OWN:continue
        grants=[x for x in d['grants'] if x['delta'].get('token')=='USDP' and float(x['delta'].get('amount',0))==100000.]
        assert len(grants)==1
        cohort.append(dict(address=d['address'].lower(),grant_time=grants[0]['time']))
        files[str(p.relative_to(ROOT))]=sha(p)
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),cohort=cohort,delay_seconds=120,
        start=START,end=END,primary='equal_wallet_consensus',minimum_active_wallets=3,
        rule='Every initially funded nonself wallet eligible only afteritsgrant. Reconstruct signed inventory from complete fills available120seconds earlier. Normalize each active wallet to unit gross, average equally. Target60%of own equity times mean vector, then unchanged asset/net/gross/variance/turnover caps. No ranking/PnL/leader identity weighting. Zero inventories do not vote. Atleast3active peers required.',
        execution='Same actual publication timestamps,6hdecisions,1hentryintent,inventory-aware turnover and close-before-reverse. Missing/stale PAS prices cannot create new exposure. Existing risk exits preserved.',
        cohort_limit='Funded cohort discovered from grant receipts, not winners/current top10. Strategy was designed after aggregate outcomes were inspected: development only. Other agents and venue liquidity treated exogenous in counterfactual replay.',
        acceptance='Full and matched-incumbent-inventory replay exceed incumbent PnL,Q and terminal-cost-adjusted PnL in base,stress,no_receipts; all failures kept. No silent parameter revision after outcome.',
        input_files=files,orders_sent=0,live_changed=False))
    save(DEST/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),DEST/'protocol.json']})
    print('Frozen full funded cohort',len(cohort),flush=True)

def check():
    verify()
    for p,h in read(DEST/'freeze.json').items():assert sha(ROOT/p)==h,p
    for p,h in read(DEST/'protocol.json')['input_files'].items():assert sha(ROOT/p)==h,p

def collect():
    check()
    for c in read(DEST/'protocol.json')['cohort']:
        addr=c['address'];out=DEST/'raw'/f'{addr}.json'
        if out.exists():continue
        old=read(PEERS/f'{addr}.json')
        if old.get('history_end_ms',0)>=END*1000 and 'fills' in old:
            rows=[r for r in old['fills']['rows'] if START*1000<=r['time']<=END*1000]
            save(out,dict(source=str((PEERS/f'{addr}.json').relative_to(ROOT)),reused=True,rows=rows));print(addr,'cached',len(rows),flush=True);continue
        cursor=START*1000;rows={};receipts=[]
        for page in range(20):
            body=dict(type='userFillsByTime',user=addr,startTime=cursor,endTime=END*1000,aggregateByTime=False)
            req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=30) as response:batch=json.load(response)
            assert isinstance(batch,list)
            receipts.append(dict(request=body,received_utc=datetime.now(timezone.utc).isoformat(),response=batch))
            for r in batch:
                if START*1000<=r['time']<=END*1000:rows.setdefault((r['coin'],r['tid']),r)
            time.sleep(4)
            if len(batch)<2000:break
            last=max(r['time'] for r in batch)
            assert last>cursor,'Full page at same millisecond cannot prove complete history'
            # Overlap the last timestamp to retain simultaneous fills; dedupe.
            cursor=last
        else:raise ValueError('History pagination incomplete')
        save(out,dict(reused=False,receipts=receipts,rows=sorted(rows.values(),key=lambda r:(r['time'],r['tid']))))
        print(addr,'fetched',len(rows),flush=True)
    check()

def prepare():
    check();cfg,d=load(OUT/'dataset');f=dict(np.load(OUT/'dataset/candidate_inputs.npz'));times=f['times']
    cohort=read(DEST/'protocol.json')['cohort'];q=np.zeros((len(cohort),10));at=0;events=[];issues=[]
    # Verify each fill's starting inventory against the entire reconstruction.
    for k,c in enumerate(cohort):
        held=np.zeros(10)
        for r in read(DEST/'raw'/f"{c['address']}.json")['rows']:
            if not r['coin'].startswith('pas:'):continue
            s=r['coin'][4:]
            if s not in SYMBOLS:raise ValueError('Unexpected competition market')
            j=SYMBOLS.index(s);delta=float(r['sz'])*(1 if r['side']=='B' else -1)
            if abs(held[j]-float(r['startPosition']))>1e-7:issues.append(dict(address=c['address'],symbol=s,time=r['time'],expected=held[j],reported=r['startPosition']))
            held[j]+=delta
            events.append((max(r['time'],c['grant_time'])/1000+120,k,j,delta))
    save(DEST/'history_audit.json',dict(issues=issues,wallets=len(cohort),pas_fills=len(events)))
    assert not issues,'Incomplete or misordered peer histories; cannot infer starting positions'
    events.sort();weights=[];counts=[];freshs=[];wallet_weights=[]
    for t in times:
        while at<len(events) and events[at][0]<=t:
            _,k,j,dq=events[at];q[k,j]+=dq;at+=1
        mi=np.searchsorted(d['times'],t*1000,side='right')-1;prices=d['close'][mi]
        valid=np.isfinite(prices)&(prices>0);fresh=valid&(d['age_ms'][mi]<=3600000)
        amounts=q*np.nan_to_num(prices,nan=0.);gross=abs(amounts).sum(1)
        active=(gross>=10)&np.all((q==0)|valid,axis=1)
        units=np.divide(amounts,gross[:,None],out=np.zeros_like(amounts),where=gross[:,None]>0)
        w=.6*units[active].mean(0) if active.sum()>=3 else np.zeros(10)
        weights.append(w);counts.append(int(active.sum()));freshs.append(fresh);wallet_weights.append(units)
    np.savez_compressed(DEST/'inputs.npz',times=times,weights=weights,active_counts=counts,fresh=freshs,peer_weights=wallet_weights)
    save(DEST/'input_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in [DEST/'inputs.npz',*(DEST/'raw').glob('*.json')]})
    print('Prepared',len(times),'points; peers',len(cohort),'fills',len(events),'active range',min(counts),max(counts),flush=True)

class PeerController:
    def __init__(self,z,incumbent,activation):self.z=z;self.incumbent=incumbent;self.activation=activation;self.last_decision=-1;self.trace=[]
    def propose(self,t,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs):
        if t<self.activation:return self.incumbent.propose(t,prices,qty,equity,raw,sigma,volume,trades,protocol,**kwargs)
        if t%21600 or t<=self.last_decision:return None
        i=int(np.searchsorted(self.z['times'],t));assert self.z['times'][i]==t
        self.last_decision=t;old=qty*prices/equity;risk=protocol['risk']
        if self.z['active_counts'][i]<3:return None
        target=self.z['weights'][i].copy()
        adding=(old*target<=0)|(abs(target)>abs(old)+1e-12)
        target=np.where(~self.z['fresh'][i]&adding,old,target)
        target=project(target,equity,sigma,risk)
        turn=abs(target-old).sum()
        if turn>risk['normal_turnover_fraction']:target=old+(target-old)*risk['normal_turnover_fraction']/turn
        target=project(target,equity,sigma,risk);target=np.where(target*old<0,0.,target)
        units=np.array([rounded(v,n) for v,n in zip(target*equity/prices,protocol['execution']['decimals'])])
        assert not risk_breaches(units,equity,prices,sigma,risk)
        self.trace.append(dict(time=int(t),active_wallets=int(self.z['active_counts'][i]),consensus=self.z['weights'][i].tolist(),target_qty=units.tolist()))
        return dict(target_qty=units.tolist(),risk_forced=False,defensive=False)

def job(scenario,switch):
    from competition_live.profile import load_profile
    from pas_performance_research.selected import controller
    check();cfg,d=load(OUT/'dataset');f=dict(np.load(OUT/'dataset/candidate_inputs.npz'));z=dict(np.load(DEST/'inputs.npz'))
    old=controller(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000)
    c=PeerController(z,old,BOUNDARY if switch else START);sc=read(OUT/'protocol.json')['scenarios'][scenario];tape=[]
    for s in SYMBOLS:
        for b in read(OUT/'dataset/raw'/f'pas_candles_{s}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and cfg['start_ms']<=at<=cfg['cutoff_ms']:
                tape.append(dict(time=at,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    r=simulate(cfg,d,read(OUT/'dataset/events.json'),tape,read(OUT/'dataset/funding.json'),load_profile()['trading'],sc,controller=c,entry_lifetime=3600,sticky_exits=True,switch_ms=BOUNDARY*1000 if switch else None)
    a=next(x for x in r['snapshots'] if x['time']==BOUNDARY*1000);b=r['snapshots'][-1];cost=(sc['fee_bps']+sc['impact_bps'])/10000
    row=dict(arm='equal_wallet_consensus',scenario=scenario,phase='same_inventory_switch' if switch else 'continuous120h',**r['metrics'],
        tail_pnl=b['equity']-a['equity'],tail_liquidation_adjusted_pnl=b['equity']-a['equity']-(b['notional']-a['notional'])*cost)
    r['controller_trace']=c.trace;persist(DEST,row,r);return row

def run():
    check();assert not (DEST/'started.json').exists();save(DEST/'started.json',dict(at=datetime.now(timezone.utc).isoformat()))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,s,sw) for s in ['base','stress','no_receipts'] for sw in [False,True]]):
            row=f.result();rows.append(row);save(DEST/'results.json',rows);print({k:row[k] for k in ['scenario','phase','net_pnl','Q','tail_pnl']},flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
