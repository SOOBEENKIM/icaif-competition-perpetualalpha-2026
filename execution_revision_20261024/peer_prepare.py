"""Validate same-millisecond fills as inventory chains, not by random trade ID."""
from collections import defaultdict
from decimal import Decimal
from itertools import groupby
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from . import peer_consensus as p

def verify_inventory(rows,symbols):
    held={s:Decimal(0) for s in symbols};batches=0;fills=0
    rows=sorted([r for r in rows if r['coin'].startswith('pas:')],key=lambda r:(r['time'],r['coin']))
    for (at,coin),group in groupby(rows,key=lambda r:(r['time'],r['coin'])):
        s=coin[4:]
        if s not in held:raise ValueError('Unknown competition market')
        batch=list(group);graph=defaultdict(list);delta=Decimal(0)
        for r in batch:
            start=Decimal(r['startPosition']);dq=Decimal(r['sz'])*(1 if r['side']=='B' else -1)
            graph[start].append(start+dq);delta+=dq
        stack=[held[s]];trail=[];used=0
        while stack:
            q=stack[-1]
            if graph[q]:stack.append(graph[q].pop());used+=1
            else:trail.append(stack.pop())
        if used!=len(batch) or trail[0]!=held[s]+delta:
            raise ValueError(f'Incomplete inventory chain:{coin}/{at}; used{used}/{len(batch)}; starting{held[s]}')
        held[s]+=delta;fills+=len(batch);batches+=1
    return dict(qty={s:float(q) for s,q in held.items()},batches=batches,fills=fills)

def main():
    p.check();dest=p.DEST
    amendment=dest/'same_timestamp_amendment.json'
    if not amendment.exists():p.save(amendment,dict(created_utc=datetime.now(timezone.utc).isoformat(),code_sha256=p.sha(__file__),
        reason='Naive trade-ID ordering failed before predictions/backtests. TradeIDs are not sequence IDs. Validate all same-time same-coin fills form a complete inventory Euler path; since observation delay is shared, their net position/cash at decision is order invariant.',outcomes_opened=False))
    assert p.read(amendment)['code_sha256']==p.sha(__file__)
    cfg,d=p.load(p.OUT/'dataset');f=dict(np.load(p.OUT/'dataset/candidate_inputs.npz'));times=f['times']
    cohort=p.read(dest/'protocol.json')['cohort'];q=np.zeros((len(cohort),10));at=0;events=[];audits=[]
    for k,c in enumerate(cohort):
        rows=p.read(dest/'raw'/f"{c['address']}.json")['rows']
        audits.append(dict(address=c['address'],**verify_inventory(rows,p.SYMBOLS)))
        for r in rows:
            if not r['coin'].startswith('pas:'):continue
            j=p.SYMBOLS.index(r['coin'][4:]);delta=float(r['sz'])*(1 if r['side']=='B' else -1)
            events.append((max(r['time'],c['grant_time'])/1000+120,k,j,delta))
    p.save(dest/'history_audit_corrected.json',dict(wallets=len(cohort),pas_fills=len(events),chains=audits,issues=[]))
    events.sort();weights=[];counts=[];freshs=[];wallet_weights=[]
    for t in times:
        while at<len(events) and events[at][0]<=t:
            _,k,j,dq=events[at];q[k,j]+=dq;at+=1
        mi=np.searchsorted(d['times'],t*1000,side='right')-1;prices=d['close'][mi]
        valid=np.isfinite(prices)&(prices>0);fresh=valid&(d['age_ms'][mi]<=3600000)
        amounts=q*np.nan_to_num(prices,nan=0.);gross=abs(amounts).sum(1)
        active=(gross>=10)&np.all((abs(q)<1e-10)|valid,axis=1)
        units=np.divide(amounts,gross[:,None],out=np.zeros_like(amounts),where=gross[:,None]>0)
        # Numerical residue from a completed close must not create a voter.
        units[gross<10]=0.
        w=.6*units[active].mean(0) if active.sum()>=3 else np.zeros(10)
        weights.append(w);counts.append(int(active.sum()));freshs.append(fresh);wallet_weights.append(units)
    np.savez_compressed(dest/'inputs.npz',times=times,weights=weights,active_counts=counts,fresh=freshs,peer_weights=wallet_weights)
    p.save(dest/'input_manifest.json',{str(x.relative_to(p.ROOT)):p.sha(x) for x in [dest/'inputs.npz',*(dest/'raw').glob('*.json')]})
    print('Verified46wallets;',len(events),'fills; active peers',min(counts),max(counts),flush=True)

if __name__=='__main__':main()
