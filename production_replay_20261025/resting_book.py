"""Known resting liquidity reconstructed from public order lifecycles.

This is an incomplete historical book, never an exact L2 reconstruction.
Only accepted non-trigger/non-reduce-only GTC/ALO orders are included. Public
fills and simulated consumption both deplete each order permanently. Canceled
orders vanish at their historical cancellation, independently of our strategy.
"""
import bisect,json,math
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
from .protocol import ROOT,OUT,save,sha

OWN='0x125778065a578d598d88bddaa3fdd635280d0e47'
DEST=OUT/'resting_book'

def reconstruct():
    orders=[];audit=Counter();files={};wallets={}
    raw=ROOT/'reports/top10_live_review_20261024/raw/extra'
    fills_root=ROOT/'reports/execution_revision_20261024/peer_consensus/raw'
    for path in sorted(raw.glob('*.json')):
        if path.stem.lower()==OWN:continue
        d=json.loads(path.read_text());history=d.get('historical_orders')
        if not history:continue
        fp=fills_root/path.name
        if not fp.exists():audit['wallet_missing_fills']+=1;continue
        files[str(path.relative_to(ROOT))]=sha(path);files[str(fp.relative_to(ROOT))]=sha(fp)
        by_order=defaultdict(list)
        for f in json.loads(fp.read_text())['rows']:
            if f['coin'].startswith('pas:'):by_order[f['oid']].append(f)
        receipt=int(datetime.fromisoformat(history['sent_kst']).timestamp()*1000)
        groups=defaultdict(list)
        for r in history['response']:
            o=r['order']
            if o['coin'].startswith('pas:'):groups[o['oid']].append(r)
        wallets[path.stem]=dict(history_rows=len(history['response']),groups=len(groups),capture_ms=receipt,
                               retention_capped=len(history['response'])>=2000)
        for oid,rows in groups.items():
            audit['groups']+=1;o=rows[0]['order']
            if o['tif'] not in ['Alo','Gtc'] or o['isTrigger'] or o['reduceOnly'] or o['orderType']!='Limit':
                audit['not_unconditional_resting_limit']+=1;continue
            ident=lambda r:tuple(r['order'][k] for k in ['coin','side','limitPx','origSz','timestamp'])
            if len({ident(r) for r in rows})!=1:audit['modified_or_inconsistent']+=1;continue
            opened=[r['statusTimestamp'] for r in rows if r['status']=='open']
            if not opened:audit['no_acceptance_record']+=1;continue
            terminals=[r['statusTimestamp'] for r in rows if r['status']!='open']
            start=min(opened);end=min(terminals,default=receipt)
            if end<=start:audit['no_resting_lifetime']+=1;continue
            size=float(o['origSz']);px=float(o['limitPx'])
            if size<=0 or px<=0:audit['invalid_price_size']+=1;continue
            fs={ (f['time'],f['tid']):f for f in by_order.get(oid,[]) }
            fs=sorted(fs.values(),key=lambda f:(f['time'],f['tid']))
            if any(f['coin']!=o['coin'] or f['side']!=o['side'] or f['crossed'] for f in fs):
                audit['ambiguous_or_taker_fill']+=1;continue
            if any(f['time']<start or f['time']>end for f in fs):
                audit['fill_outside_lifecycle']+=1;continue
            total=sum(float(f['sz']) for f in fs)
            if total>size+1e-8:audit['overfilled_history']+=1;continue
            # Missing fills would overstate known resting capacity. Check every
            # reported remaining size against all fills up to that status.
            consistent=True
            for r in rows:
                consumed=sum(float(f['sz']) for f in fs if f['time']<=r['statusTimestamp'])
                # Accepted/open record can precede a same-ms partial match.
                if r['status']=='open':continue
                remaining=float(r['order']['sz'])
                if abs(size-consumed-remaining)>1e-8:consistent=False;break
            if not consistent:audit['unreconciled_remaining_size']+=1;continue
            if any(r['status']=='filled' for r in rows) and abs(size-total)>1e-8:
                audit['filled_without_full_fill_history']+=1;continue
            cumulative=[];running=0.
            for f in fs:running+=float(f['sz']);cumulative.append([f['time'],running])
            orders.append(dict(key=path.stem+':'+str(oid),wallet=path.stem,oid=oid,symbol=o['coin'][4:],
                side=o['side'],price=px,size=size,start=start,end=end,fills=cumulative,
                terminal_observed=bool(terminals)))
            audit['accepted']+=1
    return orders,dict(counts=dict(audit),wallets=wallets,files=files)

class RestingBook:
    def __init__(self,orders,observation_delay_ms=1000,depth_fraction=.5):
        self.by=defaultdict(list);self.delay=observation_delay_ms;self.fraction=depth_fraction;self.used={}
        for o in orders:
            o=dict(o,fill_times=[f[0] for f in o['fills']])
            self.by[o['symbol']].append(o)

    def remaining(self,o,at):
        if not o['start']+self.delay<=at<o['end']:return 0.
        i=bisect.bisect_right(o['fill_times'],at)-1
        observed=o['fills'][i][1] if i>=0 else 0.
        # Haircut the certified inventory; simulated fills NEVER replenish.
        return max(0.,(o['size']-observed)*self.fraction-self.used.get(o['key'],0.))

    def levels(self,symbol,at):
        result=[]
        for side in ['B','A']:
            grouped=defaultdict(float)
            for o in self.by[symbol]:
                if o['side']==side:
                    remaining=self.remaining(o,at)
                    if remaining>1e-12:grouped[o['price']]+=remaining
            result.append([dict(px=str(p),sz=str(q),n=1) for p,q in sorted(grouped.items(),reverse=side=='B')])
        return result

    def sweep(self,symbol,at,side,quantity,limit,decimals,impact_bps=0.):
        wanted='A' if side>0 else 'B';candidates=[]
        for o in self.by[symbol]:
            available=self.remaining(o,at)
            px=o['price']*(1+side*impact_bps/10000)
            if o['side']==wanted and available>0 and side*(px-limit)<=1e-10:
                candidates.append((px,o,available))
        candidates.sort(key=lambda x:side*x[0]);left=quantity;value=0.;fills=[];factor=10**decimals
        for px,o,available in candidates:
            q=math.floor(min(left,available)*factor+1e-9)/factor
            if not q:continue
            self.used[o['key']]=self.used.get(o['key'],0.)+q
            value+=q*px;left-=q;fills.append(dict(order_key=o['key'],qty=q,price=px))
            if left<.5/factor:break
        total=sum(f['qty'] for f in fills)
        return total,value/total if total else None,fills

def main():
    import numpy as np
    from pas_strategy_audit.data import load
    from .protocol import DATA
    orders,audit=reconstruct();save(DEST/'orders.json',orders)
    cfg,d=load(DATA);book=RestingBook(orders);symbols=list(book.by);coverage={}
    for s in symbols:
        both=bid=ask=crossed=0;depth=[];total=0
        for at in range(cfg['start_ms'],cfg['cutoff_ms'],60000):
            lv=book.levels(s,at);total+=1;bid+=bool(lv[0]);ask+=bool(lv[1])
            if all(lv):
                both+=1;crossed+=float(lv[0][0]['px'])>float(lv[1][0]['px'])
                depth.append(sum(float(x['px'])*float(x['sz']) for side in lv for x in side))
        coverage[s]=dict(minutes=total,bid_minutes=bid,ask_minutes=ask,both_minutes=both,crossed_minutes=crossed,
                         median_total_known_depth=float(np.median(depth)) if depth else None)
    save(DEST/'audit.json',dict(audit,coverage=coverage,interpretation='Incomplete known-order capacity; NOT complete historical L2, selection bias in observed wallets, no exact market marks. No future outcomes used in policy.',orders_sent=0))
    print(json.dumps(dict(counts=audit['counts'],coverage=coverage),indent=2))

if __name__=='__main__':main()
