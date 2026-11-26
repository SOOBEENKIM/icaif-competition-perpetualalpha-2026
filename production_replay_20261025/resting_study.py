"""Production-policy sensitivity to certified, incomplete historical depth."""
import bisect,gzip,json,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from .engine import SyntheticVenue,run
from .resting_book import RestingBook,DEST as BOOK
from .protocol import ROOT,OUT,DATA,save,sha

DEST=OUT/'resting_execution'

class RestingVenue(SyntheticVenue):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.known=RestingBook(json.loads((BOOK/'orders.json').read_text()),
                              self.sc['latency_ms'],self.sc['depth_fraction'])

    def stream(self,symbol):
        s=super().stream(symbol);now=int(self.now*1000)
        if not s['ctx']:return s
        s.update(book=[],book_times=[])
        levels=self.known.levels(symbol,now)
        # Missing side stays missing. No fake far-away quote to pass the
        # production two-sided-book validation.
        if all(levels):
            s.update(book=[(now,0,dict(coin='pas:'+symbol,time=now,levels=levels,synthetic=True,
                                     incomplete_reconstructed_book=True))],book_times=[now])
        return s

    def send(self,order):
        from bot.execution import OrderRejected
        sent=self.now;arrival=(int(sent)//60+1)*60
        # Conservative full-minute arrival keeps the existing candle clock;
        # this is stricter than the nominal1/3s discovery delay, not a real RTT.
        if arrival>=self.end:raise OrderRejected('Historical boundary')
        self.now=float(arrival);self.settle(arrival*1000,self.account)
        if self.observe is not None:self.observe(arrival)
        symbol=order.coin[4:];j=self.symbols.index(symbol);side=1 if order.is_buy else -1
        q,px,consumed=self.known.sweep(symbol,int(arrival*1000),side,float(order.size),float(order.price),
                                     self.profile['trading']['execution']['decimals'][j],self.sc['impact_bps'])
        self.oid+=1;oid=self.oid;rows=[]
        if q:
            rows=[dict(coin=order.coin,tid=f'RESTING-SIM-{oid}',oid=oid,feeToken='USDP',side='B' if side>0 else 'A',
                       sz=str(q),px=str(px),fee=str(q*px*self.sc['fee_bps']/10000),time=int(arrival*1000),synthetic=True)]
        self.rows[oid]=rows
        self.orders.append(dict(sent=sent,arrival=arrival,symbol=symbol,side=side,requested=float(order.size),
                                limit=float(order.price),filled=q,price=px,consumed=consumed,synthetic=True))
        self.audit['conditional_fill' if q else 'known_arrival_depth_unavailable']+=1
        return dict(filled=dict(oid=oid,totalSz=str(q)))

def job(arm,scenario,switch):
    from competition_live.profile import load_profile
    from pas_strategy_audit.data import load
    cfg,d=load(DATA);p=load_profile();protocol=json.loads((DEST/'protocol.json').read_text())
    sc=protocol['scenarios'][scenario];funding=json.loads((DATA/'funding.json').read_text())
    packets=json.loads((OUT/'packets.json').read_text());t=time.time()
    result=run(d,p,sc,funding,packets,cfg['start_ms']//1000,cfg['cutoff_ms']//1000,arm,switch,RestingVenue)
    row=dict(arm=arm,scenario=scenario,switch=switch,**result['metrics'],fills=len(result['fills']),
             risk_violations=len(result['audit']['risk_violations']),seconds=time.time()-t)
    with gzip.open(DEST/f'{arm}_{scenario}_{switch or "full"}.json.gz','wt') as f:
        json.dump(result,f,allow_nan=False,default=lambda x:x.tolist())
    return row

def main():
    if (DEST/'protocol.json').exists():raise FileExistsError('Preserve prior experiment')
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        primary='fixed_entry_1h',control='incumbent',
        assumptions='Only certified non-trigger resting orders,1min delayed arrival, permanent hypothetical consumption, observed fills/cancels deplete. No transaction-volume cap. Missing two-sided book means no new order. Last completed PAS trade price valuation proxy. Unchanged production strategy/risk/account. Sparse depth audit prevents claiming complete venue reconstruction.',
        scenarios=dict(base=dict(fee_bps=9.,impact_bps=0.,depth_fraction=.5,latency_ms=1000,receipt_multiplier=1.,participation=.01),
                       stress=dict(fee_bps=12.,impact_bps=5.,depth_fraction=.25,latency_ms=3000,receipt_multiplier=.5,participation=.01)),
        source_hash=sha(__file__),book_hash=sha(BOOK/'orders.json'),exposed_development_only=True,orders_sent=0))
    rows=[];tasks=[(a,s,w) for a in ['incumbent','fixed_entry_1h'] for s in ['base','stress'] for w in ([None] if a=='incumbent' else [None,1792767600])]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,*t) for t in tasks]):
            row=f.result();rows.append(row);save(DEST/'results.json',rows);print(row,flush=True)

if __name__=='__main__':main()
