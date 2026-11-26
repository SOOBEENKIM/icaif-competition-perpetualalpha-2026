"""Replay received PAS books through production policy/account code.

Paper receipts are conditional sweeps, never exchange fills. Forecast rates in
activeAssetCtx are NOT booked as cash. Funding requires settled fundingHistory
plus an observed oracle near its settlement time; gaps invalidate the result.
"""
import bisect,copy,gzip,json,math
from contextlib import ExitStack
from datetime import datetime,timezone
from types import SimpleNamespace
from unittest.mock import patch
from collections import Counter
import numpy as np
from bot.execution import OrderRejected
from competition_live.profile import load_profile
from competition_live.runtime import private_load
from candidate_handoff.portfolio import CandidatePortfolio
from execution_revision_20261024.guarded_portfolio import adopt
from experiments.basis_carry_20261017.probe import asof,sweep
from .protocol import ROOT,OUT,save,sha

def load_books(paths,cutoff):
    streams={};connections=[];counts=Counter();files={}
    for path in sorted(paths):
        files[str(path)]=sha(path)
        try:
            with gzip.open(path,'rt') as f:
                for line in f:
                    e=json.loads(line);at=e['received_ms'];ch=e['channel'];d=e['data']
                    if at>cutoff*1000:continue
                    counts[ch]+=1
                    if ch=='connection':connections.append(at);continue
                    if ch not in ['l2Book','activeAssetCtx']:continue
                    symbol=d['coin'][4:];field='book' if ch=='l2Book' else 'ctx'
                    s=streams.setdefault(symbol,dict(book=[],ctx=[]))
                    s[field].append((at,e['epoch'],d if field=='book' else d['ctx']))
        except EOFError:
            # Collector gzip member can still be open; complete parsed lines
            # only. No interpolated observation is invented at the cutoff.
            counts['open_gzip_member']+=1
    for s in streams.values():
        s['resets']=sorted(connections)
        for field in ['book','ctx']:
            s[field].sort(key=lambda x:x[0]);s[field+'_times']=[r[0] for r in s[field]]
    return streams,dict(counts=dict(counts),files=files)


class BookVenue:
    def __init__(self,streams,profile,scenario,start,end,funding):
        self.streams=streams;self.profile=profile;self.sc=scenario;self.now=float(start);self.end=end
        self.account=None;self.gateway=self;self.rows={};self.oid=0;self.used={};self.orders=[]
        self.depth_state={}
        self.funding=sorted(funding,key=lambda x:x['time']);self.fidx=bisect.bisect_right([r['time'] for r in self.funding],start*1000)
        self.cash_records=[];self.failures=[];self.checkpoints=Counter()
        self.cfg=json.loads((ROOT/'competition_live/strategy_plan.json').read_text())['book_protocol']
        self.observe=None

    def checkpoint(self,kind,detail):self.checkpoints[kind]+=1
    def refresh(self,symbol):return self.streams.get(symbol)
    def mark(self,symbol,at=None):
        now=self.now if at is None else at;s=self.streams[symbol];i=bisect.bisect_right(s['ctx_times'],now*1000)-1
        if i<0 or now*1000-s['ctx_times'][i]>self.cfg['context_max_age_ms']:return None
        reset=bisect.bisect_right(s['resets'],now*1000)-1
        if reset>=0 and s['ctx_times'][i]<s['resets'][reset]:return None
        return s['ctx'][i][2]
    def marks(self):
        values=[self.mark(s) for s in self.profile['trading']['symbols']]
        return None if any(v is None for v in values) else np.array([float(v['markPx']) for v in values])
    def market(self,coin):
        j=self.profile['trading']['symbols'].index(coin[4:]);c=self.mark(coin[4:])
        if c is None:raise OrderRejected('No fresh received context')
        return dict(name=coin,szDecimals=self.profile['trading']['execution']['decimals'][j]),c
    def set_risk_state(self,qty,sigma):self.qty=np.array(qty);self.sigma=np.array(sigma)
    def submit(self,order):return self.gateway.send(order)
    def collect(self,oid,started,reported):return self.rows[oid]

    def depth_at(self,symbol,at):
        """Never reuse hypothetical consumption just because time advanced.

        Each price level regains capacity only when observed size increases or
        it disappears and subsequently reappears. Unchanged snapshots cannot
        refill our simulated inventory. This is deliberately conservative L2
        counterfactual accounting, not a model of unseen queue replenishment.
        """
        s=self.streams[symbol];state=self.depth_state.setdefault(symbol,dict(index=-1,sides=[{},{}]))
        end=bisect.bisect_right(s['book_times'],at)-1
        if end<state['index']:raise ValueError('Book replay clock moved backwards')
        for i in range(state['index']+1,end+1):
            book=s['book'][i][2]
            for side in [0,1]:
                previous=state['sides'][side];updated={}
                for level in book['levels'][side]:
                    px=float(level['px']);size=float(level['sz'])*self.sc['depth_fraction']
                    old,available=previous.get(px,(0.,0.))
                    updated[px]=(size,min(size,available+max(0.,size-old)))
                state['sides'][side]=updated
        state['index']=end
        return state['sides']

    def send(self,order):
        sent=self.now;symbol=order.coin[4:];s=self.streams[symbol]
        first=int(sent*1000)+self.sc['latency_ms'];i=bisect.bisect_left(s['book_times'],first)
        quote=None;arrival=None
        while i<len(s['book_times']) and s['book_times'][i]<=first+5000:
            at=s['book_times'][i]
            if at/1000>=self.end:break
            quote=asof(s,at,self.cfg)
            if quote is not None:arrival=at/1000;break
            i+=1
        if quote is None:
            self.failures.append(dict(time=sent,kind='unobserved_arrival',symbol=symbol))
            raise OrderRejected('Paper order arrival has no observed valid book')
        self.now=arrival;self.settle(int(arrival*1000),self.account)
        side=1 if order.is_buy else -1;j=self.profile['trading']['symbols'].index(symbol)
        si=1 if side>0 else 0;capacity=self.depth_at(symbol,int(arrival*1000))[si]
        levels=[dict(px=str(px),sz=str(available)) for px,(_,available) in sorted(capacity.items(),reverse=si==0)]
        size,price=sweep(levels,float(order.size),1.,side,self.profile['trading']['execution']['decimals'][j],float(order.price))
        left=size
        for level in levels:
            px=float(level['px']);displayed,available=capacity[px]
            taken=min(left,available);capacity[px]=(displayed,available-taken);left-=taken
            if left<=1e-12:break
        self.oid+=1;oid=self.oid;rows=[]
        if size:
            rows=[dict(coin=order.coin,oid=oid,tid=f'BOOK-SIM-{oid}',side='B' if side>0 else 'A',sz=str(size),px=str(price),
                fee=str(size*price*self.sc['fee_bps']/10000),feeToken='USDP',time=int(arrival*1000),synthetic=True)]
        self.rows[oid]=rows
        self.orders.append(dict(time=sent,arrival=arrival,symbol=symbol,side=side,limit=float(order.price),size=float(order.size),
            filled=size,price=price,book_received=quote['received'],server_time=quote['book']['time'],synthetic=True))
        return dict(filled=dict(oid=oid,totalSz=str(size)))

    def settle(self,at_ms,account):
        while self.fidx<len(self.funding) and self.funding[self.fidx]['time']<=at_ms:
            f=self.funding[self.fidx];self.fidx+=1;coin=f['coin'];symbol=coin[4:];j=self.profile['trading']['symbols'].index(symbol)
            qty=sum(x['signed_qty'] for x in account.physical.fills if x['asset']==j and x['time']<f['time']/1000)
            if not qty:continue
            s=self.streams[symbol];i=bisect.bisect_right(s['ctx_times'],f['time'])-1
            if i<0 or f['time']-s['ctx_times'][i]>3000:
                self.failures.append(dict(time=f['time']/1000,kind='missing_funding_oracle',symbol=symbol));continue
            oracle=float(s['ctx'][i][2]['oraclePx']);amount=-qty*oracle*float(f['fundingRate'])
            if amount>0:amount*=self.sc['receipt_multiplier']
            row=dict(time=f['time'],delta=dict(coin=coin,szi=str(qty),usdc=str(amount)),synthetic=True)
            account.apply_actual_funding([row]);self.cash_records.append(dict(row,oracle_received=s['ctx_times'][i],oracle=oracle,rate=float(f['fundingRate'])))


def run_book(streams,source,profile,scenario,start,end,seed,arm,funding):
    p=copy.deepcopy(seed)
    transform=None
    if arm=='fixed_entry_1h':p=adopt(p)
    elif arm=='size_only_2':
        from .size_study import ScaledPacket
        transform=ScaledPacket(2.,start)
    elif arm=='hourly_fresh':
        from .hourly_portfolio import adopt as adopt_hourly
        p=adopt_hourly(p)
    elif arm!='incumbent':raise ValueError(arm)
    v=BookVenue(streams,profile,scenario,start,end,funding);v.account=p.account
    a=p.account;a.executor=v;a.checkpoint=v.checkpoint;a.collect_fills=v.collect;a.refresh_market=v.refresh
    p.settlement=v.settle;p.sc['fee_bps']=scenario['fee_bps'];p.sc['depth_fraction']=scenario['depth_fraction']
    if p.pair is not None:raise ValueError('Active specialist needs its own forward renewal audit')
    for pair in p.sessions:pair.on_time=v.settle;pair.account=a;pair.owner=a.owners['pair']
    initial_fills=len(a.physical.fills);initial_events=len(p.events);initial_fees=a.physical.fees;initial_funding=a.physical.funding
    snapshots=[];missing=[];known=[r['received_at'] for r in source];risk_audit=[]
    clock=SimpleNamespace(time=lambda:v.now)
    with ExitStack() as stack:
        stack.enter_context(patch('competition_live.live_account.time',clock))
        stack.enter_context(patch('candidate_handoff.risk_priority.time',clock))
        while v.now<=end:
            now=v.now;v.settle(int(now*1000),a);marks=v.marks()
            i=bisect.bisect_right(known,now)-1;packet=source[i]['packet'] if i>=0 else None
            if packet is not None and now-packet['body']['frame']['source_time']>=21600:packet=None
            if packet is not None and transform is not None:packet=transform(packet,now)
            if marks is not None:
                eq=a.reconcile(marks);notional=float(abs(a.physical.qty*marks).sum())
                if packet is not None:
                    from replay_lab.common_comparison.engine import risk_breaches
                    breaches=risk_breaches(a.physical.qty,eq,marks,np.asarray(packet['body']['frame']['sigma']),profile['trading']['risk'])
                    if breaches:risk_audit.append(dict(time=now,limits=breaches))
                # Same idle-reserve convention; actual teammate returns excluded.
                snapshots.append(dict(time=now,equity=eq,qty=a.physical.qty.tolist(),marks=marks.tolist(),notional=notional,
                    fees=a.physical.fees-initial_fees,funding=a.physical.funding-initial_funding))
            else:missing.append(now)
            if now>=end:break
            p.step_candidate(now,streams,packet,profile)
            v.now=min(end,max(now+15,math.ceil(v.now)+1))
    if not snapshots:raise ValueError('No valid mark snapshots')
    first,last=snapshots[0],snapshots[-1];pnl=last['equity']-first['equity']
    return dict(arm=arm,net_pnl=pnl,fee_increment=last['fees'],funding_increment=last['funding'],
        price_pnl=pnl+last['fees']-last['funding'],start_equity=first['equity'],end_equity=last['equity'],
        exposure_times=[first['time'],last['time']],snapshots=snapshots,orders=v.orders,
        fills=a.physical.fills[initial_fills:],events=p.events[initial_events:],funding=v.cash_records,
        missing_mark_times=missing,failures=v.failures,
        risk_violations=risk_audit,accounting_error=max(a.maximum_error,a.physical.max_error),
        pending_orders=len(a.pending_orders),unknown=list(a.unknown),fatal=p.fatal,
        conditional=True,official_Q=None,actual_orders_sent=0)
