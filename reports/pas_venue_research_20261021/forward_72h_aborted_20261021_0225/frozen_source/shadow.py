"""Conservative received-event simulation; fills are counterfactual, not real."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .capture import OUT,ROOT,SYMBOLS,save
from .policy import propose


class Queue:
    def __init__(self,side,limit,units,ahead,arrival,cancel_ack):
        self.side,self.limit,self.remaining,self.ahead=side,limit,units,ahead
        self.arrival,self.cancel_ack=arrival,cancel_ack
        self.seen=set()

    def trade(self,trade,received):
        identity=(trade['coin'],trade['tid'])
        if identity in self.seen:return 0.
        self.seen.add(identity)
        when=trade['time']/1000
        if not self.arrival<=when<=received<=self.cancel_ack:return 0.
        if trade['side']!=('A' if self.side>0 else 'B'):return 0.
        if self.side*(float(trade['px'])-self.limit)>0:return 0.
        size=float(trade['sz']);ahead=min(self.ahead,size)
        self.ahead-=ahead;size-=ahead
        fill=min(self.remaining,size);self.remaining-=fill
        return fill


def consume(book,side,units,limit=None,fraction=.25):
    left=units;cash=0.
    for row in book['levels'][1 if side>0 else 0]:
        price=float(row['px'])
        if limit is not None and side*(price-limit)>1e-9:break
        take=min(left,float(row['sz'])*fraction)
        left-=take;cash+=take*price
        if left<=units*1e-12:break
    filled=units-left
    return filled,cash/filled if filled else None


def ordered_events(paths):
    previous=-float('inf')
    for path in paths:
        with gzip.open(path,'rt') as f:
            for line in f:
                event=json.loads(line)
                if event['received']<previous:
                    raise ValueError('Receipt clock moved backwards; cannot claim causal replay')
                previous=event['received']
                yield event


def main(directory=OUT,scenario='base'):
    directory=Path(directory)
    protocol=json.loads((directory/('protocol.json' if directory==OUT else 'plan.json')).read_text())
    manifest=json.loads((directory/('capture_manifest.json' if directory==OUT else 'completion.json')).read_text())
    if 'end' not in manifest:manifest['end']=manifest['at']
    if manifest['end']<protocol['end']-1:raise ValueError('Incomplete planned capture')
    for name,digest in protocol['code_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen source changed: '+name)
    paths=[OUT/'synchronized_public.jsonl.gz'] if directory==OUT else sorted((directory/'raw').glob('*.jsonl.gz'))
    events=ordered_events(paths)
    cfg=dict(protocol['config']);cfg.update(protocol.get('scenarios',{}).get(scenario,{}))
    start,end=protocol['start'],protocol['end']
    meta,context=json.loads((OUT/'raw/pas_meta.json').read_text())['response']
    decimals={m['name'].split(':')[1]:m['szDecimals'] for m in meta['universe']}
    sigma={}
    for s in SYMBOLS:
        bars=json.loads((OUT/'raw'/f'main_candles_{s}.json').read_text())['response']
        prices=[float(c['c']) for c in bars if c['T']/1000<protocol['reference_risk_fit_end']]
        sigma[s]=max(.0005,float(np.std(np.diff(np.log(prices)),ddof=1)*math.sqrt(12)))
    books={};contexts={};states={};rejections=collections.Counter();checks=[];fills=[];closed=[]
    next_decision=math.ceil(start/15)*15
    funding_exposure_unknown=False;previous_hour=int(start//3600)
    allow_until=end-cfg['order_ttl_seconds']-cfg['cancel_latency_seconds']-cfg['latency_seconds']-cfg['holding_seconds']-15
    def fresh(s,now):
        if s not in books or s not in contexts:return None
        at,b=books[s];ct,c=contexts[s]
        if not 0<=now-at<=cfg['book_max_age_seconds'] or not 0<=now-ct<=cfg['context_max_age_seconds']:return None
        if not -1<=now-b['time']/1000<=cfg['book_max_age_seconds']:return None
        return at,b,ct,c
    def enter_fill(state,qty,price,now,fee):
        if qty<=0:return
        state['qty']+=qty;state['entry_value']+=qty*price;state['fees']+=qty*price*fee/10000
        state.setdefault('first_fill',now)
        fills.append(dict(mode=state['mode'],symbol=state['symbol'],time=now,kind='entry',qty=qty,
                          price=price,fee_bps=fee,counterfactual=True))
    for event in events:
        now=event['received'];channel=event['channel'];data=event.get('data')
        if now>end:break
        if now>=start:
            if int(now//3600)>previous_hour and any(state['qty'] for state in states.values()):
                funding_exposure_unknown=True
            previous_hour=int(now//3600)
        if event['network']=='pas':
            if channel=='l2Book':books[data['coin'].split(':')[1]]=(now,data)
            elif channel=='activeAssetCtx':contexts[data['coin'].split(':')[1]]=(now,data['ctx'])
            elif channel=='trades':
                for trade in data:
                    s=trade['coin'].split(':')[1]
                    state=states.get(('maker_basis',s))
                    if state and state.get('queue') is not None:
                        q=state['queue'].trade(trade,now)
                        enter_fill(state,q,state['proposal']['limit'],now,cfg['maker_fee_bps'])
        if now<start:continue
        for key,state in list(states.items()):
            s=state['symbol'];view=fresh(s,now)
            if state['phase']=='pending' and now>=state['arrival']:
                if view is None:
                    if now>=state['arrival']+cfg['book_max_age_seconds']:states.pop(key)
                    continue
                _,b,_,_=view;proposal=state['proposal'];side=proposal['side']
                if state['mode']=='taker_basis':
                    qty,price=consume(b,side,proposal['units'],proposal['limit'],cfg['depth_fraction'])
                    if qty:enter_fill(state,qty,price,now,cfg['taker_fee_bps'])
                    state['phase']='holding'
                    if not state['qty']:states.pop(key);continue
                else:
                    opposite=float(b['levels'][1 if side>0 else 0][0]['px'])
                    if side*(proposal['limit']-opposite)>=0:
                        rejections['maker_post_only_would_cross']+=1;states.pop(key);continue
                    same=b['levels'][0 if side>0 else 1]
                    ahead=sum(float(row['sz']) for row in same if side*(float(row['px'])-proposal['limit'])>=0)
                    state['queue']=Queue(side,proposal['limit'],proposal['units'],ahead,now,
                        now+cfg['order_ttl_seconds']+cfg['cancel_latency_seconds'])
                    state['phase']='resting'
            if state['phase']=='resting':
                if now>state['queue'].cancel_ack or state['queue'].remaining<=1e-12:
                    state['queue']=None;state['phase']='holding'
                    if not state['qty']:states.pop(key);continue
            if state['phase']=='holding' and state['qty'] and now>=state['first_fill']+cfg['holding_seconds']:
                if view is None:continue
                _,book,_,_=view
                qty,price=consume(book,-state['proposal']['side'],state['qty'],fraction=cfg['depth_fraction'])
                if qty<=0:continue
                side=state['proposal']['side']
                average=state['entry_value']/state['qty']
                state['pnl']+=side*qty*(price-average)-qty*price*cfg['exit_impact_bps']/10000
                state['fees']+=qty*price*cfg['taker_fee_bps']/10000
                state['entry_value']-=qty*average;state['qty']-=qty
                fills.append(dict(mode=state['mode'],symbol=s,time=now,kind='exit',qty=qty,
                    price=price,fee_bps=cfg['taker_fee_bps'],additional_impact_bps=cfg['exit_impact_bps'],counterfactual=True))
                if state['qty']<1e-10:
                    closed.append(dict(mode=state['mode'],symbol=s,entry_time=state['first_fill'],
                        exit_time=now,net_pnl=state['pnl']-state['fees'],fees=state['fees']))
                    states.pop(key)
        if now<next_decision or now>allow_until:continue
        if cfg.get('avoid_hourly_funding_boundaries'):
            next_hour=(int(now)//3600+1)*3600
            if next_hour-now<cfg['order_ttl_seconds']+cfg['cancel_latency_seconds']+cfg['latency_seconds']+cfg['holding_seconds']+15:
                continue
        next_decision=math.floor(now/15)*15+15
        for s in SYMBOLS:
            view=fresh(s,now)
            if view is None:rejections['missing_or_stale_sample']+=1;continue
            at,book,ct,ctx=view
            sample=dict(now=now,book_received=at,context_received=ct,book=book,context=ctx,
                        size_decimals=decimals[s],reference_hourly_sigma=sigma[s])
            for mode in protocol['models']:
                key=mode,s
                if key in states:continue
                proposal=propose(sample,cfg,mode)
                checks.append(dict(at=now,symbol=s,**proposal))
                if not proposal['accepted']:
                    rejections[mode+':'+proposal['reason']]+=1;continue
                reserved=sum(state['proposal']['notional'] for (m,_),state in states.items() if m==mode)
                if reserved+proposal['notional']>cfg['maximum_total_gross'] or proposal['notional']>cfg['maximum_asset_gross']:
                    rejections[mode+':capacity_reserved']+=1;continue
                states[key]=dict(mode=mode,symbol=s,phase='pending',proposal=proposal,
                    arrival=now+cfg['latency_seconds'],qty=0.,entry_value=0.,fees=0.,pnl=0.)
    # This capture is within one funding interval. Never invent settlement cash.
    no_funding_boundary=(int(start//3600)==int(end//3600) or
        (cfg.get('avoid_hourly_funding_boundaries',False) and not funding_exposure_unknown))
    rows=[]
    for mode in protocol['models']:
        trades=[r for r in closed if r['mode']==mode]
        blocks={int(r['entry_time']//protocol.get('block_seconds',60)) for r in trades}
        counts=collections.Counter(r['reason'] for r in checks if r['mode']==mode)
        residual=[dict(symbol=s,qty=state['qty'],phase=state['phase']) for (m,s),state in states.items() if m==mode]
        contributions=collections.defaultdict(float)
        for trade in trades:contributions[int(trade['entry_time']//protocol.get('block_seconds',60))]+=trade['net_pnl']
        confidence=None
        if len(contributions)>=30:
            values=np.array(list(contributions.values()))
            rng=np.random.default_rng(20261021)
            bootstrap=rng.choice(values,size=(5000,len(values)),replace=True).mean(1)
            confidence=np.quantile(bootstrap,[.025,.975]).tolist()
        component_pass=bool(len(blocks)>=30 and confidence and confidence[0]>0
            and sum(r['net_pnl'] for r in trades)>0 and not residual
            and not manifest['errors'] and no_funding_boundary)
        rows.append(dict(mode=mode,checks=sum(counts.values()),reasons=dict(counts),
            accepted_proposals=sum(r['accepted'] for r in checks if r['mode']==mode),
            completed_trades=len(trades),entry_fill_count=sum(f['mode']==mode and f['kind']=='entry' for f in fills),
            completed_net_pnl=sum(r['net_pnl'] for r in trades),nonoverlapping_time_blocks=len(blocks),
            block_bootstrap_mean_pnl_95pct=confidence,component_evidence_gate_pass=component_pass,
            portfolio_Q_evaluated=False,
            residual=residual,fees=sum(r['fees'] for r in trades),activate=False,
            reason='No automatic adoption; component evidence does not establish whole-portfolio improvement'))
    result=dict(start=start,end=end,scenario=scenario,models=rows,checks=checks,fills=fills,closed=closed,
        capture_errors=manifest['errors'],funding_known_zero=no_funding_boundary,
        residual_or_unknown=bool(states) or bool(manifest['errors']) or not no_funding_boundary,
        actual_exchange_fills=0,orders_sent=0,comparison='incremental overlay versus no new trade',
        future_exit_liquidity_guaranteed=False,policy_frozen_before_start=True)
    save(directory/('shadow_results.json' if scenario=='base' else 'shadow_stress_results.json'),result)
    print(json.dumps(dict(models=rows,capture_errors=manifest['errors'],orders_sent=0)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=OUT)
    parser.add_argument('--scenario',default='base',choices=('base','stress'))
    args=parser.parse_args();main(args.directory,args.scenario)
