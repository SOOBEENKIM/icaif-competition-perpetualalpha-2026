"""Evidence-based comparison; no inferred team names or invented strategy labels."""
import csv
from collections import Counter, defaultdict
from decimal import Decimal as D
import datetime as dt
import html
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.build_public_wallet_leaderboard import ordered_fills
OUT=Path(__file__).resolve().parent; RAW=OUT/'raw'
USER='0x125778065a578d598d88bddaa3fdd635280d0e47'
KST=dt.timezone(dt.timedelta(hours=9));START=1792411200000

def stamp(ms):
    return dt.datetime.fromtimestamp(ms/1000,KST).strftime('%m/%d %H:%M:%S') if ms else ''

def csvout(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def summarize(w,extra):
    a=w['address'];fills,issues=ordered_fills([f for f in w['fills']['rows'] if f['coin'].startswith('pas:')])
    ps=[x['position'] for x in w['positions'].get('response',{}).get('assetPositions',[])]
    by={};posrows=[]
    def asset(c):
        return by.setdefault(c,dict(address=a,coin=c,closed_pnl=0.,fees=0.,funding=0.,unrealized=0.,volume=0.,fills=0,open_long_notional=0.,open_short_notional=0.,close_long_notional=0.,close_short_notional=0.,maker_notional=0.,qty=0.,duration_hours=[]))
    open_at={};events=[]
    for f in fills:
        c=f['coin'];s=asset(c);q=float(f['sz'])*(1 if f['side']=='B' else -1);old=float(f['startPosition']);new=old+q;px=float(f['px']);nt=abs(q)*px
        s['closed_pnl']+=float(f['closedPnl']);s['fees']+=float(f['fee']);s['volume']+=nt;s['fills']+=1;s['qty']+=q
        if not f['crossed']:s['maker_notional']+=nt
        if f['feeToken']!='USDP':issues.append('non_USDP_fee:'+c)
        closing=min(abs(old),abs(q)) if old*q<0 else 0.
        opening=abs(q)-closing
        if closing:s['close_long_notional' if old>0 else 'close_short_notional']+=closing*px
        if opening:s['open_long_notional' if q>0 else 'open_short_notional']+=opening*px
        if not old or old*new<0:
            if old*new<0 and c in open_at:s['duration_hours'].append((f['time']-open_at[c])/3600000)
            open_at[c]=f['time']
        if abs(new)<1e-10:
            if c in open_at:s['duration_hours'].append((f['time']-open_at.pop(c))/3600000)
        events.append(dict(address=a,time_kst=stamp(f['time']),notional=nt,fee_bps=float(f['fee'])/nt*10000 if nt else None,**f))
    for r in w['funding']['rows']:
        d=r['delta'];c=d.get('coin','')
        if c.startswith('pas:'):asset(c)['funding']+=float(d['usdc'])
    for p in ps:
        s=asset(p['coin']);s['unrealized']=float(p['unrealizedPnl'])
        if abs(s['qty']-float(p['szi']))>1e-7:issues.append('inventory_mismatch:'+p['coin'])
        posrows.append(dict(address=a,coin=p['coin'],side='롱' if float(p['szi'])>0 else '숏',qty=float(p['szi']),entry=float(p['entryPx']),mark=float(p['positionValue'])/abs(float(p['szi'])),notional=float(p['positionValue']),unrealized=float(p['unrealizedPnl']),leverage=p['leverage']['value'],margin_type=p['leverage']['type'],margin_used=float(p['marginUsed']),liquidation_price=p.get('liquidationPx'),snapshot_kst=w['positions']['received_kst']))
    for c,s in by.items():
        if c not in {p['coin'] for p in ps} and abs(s['qty'])>1e-7:issues.append('missing_current_position:'+c)
        s['net_pnl']=s['closed_pnl']-s['fees']+s['funding']+s['unrealized']
        s['median_completed_position_hours']=statistics.median(s['duration_hours']) if s['duration_hours'] else None
        s['current_position_hours']=(w['history_end_ms']-open_at[c])/3600000 if c in open_at and abs(s['qty'])>1e-7 else None
    flows=[];transfer_fees=0.;unclassified=[]
    for r in w['ledger']['rows']:
        d=r['delta']
        if d.get('token')!='USDP':continue
        sender=d.get('user','').lower();dest=d.get('destination','').lower();amount=float(d.get('amount',0))
        if sender==dest==a:continue
        if dest==a:flows.append(dict(time=r['time'],amount=amount))
        elif sender==a:flows.append(dict(time=r['time'],amount=-amount))
        elif d.get('type')=='spotGenesis':flows.append(dict(time=r['time'],amount=amount))
        else:unclassified.append(d)
        if sender==a and d.get('feeToken')=='USDP':transfer_fees+=float(d.get('fee',0))
    sums={k:sum(s[k] for s in by.values()) for k in ('closed_pnl','fees','funding','unrealized','volume','open_long_notional','open_short_notional','close_long_notional','close_short_notional','maker_notional')}
    cash=sum(x['amount'] for x in flows);net=sums['closed_pnl']-sums['fees']+sums['funding']+sums['unrealized']-transfer_fees
    mode=extra.get('mode',{}).get('response');spot=next((b for b in w['spot'].get('response',{}).get('balances',[]) if b['coin']=='USDP'),{})
    if mode=='unifiedAccount':equity=float(spot.get('total',0));equity_source='unified USDP total, includes PnL'
    elif mode in ('disabled','default','standard'):equity=float(spot.get('total',0))+float(w['positions']['response']['marginSummary']['accountValue']);equity_source='USDP spot + PAS margin account value'
    else:equity=None;equity_source='unverified account mode'
    one_grant=len(flows)==1 and abs(flows[0]['amount']-100000)<1e-8 and bool(w['grants'])
    if not all(w[k]['pagination_complete'] for k in ('fills','funding','ledger')):issues.append('incomplete_pagination')
    if w['fills']['retention_limit_possible']:issues.append('10000_fill_retention_possible')
    if unclassified:issues.append('unclassified_USDP_transfer')
    # Current accounting, never label this a daily official score.
    equity_pnl=equity-cash if equity is not None else None
    gross=sum(p['notional'] for p in posrows);long=sum(p['notional'] for p in posrows if p['side']=='롱');short=gross-long
    levs=sorted({p['leverage'] for p in posrows});long_seen=sums['open_long_notional']>0;short_seen=sums['open_short_notional']>0
    current='롱+숏' if long and short else '롱만' if long else '숏만' if short else '무포지션'
    historic='롱+숏 진입' if long_seen and short_seen else '롱 진입만' if long_seen else '숏 진입만' if short_seen else '미거래'
    hist=extra.get('historical_orders',{}).get('response',[]);hist=hist if isinstance(hist,list) else []
    hist=[r for r in hist if r.get('order',{}).get('coin','').startswith('pas:') and r['order'].get('timestamp',0)>=START]
    # The endpoint can return open and filled events for the same oid.
    # Count one most-recent state per order, retaining raw events separately.
    latest_orders={}
    for h in sorted(hist,key=lambda h:(h.get('statusTimestamp',0),h.get('status')!='open')):
        latest_orders[h['order']['oid']]=h
    row=dict(address=a,name='SFINX' if a==USER else '',funded_team=bool(w['grants']),single_grant_100k=one_grant,first_fill=stamp(fills[0]['time']) if fills else '',last_fill=stamp(fills[-1]['time']) if fills else '',snapshot_kst=w['positions'].get('received_kst'),history_end_kst=stamp(w['history_end_ms']),fills=len(fills),filled_orders=len({f['oid'] for f in fills}),trading_days=len({int((f['time']/1000-43200)//86400) for f in fills}),markets_traded=sorted(by),current=current,historical=historic,positions=len(ps),long_notional=long,short_notional=short,gross_notional=gross,net_notional=long-short,configured_leverages=levs,mode=mode,net_external_funding=cash,equity=equity,equity_source=equity_source,balance_net_pnl=equity_pnl,trade_net_pnl=net,return_pct=(equity_pnl/100000*100 if equity_pnl is not None else net/100000*100) if one_grant else None,accounting_snapshot_difference=equity_pnl-net if equity_pnl is not None else None,gross_equity_pct=gross/equity*100 if equity else None,effective_gross_leverage=gross/equity if equity else None,net_equity_pct=(long-short)/equity*100 if equity else None,margin_used=sum(p['margin_used'] for p in posrows),open_orders=len(w['orders'].get('response',[])),maker_notional_pct=sums['maker_notional']/sums['volume']*100 if sums['volume'] else None,order_types=dict(Counter(r['order'].get('orderType') for r in hist)),order_statuses=dict(Counter(r.get('status') for r in hist)),historical_orders_returned=len(hist),history_issues=sorted(set(issues)),**sums)
    row['order_types']=dict(Counter(h['order'].get('orderType') for h in latest_orders.values()))
    row['order_statuses']=dict(Counter(h.get('status') for h in latest_orders.values()))
    row['time_in_force']=dict(Counter(h['order'].get('tif') for h in latest_orders.values()))
    row['historical_unique_orders']=len(latest_orders)
    if not w['grants']:
        # These counterparties may already hold inventory/capital before START.
        # Current equity minus this period's transfers is NOT their period PnL.
        row['balance_net_pnl']=None
        row['trade_net_pnl']=None
        row['accounting_snapshot_difference']=None
        row['history_issues']=sorted(set(row['history_issues']+['opening_capital_and_inventory_not_established']))
        for s in by.values():s['net_pnl']=None
    return row,events,posrows,list(by.values()),hist

def main():
    wallets=[json.loads(p.read_text()) for p in sorted((RAW/'wallets').glob('*.json'))]
    rows=[];fills=[];positions=[];assets=[];orders=[];funding=[];hist=[]
    for w in wallets:
        p=RAW/'extra'/(w['address']+'.json');ex=json.loads(p.read_text()) if p.exists() else {}
        r,f,ps,ss,hs=summarize(w,ex);rows.append(r);fills.extend(f);positions.extend(ps);assets.extend(ss)
        orders.extend(dict(address=w['address'],**o) for o in w['orders'].get('response',[]))
        funding.extend(dict(address=w['address'],time_kst=stamp(f['time']),**f) for f in w['funding']['rows'] if f['delta'].get('coin','').startswith('pas:'))
        hist.extend(dict(address=w['address'],**h) for h in hs)
    sides=defaultdict(dict)
    for f in fills:sides[(f['coin'],f['time'],f['tid'])][f['side']]=f['address']
    for f in fills:f['counterparty']=sides[(f['coin'],f['time'],f['tid'])].get('A' if f['side']=='B' else 'B')
    # Public recentTrades supplies additional counterparties, even when a maker's old fills expired.
    recent=json.loads((RAW/'recent_trades.json').read_text())
    for rec in recent.values():
        for t in rec.get('response',[]):
            u=t.get('users',[])
            if len(u)==2:
                for f in fills:
                    if not f['counterparty'] and f['coin']==t['coin'] and f['tid']==t['tid'] and f['address'] in [x.lower() for x in u]:
                        f['counterparty']=u[1] if f['address']==u[0].lower() else u[0]
    ranked=sorted([r for r in rows if r['funded_team'] and r['fills'] and r['return_pct'] is not None and not r['history_issues']],key=lambda r:-r['return_pct'])
    for i,r in enumerate(ranked,1):r['wallet_snapshot_pnl_rank']=i
    official=json.loads((RAW/'leaderboard.json').read_text())['response'];teams=sorted(official['teams'],key=lambda x:(x['rank'] is None,x['rank'] or 999,x['name']))
    for t in teams:
        if t['q'] is not None:
            q=(max(t['return'],0)+.005)/(abs(t['return'])+1.5*t['volatility_horizon']+.05)
            assert abs(q-t['q'])<1e-12
            t['jury_implied']=(t['score']-.8*t['q'])/.2
    active=[r for r in rows if r['funded_team'] and r['fills']]
    market_counts=[]
    for c in json.loads((RAW/'market.json').read_text())['response'][0]['universe']:
        coin=c['name'];ps=[p for p in positions if p['coin']==coin and any(r['address']==p['address'] and r['funded_team'] for r in rows)]
        ss=[s for s in assets if s['coin']==coin and any(r['address']==s['address'] and r['funded_team'] for r in rows)]
        market_counts.append(dict(coin=coin,teams_traded=sum(s['fills']>0 for s in ss),long_holders=sum(p['side']=='롱' for p in ps),short_holders=sum(p['side']=='숏' for p in ps),long_notional=sum(p['notional'] for p in ps if p['side']=='롱'),short_notional=sum(p['notional'] for p in ps if p['side']=='숏')))
    summary=dict(collected_wallets=len(rows),funded_wallets=sum(r['funded_team'] for r in rows),funded_trading_wallets=len(active),current_direction_counts=dict(Counter(r['current'] for r in active)),historical_direction_counts=dict(Counter(r['historical'] for r in active)),official_generated_at=official['generated_at'],official_ranked_teams=sum(t['eligible'] for t in teams),sfinx=next(r for r in rows if r['address']==USER),sfinx_official=next(t for t in teams if t['name']=='SFINX'),median_gross_equity_pct=statistics.median(r['gross_equity_pct'] for r in active if r['gross_equity_pct'] is not None),wallet_history_issues=[dict(address=r['address'],issues=r['history_issues']) for r in rows if r['history_issues']],name_mapping_available=False)
    result=dict(summary=summary,wallets=rows,official=teams,positions=positions,assets=assets,market_counts=market_counts)
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for n,rs in [('wallets',rows),('official_teams',teams),('fills',fills),('positions',positions),('by_market',assets),('open_orders',orders),('funding',funding),('historical_orders',hist),('market_counts',market_counts)]:csvout(n,rs)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
