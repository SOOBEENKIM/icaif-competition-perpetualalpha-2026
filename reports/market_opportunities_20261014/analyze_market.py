"""Descriptive market research only; no fitted strategy, order submission or holdout claim."""
import argparse, bisect, collections, csv, datetime, gzip, json, math, pathlib
import numpy as np

HERE=pathlib.Path(__file__).resolve().parent
RAW=HERE/'raw'
PREVIOUS=HERE.parent/'combined_leaderboard_20261014'
COINS=['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY']
REF=dict(zip(COINS,['BTC','ETH','HYPE','xyz:SP500','xyz:EWY','xyz:EWJ','xyz:BRENTOIL','xyz:GOLD','xyz:EUR','xyz:JPY']))
START=int(datetime.datetime(2026,10,5,tzinfo=datetime.timezone.utc).timestamp()*1000)

def read_json(name):return json.loads((RAW/name).read_text())
def percentiles(xs):
    a=np.array([x for x in xs if x is not None and np.isfinite(x)],dtype=float)
    if len(a)==0:return None
    return dict(n=len(a),mean=float(a.mean()),p05=float(np.quantile(a,.05)),median=float(np.median(a)),p95=float(np.quantile(a,.95)),min=float(a.min()),max=float(a.max()))
def csvout(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def stream(name,partial):
    rows=[]
    try:
        with gzip.open(RAW/name,'rt') as f:
            for line in f: rows.append(json.loads(line))
    except (EOFError, json.JSONDecodeError):
        if not partial:raise
    return rows
def vwap(levels,quantity):
    remaining=quantity;cash=0.0
    for row in levels:
        take=min(float(row['sz']),remaining);cash+=take*float(row['px']);remaining-=take
        if remaining<quantity*1e-10:return cash/quantity
    return None
def corr(a,b):
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:return None
    return float(np.corrcoef(a,b)[0,1])

def history_stats():
    fill_path=RAW/'practice_wallet_fills.csv'
    if not fill_path.exists():fill_path=PREVIOUS/'fills.csv'
    with fill_path.open(encoding='utf-8-sig',newline='') as f:
        fills=[r for r in csv.DictReader(f) if int(r['time_ms'])>=START]
    groups=collections.defaultdict(list)
    for r in fills:groups[r['coin'].split(':')[1]].append(r)
    end_path=RAW/'historical_cutoff.json'
    if not end_path.exists():end_path=PREVIOUS/'raw'/'market_now.json'
    end=int(json.loads(end_path.read_text())['queried_at_ms'])
    rows=[];funding_rows=[];hourly_rows=[];trades=[]
    for coin in COINS:
        fs=groups[coin];takers=[r for r in fs if r['crossed']=='True'];makers=[r for r in fs if r['crossed']=='False']
        assert len(takers)==len(makers)
        ntl=[float(r['px'])*float(r['sz']) for r in takers]
        maker_ntl=collections.Counter()
        for r in makers:maker_ntl[r['address']]+=float(r['px'])*float(r['sz'])
        top=maker_ntl.most_common(3)
        fpath='funding_final_'+coin+'.json' if (RAW/('funding_final_'+coin+'.json')).exists() else 'funding_'+coin+'.json'
        funding=[r for r in read_json(fpath)['response'] if r['time']>=START]
        fr=np.array([float(r['fundingRate'])*1e4 for r in funding])
        premium=[float(r['premium'])*1e4 for r in funding]
        for r in funding:funding_rows.append({**r,'coin':coin})
        pairs=[(a,b) for a,b in zip(funding,funding[1:]) if abs(b['time']-a['time']-3600000)<5000]
        nonzero=[(float(a['fundingRate']),float(b['fundingRate'])) for a,b in pairs if float(a['fundingRate']) and float(b['fundingRate'])]
        candles=[r for r in read_json('candles_par_'+coin+'.json')['response'] if r['T']<end and r['t']>=START]
        refs={r['t']:r for r in read_json('candles_reference_'+coin+'.json')['response'] if r['T']<end}
        c_by_time={r['t']:r for r in candles}
        hour_returns=[];reference_returns=[];relative_basis=[]
        for r in candles:
            prior=c_by_time.get(r['t']-3600000);ref=refs.get(r['t']);pref=refs.get(r['t']-3600000)
            if prior:
                ret=(float(r['c'])/float(prior['c'])-1)*1e4
                hour_returns.append(ret)
                if ref and pref:reference_returns.append((ret,(float(ref['c'])/float(pref['c'])-1)*1e4))
            if ref:relative_basis.append((float(r['c'])/float(ref['c'])-1)*1e4)
            hourly_rows.append(dict(coin=coin,**r,reference_close=ref['c'] if ref else None))
        maker_fee=sum(float(r['fee']) for r in makers)/sum(float(r['px'])*float(r['sz']) for r in makers)*1e4
        taker_fee=sum(float(r['fee']) for r in takers)/sum(ntl)*1e4
        median_abs=float(np.median(np.abs(fr)))
        rows.append(dict(coin=coin,unique_trades=len(takers),notional=sum(ntl),median_trade_notional=float(np.median(ntl)),
                         trades_per_hour=len(takers)/((end-START)/3600000),maker_fee_bps=maker_fee,taker_fee_bps=taker_fee,
                         top3_maker_notional_share=sum(v for a,v in top)/sum(ntl),top3_makers=top,
                         funding_observations=len(fr),funding_bps_per_hour=percentiles(fr),abs_funding_bps_per_hour=percentiles(abs(fr)),
                         funding_negative_fraction=float(np.mean(fr<0)),funding_positive_fraction=float(np.mean(fr>0)),
                         funding_sign_persistence_nonzero=sum(a*b>0 for a,b in nonzero)/len(nonzero) if nonzero else None,
                         funding_lag1_corr=corr([float(a['fundingRate']) for a,b in pairs],[float(b['fundingRate']) for a,b in pairs]),
                         latest_settled_funding_bps=float(fr[-1]),latest_funding_time=funding[-1]['time'],
                         premium_bps=percentiles(premium),maker_fee_only_recovery_hours_at_median_abs=6/median_abs if median_abs else None,
                         taker_fee_only_recovery_hours_at_median_abs=18/median_abs if median_abs else None,
                         hourly_abs_move_bps=percentiles(np.abs(hour_returns)),hourly_return_corr_to_reference=corr(*zip(*reference_returns)) if reference_returns else None,
                         trade_close_to_reference_close_basis_bps=percentiles(relative_basis),completed_candles=len(candles)))
        for r in takers:trades.append({k:r[k] for k in ['coin','time_ms','px','sz','side','address','counterparty_address','fee']})
    csvout('settled_funding.csv',funding_rows);csvout('hourly_price_panel.csv',hourly_rows);csvout('historical_unique_trades.csv',trades)
    return rows,end

def live_stats(partial):
    par=stream('par_stream.jsonl.gz',partial);reference=stream('reference_stream.jsonl.gz',partial)
    events=collections.defaultdict(list);trades={};counts=collections.Counter();errors=[]
    start=read_json('capture_manifest.json')['started_ms']
    for label,rs in [('par',par),('reference',reference)]:
        for row in rs:
            msg=row['message'];channel=msg['channel'];data=msg.get('data');counts[label+':'+channel]+=1
            if channel in ('l2Book','activeAssetCtx','bbo'):
                events[(label,channel,data['coin'])].append((row['received_ms'],data))
            if channel=='trades':
                for trade in data:
                    if trade['time']>=start:trades[(trade['coin'],trade['time'],trade['tid'])]=dict(trade,received_ms=row['received_ms'])
            if channel=='error':errors.append(row)
    for key in events:events[key].sort(key=lambda x:x[0])
    end=min(max(r['received_ms'] for r in par),max(r['received_ms'] for r in reference))
    panel=[];summary=[];markouts=[]
    capture_status=read_json('capture_status.json')
    disconnects=[int(datetime.datetime.fromisoformat(e['time']).timestamp()*1000) for e in capture_status.get('errors',[]) if e['stream']=='par']
    sizes={a['name'].split(':')[1]:a['szDecimals'] for a in read_json('par_meta.json')['response'][0]['universe']}
    for coin in COINS:
        books=events['par','l2Book','par:'+coin];contexts=events['par','activeAssetCtx','par:'+coin]
        rb=events['reference','bbo',REF[coin]];rc=events['reference','activeAssetCtx',REF[coin]]
        if not books or not contexts:raise ValueError('Missing stream for '+coin)
        times=[[t for t,d in rs] for rs in [books,contexts,rb,rc]]
        first=max(books[0][0],contexts[0][0]);rows=[];gaps=[]
        for disconnected in disconnects:
            b_next=next((t for t,d in books if t>disconnected),end)
            c_next=next((t for t,d in contexts if t>disconnected),end)
            gaps.append((disconnected,max(b_next,c_next)))
        for t in range((first//1000+1)*1000,end//1000*1000,1000):
            if any(a<=t<b for a,b in gaps):continue
            indices=[bisect.bisect_right(ts,t)-1 for ts in times]
            i,j,bi,ci=indices
            if i<0 or j<0:continue
            bt,b=books[i];ct,c=contexts[j];ctx=c['ctx'];levels=b['levels']
            if not levels[0] or not levels[1]:continue
            bid=float(levels[0][0]['px']);ask=float(levels[1][0]['px']);mid=(bid+ask)/2;oracle=float(ctx['oraclePx']);mark=float(ctx['markPx'])
            if ask<bid:raise ValueError('Crossed book '+coin)
            if t-ct>5000:continue
            refmid=None;ref_time=None
            if bi>=0 and rb[bi][1].get('bbo') and all(rb[bi][1]['bbo']):
                ref_time=rb[bi][0];refmid=sum(float(v['px']) for v in rb[bi][1]['bbo'])/2
            q_bid=float(levels[0][0]['sz']);q_ask=float(levels[1][0]['sz'])
            row=dict(coin=coin,time_ms=t,book_received_ms=bt,context_received_ms=ct,book_exchange_ms=b['time'],bid=bid,ask=ask,mid=mid,oracle=oracle,mark=mark,
                     spread_bps=(ask-bid)/mid*1e4,mid_oracle_bps=(mid/oracle-1)*1e4,
                     running_funding_bps_per_hour=float(ctx['funding'])*1e4,
                     buy_at_best_to_oracle_bps=(oracle/ask-1)*1e4,sell_at_best_to_oracle_bps=(bid/oracle-1)*1e4,
                     best_bid_notional=bid*q_bid,best_ask_notional=ask*q_ask,
                     imbalance=(q_bid-q_ask)/(q_bid+q_ask),microprice=(ask*q_bid+bid*q_ask)/(q_bid+q_ask),
                     reference_mid=refmid,reference_received_ms=ref_time,
                     oracle_reference_bps=(oracle/refmid-1)*1e4 if refmid else None,
                     mid_reference_bps=(mid/refmid-1)*1e4 if refmid else None,
                     depth_bid_10bps=sum(float(x['px'])*float(x['sz']) for x in levels[0] if float(x['px'])>=mid*(1-.001)),
                     depth_ask_10bps=sum(float(x['px'])*float(x['sz']) for x in levels[1] if float(x['px'])<=mid*(1+.001)),
                     book_age_seconds=(t-bt)/1000,context_age_seconds=(t-ct)/1000)
            for notional in [100,1000,10000]:
                lot=10**(-sizes[coin]);qty=math.floor(notional/oracle/lot)*lot
                buy=vwap(levels[1],qty);sell=vwap(levels[0],qty)
                row['buy_vwap_'+str(notional)]=buy;row['sell_vwap_'+str(notional)]=sell
                row['buy_to_oracle_net18_bps_'+str(notional)]=(oracle-buy)/buy*1e4-9-9*oracle/buy if buy else None
                row['sell_to_oracle_net18_bps_'+str(notional)]=(sell-oracle)/sell*1e4-9-9*oracle/sell if sell else None
                row['roundtrip_cost_bps_'+str(notional)]=(buy-sell)/((buy+sell)/2)*1e4+18 if buy and sell else None
            rows.append(row);panel.append(row)
        own_trades=sorted([r for r in trades.values() if r['coin']=='par:'+coin],key=lambda r:r['time'])
        s=dict(coin=coin,samples=len(rows),book_messages=len(books),context_messages=len(contexts),excluded_disconnect_intervals_ms=gaps,
               first_sample_ms=rows[0]['time_ms'] if rows else None,last_sample_ms=rows[-1]['time_ms'] if rows else None,
               observed_new_trades=len(own_trades),observed_new_trade_notional=sum(float(r['px'])*float(r['sz']) for r in own_trades))
        fields=['spread_bps','mid_oracle_bps','running_funding_bps_per_hour','best_bid_notional','best_ask_notional',
                'oracle_reference_bps','mid_reference_bps','depth_bid_10bps','depth_ask_10bps','book_age_seconds',
                'buy_to_oracle_net18_bps_1000','sell_to_oracle_net18_bps_1000','roundtrip_cost_bps_1000','roundtrip_cost_bps_10000']
        for key in fields:s[key]=percentiles([r.get(key) for r in rows])
        s['spread_above_maker_roundtrip_fee_fraction']=sum(r['spread_bps']>6 for r in rows)/len(rows)
        s['any_1000_oracle_gap_after18fees_fraction']=sum(max(r['buy_to_oracle_net18_bps_1000'] or -1e9,r['sell_to_oracle_net18_bps_1000'] or -1e9)>0 for r in rows)/len(rows)
        s['oracle_change_count']=sum(a['oracle']!=b['oracle'] for a,b in zip(rows,rows[1:]))
        # Explanatory non-overlapping 60/300-second blocks. No simulated fills or PnL attribution.
        for horizon in [60,300]:
            by_time={r['time_ms']:r for r in rows}
            pairs=[(by_time[t],by_time[t+horizon*1000]) for t in range(rows[0]['time_ms'],rows[-1]['time_ms']-horizon*1000+1,horizon*1000) if t in by_time and t+horizon*1000 in by_time and not any(t<a<t+horizon*1000 for a,b in gaps)]
            s['nonoverlap_'+str(horizon)+'s']=dict(n=len(pairs),
                absolute_mid_move_bps=percentiles([abs(b['mid']/a['mid']-1)*1e4 for a,b in pairs]),
                basis_contraction_bps=percentiles([abs(a['mid_oracle_bps'])-abs(b['mid_oracle_bps']) for a,b in pairs]),
                imbalance_forward_mid_corr=corr([a['imbalance'] for a,b in pairs],[(b['mid']/a['mid']-1)*1e4 for a,b in pairs]))
        # Observed trades -> future known oracle / mid. These are markouts, not closed profits.
        row_times=[r['time_ms'] for r in rows]
        for trade in own_trades:
            sign=1 if trade['side']=='B' else -1;price=float(trade['px'])
            for horizon in [60,300]:
                target=trade['time']+horizon*1000;idx=bisect.bisect_left(row_times,target)
                if idx>=len(rows) or rows[idx]['time_ms']-target>2000:continue
                future=rows[idx]
                markouts.append(dict(coin=coin,trade_time_ms=trade['time'],tid=trade['tid'],horizon_seconds=horizon,
                    taker_side=trade['side'],price=price,size=trade['sz'],future_oracle=future['oracle'],future_mid=future['mid'],
                    taker_oracle_markout_after_entry9_bps=sign*(future['oracle']-price)/price*1e4-9,
                    maker_oracle_markout_after_entry3_bps=-sign*(future['oracle']-price)/price*1e4-3,
                    note='Entry fee assumed at user base tier; no exit fee/funding. Not realized PnL or strategy performance.'))
        summary.append(s)
    csvout('live_panel_1s.csv',panel);csvout('live_unique_trades.csv',list(trades.values()));csvout('trade_markouts.csv',markouts)
    return summary,dict(start_ms=start,end_ms=end,duration_seconds=(end-start)/1000,channel_counts=dict(counts),stream_error_messages=errors,collector_errors=capture_status.get('errors',[]),unique_new_trades=len(trades),resampled_rows=len(panel))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    historical,end=history_stats();live,capture=live_stats(args.partial)
    result=dict(research_status='hypothesis discovery, descriptive evidence; not verified strategy profitability',
                historical_start_ms=START,historical_fill_end_ms=end,historical=historical,live=live,capture=capture,
                limitations=['15-minute single-session order-book sample cannot establish long-run profitability or queue fill probabilities.',
                'Funding history is settled; live context funding is a running estimate, not a locked future payment.',
                'Oracle-gap values assume hypothetical valuation at current oracle, which is not an executable exit quote.',
                'Resting book amounts are displayed liquidity, not guaranteed fills; sweep calculations exclude latency and subsequent impact.',
                'Mainnet reference prices and hourly closes are not historical competition oracle prices.',
                'Venue pa was not found in perpDexs at discovery; conclusions are from practice venue par.',
                'One-second samples are correlated; message/row counts are not independent trial counts.'])
    name='analysis_preview.json' if args.partial else 'analysis.json'
    (HERE/name).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(dict(capture=capture,historical_trades=sum(r['unique_trades'] for r in historical)),indent=2))
    for s in live:
        print(s['coin'],'spread',round(s['spread_bps']['median'],3),'basis',round(s['mid_oracle_bps']['median'],3),
              'bestdepth',round(s['best_bid_notional']['median']),round(s['best_ask_notional']['median']),
              'rt1000',round(s['roundtrip_cost_bps_1000']['median'],2),
              'gaplong',round(s['buy_to_oracle_net18_bps_1000']['median'],2),'gapshort',round(s['sell_to_oracle_net18_bps_1000']['median'],2),
              'newtrades',s['observed_new_trades'])

if __name__=='__main__':main()
