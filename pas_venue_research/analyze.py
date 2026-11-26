"""Cross-check prices, activity and counterparties without inventing identities."""
import collections
import json
import math
import numpy as np
from .capture import OUT,SYMBOLS,REFERENCES,save


def raw(name):
    return json.loads((OUT/'raw'/f'{name}.json').read_text())


def contexts(name):
    meta,ctx=raw(name)['response']
    return {m['name']:(m,c) for m,c in zip(meta['universe'],ctx)}


def main():
    pas=contexts('pas_meta');ref={**contexts('main_meta'),**contexts('xyz_meta')}
    manifest=json.loads((OUT/'participants_capture.json').read_text())
    wallets=manifest['wallets'];market_fills=collections.defaultdict(dict);activity=[]
    net=collections.defaultdict(float);absolute=collections.defaultdict(float)
    all_records=[]
    for wallet in wallets:
        records={}
        for path in (OUT/'raw').glob(wallet+'_fills_*.json'):
            for f in json.loads(path.read_text())['response']:
                if f['coin'].startswith('pas:') and f['time']<=manifest['cutoff_ms']:
                    records[f['coin'],f['tid']]=f
        fills=list(records.values())
        value=sum(float(f['px'])*float(f['sz']) for f in fills)
        activity.append(dict(wallet=wallet,traded=bool(fills),fill_count=len(fills),
            notional=value,maker_notional=sum(float(f['px'])*float(f['sz']) for f in fills if not f['crossed']),
            fees=sum(float(f['fee']) for f in fills),first_fill=min([f['time'] for f in fills],default=None)))
        for f in fills:
            key=f['coin'],f['tid']
            market_fills[f['coin']].setdefault(key,[]).append(dict(wallet=wallet,**f))
            all_records.append(dict(wallet=wallet,**f))
        positions=raw(wallet+'_positions')['response']['assetPositions']
        for row in positions:
            p=row['position'];net[p['coin']]+=float(p['szi']);absolute[p['coin']]+=abs(float(p['szi']))
    market_rows=[]
    for symbol in SYMBOLS:
        coin='pas:'+symbol;meta,ctx=pas[coin];_,rctx=ref[REFERENCES[symbol]]
        local=raw('pas_book_'+symbol);external=raw('main_book_'+symbol)
        b=local['response'];r=external['response']
        bid,ask=[float(level[0]['px']) for level in b['levels']]
        rb,ra=[float(level[0]['px']) for level in r['levels']]
        mid=(bid+ask)/2;rmid=(rb+ra)/2;oracle=float(ctx['oraclePx'])
        unique=list(market_fills[coin].values())
        matches=sum(len(x)==2 for x in unique);unmatched=sum(len(x)==1 for x in unique)
        trade_volume=sum(float(x[0]['px'])*float(x[0]['sz']) for x in unique)
        participants=collections.defaultdict(float)
        for x in unique:
            for f in x:participants[f['wallet']]+=float(f['px'])*float(f['sz'])
        total=sum(participants.values());shares=np.array(list(participants.values()))/total if total else np.array([])
        candles=raw('pas_candles_'+symbol)['response']
        ref_candles={c['t']:c for c in raw('main_candles_'+symbol)['response']}
        cutoff=int(local['started']*1000)
        completed=[c for c in candles if c['T']<cutoff and c['t']>=1792411200000]
        all_complete=[c for c in candles if c['T']<cutoff]
        active=[c for c in all_complete if c['n']>0 and float(c['v'])>0 and c['t'] in ref_candles
                and ref_candles[c['t']]['n']>0]
        lookup={c['t']:c for c in active}
        consecutive=[(lookup[c['t']-300000],c) for c in active if c['t']-300000 in lookup]
        gap=[math.log(float(c['c'])/float(ref_candles[c['t']]['c'])) for c in active]
        pair_x=np.array([math.log(float(a['c'])/float(ref_candles[a['t']]['c'])) for a,b in consecutive])
        pair_y=np.array([math.log(float(b['c'])/float(ref_candles[b['t']]['c'])) for a,b in consecutive])
        phi=float(np.cov(pair_x,pair_y,ddof=0)[0,1]/np.var(pair_x)) if len(pair_x)>=30 and np.var(pair_x)>1e-12 else None
        funding=raw('pas_funding_'+symbol)['response']
        recent=[float(f['fundingRate'])*10000 for f in funding if f['time']>=1792411200000]
        oi=float(ctx['openInterest'])
        market_rows.append(dict(symbol=symbol,pas_bid=bid,pas_ask=ask,pas_mid=mid,
            pas_oracle=oracle,pas_mark=float(ctx['markPx']),main_mid=rmid,main_oracle=float(rctx['oraclePx']),
            book_server_time_difference_ms=abs(b['time']-r['time']),
            receipt_time_difference_ms=abs(local['received']-external['received'])*1000,
            pas_mid_minus_main_bps=(mid/rmid-1)*10000,
            pas_mid_minus_oracle_bps=(mid/oracle-1)*10000,
            pas_oracle_minus_main_mid_bps=(oracle/rmid-1)*10000,
            pas_mark_minus_oracle_bps=(float(ctx['markPx'])/oracle-1)*10000,
            spread_bps=(ask-bid)/mid*10000,
            top_bid_notional=bid*float(b['levels'][0][0]['sz']),top_ask_notional=ask*float(b['levels'][1][0]['sz']),
            funding_bps_hour=float(ctx['funding'])*10000,
            realized_funding_hours=len(recent),realized_funding_mean_bps_hour=float(np.mean(recent)) if recent else None,
            realized_funding_min_max_bps=[min(recent),max(recent)] if recent else None,
            estimated_team_net_qty=net[coin],team_abs_qty=absolute[coin],market_oi_units=oi,
            estimated_team_share_of_two_sided_oi=absolute[coin]/(2*oi) if oi else None,
            participants_with_fills=len(participants),unique_trades_with_team_side=len(unique),
            both_sides_in_grant_roster=matches,one_side_in_grant_roster=unmatched,
            observed_trade_notional=trade_volume,
            top_participant_side_notional_fraction=float(shares.max()) if len(shares) else None,
            participant_notional_hhi=float(shares@shares) if len(shares) else None,
            scored_completed_5m_bars=len(completed),scored_active_5m_bars=sum(c['n']>0 and float(c['v'])>0 for c in completed),
            all_active_reference_aligned_bars=len(active),consecutive_active_basis_pairs=len(consecutive),
            unselected_ar1_phi_diagnostic=phi,
            active_candle_basis_median_bps=float(np.median(gap)*10000) if gap else None,
            latest_pas_candle_end=max([c['T'] for c in candles],default=None)))
    board=raw('leaderboard')['response']
    unknown=sum(r['one_side_in_grant_roster'] for r in market_rows)
    matched=sum(r['both_sides_in_grant_roster'] for r in market_rows)
    result=dict(snapshot_received=raw('pas_meta')['received'],wallet_cutoff_ms=manifest['cutoff_ms'],
        official_generated_at=board['generated_at'],official_roster=len(board['teams']),
        official_ranked=sum(r['ranked'] for r in board['teams']),
        grant_recipients=len(wallets),observed_traded_wallets=sum(a['traded'] for a in activity),
        no_observed_fill_wallets=sum(not a['traded'] for a in activity),
        unique_trades=unknown+matched,both_sides_registered=matched,one_side_registered=unknown,
        one_side_registered_fraction=unknown/(unknown+matched) if unknown+matched else None,
        participant_notional=sum(a['notional'] for a in activity),
        participant_maker_notional_fraction=sum(a['maker_notional'] for a in activity)/max(sum(a['notional'] for a in activity),1),
        total_participant_fees=sum(a['fees'] for a in activity),markets=market_rows,activity=activity,
        limitations=['The grant roster is not a list of all possible trading counterparties.',
            'An unmatched counterparty is not individually identified as an organizer bot.',
            'Position and OI observations are not atomic; concentration ratios are estimates.',
            '5m close prices are last-trade observations, not synchronous executable bid/ask or oracle prices.',
            'AR1 is a diagnostic on reused data, not a selected profitable strategy.',
            'A visible book is cancelable; a touched passive quote does not prove a fill.'])
    save(OUT/'market_structure.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('markets','activity','limitations')}))
    for m in market_rows:
        print(json.dumps({k:m[k] for k in ('symbol','pas_mid_minus_main_bps','pas_mid_minus_oracle_bps',
            'spread_bps','funding_bps_hour','participants_with_fills','unique_trades_with_team_side',
            'one_side_in_grant_roster','scored_active_5m_bars','scored_completed_5m_bars','consecutive_active_basis_pairs')}))


if __name__=='__main__':
    main()
