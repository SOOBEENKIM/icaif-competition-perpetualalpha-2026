"""Cross-check public wallet fills against market candles and later positions."""
import argparse
from collections import Counter
from decimal import Decimal as D
import json
from pathlib import Path

from refresh_public_wallets import api, save
from build_public_wallet_leaderboard import ordered_fills, reconstruct, dec

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    raw=args.report/'raw';market=json.loads((raw/'market_now.json').read_text());end=market['queried_at_ms']
    coins=[u['name'] for u in market['response'][0]['universe']]
    wallets=[json.loads(p.read_text()) for p in (raw/'wallets').glob('*.json')]
    unique={}
    for wallet in wallets:
        for f in wallet['fills']['rows']:
            if f['coin'].startswith('par:') and f['time']<=end:
                unique[f['coin'],f['time'],f['tid']]=f
    hourly=Counter((f['coin'],f['time']//3600000*3600000) for f in unique.values())
    candle_dir=raw/'coverage_candles';candle_dir.mkdir(exist_ok=True)
    counts, mismatches, errors=[],[],[]
    for coin in coins:
        path=candle_dir/(coin.replace(':','_')+'.json')
        if not path.exists():
            save(path,api({'type':'candleSnapshot','req':{'coin':coin,'interval':'1h','startTime':1788220800000,'endTime':end}}))
        response=json.loads(path.read_text())
        if not isinstance(response.get('response'),list):
            errors.append(response);continue
        for bar in response['response']:
            if bar['t']+3600000>end:continue
            observed=hourly.get((coin,bar['t']),0)
            row={'coin':coin,'hour_ms':bar['t'],'exchange_executions':bar['n'],'observed_executions':observed}
            counts.append(row)
            if observed!=bar['n']:mismatches.append(row)
    tails=raw/'position_reconciliation';tails.mkdir(exist_ok=True)
    position_checks=[]
    for wallet in wallets:
        state=wallet.get('positions',{}).get('response')
        if not state:continue
        old=[f for f in wallet['fills']['rows'] if f['coin'].startswith('par:') and f['time']<=end]
        live={p['position']['coin']:dec(p['position']['szi']) for p in state['assetPositions']}
        qty=Counter()
        for f in old:qty[f['coin']]+=dec(f['sz'])*(1 if f['side']=='B' else -1)
        changed=any(qty.get(c,D(0))!=live.get(c,D(0)) for c in set(qty)|set(live))
        extra=[]
        if changed and state['time']>end:
            path=tails/(wallet['address']+'.json')
            if not path.exists():
                save(path,api({'type':'userFillsByTime','user':wallet['address'],'startTime':end+1,'endTime':state['time'],'aggregateByTime':False}))
            response=json.loads(path.read_text())
            if not isinstance(response.get('response'),list):errors.append(response)
            else:extra=[f for f in response['response'] if f['coin'].startswith('par:') and end<f['time']<=state['time']]
        fs,chain_issues=ordered_fills(old+extra)
        final=Counter()
        for f in fs:final[f['coin']]+=dec(f['sz'])*(1 if f['side']=='B' else -1)
        different=[{'coin':c,'replayed':str(final.get(c,D(0))),'queried':str(live.get(c,D(0)))} for c in set(final)|set(live) if final.get(c,D(0))!=live.get(c,D(0))]
        position_checks.append({'address':wallet['address'],'later_fills':len(extra),'query_state_time_ms':state['time'],
                                'differences':different,'chain_issues':chain_issues})
    output={'cutoff_ms':end,'completed_coin_hours_checked':len(counts),
        'completed_hour_market_executions':sum(r['exchange_executions'] for r in counts),
        'completed_hour_observed_executions':sum(r['observed_executions'] for r in counts),
        'candle_count_mismatches':mismatches,'position_wallets_checked':len(position_checks),
        'position_checks':position_checks,'request_errors':errors,
        'passed':not mismatches and not errors and all(not p['differences'] and not p['chain_issues'] for p in position_checks),
        'limits':'Only completed hourly candles are checked against market totals. The final partial hour is not covered by this candle-count assertion. Matching does not identify wallet owners.'}
    save(args.report/'coverage_validation.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='position_checks'},indent=2))
    print('Position differences',json.dumps([p for p in position_checks if p['differences'] or p['chain_issues']],indent=2))

if __name__=='__main__':main()
