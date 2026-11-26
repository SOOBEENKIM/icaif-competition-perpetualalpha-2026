"""Independent checks of units, two-sided deduplication, causality and historical coverage."""
import collections, csv, datetime, json, math
from pathlib import Path
from analyze_market import vwap

HERE=Path(__file__).resolve().parent
checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)
def rows(name):
    with (HERE/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    a=json.loads((HERE/'analysis.json').read_text())
    check('VWAP walks multiple levels',math.isclose(vwap([{'px':100,'sz':1},{'px':102,'sz':2}],2),101))
    check('VWAP rejects insufficient displayed depth',vwap([{'px':100,'sz':1}],2) is None)
    fs=rows('raw/practice_wallet_fills.csv');groups=collections.defaultdict(list)
    for r in fs:groups[(r['coin'],r['time_ms'],r['tid'])].append(r)
    check('Each historical execution has exactly one maker and one taker',all(len(g)==2 and {r['crossed'] for r in g}=={'True','False'} for g in groups.values()))
    check('Both sides agree on price/size, opposite directions and counterparties',all(g[0]['px']==g[1]['px'] and g[0]['sz']==g[1]['sz'] and {g[0]['side'],g[1]['side']}=={'B','A'} and g[0]['address']==g[1]['counterparty_address'] and g[1]['address']==g[0]['counterparty_address'] for g in groups.values()))
    check('Historical unique execution count matches',len(groups)==sum(r['unique_trades'] for r in a['historical']))
    check('Historical fee token USDPR',all(r['feeToken']=='USDPR' for r in fs))
    check('Observed maker/taker base fees within rounding tolerance',all(abs(float(r['fee'])-float(r['px'])*float(r['sz'])*(.0009 if r['crossed']=='True' else .0003))<.000002 for r in fs))
    candle_counts=collections.Counter()
    for r in rows('hourly_price_panel.csv'):candle_counts[(r['coin'],int(r['t']))]=int(r['n'])
    trade_counts=collections.Counter((k[0].split(':')[1],int(k[1])//3600000*3600000) for k in groups)
    check('Every completed candle count reconciles to unique public fills',all(trade_counts[k]==v for k,v in candle_counts.items()))
    funding=rows('settled_funding.csv');fg=collections.defaultdict(list)
    for r in funding:fg[r['coin']].append(int(r['time']))
    check('All ten assets have 225 settled hourly funding observations',len(fg)==10 and all(len(v)==225 for v in fg.values()))
    check('Funding timestamps unique and hourly within 5 seconds',all(len(set(v))==len(v) and all(abs(y-x-3600000)<5000 for x,y in zip(v,v[1:])) for v in fg.values()))
    panel=rows('live_panel_1s.csv');live={r['coin']:r for r in a['live']}
    check('Panel uses only already-received books and contexts',all(float(r['book_received_ms'])<=float(r['time_ms']) and float(r['context_received_ms'])<=float(r['time_ms']) for r in panel))
    check('Panel excludes disconnected intervals and stale contexts',all(float(r['context_age_seconds'])<=5 and not any(lo<=int(r['time_ms'])<hi for lo,hi in live[r['coin']]['excluded_disconnect_intervals_ms']) for r in panel))
    check('All displayed spreads nonnegative',all(float(r['ask'])>=float(r['bid'])>0 for r in panel))
    check('Instantaneous roundtrip calculation includes both taker fees',all(math.isclose(float(r['roundtrip_cost_bps_1000']), (float(r['buy_vwap_1000'])-float(r['sell_vwap_1000']))/((float(r['buy_vwap_1000'])+float(r['sell_vwap_1000']))/2)*10000+18,abs_tol=1e-9) for r in panel if r['roundtrip_cost_bps_1000']))
    ts=rows('live_unique_trades.csv')
    check('Live trades are unique and exclude subscription history before capture',len({(r['coin'],r['time'],r['tid']) for r in ts})==len(ts)==a['capture']['unique_new_trades'] and all(a['capture']['start_ms']<=int(r['time'])<=a['capture']['end_ms'] for r in ts))
    summary=dict(status='passed',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,checks_count=len(checks),historical_unique_trades=len(groups),completed_candle_buckets=len(candle_counts),funding_rows=len(funding),live_rows=len(panel),live_trades=len(ts),scope='Data and financial arithmetic checks. Not a strategy profitability validation.')
    (HERE/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
