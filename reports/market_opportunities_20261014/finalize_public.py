"""Refresh only public funding and venue metadata; freeze portable historical inputs."""
import concurrent.futures, csv, json, shutil, time
from collect_public import COINS, RAW, fetch

def main():
    endpoint='https://api.hyperliquid-testnet.xyz'
    end=time.time_ns()//1000000
    start=1791158400000  # 2026-10-05 00:00:00 UTC
    def funding(coin):
        result=fetch(endpoint,dict(type='fundingHistory',coin='par:'+coin,startTime=start,endTime=end))
        (RAW/('funding_final_'+coin+'.json')).write_text(json.dumps(result))
        print(coin,len(result.get('response',[])),result.get('error'),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(funding,COINS))
    for name,payload in [('perpDexs_final',dict(type='perpDexs')),('pa_meta_probe',dict(type='metaAndAssetCtxs',dex='pa'))]:
        result=fetch(endpoint,payload)
        (RAW/(name+'.json')).write_text(json.dumps(result))
        print(name,result.get('error','OK'),flush=True)
    previous=RAW.parent.parent/'combined_leaderboard_20261014'
    with (previous/'fills.csv').open(encoding='utf-8-sig',newline='') as src,(RAW/'practice_wallet_fills.csv').open('w',encoding='utf-8-sig',newline='') as dst:
        reader=csv.DictReader(src);writer=csv.DictWriter(dst,fieldnames=reader.fieldnames);writer.writeheader()
        writer.writerows(r for r in reader if int(r['time_ms'])>=start)
    shutil.copyfile(previous/'raw'/'market_now.json',RAW/'historical_cutoff.json')

if __name__=='__main__':main()
