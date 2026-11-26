"""Public read-only account refresh for a fixed, previously selected peer set."""
import importlib.util,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'artifacts'
PEERS=['0x125778065a578d598d88bddaa3fdd635280d0e47','0xdd0da0d42b4d19c8ebe86e8f3d33e41c2ef75f84','0xffac3f6faa48163bfe7c6ea6207bdbab34139c3f','0x06a336acf9d4ed8fae9e60c453fff4ecda409a11','0x74ca48ffc4ce36072057609f6d2e1732fece948a']

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'public_snapshot.json').exists():raise FileExistsError('Snapshot already complete')
    path=ROOT/'reports/wallet_leaderboard_20261016/source/refresh_public_wallets.py'
    spec=importlib.util.spec_from_file_location('public_snapshot',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    market=module.api({'type':'metaAndAssetCtxs','dex':'par'})
    if not isinstance(market.get('response'),list):raise ValueError(market)
    end=int(time.time()*1000)
    def refresh(address):
        previous=json.loads((ROOT/'reports/wallet_leaderboard_20261016/raw/wallets'/f'{address}.json').read_text())
        result={'address':address,'end_ms':end}
        for field,kind in [('fills','userFillsByTime'),('funding','userFunding'),('ledger','userNonFundingLedgerUpdates')]:
            result[field]=module.history(kind,address,previous,end)
            if not result[field]['pagination_complete']:raise ValueError((address,field,result[field]['note']))
        result['positions']=module.api({'type':'clearinghouseState','user':address,'dex':'par'})
        if not isinstance(result['positions'].get('response'),dict):raise ValueError('Account query failed')
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:wallets=list(pool.map(refresh,PEERS))
    module.save(OUT/'public_snapshot.json',{'market':market,'end_ms':end,'wallets':wallets,'live_orders_sent':0})
    print(json.dumps({'public_wallets':len(wallets),'end_ms':end,'live_orders_sent':0}))

if __name__=='__main__':main()
