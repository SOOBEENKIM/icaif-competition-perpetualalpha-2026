"""Read public portfolio histories; never load wallet keys or place orders."""
import concurrent.futures
import datetime as dt
import json
from pathlib import Path
import sys
import urllib.request

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / 'scripts'))
from refresh_public_wallets import api, save, USER

ROOT = Path(__file__).resolve().parents[1]
PRIOR = PROJECT / 'reports/wallet_leaderboard_20261016'


def main():
    raw = ROOT / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    wallets = json.loads((PRIOR / 'analysis.json').read_text())['wallets']
    wallets.sort(key=lambda w: (w['address'] != USER, w['rank'] or 999))
    url = 'https://stats-data.hyperliquid.xyz/Testnet/leaderboard'
    with urllib.request.urlopen(url, timeout=30) as response:
        leaderboard = json.load(response)
        headers = dict(response.headers)
    save(raw / 'exchange_leaderboard.json', {
        'source': url, 'received_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'headers': headers, 'response': leaderboard,
    })

    def collect(wallet):
        address = wallet['address']
        path = raw / (address + '.json')
        if path.exists():
            value = json.loads(path.read_text())
            if isinstance(value.get('response'), list):
                return value
        value = api({'type': 'portfolio', 'user': address})
        save(path, value)
        print(json.dumps({'wallet': wallet['wallet_id'], 'address': address,
                          'ok': isinstance(value.get('response'), list),
                          'error': value.get('error')}), flush=True)
        return value

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(collect, wallets))
    summary = {'wallets': len(values),
               'ok': sum(isinstance(v.get('response'), list) for v in values),
               'finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    save(ROOT / 'collection.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
