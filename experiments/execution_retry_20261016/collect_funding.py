"""Read-only settled funding and public metadata, with raw receipts."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from refresh_public_wallets import api, save


def main():
    raw = HERE / 'raw';raw.mkdir(exist_ok=True)
    meta = api({'type': 'metaAndAssetCtxs', 'dex': 'par'})
    if 'response' not in meta:
        raise RuntimeError(meta)
    save(raw / 'meta.json', meta)
    def collect(coin):
        value = api({'type': 'fundingHistory', 'coin': coin, 'startTime': 1791975600000, 'endTime': 1792041600000})
        if not isinstance(value.get('response'), list):
            raise RuntimeError(value)
        save(raw / ('funding_' + coin.replace(':', '_') + '.json'), value)
        return coin, len(value['response'])
    with ThreadPoolExecutor(max_workers=3) as pool:
        print(list(pool.map(collect, [x['name'] for x in meta['response'][0]['universe']])))


if __name__ == '__main__':
    main()
