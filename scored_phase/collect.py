"""Public /info queries only; no signer, exchange actions or secret files."""
import json
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
WALLET = '0x125778065a578d598D88bddaA3FdD635280d0E47'
URL = 'https://api.hyperliquid-testnet.xyz/info'


def main():
    directory = ROOT / 'scored_phase/artifacts/public_preflight'
    directory.mkdir(parents=True, exist_ok=True)
    queries = {
        'dexs': {'type': 'perpDexs'},
        'meta': {'type': 'meta', 'dex': 'pas'},
        'contexts': {'type': 'metaAndAssetCtxs', 'dex': 'pas'},
        'limits': {'type': 'perpDexLimits', 'dex': 'pas'},
        'spot_meta': {'type': 'spotMeta'},
        'mode': {'type': 'userAbstraction', 'user': WALLET},
        'positions': {'type': 'clearinghouseState', 'user': WALLET, 'dex': 'pas'},
        'spot': {'type': 'spotClearinghouseState', 'user': WALLET},
        'orders': {'type': 'openOrders', 'user': WALLET, 'dex': 'pas'},
        'fees': {'type': 'userFees', 'user': WALLET},
        'agents': {'type': 'extraAgents', 'user': WALLET},
        'rate_limit': {'type': 'userRateLimit', 'user': WALLET},
    }
    result = {'queried_at_unix': time.time(), 'orders_sent': 0, 'wallet': WALLET}
    with requests.Session() as session:
        for name, payload in queries.items():
            response = session.post(URL, json=payload, timeout=20)
            response.raise_for_status()
            result[name] = response.json()
            print('READ', name, flush=True)
    (directory / 'snapshot.json').write_text(json.dumps(result, indent=2)+'\n')
    token = result['meta'].get('collateralToken')
    print(json.dumps({'at': result['queried_at_unix'],
        'dex': [d for d in result['dexs'] if d and d.get('name')=='pas'],
        'collateral': [t for t in result['spot_meta']['tokens'] if t['index']==token],
        'markets': result['meta']['universe'], 'limits': result['limits'],
        'mode': result['mode'], 'positions': result['positions'],
        'USDP_balances': [b for b in result['spot']['balances'] if b['coin']=='USDP'],
        'orders': result['orders'], 'fees': result['fees']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
