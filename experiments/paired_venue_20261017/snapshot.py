"""A timestamped public REST snapshot, never a historical fill reconstruction."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import time
import urllib.request
from .forward import HERE, published_pairs, current_snapshots
from .engine import quote_pair
from experiments.pairs_research_20261017.research import save


def request(payload):
    if payload['type'] not in ('metaAndAssetCtxs', 'l2Book'): raise ValueError('Read-only endpoint allowlist')
    sent = time.time()
    req = urllib.request.Request('https://api.hyperliquid-testnet.xyz/info', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=20) as response: data = json.load(response)
    return dict(request=payload, sent_at=sent, received_at=time.time(), response=data)


def main():
    cfg = json.loads((HERE / 'protocol.json').read_text()); frame, obj = published_pairs()
    coins = sorted(set(c for pair in obj['frame']['paired_metrics'] for c in pair['coins']))
    requests = [dict(type='metaAndAssetCtxs', dex='par')] + [dict(type='l2Book', coin=c) for c in coins]
    with ThreadPoolExecutor(max_workers=7) as pool: receipts = list(pool.map(request, requests))
    meta_receipt = receipts[0]; meta, ctx = meta_receipt['response']
    metadata = {r['name']: r for r in meta['universe']}
    contexts = {m['name']: dict(received_ms=int(meta_receipt['received_at']*1000), ctx=c) for m,c in zip(meta['universe'], ctx)}
    books = {r['response']['coin']:dict(received_ms=int(r['received_at']*1000),data=r['response']) for r in receipts[1:]}
    now = int(time.time()*1000); snapshots = current_snapshots(now, books, contexts, cfg)
    rows = []
    for pair in obj['frame']['paired_metrics']:
        if pair['correlation'] < cfg['minimum_correlation'] or any(c not in snapshots for c in pair['coins']): continue
        for horizon in cfg['horizons_hours']:
            for name, scenario in cfg['scenarios'].items():
                choices = [quote_pair(pair['coins'], snapshots, metadata, pair['beta'], direction, horizon, cfg, scenario) for direction in (-1,1)]
                choices = [r for r in choices if r]
                if choices: rows.append(dict(scenario=name, proposed=max(choices, key=lambda p:p['predicted_net_bps'])))
    path = HERE / 'artifacts' / ('snapshot_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    save(path, dict(at_ms=now, signal_time=frame.source_time, receipts=receipts, rows=rows,
        no_orders_sent=True, no_realized_returns=True, fresh_coins=sorted(snapshots)))
    print(json.dumps(dict(path=str(path), fresh_coins=len(snapshots), proposals=len(rows),
        passing=[r for r in rows if r['proposed']['predicted_net_bps'] > cfg['entry_buffer_bps']]), indent=2))


if __name__ == '__main__': main()
