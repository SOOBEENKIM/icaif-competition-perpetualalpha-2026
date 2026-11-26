"""Capture fresh public PAS candles/funding at a precommitted end timestamp."""
import json
import time
import urllib.request
from .protocol import OUT, freeze
from all_market_strategy.signals import SYMBOLS

TN = 'https://api.hyperliquid-testnet.xyz/info'


def main():
    spec = freeze()
    target = OUT / 'pas'
    (target / 'raw').mkdir(parents=True, exist_ok=True)
    cfg = {'start_ms': spec['pas_start_ms'], 'cutoff_ms': spec['pas_cutoff_ms'],
           'warmup_start_ms': spec['pas_start_ms'] - 6 * 3600000, 'symbols': list(SYMBOLS)}
    (target / 'dataset.json').write_text(json.dumps(cfg, indent=2))
    for symbol in SYMBOLS:
        requests = [
            ('pas_candles_' + symbol, {'type': 'candleSnapshot', 'req': {'coin': 'pas:' + symbol,
              'interval': '1m', 'startTime': cfg['warmup_start_ms'], 'endTime': cfg['cutoff_ms'] - 1}}),
            ('pas_funding_' + symbol, {'type': 'fundingHistory', 'coin': 'pas:' + symbol,
              'startTime': cfg['warmup_start_ms'], 'endTime': cfg['cutoff_ms']}),
        ]
        for name, body in requests:
            path = target / 'raw' / (name + '.json')
            if path.exists():
                assert json.loads(path.read_text())['request'] == body
                continue
            request = urllib.request.Request(TN, data=json.dumps(body).encode(),
                headers={'Content-Type': 'application/json', 'User-Agent': 'SFINX-readonly-recheck/1'})
            started = time.time()
            with urllib.request.urlopen(request, timeout=25) as response:
                data = json.load(response)
            assert isinstance(data, list) and len(data) < 5000
            path.write_text(json.dumps({'url': TN, 'request': body, 'started_ms': int(started * 1000),
                                       'received_ms': int(time.time() * 1000), 'data': data}))
            print(name, len(data), flush=True)
            time.sleep(1.2)


if __name__ == '__main__':
    main()
