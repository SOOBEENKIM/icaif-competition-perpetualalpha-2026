"""Unmodified official CPU inference, pinned weights, causal cached forecasts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

from .data import HERE, ROOT, DAY, HOUR, PRICE_COLS, context, daily_bars, load, make_plan, sha, write


def official(threads=4):
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    source = HERE / 'vendor/Kronos'
    sys.path.insert(0, str(source))
    from model import Kronos, KronosTokenizer, KronosPredictor
    provenance = json.loads((HERE / 'model_provenance.json').read_text())
    for name, entry in provenance.items():
        for file, record in entry['files'].items():
            if sha(HERE / 'models' / name / file) != record['sha256']:
                raise ValueError('Model changed: ' + name + '/' + file)
    tokenizer = KronosTokenizer.from_pretrained(str(HERE / 'models/Kronos-Tokenizer-base'), local_files_only=True).eval()
    model = Kronos.from_pretrained(str(HERE / 'models/Kronos-base'), local_files_only=True).eval()
    return KronosPredictor(model, tokenizer, device='cpu', max_context=512, clip=5)


def predict(predictor, bars, origin, lookback, horizon, step, count=10):
    from .cached_decoder import CachedDecoder
    if isinstance(predictor.model.decode_s1, CachedDecoder):
        predictor.model.decode_s1.reset()
    frames, past, future = zip(*(context(bars, origin, j, lookback, horizon, step) for j in range(count)))
    torch.manual_seed(100 + origin // DAY)
    with torch.inference_mode():
        result = predictor.predict_batch(list(frames), list(past), list(future), pred_len=horizon,
                                         T=.6, top_p=.9, top_k=0, sample_count=10, verbose=False)
    values = np.stack([x[PRICE_COLS].to_numpy() for x in result])
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Nonfinite/nonpositive forecast')
    last = np.array([f.close.iloc[-1] for f in frames])
    return values, last


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['daily', 'hourly'], default='daily')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--cached', action='store_true')
    args = p.parse_args()
    plan = make_plan()
    m, z, _ = load()
    bars = daily_bars(z) if args.kind == 'daily' else z
    lookback, horizon, step = (90, 10, DAY) if args.kind == 'daily' else (80, 12, HOUR)
    first = int(np.datetime64(plan['calibration']['start'].removesuffix('Z'), 's').astype('int64'))
    origins = bars['known'][(bars['known'] >= first) & (bars['known'] % DAY == 0)]
    if args.limit:
        origins = origins[:args.limit]
    predictor = official(args.threads)
    if args.cached:
        if not args.benchmark:
            proof = json.loads((HERE/'model_parity.json').read_text())
            if not proof['passed'] or proof['cached_decoder_sha256'] != sha(HERE/'cached_decoder.py'):
                raise ValueError('Official inference parity must pass first')
        from .cached_decoder import install
        install(predictor)
    start = time.monotonic()
    directory = HERE / 'forecasts' / args.kind
    directory.mkdir(parents=True, exist_ok=True)
    if args.benchmark:
        for count in [1, 10]:
            before = time.monotonic()
            values, last = predict(predictor, bars, int(origins[0]), lookback, horizon, step, count)
            print(json.dumps({'assets': count, 'threads': args.threads, 'seconds': time.monotonic()-before,
                              'shape': list(values.shape), 'finite': bool(np.isfinite(values).all())}), flush=True)
        return
    frozen = {'plan_sha256': sha(HERE/'plan.json'), 'source_sha256': plan['source_sha256'],
              'model_provenance_sha256': sha(HERE/'model_provenance.json'),
              'upstream_sha256': sha(HERE/'upstream.json'), 'threads': args.threads,
              'cached_decoder': args.cached,
              'cached_decoder_sha256': sha(HERE/'cached_decoder.py') if args.cached else None,
              'forecast_adapter_sha256': sha(Path(__file__)), 'data_adapter_sha256': sha(HERE/'data.py')}
    meta_path = directory / 'manifest.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != frozen:
        raise ValueError('Do not reuse cache from changed implementation/settings')
    write(meta_path, frozen)
    for n, origin in enumerate(origins):
        origin = int(origin)
        path = directory / f'{origin}.npz'
        if path.exists():
            continue
        before = time.monotonic()
        values, last = predict(predictor, bars, origin, lookback, horizon, step)
        tmp = path.with_suffix('.partial.npz')
        np.savez_compressed(tmp, origin=origin, forecast=values, last_close=last,
                            seed=100 + origin//DAY, context_end=origin)
        tmp.replace(path)
        print(json.dumps({'kind': args.kind, 'done': n+1, 'total': len(origins),
                          'date': str(np.datetime64(origin,'s')), 'seconds': round(time.monotonic()-before,2),
                          'elapsed': round(time.monotonic()-start,2)}), flush=True)
    write(directory / 'completed.json', {'origins': len(origins), 'elapsed_seconds': time.monotonic()-start,
                                          'last_origin': int(origins[-1]), 'live_orders_sent': 0})


if __name__ == '__main__':
    main()
