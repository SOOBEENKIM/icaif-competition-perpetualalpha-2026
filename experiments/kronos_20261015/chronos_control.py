"""Existing pinned Chronos-2 checkpoint under the same new daily strategy."""
import os
import json
import time
import numpy as np
import torch
from .data import HERE, ROOT, DAY, load, daily_bars, write, sha


def main():
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    from chronos import Chronos2Pipeline
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    checkpoint = ROOT/'replay_lab/output/tail_calibration/20261014-v1/runtime/pretrained'
    manifest = json.loads((checkpoint/'provenance.json').read_text())
    for name, record in manifest['files'].items():
        if sha(checkpoint/name) != record['sha256']:
            raise ValueError('Changed Chronos checkpoint')
    model = Chronos2Pipeline.from_pretrained(str(checkpoint), device_map='cpu', local_files_only=True)
    m, z, _ = load()
    bars = daily_bars(z)
    start = int(np.datetime64('2026-06-17','s').astype('int64'))
    directory = HERE/'forecasts/chronos_daily'
    directory.mkdir(parents=True, exist_ok=True)
    write(directory/'manifest.json', dict(checkpoint=manifest['revision'], model='amazon/chronos-2',
         lookback=90, horizon=10, point_forecast='median quantile', inputs='raw daily close of all10assets',
         plan_sha256=sha(HERE/'plan.json'), adapter_sha256=sha(__file__)))
    begin = time.monotonic()
    for i, origin in enumerate(bars['known']):
        if origin < start:
            continue
        path = directory/f'{int(origin)}.npz'
        if path.exists():
            continue
        assert i >= 89
        context = bars['close'][i-89:i+1].T.astype('float32')
        q, _ = model.predict_quantiles([context], prediction_length=10, quantile_levels=[.1,.5,.9],
                                      context_length=90, batch_size=10, cross_learning=False)
        forecast = q[0].detach().cpu().numpy()[:, :, 1].astype('float64')
        if not np.isfinite(forecast).all() or np.any(forecast <= 0):
            raise ValueError('Invalid Chronos forecast')
        np.savez_compressed(path, origin=origin, forecast=forecast, last_close=bars['close'][i])
    write(directory/'completed.json', dict(elapsed_seconds=time.monotonic()-begin, live_orders_sent=0))
    print('Chronos daily control complete', flush=True)


if __name__ == '__main__':
    main()
