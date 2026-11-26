"""Compare every sampled token and decoded value against the official loop."""
import json
import time
import numpy as np
from .data import HERE, DAY, HOUR, load, daily_bars, write, sha
from .forecast import official, predict
from .cached_decoder import install


def main():
    predictor = official(4)
    import model.kronos as upstream
    original_sampler = upstream.sample_from_logits
    original_decoder = predictor.model.decode_s1
    m, z, _ = load()
    daily = daily_bars(z)
    origin = int(np.datetime64('2026-06-17', 's').astype('int64'))
    checks = []
    for label, bars, lookback, horizon, step, count in [
        ('daily_all10', daily, 90, 10, DAY, 10),
        ('hourly_one', z, 80, 12, HOUR, 1),
        ('daily_later_one', daily, 90, 10, DAY, 1),
    ]:
        current = origin + (7*DAY if label == 'daily_later_one' else 0)
        traces = []
        def traced(*args, **kwargs):
            out = original_sampler(*args, **kwargs)
            traces.append(out.detach().cpu().numpy().copy())
            return out
        upstream.sample_from_logits = traced
        predictor.model.decode_s1 = original_decoder
        begin = time.monotonic()
        reference, last = predict(predictor, bars, current, lookback, horizon, step, count)
        elapsed_reference = time.monotonic() - begin
        reference_tokens = [x.copy() for x in traces]
        traces.clear()
        install(predictor)
        begin = time.monotonic()
        actual, actual_last = predict(predictor, bars, current, lookback, horizon, step, count)
        elapsed_cached = time.monotonic() - begin
        assert len(reference_tokens) == len(traces) == 2*horizon
        for left, right in zip(reference_tokens, traces):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(reference, actual)
        np.testing.assert_array_equal(last, actual_last)
        record = dict(case=label, origin=current, assets=count, paths_per_asset=10,
                      all_sampled_tokens_exact=True, decoded_forecasts_exact=True,
                      official_seconds=elapsed_reference, cached_seconds=elapsed_cached)
        checks.append(record)
        print(json.dumps(record), flush=True)
        np.savez_compressed(HERE/f'reference_{label}.npz', forecast=reference, last_close=last,
                            sampled_tokens=np.stack(reference_tokens), origin=current)
    upstream.sample_from_logits = original_sampler
    write(HERE/'model_parity.json', dict(passed=True, checks=checks,
          cached_decoder_sha256=sha(HERE/'cached_decoder.py'),
          forecast_adapter_sha256=sha(HERE/'forecast.py'),
          source_manifest_sha256=sha(HERE/'upstream.json'),
          model_manifest_sha256=sha(HERE/'model_provenance.json'),
          scope='Same CPU float32, seed, fixed batch, reference loop and sampled paths. Not a proof of equality for every possible input.'))


if __name__ == '__main__':
    main()
