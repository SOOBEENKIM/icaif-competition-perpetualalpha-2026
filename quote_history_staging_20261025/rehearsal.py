"""Unsigned feature/target parity on the already exposed first trial only."""
import json
import time

import numpy as np

from .adapter import attach_history, prepare_unsigned
from .archive import PointCache
from production_replay_20261025.book_replay import load_books
from production_replay_20261025.quote_history import quote_history, quote_target
from production_replay_20261025.replication import check, PREVIOUS, OUT
from competition_live.runtime import private_load
from experiments.basis_carry_20261017.probe import asof


def run():
    contract = check()
    # Do not open the second trial's archive or any second-trial performance.
    source = 1792908000  # 2026-10-25 06:00 UTC, already exposed development data.
    cutoff = 1792915500  # First trial's fixed end.
    streams, manifest = load_books([
        PREVIOUS / 'sealed_forward_capture/received_through_fixed_end.jsonl.gz'], cutoff)
    packets = [json.loads(p.read_text()) for p in (PREVIOUS / 'forward_signals').glob('*.json')]
    packet = next(p for p in packets if p['packet']['body']['frame']['source_time'] == source)
    now = packet['received_at'] + 1
    profile = contract['profile']
    symbols = profile['trading']['symbols']
    seed = private_load(PREVIOUS / 'forward_seed.pkl')['portfolio']
    candidate = prepare_unsigned(seed)
    # Search only for an old, valid public observation to test adapter parity.
    # This is an implementation check, never a performance-selected start.
    for _ in range(60):
        quotes = {s: asof(streams[s], now * 1000, candidate.cfg) for s in symbols}
        if all(q is not None for q in quotes.values()):
            break
        now += 1
    else:
        raise ValueError('No old complete quote observation for parity rehearsal')
    current = {}
    for symbol, q in quotes.items():
        s = dict(resets=[])
        for field in ('book', 'ctx'):
            at = q['received'][field]
            s[field] = [(at, 0, q[field])]
            s[field + '_times'] = [at]
        current[symbol] = s
    joined, evidence = attach_history(current, streams, symbols, source, now, candidate.cfg)
    cache = PointCache(OUT / 'staging_old_points_v2', symbols, candidate.cfg)
    cache_started = time.perf_counter()
    cached_streams, cache_hashes = cache.history(source, [
        PREVIOUS / 'sealed_forward_capture/received_through_fixed_end.jsonl.gz'])
    cache_seconds = time.perf_counter() - cache_started
    cached_joined, _ = attach_history(current, cached_streams, symbols, source, now, candidate.cfg)
    expected, _ = quote_history(streams, symbols, source, candidate.cfg)
    actual, _ = quote_history(joined, symbols, source, candidate.cfg)
    np.testing.assert_array_equal(expected, actual)
    cached_prices, _ = quote_history(cached_joined, symbols, source, candidate.cfg)
    np.testing.assert_array_equal(expected, cached_prices)
    frame = packet['packet']['body']['frame']
    marks = np.array([float(quotes[s]['ctx']['markPx']) for s in symbols])
    equity = candidate.account.reconcile(marks)
    args = (frame, marks, candidate.account.owners['base'].qty, equity, candidate.p)
    target1, trace1 = quote_target(*args, streams, candidate.cfg)
    target2, trace2 = quote_target(*args, joined, candidate.cfg)
    target3, _ = quote_target(*args, cached_joined, candidate.cfg)
    if target1 is None or target2 is None:
        raise ValueError('Parity must exercise a real target, not two skipped decisions')
    np.testing.assert_array_equal(target1['target_qty'], target2['target_qty'])
    np.testing.assert_array_equal(target1['target_qty'], target3['target_qty'])
    for symbol in symbols:
        assert asof(current[symbol], now * 1000, candidate.cfg) == asof(joined[symbol], now * 1000, candidate.cfg)
    result = dict(at=time.time(),source=source,observed_at=now,
        archive_sha=manifest['files'],all_ten_feature_points_identical=True,
        all_ten_targets_identical=True,all_ten_execution_observations_unchanged=True,
        persistent_cache_features_and_targets_identical=True,
        cache_seconds=cache_seconds,cache_hashes=cache_hashes,
        target_qty=target2['target_qty'],history=evidence,
        new_performance_evidence=False,live_changed=False,orders_sent=0,
        limitation='Unsigned integration parity on old data only. No active-pair migration, worker launch, or live deployment.')
    (OUT / 'staging_old_data_rehearsal.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('history','target_qty')}))
    check()


if __name__ == '__main__':
    run()
