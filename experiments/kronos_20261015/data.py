from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'replay_lab/output/tail_calibration/20261014-v1/runtime/frozen_inputs.npz'
BASELINE = ROOT / 'replay_lab/output/tail_calibration/20261014-v1/baseline.json'
HOUR, DAY = 3600, 86400
PRICE_COLS = ['open', 'high', 'low', 'close']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def load():
    from replay_lab.common_comparison.models import Market
    z = dict(np.load(SOURCE, allow_pickle=False))
    b = json.loads(BASELINE.read_text())
    symbols = tuple(b['protocol']['symbols'])
    m = Market(z['market_times'], z['close'], z['volume'], z['trades'], z['funding'], z['observed'], symbols)
    assert np.array_equal(z['times'] + HOUR, z['known'])
    assert np.array_equal(z['known'], m.times)
    assert np.all(np.diff(m.times) == HOUR)
    assert m.observed.all() and np.isfinite(m.funding).all()
    return m, z, b['protocol']


def daily_bars(z):
    """Use exactly 24 completed UTC hours; discard both partial edge days."""
    times = z['times']
    groups, known = [], []
    for day in np.unique(times // DAY):
        ids = np.flatnonzero(times // DAY == day)
        if len(ids) != 24 or not np.array_equal(times[ids], day * DAY + np.arange(24) * HOUR):
            continue
        groups.append(ids)
        known.append(int((day + 1) * DAY))
    bars = {k: [] for k in PRICE_COLS}
    for ids in groups:
        bars['open'].append(z['open'][ids[0]])
        bars['high'].append(z['high'][ids].max(0))
        bars['low'].append(z['low'][ids].min(0))
        bars['close'].append(z['close'][ids[-1]])
    return {'known': np.asarray(known, dtype=np.int64),
            'times': np.asarray(known, dtype=np.int64) - DAY,
            **{k: np.asarray(v) for k, v in bars.items()}}


def context(bars, origin, asset, lookback, horizon, step):
    ids = np.flatnonzero(bars['known'] <= origin)
    if len(ids) < lookback or bars['known'][ids[-1]] != origin:
        raise ValueError('Insufficient or stale completed bars')
    ids = ids[-lookback:]
    if not np.all(np.diff(bars['times'][ids]) == step):
        raise ValueError('Noncontiguous context')
    frame = pd.DataFrame({k: bars[k][ids, asset] for k in PRICE_COLS})
    values = frame.to_numpy()
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Invalid OHLC context')
    if np.any(frame.high < frame[['open', 'close']].max(axis=1)) or np.any(frame.low > frame[['open', 'close']].min(axis=1)):
        raise ValueError('Invalid OHLC bounds')
    # Future timestamps are calendar arithmetic, never slices of future data.
    past = pd.Series(pd.to_datetime(bars['times'][ids], unit='s', utc=True).tz_localize(None))
    future = pd.Series(pd.to_datetime(origin + np.arange(horizon) * step, unit='s', utc=True).tz_localize(None))
    return frame, past, future


def make_plan():
    m, z, protocol = load()
    plan = {
        'name': 'Kronos paper-aligned adaptation v1',
        'paper': 'https://arxiv.org/html/2508.02739v1',
        'upstream_commit': '67b630e67f6a18c9e9be918d9b4337c960db1e9a',
        'model': 'NeoQuasar/Kronos-base', 'tokenizer': 'NeoQuasar/Kronos-Tokenizer-base',
        'mode': 'official pretrained zero-shot; no fine-tuning or pretraining',
        'model_last_modified': '2025-09-09; exact revisions in model_provenance.json',
        'source_sha256': sha(SOURCE), 'symbols': list(m.symbols),
        'inference': {'T': 0.6, 'top_p': 0.9, 'top_k': 0, 'sample_count': 10,
                      'max_context': 512, 'clip': 5.0, 'dtype': 'float32', 'device': 'cpu',
                      'seed': 100, 'seed_scheme': '100 + origin UTC-day number; reset every origin',
                      'batch': 'all 10 assets at one origin, fixed symbol order'},
        'primary': {'lookback': 90, 'horizon': 10, 'frequency': 'complete UTC daily bars',
                    'input': 'OHLC only; official missing-volume/amount=zero convention',
                    'score': 'mean(predicted close[1:11])/last observed close - 1; paper Eq13',
                    'top_k': 2, 'drop_n': 1, 'minimum_holding_days': 5,
                    'gross_before_calibration': 0.27, 'long_only': True,
                    'positive_forecast_filter': False, 'rebalance_buffer': 0.2,
                    'selection': 'daily top-k/drop-n based on actual filled inventory; deterministic ties'},
        'hourly_diagnostic': {'lookback': 80, 'horizon': 12, 'sample_origins': 'daily UTC midnight',
                              'input': 'OHLC only', 'score': 'last predicted close/current close-1; paper Eq11',
                              'used_for_trading_or_selection': False},
        'calibration': {'start': '2026-06-17T00:00:00Z', 'end': '2026-07-31T23:00:00Z',
                        'rule': 'min(1, 0.0009 / development daily population STD); one fixed scale per model',
                        'daily_std_budget': 0.0009, 'parameter_grid': False},
        'validation': {'start': '2026-08-01T00:00:00Z', 'end': '2026-08-31T23:00:00Z',
                       'used_to_select_or_modify': False},
        'comparison': {'start': '2026-09-01T00:00:00Z', 'end': '2026-10-13T03:00:00Z'},
        'comparators': ['selected_default_unchanged', 'selected_original_unchanged',
                        'chronos2_same_daily_topk_policy', 'equal_weight_same_max_gross'],
        'success_rule': 'Report all results. Improvement requires higher ROI and no higher daily STD than selected_default, positive base and stress ROI, plus all engineering checks. No automatic replacement.',
        'protocol': protocol,
        'adaptations': [
            'Ten mixed reference markets replace CSI300/CSI800 stocks; k50/drop5 scaled to k2/drop1.',
            'Calendar UTC days replace A-share trading days; actual competition daily snapshot time not established.',
            'Account risk caps and turnover buffer replace full-capital stock allocation.',
            'Existing shared hourly ledger: one-hour-delayed reference-close proxy fills, actual reference funding, 9bp fee and modeled impact.',
            'OHLC-only is paper crypto/FX convention, extended to mixed proxy universe; no invented traded amount.',
            '10 sampled paths from paper Table6; official fine-tune demo defaults to5.',
            'Paper Eq13 de-normalized relative return; official qlib_test.py demo instead ranks normalized close differences.',
            'Released base102M used; unreleased large499M and original multi-billion-row training are not reproduced.',
            'No original CSI stock data or full 25-baseline research replication claim.'
        ],
        'historical_reuse': True, 'independent_holdout': False,
        'new_market_collection': False, 'live_orders_enabled': False,
        'parameters_changed_after_comparison': False,
    }
    path = HERE / 'plan.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('Frozen plan differs')
    else:
        write(path, plan)
    return plan
