"""Small, dependency-free supervised baseline with chronological purging."""

import csv
import hashlib
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SYMBOLS = ('BTC', 'ETH', 'HYPE', 'SP500', 'EWY', 'EWJ', 'BRENT', 'GOLD', 'USDEUR', 'USDJPY')
SIZE_DECIMALS = dict(zip(SYMBOLS, (4, 4, 2, 3, 3, 3, 2, 3, 1, 2)))
FEATURE_NAMES = ('1시간 수익률', '3시간 수익률', '6시간 수익률', '12시간 수익률',
                 '24시간 수익률', '48시간 수익률', '24시간 변동성', '현재 봉 가격범위',
                 '평균 대비 거래량', 'UTC 시간 sin', 'UTC 시간 cos')
VAL_START = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp())
TEST_START = int(datetime(2026, 8, 15, tzinfo=timezone.utc).timestamp())


@dataclass(frozen=True, slots=True)
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int

    def packed(self):
        return [self.time, self.open, self.high, self.low, self.close, self.volume]


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace('+00:00', 'Z')


def load_history(directory: Path):
    data, manifest = {}, {}
    for symbol in SYMBOLS:
        path = directory / (symbol + '.csv')
        candles = []
        with path.open(newline='') as source:
            for row in csv.DictReader(source):
                t = int(datetime.fromisoformat(row['time'].replace('Z', '+00:00')).timestamp())
                values = [float(row[key]) for key in ('open', 'high', 'low', 'close', 'volume')]
                o, h, low, c, volume = values
                trades = int(row['trades'])
                if (not all(math.isfinite(v) for v in values) or min(o, h, low, c) <= 0
                        or not low <= min(o, c) <= max(o, c) <= h or volume < 0 or trades < 0):
                    raise ValueError(f'{symbol}: invalid OHLCV at {row["time"]}')
                if candles and t - candles[-1].time != 3600:
                    raise ValueError(f'{symbol}: duplicate, missing or unsorted hour at {row["time"]}')
                candles.append(Candle(t, o, h, low, c, volume, trades))
        if len(candles) < 100 or candles[-1].time < TEST_START:
            raise ValueError(f'{symbol}: insufficient history for the documented split')
        data[symbol] = candles
        manifest[symbol] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'rows': len(candles), 'start': iso(candles[0].time),
                            'end': iso(candles[-1].time + 3600),
                            'zero_activity': sum(c.volume == 0 or c.trades == 0 for c in candles)}
    common = set.intersection(*(set(c.time for c in rows) for rows in data.values()))
    if not common:
        raise ValueError('No common time range')
    data = {s: [c for c in rows if c.time in common] for s, rows in data.items()}
    return data, manifest


def features(rows):
    result = [None] * len(rows)
    for i in range(48, len(rows)):
        c = rows[i]
        rets = [math.log(c.close / rows[i - lag].close) for lag in (1, 3, 6, 12, 24, 48)]
        hourly = [math.log(rows[k].close / rows[k - 1].close) for k in range(i - 23, i + 1)]
        avg_volume = statistics.fmean(r.volume for r in rows[i - 23:i + 1])
        hour = ((c.time + 3600) // 3600) % 24
        result[i] = rets + [statistics.pstdev(hourly), (c.high - c.low) / c.close,
                            math.log1p(c.volume) - math.log1p(avg_volume),
                            math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)]
    return result


def solve(matrix, vector):
    """Partial-pivot Gaussian elimination; positive ridge removes singularity."""
    n = len(vector)
    a = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
    for j in range(n):
        p = max(range(j, n), key=lambda k: abs(a[k][j]))
        a[j], a[p] = a[p], a[j]
        if abs(a[j][j]) < 1e-14:
            raise ValueError('Singular model matrix')
        factor = a[j][j]
        a[j] = [v / factor for v in a[j]]
        for k in range(n):
            if k == j:
                continue
            factor = a[k][j]
            a[k] = [v - factor * pivot for v, pivot in zip(a[k], a[j])]
    return [row[-1] for row in a]


class RidgeModel:
    """Replace this fit/predict/explain contract with a future forecasting model."""

    def fit(self, x, y, alpha):
        n, p = len(x), len(x[0])
        self.means = [statistics.fmean(row[j] for row in x) for j in range(p)]
        self.scales = [max(statistics.pstdev(row[j] for row in x), 1e-10) for j in range(p)]
        self.intercept = statistics.fmean(y)
        self.alpha = alpha
        gram = [[0.0] * p for _ in range(p)]
        rhs = [0.0] * p
        for row, target in zip(x, y):
            z = [(v - m) / s for v, m, s in zip(row, self.means, self.scales)]
            for j in range(p):
                rhs[j] += z[j] * (target - self.intercept) / n
                for k in range(j, p):
                    gram[j][k] += z[j] * z[k] / n
        for j in range(p):
            for k in range(j):
                gram[j][k] = gram[k][j]
            gram[j][j] += alpha
        self.weights = solve(gram, rhs)
        return self

    def predict(self, row):
        return self.intercept + sum(w * (v - m) / s for v, w, m, s in
                                    zip(row, self.weights, self.means, self.scales))

    def explain(self, row):
        contributions = [(name, w * (v - m) / s) for name, v, w, m, s in
                         zip(FEATURE_NAMES, row, self.weights, self.means, self.scales)]
        return sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:3]

    def state(self):
        return dict(alpha=self.alpha, intercept=self.intercept, weights=self.weights,
                    means=self.means, scales=self.scales, features=FEATURE_NAMES)


def prediction_metrics(pred, actual):
    if not actual:
        return {'n': 0, 'mae': None, 'rmse': None, 'direction_accuracy': None, 'zero_mae': None}
    nonzero = [(p, y) for p, y in zip(pred, actual) if abs(y) > 1e-12]
    return {'n': len(actual), 'mae': statistics.fmean(abs(p-y) for p, y in zip(pred, actual)),
            'rmse': math.sqrt(statistics.fmean((p-y)**2 for p, y in zip(pred, actual))),
            'direction_accuracy': statistics.fmean((p > 0) == (y > 0) for p, y in nonzero) if nonzero else None,
            'zero_mae': statistics.fmean(abs(y) for y in actual)}


def split_samples(rows, horizon, validation_start=VAL_START, test_start=TEST_START):
    """Labels must finish before the next chronological partition starts."""
    train, validation, test = [], [], []
    for i in range(48, len(rows) - horizon):
        available_at = rows[i].time + 3600
        label_at = rows[i + horizon].time + 3600
        sample = (i, math.log(rows[i + horizon].close / rows[i].close))
        if available_at < validation_start and label_at <= validation_start:
            train.append(sample)
        elif validation_start <= available_at < test_start and label_at <= test_start:
            validation.append(sample)
        elif available_at >= test_start:
            test.append(sample)
    if min(len(train), len(validation), len(test)) < 24:
        raise ValueError('Insufficient chronological train/validation/test samples')
    return train, validation, test


def train_models(data, horizon, validation_start=VAL_START, test_start=TEST_START):
    models, forecasts, explanations, summary = {}, {}, {}, {}
    for symbol, rows in data.items():
        x = features(rows)
        train, validation, test = split_samples(rows, horizon, validation_start, test_start)
        candidates = []
        for alpha in (0.001, 0.01, 0.1, 1.0):
            model = RidgeModel().fit([x[i] for i, _ in train], [y for _, y in train], alpha)
            metric = prediction_metrics([model.predict(x[i]) for i, _ in validation], [y for _, y in validation])
            candidates.append((metric['mae'], model, metric))
        _, best, validation_metric = min(candidates, key=lambda item: item[0])
        models[symbol] = best
        forecasts[symbol] = [best.predict(row) if row is not None else None for row in x]
        explanations[symbol] = [best.explain(row) if row is not None else [] for row in x]
        summary[symbol] = {'train_n': len(train), 'validation': validation_metric,
                           'test': prediction_metrics([forecasts[symbol][i] for i, _ in test], [y for _, y in test]),
                           'train_last_label_at': iso(rows[train[-1][0] + horizon].time + 3600),
                           'validation_last_label_at': iso(rows[validation[-1][0] + horizon].time + 3600),
                           'model': best.state()}
    return forecasts, explanations, summary
