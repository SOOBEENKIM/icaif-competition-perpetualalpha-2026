"""Daily formation screening, distinct from forecasts and executable entry choice.

No profit labels or future observations enter this screening. All unordered
pairs have stable IDs, including excluded pairs and currently held pairs.
"""
from dataclasses import asdict, dataclass
from itertools import combinations
import numpy as np

SYMBOLS = ('BTC', 'ETH', 'HYPE', 'SP500', 'EWY', 'EWJ', 'BRENT', 'GOLD', 'USDEUR', 'USDJPY')
PAIRS = tuple(combinations(SYMBOLS, 2))
IDS = tuple((SYMBOLS.index(a), SYMBOLS.index(b)) for a, b in PAIRS)


@dataclass(frozen=True)
class SelectionSpec:
    lookback_days: int = 90
    blocks: int = 3
    minimum_coverage: float = .8
    minimum_correlation: float = .2
    minimum_beta: float = .25
    maximum_beta: float = 4.
    maximum_beta_ratio: float = 2.
    minimum_median_leg_notional: float = 4000.

    def __post_init__(self):
        if (self.lookback_days <= 0 or self.blocks <= 0 or self.lookback_days % self.blocks
                or not 0 < self.minimum_coverage <= 1
                or not 0 <= self.minimum_correlation < 1
                or not 0 < self.minimum_beta < self.maximum_beta
                or self.maximum_beta_ratio < 1 or self.minimum_median_leg_notional < 0):
            raise ValueError('Invalid formation specification')


def validate_snapshot(d):
    t = np.asarray(d['market_times'])
    if (t.ndim != 1 or not len(t) or np.any(t % 3600) or np.any(np.diff(t) != 3600)):
        raise ValueError('Contiguous completed hourly timestamps required')
    if tuple(d.get('symbols', SYMBOLS)) != SYMBOLS:
        raise ValueError('Asset order differs from the canonical universe')
    for key in ('close', 'volume', 'trades'):
        a = np.asarray(d[key])
        if a.shape != (len(t), len(SYMBOLS)) or not np.isfinite(a).all():
            raise ValueError('Invalid snapshot matrix: ' + key)
        if np.any(a <= 0) if key == 'close' else np.any(a < 0):
            raise ValueError('Invalid snapshot values: ' + key)
    return t


def select(d, source, spec=SelectionSpec()):
    """Use only bars completed by source, even when given a longer replay array.

    Screening diagnostics are not p-values or a profitability certificate.
    Positive stable beta is required because the retained execution contract
    is an opposite-side hedge with beta in [.25, 4], not an arbitrary spread.
    """
    raw_times = np.asarray(d['market_times'])
    if source % 86400:
        raise ValueError('Formation selection runs at 00:00 UTC')
    end = int(np.searchsorted(raw_times, source, side='right'))
    if end == 0 or raw_times[end-1] != source:
        raise ValueError('Selection source bar is missing')
    prefix = {k: np.asarray(d[k])[:end] for k in ('market_times', 'close', 'volume', 'trades')}
    if 'symbols' in d:
        prefix['symbols'] = d['symbols']
    t = validate_snapshot(prefix)
    close, volume, trades = (prefix[k] for k in ('close', 'volume', 'trades'))
    r = np.diff(np.log(close), axis=0)
    active = (volume > 0) & (trades > 0)
    fresh = active[1:] & active[:-1]
    width = spec.lookback_days // spec.blocks * 24
    selected = np.zeros(len(PAIRS), bool)
    audit = []
    for k, ((a, b), names) in enumerate(zip(IDS, PAIRS)):
        blocks, reasons = [], []
        for block in range(spec.blocks):
            lo = source - (spec.blocks-block)*width*3600
            hi = lo + width*3600
            mask = (t[1:] > lo) & (t[1:] <= hi) & fresh[:, a] & fresh[:, b]
            n = int(mask.sum()); coverage = n / width
            row = dict(block=block, start_exclusive=lo, end_inclusive=hi, rows=n, coverage=coverage)
            if coverage < spec.minimum_coverage:
                reasons.append('insufficient_fresh_history')
            if n < 2:
                row.update(correlation=None, beta=None, median_leg_notional=None)
                reasons.append('insufficient_return_rows')
            else:
                x, y = r[mask, a], r[mask, b]
                x, y = x-x.mean(), y-y.mean()
                xx, yy, xy = float(x@x), float(y@y), float(x@y)
                corr = xy / np.sqrt(xx*yy) if xx > 1e-20 and yy > 1e-20 else None
                beta = xy / yy if yy > 1e-20 else None
                notionals = volume[1:][mask][:, [a, b]] * close[1:][mask][:, [a, b]]
                liquid = float(np.median(notionals, axis=0).min())
                row.update(correlation=corr, beta=beta, median_leg_notional=liquid)
                if corr is None or corr < spec.minimum_correlation:
                    reasons.append('weak_or_negative_return_relationship')
                if beta is None or not spec.minimum_beta <= beta <= spec.maximum_beta:
                    reasons.append('hedge_beta_outside_supported_range')
                if liquid < spec.minimum_median_leg_notional:
                    reasons.append('low_reference_leg_liquidity')
            blocks.append(row)
        betas = [x['beta'] for x in blocks]
        ratio = max(betas)/min(betas) if all(b is not None and b > 0 for b in betas) else None
        if ratio is None or ratio > spec.maximum_beta_ratio:
            reasons.append('unstable_hedge_beta')
        reasons = sorted(set(reasons))
        selected[k] = not reasons
        audit.append(dict(pair=list(names), pair_index=k, selected=bool(selected[k]),
                          reasons=reasons, beta_ratio=ratio, blocks=blocks))
    return dict(source_time=int(source), known_through=int(t[-1]),
                selection_valid_until=int(source+86400), spec=asdict(spec),
                universe_size=len(PAIRS), selected=selected.tolist(), pairs=audit,
                interpretation='Formation eligibility only; model direction and executable costs are separate gates')
