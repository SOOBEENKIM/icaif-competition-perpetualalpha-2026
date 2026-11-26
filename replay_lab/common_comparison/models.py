"""Causal inputs and three explicitly budgeted ridge fits; no execution code."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


HOUR = 3600
SYMBOLS = ("BTC", "ETH", "HYPE")


def timestamp(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def iso(value):
    return datetime.fromtimestamp(int(value), timezone.utc).isoformat().replace("+00:00", "Z")


def funding_bucket(milliseconds):
    """Normalize <1s reporting jitter to its stated hour, never arbitrary late events."""
    offset = int(milliseconds) % (HOUR * 1000)
    if offset >= 1000:
        raise ValueError("Funding timestamp is outside the declared subsecond tolerance")
    return int(milliseconds) // (HOUR * 1000) * HOUR


@dataclass
class Market:
    times: np.ndarray  # candle CLOSE / reference valuation time
    close: np.ndarray
    volume: np.ndarray
    trades: np.ndarray
    funding: np.ndarray
    observed: np.ndarray
    symbols: tuple = SYMBOLS

    def source(self, decision, delay=HOUR):
        return int(np.searchsorted(self.times, decision - delay, side="right") - 1)

    def exact(self, time):
        index = int(np.searchsorted(self.times, time))
        if index >= len(self.times) or self.times[index] != time:
            raise ValueError(f"Missing reference close at {iso(time)}")
        return index


def load_inputs(root):
    from ..model import load_history
    from ..maker.model import arrays

    data, data_manifest = load_history(root / "history")
    original = arrays(data)
    times = original["known"]
    rates = np.zeros((len(times), 3))
    observed = np.zeros_like(rates, dtype=bool)
    archive = root / "replay_lab/output/retained_strategies/manifest.json"
    manifest = json.loads(archive.read_text())
    # Use the exact, already-hashed retained inputs, never a moving glob of new captures.
    lookup = {int(t): i for i, t in enumerate(times)}
    for j, symbol in enumerate(SYMBOLS):
        paths = [name for name in manifest["data_hashes"] if name.endswith(f"/{symbol}-funding.json")]
        for name in sorted(paths):
            for batch in json.loads((root / name).read_text()):
                for row in batch.get("response", []):
                    milliseconds = int(row["time"])
                    time = funding_bucket(milliseconds)
                    if time not in lookup:
                        continue
                    rate = float(row["fundingRate"])
                    if not np.isfinite(rate):
                        raise ValueError("Nonfinite funding rate")
                    i = lookup[time]
                    if observed[i, j] and rates[i, j] != rate:
                        raise ValueError("Conflicting observed funding")
                    rates[i, j], observed[i, j] = rate, True
    market = Market(times.copy(), original["close"][:, :3].copy(),
                    original["volume"][:, :3].copy(), original["trades"][:, :3].copy(),
                    rates, observed)
    return market, original, data_manifest


def features(market):
    close = market.close
    x = np.full((len(close), 3, 7), np.nan)
    returns = np.diff(np.log(close), axis=0)
    for i in range(168, len(close)):
        x[i, :, :5] = np.stack([np.log(close[i] / close[i - h]) for h in (1, 4, 24, 72, 168)], axis=1)
        x[i, :, 5] = returns[i - 24:i].std(axis=0, ddof=0)
        x[i, :, 6] = returns[i - 168:i].std(axis=0, ddof=0)
    return x


def covariance(market, source, risk):
    if source < 168:
        raise ValueError("Covariance requires 169 completed closes")
    returns = np.diff(np.log(market.close[source - 168:source + 1]), axis=0)
    c = np.cov(returns, rowvar=False, ddof=0)
    diag_weight = risk["covariance_diagonal_weight"]
    sigma = (1 - diag_weight) * c + diag_weight * np.diag(np.diag(c))
    np.fill_diagonal(sigma, np.maximum(np.diag(sigma), risk["hourly_sigma_floor"] ** 2))
    return c, sigma


def training_rows(market, x, start, end):
    rows = []
    for decision in range(((start + HOUR - 1) // HOUR) * HOUR, end, HOUR):
        if decision % (4 * HOUR):
            continue
        source = market.source(decision)
        maturity = decision + 6 * HOUR
        if source < 168 or maturity >= end:
            continue
        first = market.exact(decision + HOUR)
        last = market.exact(decision + 5 * HOUR)
        for j in range(3):
            if (market.volume[first, j] <= 0 or market.volume[last, j] <= 0
                    or market.trades[first, j] <= 0 or market.trades[last, j] <= 0):
                continue
            if not np.isfinite(x[source, j]).all():
                continue
            rows.append((j, source, decision, maturity,
                         float(np.log(market.close[last, j] / market.close[first, j]))))
    return rows


def fit_ridge(market, x, start, end, alpha):
    rows = training_rows(market, x, start, end)
    if not rows:
        raise ValueError("Empty fitting set")
    means, scales, target_means, target_scales = [], [], [], []
    counts, design, response = [], [], []
    for j in range(3):
        subset = [r for r in rows if r[0] == j]
        if len(subset) < 30:
            raise ValueError("Insufficient asset-specific training rows")
        values = np.stack([x[r[1], j] for r in subset])
        labels = np.array([r[4] for r in subset])
        mean, scale = values.mean(0), values.std(0, ddof=0)
        y_mean, y_scale = float(labels.mean()), float(labels.std(ddof=0))
        safe = np.where(scale > 0, scale, 1)
        z = (values - mean) / safe
        z[:, scale == 0] = 0
        design.append(z)
        response.append((labels - y_mean) / (y_scale if y_scale else 1))
        means.append(mean.tolist()); scales.append(scale.tolist())
        target_means.append(y_mean); target_scales.append(y_scale); counts.append(len(subset))
    z, y = np.concatenate(design), np.concatenate(response)
    z = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(z.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(z.T @ z / len(z) + penalty, z.T @ y / len(z))
    return dict(alpha=alpha, beta=beta.tolist(), means=means, scales=scales,
                target_means=target_means, target_scales=target_scales,
                samples=len(rows), samples_per_asset=counts, fit_start=start, fit_end=end,
                last_label_available=max(r[3] for r in rows),
                last_decision=max(r[2] for r in rows), refitted_after_selection=False)


def predict_ridge(state, values):
    scales = np.array(state["scales"])
    z = (values - state["means"]) / np.where(scales > 0, scales, 1)
    z[scales == 0] = 0
    beta = np.array(state["beta"])
    return (beta[0] + z @ beta[1:]) * state["target_scales"] + state["target_means"]


def retained_models(original, manifest):
    from ..trend_repair.adaptive_weight_model import AdaptiveWeightPolicy, fit

    result, states = {}, {}
    for version in manifest["versions"]:
        model, state = fit(original, AdaptiveWeightPolicy(version["model"]), None, None)
        if "target_weight" in model:
            model["target_weight"] *= version["scale"]
        result[version["policy"]] = model
        states[version["policy"]] = state
    return result, states
