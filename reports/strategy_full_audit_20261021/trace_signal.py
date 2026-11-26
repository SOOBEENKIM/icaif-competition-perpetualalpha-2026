"""Replay the published baseline's composition using its saved model outputs.

This is target reconstruction, not a performance backtest or new model fit.
Run in a separate Python process to preserve the producer's root import paths.
"""
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from replay_lab.common_comparison.models import Market, covariance
from replay_lab.portfolio_development.candidate_agent import FundingTrendAgent
from replay_lab.competition_rebuild.regimes import paths
from replay_lab.competition_rebuild.persistence import episode_overlay
from replay_lab.competition_rebuild.causal_risk import normalize_risk
from replay_lab.v10_revision.core import sizing
from replay_lab.reversion_blend.model import features, signal
from replay_lab.portfolio_rebuild_v10.lifecycle_allocation import targets
from replay_lab.alpha_rebuild.common import limits


def main():
    config = json.loads((ROOT / "competition_live/config.json").read_text())
    pointer = json.loads((OUT / "raw/reference_pointer.json").read_text())
    run = ROOT / pointer["run"]
    with np.load(run / "inputs.npz") as archive:
        data = {k: archive[k].copy() for k in archive.files}
    with np.load(run / "targets.npz") as archive:
        saved = {k: archive[k].copy() for k in archive.files}
    tail = ROOT / "replay_lab/output/tail_calibration/20261014-v1"
    bundle = json.loads((tail / "baseline.json").read_text())
    selected = json.loads((tail / "selected.json").read_text())
    market = Market(data["market_times"], data["close"], data["volume"], data["trades"],
                    data["funding"], data["observed"], tuple(config["trading"]["symbols"]))
    original = {k: v for k, v in data.items() if k not in ("market_times", "funding", "observed")}
    agent = FundingTrendAgent(bundle["signal_bundle"]["v7"])
    agent.prepare_history(market, original, funding_source="mainnet_reference_settled")
    base = agent.weights
    sigma = np.full((len(market.times), 10, 10), np.nan)
    for i in range(168, len(market.times)):
        sigma[i] = covariance(market, i, bundle["protocol"]["risk"])[1]
    _, jm = paths(market, saved["frozen_regime_probability"], base)
    chronos = episode_overlay(market, base, saved["chronos_forecast"], "sticky")
    filtered = (jm + chronos) / 2
    normalized = normalize_risk(base, filtered, sigma, market.times)
    sized = sizing(normalized, sigma, market.times)
    sized = np.clip(sized, -.135, .135)
    divisor = np.maximum(1., np.maximum(abs(sized).sum(1) / .54, abs(sized.sum(1)) / .36))
    sized /= divisor[:, None]
    f = features(market, 7)
    reversion = signal(market, f, 7, "crypto", "aligned")
    raw = np.nan_to_num(reversion / np.maximum(f["vol"], .003), nan=0)
    reversal_weight = .9 * targets(raw, sigma, market.times, days=3, risk=.001)
    combined = limits(sized + .5 * reversal_weight)
    final = combined * selected["alpha_selection"]["risk_scale"] * selected["incremental_scale"]
    np.testing.assert_allclose(final, saved["default"], rtol=1e-12, atol=1e-15)
    arrays = {
        "trend_carry": base, "jm_filtered": jm, "chronos_filtered": chronos,
        "mean_filtered": filtered, "risk_normalized": normalized,
        "risk_sized": sized, "reversion_added": .5 * reversal_weight,
        "combined": combined, "final": final,
    }
    rows = []
    for i in np.flatnonzero((market.times >= config["start_unix"]) & (market.times % 21600 == 0)):
        row = {
            "time": int(market.times[i]),
            "kst": dt.datetime.fromtimestamp(int(market.times[i]), dt.timezone(dt.timedelta(hours=9))).isoformat(),
            "stages": {k: v[i].tolist() for k, v in arrays.items()},
            "stage_daily_risk_pct": {k: float(np.sqrt(24 * v[i] @ sigma[i] @ v[i]) * 100) for k, v in arrays.items()},
            "regime_bull_state_weight": saved["frozen_regime_probability"][i].tolist(),
            "chronos_log_prediction": saved["chronos_forecast"][i].tolist(),
        }
        rows.append(row)
    result = {
        "reconstructed_published_targets": True,
        "maximum_absolute_error": float(abs(final - saved["default"]).max()),
        "rows_checked": len(final), "chronos_inference_run": False, "orders_sent": 0,
        "source_module": sys.modules["replay_lab.competition_rebuild.causal_risk"].__file__,
        "rows": rows,
    }
    (OUT / "signal_trace.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
