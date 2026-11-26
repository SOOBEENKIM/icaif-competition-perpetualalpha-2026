"""Feed validated new reference history into the unchanged selected agent.

Outputs are raw strategy targets, not venue orders. Current wallet positions,
par quotes, order sizing and execution are deliberately outside this data task.
"""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

from .refresh import FROZEN, FROZEN_SHA, HOUR, OUT, REFERENCES, ROOT, iso, load_inputs, save_json, sha


def kst(seconds):
    return datetime.fromtimestamp(seconds, ZoneInfo("Asia/Seoul")).isoformat()


def protected_hashes():
    manifest = ROOT / "experiments/kronos_20261015/preserved_before.json"
    expected = json.loads(manifest.read_text())
    changed = [name for name, digest in expected.items() if sha(ROOT / name) != digest]
    if changed:
        raise ValueError(f"Preserved strategy/model/data files changed: {changed}")
    return expected


def verify_inputs(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "data_validated_pending_inference":
        raise ValueError("Run must contain validated data pending inference")
    if manifest["source"] != "mainnet_reference_settled" or manifest["source_mappings"] != REFERENCES:
        raise ValueError("Wrong source or mapping")
    if sha(run / "inputs.npz") != manifest["inputs_sha256"] or sha(FROZEN) != FROZEN_SHA:
        raise ValueError("Input hash mismatch")
    for symbol, digest in manifest["receipts_sha256"].items():
        if sha(run / "receipts" / f"{symbol}.json") != digest:
            raise ValueError("Receipt hash mismatch")
    z = load_inputs(run / "inputs.npz")
    frozen = load_inputs(FROZEN)
    if set(z) != set(frozen):
        raise ValueError("Input schema changed")
    for key, old in frozen.items():
        np.testing.assert_array_equal(z[key][:len(old)], old)
    completed = int(z["market_times"][-1])
    if completed != int(time.time() // HOUR) * HOUR:
        raise ValueError("Inputs do not end at the latest completed hour; refresh first")
    return manifest, z, frozen


def infer(run):
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    from replay_lab.common_comparison.models import Market
    from strategy.agent import TAIL, load_agent, profiles

    run = Path(run).resolve()
    started = time.monotonic()
    manifest, z, frozen = verify_inputs(run)
    preserved = protected_hashes()
    symbols = tuple(REFERENCES)
    market = Market(z["market_times"], z["close"], z["volume"], z["trades"],
                    z["funding"], z["observed"], symbols)
    original = {k: v for k, v in z.items() if k not in ("market_times", "funding", "observed")}
    print(f"Running unchanged default agent on {len(market.times)} completed hourly rows", flush=True)
    agent = load_agent("default")
    agent.prepare_history(market, original, funding_source=manifest["source"])
    # Extending history must not revise any previously published target.
    with np.load(TAIL / "final_targets.npz", allow_pickle=False) as archive:
        expected = archive["tail_calibrated"]
    np.testing.assert_array_equal(agent.weights[:len(expected)], expected)
    if agent.weights.shape != market.close.shape or not np.isfinite(agent.weights).all():
        raise ValueError("Invalid target weights")
    if protected_hashes() != preserved:
        raise ValueError("Frozen artifacts changed during inference")
    completed = int(market.times[-1])
    scheduled = np.flatnonzero(market.times % (6 * HOUR) == 0)[-1]
    scheduled_time = int(market.times[scheduled])
    next_time = (completed // (6 * HOUR) + 1) * 6 * HOUR
    weights = agent.weights[-1]
    prediction = agent.agent.signal.forecast[-1]
    probability = agent.agent.frozen_probability[-1]
    assets = [{"symbol": s, "reference_coin": REFERENCES[s],
               "reference_close": float(market.close[-1, j]),
               "settled_hourly_reference_funding": float(market.funding[-1, j]),
               "raw_target_weight": float(weights[j]),
               "raw_target_percent_of_equity": float(weights[j] * 100),
               "last_scheduled_raw_target_weight": float(agent.weights[scheduled, j]),
               "direction": "long" if weights[j] > 0 else "short" if weights[j] < 0 else "flat",
               "chronos_median_log_difference_steps_25_minus_1": float(prediction[j]),
               "frozen_regime_bull_probability": float(probability[j])}
              for j, s in enumerate(symbols)]
    fresh = completed == int(time.time() // HOUR) * HOUR
    report = {"schema": 1, "created_at_utc": iso(), "profile": "default",
              "policy": "trend_plus_7day_reversion_tail_calibrated",
              "source": manifest["source"], "inputs_sha256": manifest["inputs_sha256"],
              "input_completed_bar_utc": iso(completed), "input_completed_bar_kst": kst(completed),
              "first_new_completed_bar_kst": kst(int(frozen["market_times"][-1]) + HOUR),
              "new_hours_since_frozen": len(market.times) - len(frozen["market_times"]),
              "total_hours": len(market.times), "all_assets_have_settled_funding": bool(market.observed.all()),
              "fresh_at_publication": fresh, "parameters_unchanged": True,
              "previous_5006_targets_exact": True, "protected_files_unchanged": len(preserved),
              "fixed_relative_scale": profiles()["default"]["relative_to_original"],
              "gross_raw_target_percent": float(abs(weights).sum() * 100),
              "net_raw_target_percent": float(weights.sum() * 100),
              "last_scheduled_signal_kst": kst(scheduled_time),
              "chronos_forecast_origin_kst": kst(scheduled_time),
              "next_normal_rebalance_kst": kst(next_time),
              "normal_rebalance_due_at_input_hour": completed % (6 * HOUR) == 0,
              "assets": assets, "orders_sent": 0, "wallet_state_loaded": False,
              "background_loop_started": False,
              "output_interpretation": "Raw reference-market strategy weights before account/venue execution constraints; not actual holdings or submitted orders.",
              "model_interpretation": "Frozen Chronos-2 pretrained checkpoint and saved selected regime/strategy parameters; no new parameter selection. Existing prepare_history recomputes historical auxiliary fits; the selected post-cutoff regime uses its saved frozen fit.",
              "elapsed_inference_seconds": time.monotonic() - started}
    np.savez_compressed(run / "targets.npz", times=market.times, default=agent.weights,
                        chronos_forecast=agent.agent.signal.forecast,
                        frozen_regime_probability=agent.agent.frozen_probability)
    save_json(run / "signals.json", report)
    with (run / "targets_latest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(assets[0]))
        writer.writeheader()
        writer.writerows(assets)
    save_json(run / "verification.json", {
        "old_input_prefix_exact": True, "old_target_prefix_exact": True,
        "missing_bars": 0, "missing_funding": 0, "frozen_parameters_unchanged": True,
        "protected_files_unchanged": len(preserved), "orders_sent": 0,
        "inference_sources_sha256": {str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob("*.py")},
        "targets_sha256": sha(run / "targets.npz"), "signals_sha256": sha(run / "signals.json")})
    manifest.update(status="ready" if fresh else "inference_completed_but_stale", inference_completed_at_utc=iso())
    save_json(run / "manifest.json", manifest)
    if fresh:
        save_json(OUT / "latest.json", {"schema": 1, "run": str(run.relative_to(ROOT)),
            "inputs": str((run / "inputs.npz").relative_to(ROOT)), "inputs_sha256": sha(run / "inputs.npz"),
            "signals": str((run / "signals.json").relative_to(ROOT)), "signals_sha256": sha(run / "signals.json"),
            "completed_bar_utc": iso(completed), "published_at_utc": iso(), "orders_sent": 0})
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not fresh:
        raise ValueError("Hour advanced during inference: outputs saved but not published as latest; refresh again")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    infer(parser.parse_args().run)
