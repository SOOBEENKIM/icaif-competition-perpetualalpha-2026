"""Offline preservation check against saved targets, without new market access."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from replay_lab.common_comparison.models import Market
from .agent import ROOT, RUNTIME, TAIL, ORIGINAL, load_agent, profiles, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5006)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = profiles()
    start = time.monotonic()
    z = dict(np.load(RUNTIME / "frozen_inputs.npz", allow_pickle=False))
    if not 1441 <= args.rows <= len(z["market_times"]):
        raise ValueError("Use 1441 through the number of frozen observations")
    n = args.rows
    bundle = read(TAIL / "baseline.json")
    symbols = tuple(bundle["signal_bundle"]["v7"]["control_bundle"]["symbols"])
    m = Market(z["market_times"][:n], z["close"][:n], z["volume"][:n], z["trades"][:n],
               z["funding"][:n], z["observed"][:n], symbols)
    original = {k: v[:n] for k, v in z.items() if k not in ("market_times", "funding", "observed")}
    expected = {
        "default": np.load(TAIL / "final_targets.npz")["tail_calibrated"][:n],
        "original": np.load(ORIGINAL / "final_targets.npz")["crypto_d7_aligned_add50"][:n],
    }
    results = []
    for profile in ("default", "original"):
        agent = load_agent(profile)
        agent.prepare_history(m, original, funding_source="mainnet_reference_settled")
        np.testing.assert_array_equal(agent.weights, expected[profile])
        assert agent.decide(read(TAIL / "selected.json")["test_start"] - 3600, np.zeros(10), 1e6) is None
        state = agent.snapshot()
        resumed = load_agent(profile, saved=state)
        assert resumed.snapshot() == state
        try:
            load_agent("original" if profile == "default" else "default", saved=state)
        except ValueError:
            pass
        else:
            raise AssertionError("Cross-profile restart must fail")
        results.append({"profile": profile, "rows": n, "exact_targets": True,
                        "restart_state_preserved": True, "wrong_profile_restart_rejected": True})
        print(json.dumps(results[-1]), flush=True)
    np.testing.assert_array_equal(expected["default"], expected["original"] * config["default"]["relative_to_original"])
    sources = sorted({str(Path(module.__file__).resolve().relative_to(ROOT))
                      for module in list(sys.modules.values())
                      if getattr(module, "__file__", None) and Path(module.__file__).resolve().is_relative_to(ROOT)
                      and Path(module.__file__).is_file()
                      and Path(module.__file__).resolve().relative_to(ROOT).parts[0] in {"strategy", "replay_lab", "bot", "scripts"}})
    result = {"profiles": results, "same_signal_fixed_scale_exact": True,
              "loaded_project_sources": sources, "elapsed_seconds": time.monotonic() - start,
              "new_market_data": False, "live_orders_sent": 0, "new_performance_claim": False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "loaded_project_sources"}), flush=True)


if __name__ == "__main__":
    main()
