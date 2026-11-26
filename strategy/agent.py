"""One entry point; preserve the two frozen implementations without refitting.

Only computes target positions/order intents. There is no exchange client here.
"""
import json
from pathlib import Path

from replay_lab.reversion_blend.agent import ReversionBlendAgent
from replay_lab.tail_calibration.agent import TailCalibratedAgent

ROOT = Path(__file__).resolve().parents[1]
TAIL = ROOT / "replay_lab/output/tail_calibration/20261014-v1"
ORIGINAL = ROOT / "replay_lab/output/reversion_blend/20261014-v1"
RUNTIME = TAIL / "runtime"


def read(path):
    return json.loads(path.read_text())


def profiles():
    config = read(Path(__file__).with_name("profiles.json"))
    tail, parent = read(TAIL / "selected.json"), read(ORIGINAL / "selected.json")
    assert config["default_profile"] == "default" and not config["live_orders_enabled"]
    assert config["common_policy"] == parent["policy"] == tail["alpha_selection"]["policy"]
    assert config["default"]["relative_to_original"] == tail["incremental_scale"]
    assert config["default"]["total_scale"] == tail["total_scale"]
    assert config["original"]["total_scale"] == parent["risk_scale"]
    assert config["original"]["relative_to_original"] == 1.0
    return config


class SelectedAgent:
    """Tag restart state so profiles cannot accidentally share a controller."""

    def __init__(self, profile="default", saved=None):
        config = profiles()
        if profile not in ("default", "original"):
            raise ValueError("profile must be default or original")
        if saved is not None and (saved.get("profile") != profile or saved.get("schema") != 1):
            raise ValueError("Restart state belongs to a different profile/schema")
        self.profile = profile
        selection = read((TAIL if profile == "default" else ORIGINAL) / "selected.json")
        cls = TailCalibratedAgent if profile == "default" else ReversionBlendAgent
        self.agent = cls(read(TAIL / "baseline.json"), RUNTIME / "pretrained",
                         read(TAIL / "regime_fit.json"), read(TAIL / "allocation_fit.json"),
                         selection, saved["agent"] if saved is not None else None)
        self.relative_scale = config[profile]["relative_to_original"]

    def prepare_history(self, market, original, *, funding_source):
        return self.agent.prepare_history(market, original, funding_source=funding_source)

    @property
    def weights(self):
        return self.agent.weights

    def decide(self, time, qty, equity, valid_input=True):
        return self.agent.decide(time, qty, equity, valid_input)

    def snapshot(self):
        return {"schema": 1, "profile": self.profile, "agent": self.agent.snapshot()}


def load_agent(profile="default", saved=None):
    return SelectedAgent(profile, saved)
