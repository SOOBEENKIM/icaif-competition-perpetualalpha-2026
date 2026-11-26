"""Load an independent copy of the frozen replay module with a controller hook.

The selected engine's source and its imported live classes are not modified.
"""
import functools
import importlib.util
from competition_live.profile import ROOT
from .controller import RiskPreservingController


def reference_replay(times,weights,protected,enabled):
    name='experiments.decision_gate_pairs_20261017._isolated_repair_replay'
    path=ROOT/'experiments/decision_gate_pairs_20261017/portfolio.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if enabled:
        module.Controller=functools.partial(RiskPreservingController,times,weights,protected)
    return module.replay


def pas_replay(times,weights,protected,enabled):
    spec=importlib.util.spec_from_file_location('strategy_recheck._isolated_pas_replay',
                                              ROOT/'strategy_recheck/pas.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if enabled:
        module.Controller=functools.partial(RiskPreservingController,times,weights,protected)
    return module.simulate
