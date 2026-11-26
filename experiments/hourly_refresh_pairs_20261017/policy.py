"""Hourly inference uses completed bars; trading logic remains frozen."""
from experiments.paid_rehedge_pairs_20261017.portfolio import replay as paid_replay
from experiments.forecast_validity_pairs_20261017.policy import replay as prior_replay


def replay(d, de, pred, cfg, protocol, arm, scenario, start, end, **kwargs):
    if '_hourlyhedge_' not in arm:
        return prior_replay(d, de, pred, cfg, protocol, arm, scenario, start, end, **kwargs)
    run=paid_replay(d, de, pred, cfg, protocol, arm.replace('_hourlyhedge_','_renewhedge_'), scenario, start, end, **kwargs)
    run['arm']=arm
    return run
