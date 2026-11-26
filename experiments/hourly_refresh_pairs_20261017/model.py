"""Reuse frozen weights; only inference/calibration evaluation frequency changes."""
from experiments.adaptive_residual_pairs_20261017.model_v2 import build
from experiments.dense_history_pairs_20261017.model import map_state


def infer(times, reference, features, models, pairs, horizon, spec):
    if spec['decision_hours']!=1:
        raise ValueError('Hourly evaluation must be explicitly configured')
    forecasts,records=build(reference,features,models,pairs,horizon,spec)
    hourly={kind:map_state(times,reference,features,forecast,features['pairs'],source_hours=1,lifetime_hours=1)
            for kind,forecast in forecasts.items() if kind in ['ridge','bias','affine','recent']}
    six_hour={kind:map_state(times,reference,features,forecast,features['pairs'],source_hours=6,lifetime_hours=1)
              for kind,forecast in forecasts.items() if kind in ['ridge','bias','affine','recent']}
    return hourly,six_hour,records
