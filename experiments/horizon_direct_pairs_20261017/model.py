"""Same daily per-pair price model with an explicitly matched target horizon."""
import copy
from experiments.sector_pairs_20261017.model import infer as direct_infer


def infer(times, reference, symbols, pairs, direct_spec, horizon):
    if horizon not in (6,24,72):
        raise ValueError('Only pre-specified target horizons are allowed')
    spec=copy.deepcopy(direct_spec);spec['horizon']=horizon
    state,records,arrays=direct_infer(times,reference,symbols,pairs,spec)
    for record in records:record['horizon']=horizon
    return state,records,arrays
