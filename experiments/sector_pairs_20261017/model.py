"""Extend only the pair universe; direct coefficients do not use pretraining."""
import copy
import numpy as np
from experiments.dense_history_pairs_20261017.model import design, labels, map_state
from experiments.rolling_feature_pairs_20261017.model import rolling


def infer(times, reference, symbols, pairs, direct_spec):
    features=design(reference['market_times'],reference['close'],reference['volume'],reference['trades'],symbols,pairs)
    spec=copy.deepcopy(direct_spec);spec['candidates']=['direct'];spec['diagnostic_variants']=[]
    h=spec['horizon'];target,maturity=labels(reference['market_times'],reference['close'],features,h)
    terminal=np.zeros_like(features['fresh']);terminal[:-h]=features['fresh'][h:]
    # The old function also fits a residual column. Its raw prediction is not
    # an input to the direct column; zeroing this unused column must preserve
    # the original pair forecasts and avoids inventing pretraining for equities.
    unused=np.where(np.isfinite(features['x']).all(-1),0.,np.nan)
    forecasts,stamps,records=rolling(reference['market_times'],features,target,maturity,unused,terminal,spec)
    state=map_state(times,reference,features,forecasts['direct'],features['pairs'],source_hours=1,lifetime_hours=1)
    return state,records,dict(times=reference['market_times'],target=target,maturity=maturity,
        predictions=forecasts['direct'],fit_stamps=stamps,unit=features['vol']*np.sqrt(h),
        fresh=features['fresh'],terminal_fresh=terminal)
