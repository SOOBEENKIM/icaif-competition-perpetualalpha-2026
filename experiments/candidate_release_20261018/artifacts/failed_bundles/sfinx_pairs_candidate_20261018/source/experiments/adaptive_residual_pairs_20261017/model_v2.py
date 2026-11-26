"""Preserve frozen baseline multiplication order exactly."""
import numpy as np
from experiments.dense_history_pairs_20261017.model import labels
from .model import frozen_predictions,fit_correction,adapt


def build(reference,d,models,pairs,h,spec):
    times=reference['market_times'];target,maturity=labels(times,reference['close'],d,h)
    raw=frozen_predictions(d,models,pairs,h);unit=d['vol']*np.sqrt(h)
    corrected,records=adapt(times,target,maturity,raw,d['fresh'],unit,h,spec)
    # The original control evaluates (raw * vol) * sqrt(h). Reassociation to
    # raw * (vol * sqrt(h)) differs by roundoff, so retain its exact order.
    return dict(ridge=raw*d['vol']*np.sqrt(h),zero=np.where(np.isfinite(raw),0.,np.nan),**corrected),records
