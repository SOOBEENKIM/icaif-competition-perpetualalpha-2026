"""Apply existing risk decisions after every base-owner alpha component."""
import numpy as np
from replay_lab.competition_rebuild.regimes import paths
from replay_lab.competition_rebuild.persistence import episode_overlay
from replay_lab.alpha_rebuild.common import limits


def final_mix(market, sized_trend, half_reversion, scale, probability, forecast):
    before=limits(np.asarray(sized_trend)+np.asarray(half_reversion))*scale
    if not np.isfinite(before).all() or not np.isfinite(scale) or not 0<scale<=1:
        raise ValueError('Invalid final mix or fixed scale')
    _,jm=paths(market,probability,before)
    chronos=episode_overlay(market,before,forecast,'sticky')
    final=(jm+chronos)/2
    ratio=np.divide(final,before,out=np.ones_like(final),where=before!=0)
    if np.any((ratio<0)|(ratio>1)) or not np.isfinite(final).all():
        raise ValueError('Risk overlay may only reduce final exposure')
    protected=(ratio<1-1e-12)&(before!=0)
    return final,dict(pre_gate=before,jm=jm,chronos=chronos,ratio=ratio,protected=protected)
