"""Let the costed planner price the full opportunity, not price alpha alone."""
import numpy as np


def state(source,*,carry_only=False):
    out={k:np.array(v,copy=True) for k,v in source.items()}
    available=out['economic_available'] & np.isfinite(out['mu']) & np.isfinite(out['beta']) & (out['vol']>0)
    if 'relationship_allowed' in out:available &= out['relationship_allowed']
    if 'regime_allowed' in out:available &= out['regime_allowed']
    # Both signs will still have to pass the retained reservation-limit,
    # funding, physical-risk, depth and lot-size checks independently.
    out['directions']=available.astype(int)
    out.pop('economic_action',None);out.pop('economic_available',None)
    if carry_only:out['mu']=np.where(available,0.,np.nan)
    return out
