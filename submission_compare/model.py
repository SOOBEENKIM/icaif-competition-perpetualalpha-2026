import numpy as np


def candidates(incumbent,symmetric,forecast,bull_probability,agreement):
    """Use already frozen predictions; never outcome prices or fitted cutoffs.

    The 0.4 bearish cutoff is inherited from the existing regime model.
    Changes apply only to BTC/ETH/HYPE; other directional sleeves are retained.
    """
    arrays=[np.asarray(x,float) for x in
            (incumbent,symmetric,forecast,bull_probability,agreement)]
    base,short,pred,prob,agree=arrays
    if any(x.shape!=base.shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Invalid aligned signals')
    if base.ndim!=2 or base.shape[1]!=10:raise ValueError('Expected ten assets')
    crypto=np.arange(10)<3
    confirmed=(short<0)&(pred<0)&(prob<.4)&crypto[None,:]
    a=np.where(confirmed,short,base)
    b=base.copy()
    b[(base*pred<0)&crypto[None,:]]=0.
    c=np.where(confirmed,short,agree)
    return dict(confirmed_short=a,forecast_filter=b,agreement_confirmed_short=c)
