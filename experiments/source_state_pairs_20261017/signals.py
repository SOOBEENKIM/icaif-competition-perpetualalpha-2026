"""Keep reference confirmation separate from the current venue opportunity."""
import numpy as np
from experiments.adaptive_mean_pairs_20261017.signals import crossing


def reference_confirmation(d,de,reference,threshold):
    out={k:v.copy() if hasattr(v,'copy') else v for k,v in de.items()}
    source=de['forecast_times'];times=reference['market_times']
    ix=np.searchsorted(times,source)
    assert np.all(times[ix]==source) and np.all(source<=d['times'])
    lp=np.log(reference['close'][ix]);z=np.full_like(de['z'],np.nan)
    for k,(a,b) in enumerate(de['pairs']):z[:,k]=(lp[:,a]-de['beta'][:,k]*lp[:,b]-de['mean'][:,k])/de['sd'][:,k]
    si=np.searchsorted(d['times'],source);pi=np.searchsorted(d['times'],source-3600)
    valid=(source-3600>=d['times'][0])
    out['reference_z']=z
    for key,delayed in [('standard',False),('delayed',True)]:
        out[key]=np.zeros_like(de[key])
        for i in np.flatnonzero(valid):
            assert d['times'][si[i]]==source[i] and d['times'][pi[i]]==source[i]-3600
            for k in range(z.shape[1]):out[key][i,k]=crossing(z[pi[i],k],z[si[i],k],threshold,delayed)
    # The current par z, stop and cost logic are intentionally retained.
    return out
