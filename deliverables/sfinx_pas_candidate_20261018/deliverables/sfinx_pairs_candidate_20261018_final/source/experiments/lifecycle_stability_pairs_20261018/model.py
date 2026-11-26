"""Causal deletion-of-overlapping-weeks stability, without coverage claims."""
import numpy as np
from experiments.lifecycle_value_pairs_20261018.model import eligible,fit,predict


def intervals(rows,spec):
    width=spec['block_days']*86400;anchor=spec['block_anchor_unix']
    begin=min(r['source'] for r in rows);end=max(r['outcome_time'] for r in rows)
    first=(begin-anchor)//width*width+anchor
    return [(left,left+width) for left in range(first,end+1,width)
            if any(overlaps(r,left,left+width) for r in rows)]


def overlaps(row,left,right):
    # Conservatively purge a closing observation at the week's left boundary.
    return row['source']<right and row['outcome_time']>=left


def bundle(rows,full,learning_spec,spec):
    now=full['fit_time'];past=eligible(rows,now,learning_spec)
    assert [r['sample_id'] for r in past]==full['training_ids']
    assert max(r['outcome_time'] for r in past)<now
    blocks=intervals(past,spec);fits=[]
    for left,right in blocks:
        remaining=[r for r in past if not overlaps(r,left,right)]
        model=fit(remaining,now,learning_spec)
        fits.append(dict(left=left,right=right,deleted=len(past)-len(remaining),model=model))
    complete=len(blocks)>=spec['minimum_blocks'] and all(b['model'] is not None for b in fits)
    return dict(fit_time=now,full=full,blocks=fits,complete=complete,
                minimum_blocks=spec['minimum_blocks'],last_outcome=max(r['outcome_time'] for r in past))


def assess(value_bundle,x,now):
    if value_bundle is None:return dict(allowed=True,reason='no_value_model',full=None,lower=None,upper=None,blocks=0)
    if value_bundle['fit_time']>now or value_bundle['last_outcome']>=value_bundle['fit_time']:
        raise ValueError('Strictly mature fit required')
    full=predict(value_bundle['full'],x)
    if not value_bundle['complete']:
        return dict(allowed=True,reason='incomplete_block_models',full=full,lower=None,upper=None,
                    blocks=len(value_bundle['blocks']))
    values=np.asarray([full,*[predict(b['model'],x) for b in value_bundle['blocks']]])
    if not np.isfinite(values).all():raise ValueError('Nonfinite stability prediction')
    upper=float(values.max());lower=float(values.min())
    return dict(allowed=bool(upper>=0),reason='stable_negative_value' if upper<0 else 'negative_value_not_stable',
                full=full,lower=lower,upper=upper,blocks=len(value_bundle['blocks']))
