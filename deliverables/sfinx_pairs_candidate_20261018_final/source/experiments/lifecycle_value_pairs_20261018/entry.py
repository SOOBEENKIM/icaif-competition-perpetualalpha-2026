"""Isolated training opportunity or supplementary entry-only value filter."""
from experiments.decision_gate_pairs_20261017.entry import proposals as original
from .model import choose


def proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,*,risk_qty=None):
    opportunity=cfg.get('lifecycle_training_opportunity')
    if opportunity is not None and int(d['times'][i])!=opportunity['source']:return []
    choices=original(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,risk_qty=risk_qty)
    if opportunity is not None:
        return [c for c in choices if c['pair_index']==opportunity['pair_index']]
    kind=cfg.get('lifecycle_kind')
    if kind is None:return choices
    stamp=int(d['times'][i])//86400*86400
    model=cfg['lifecycle_models'].get(stamp)
    return choose(choices,d,de,i,p['initial'],model,kind)
