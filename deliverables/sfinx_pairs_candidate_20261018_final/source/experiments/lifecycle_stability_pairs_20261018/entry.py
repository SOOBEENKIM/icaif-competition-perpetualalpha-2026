"""Supplementary entry veto; keep original ordering, prices and quantities."""
from experiments.lifecycle_value_pairs_20261018.entry import proposals as parent
from experiments.decision_gate_pairs_20261017.entry import proposals as incumbent
from experiments.lifecycle_value_pairs_20261018.model import feature
from .model import assess


def proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,*,risk_qty=None):
    if 'stability_models' not in cfg:
        return parent(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,risk_qty=risk_qty)
    choices=incumbent(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,risk_qty=risk_qty)
    choices=[choice for choice in choices if choice['pair_index']<2]
    now=int(d['times'][i]);value_bundle=cfg['stability_models'].get(now//86400*86400);kept=[]
    for proposal in choices:
        x,unit,gross=feature(proposal,d,de,i,p['initial'])
        check=assess(value_bundle,x,now)
        if 'stability_checks' in cfg:
            cfg['stability_checks'].append(dict(time=now,pair=proposal['pair'],source=proposal['forecast_source'],
                direction=proposal['direction'],unit=unit,gross=gross,expected_net_cash=proposal['expected_net_cash'],
                model_time=value_bundle['fit_time'] if value_bundle else None,**check))
        if check['allowed']:kept.append(proposal)
    return kept
