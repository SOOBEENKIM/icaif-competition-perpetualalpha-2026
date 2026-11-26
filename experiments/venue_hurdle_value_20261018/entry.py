"""Only change the entry value model; source orders and paid exits stay fixed."""
import numpy as np
from experiments.venue_execution_value_20261018.entry import proposals as original,plan
from experiments.venue_execution_value_20261018.model import adjusted_net
from .features import features
from .model import predict


def proposals(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,*,risk_qty=None):
    if 'hurdle_model_kind' not in cfg:return original(i,d,de,pred,h,cfg,p,sc,qty,equity,reserved,directions,arm,risk_qty=risk_qty)
    now=int(d['times'][i]);model=cfg['hurdle_models'].get((h,now))
    if model is None:return []
    risk_qty=qty if risk_qty is None else risk_qty;result=[]
    for k,ids in enumerate(de['pairs']):
        if not directions[k] or np.any(qty[ids]) or np.any(abs(reserved[ids])>1e-12):continue
        for direction in (-1,1):
            proposal=plan(i,k,direction,h,d,de,cfg,p,sc,risk_qty,equity)
            if proposal is None:continue
            x,gross=features(proposal,d,de,i,p)
            value,components=predict(model,x,now,cfg['hurdle_model_kind'])
            net=adjusted_net(value,cfg['scenarios']['base'],sc)
            if net<=cfg['venue_cash']['entry_buffer_bps']/10000:continue
            result.append(dict(proposal,expected_net_cash=net*gross,expected_cash=net*gross,expected_net_bps=net*10000,
                hurdle_fit_time=model['fit_time'],execution_value_components=components))
    return sorted(result,key=lambda r:(-r['expected_cash'],r['pair']))
