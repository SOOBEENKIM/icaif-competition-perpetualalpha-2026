import copy
import numpy as np
from competition_live.runtime import fingerprint
from .venue_funding_packet import VenueFundingPacket


def fixture():
    symbols=['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY']
    source=100*3600;body=dict(frame=dict(source_time=source,symbols=symbols,
        anchor_weights=[.05]*10,protected=[True]+[False]*9,
        price_forecast_168h=[.02]*10,reference_funding_forecast_168h=[.002]*10),original_signal={'unchanged':1})
    p=dict(body=body,sha256=fingerprint(body))
    rows=[dict(time=i*3600000,coin='pas:'+s,rate=(j-5)*.0001) for i in range(70,101) for j,s in enumerate(symbols)]
    return source,p,rows


def test_future_funding_poison_cannot_change_packet():
    s,p,rows=fixture();a=VenueFundingPacket(rows,0)(p,s+180)
    bad=copy.deepcopy(rows)
    for r in bad:
        if r['time']/1000+60>s:r['rate']=9999
    assert a==VenueFundingPacket(bad,0)(p,s+180)


def test_handover_preserves_prefix_and_immutable_original():
    s,p,rows=fixture();before=copy.deepcopy(p);f=VenueFundingPacket(rows,s+1)
    assert f(p,s) is p
    out=f(p,s+2)
    assert p==before and out['body']['original_signal']==before['body']['original_signal']
    assert fingerprint(out['body'])==out['sha256']


def test_cost_gate_never_increases_anchor_or_removes_protection():
    s,p,rows=fixture();out=VenueFundingPacket(rows,0)(p,s+100)['body']['frame']
    assert all(abs(x)<=.05 for x in out['anchor_weights']) and out['protected'][0]
    assert len(out['anchor_weights'])==10


def test_replace_only_preserves_all_anchor_and_protection_values():
    s,p,rows=fixture();out=VenueFundingPacket(rows,0,filter_core=False)(p,s+100)['body']['frame']
    for name in ['anchor_weights','protected','price_forecast_168h']:assert out[name]==p['body']['frame'][name]
    assert np.array(out['reference_funding_forecast_168h']).shape==(10,)
