"""Check an illustrative overlapping ETH order against real public PAS state.

This is an execution integration probe, NOT a candidate trading decision.
There is no signer and no order client. It never calls the submission method.
"""
import json,time
from decimal import Decimal
import numpy as np
from .book import JointBook
from .pas_gateway import PASGateway


def main():
    from competition_live.profile import ROOT,load_profile
    from competition_live.venue import PublicPAS
    from bot.execution import rounded_price
    from replay_lab.common_comparison.engine import risk_breaches
    p=load_profile();public=PublicPAS(p)
    folder=ROOT/'reports/joint_wallet_20261022'
    book=JointBook.restore(json.loads((folder/'book.json').read_text()))
    meta,ctx=public.market();j=book.symbols.index('ETH');coin='pas:ETH'
    levels=public.post(dict(type='l2Book',coin=coin))['levels']
    px=float(rounded_price(Decimal(levels[1][0]['px'])*Decimal('1.001'),meta['universe'][j]['szDecimals'],True))
    frame=json.loads((ROOT/'reports/strategy_replacement_review_20261022/candidate_frame.json').read_text())
    q=book.owners['ours'].qty.copy();target=float(q[j]+.01)
    plan=book.plan('ours','ETH',target,px,meta['universe'][j]['szDecimals'],
        max_notional=p['execution']['maximum_single_order_notional'],slot='unsigned_overlap_probe')
    marks=np.array([float(c['markPx']) for c in ctx]);q[j]+=plan['delta']
    assert not risk_breaches(q,book.owners['ours'].equity(marks),marks,np.array(frame['sigma']),p['trading']['risk'])
    gateway=PASGateway(p,public,client=None);gateway.preflight(plan,book)
    result=dict(at=time.time(),preflight_pass=True,probe_kind='illustrative_execution_probe_NOT_alpha_decision',
        illustrative_plan=plan,teammate_qty_unchanged=book.owners['teammate'].qty.tolist(),
        signer_loaded=False,orders_sent=0,live_changed=False,
        scope='Real PAS market identity, precision, OI, wallet inventory and order capacity preflight. No live fill or improved PAS PnL claimed.')
    (folder/'public_preflight.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
