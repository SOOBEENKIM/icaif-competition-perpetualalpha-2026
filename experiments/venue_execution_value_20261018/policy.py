"""Fixed holding policy and bounded completion for venue cash outcomes."""
import numpy as np
from experiments.paid_rehedge_pairs_20261017.policy import held_state as original_held
from experiments.conditional_completion_pairs_20261017.completion import completion_plan as original_completion,after_completion
from replay_lab.common_comparison.engine import rounded,risk_breaches
from replay_lab.pullback.broker import execution_price


def held_state(pair,i,d,de,cfg,*,ledger=None):
    if not pair.get('venue_cash'):return original_held(pair,i,d,de,cfg,ledger=ledger)
    now=int(d['times'][i]);step=int(d['times'][1]-d['times'][0])
    maturity=pair['first_fill']+pair['horizon']*3600
    reason='fixed_horizon' if now>=maturity-step else None
    bound=cfg['dense_policy']['stop_sigma']*pair['entry_vol']*np.sqrt(pair['horizon']);value=None
    if 'risk_entry_prices' in pair:
        q=np.asarray(pair['risk_quantities']);entry=np.asarray(pair['risk_entry_prices'])
        value=float(q@(d['par_close'][i,pair['ids']]-entry)/(abs(q)@entry))
        if value<=-bound and reason is None:reason='pair_return_stop'
    return dict(reason=reason,maturity=maturity,age_hours=(now-pair['first_fill'])/3600,
                pair_return=value,return_stop_bound=bound)


def completion_plan(pair,ledger,i,d,de,cfg,p,sc,*,risk_ledger=None):
    if not pair.get('venue_cash'):return original_completion(pair,ledger,i,d,de,cfg,p,sc,risk_ledger=risk_ledger)
    now=int(d['times'][i]);ids=pair['ids'];owned=ledger.qty[ids].copy();want=np.asarray(pair['target'])
    if now>=pair['completion_deadline'] or np.count_nonzero(owned)!=1:return None
    have=int(np.flatnonzero(owned)[0]);missing=1-have;ratio=owned[have]/want[have]
    if not 0<ratio<=1+1e-8:return None
    target=want*ratio;target[have]=owned[have]
    target[missing]=rounded(target[missing],p['execution']['decimals'][ids[missing]])
    ratios=target/want
    if min(ratios)<=0 or abs(ratios[0]-ratios[1])/max(ratios)>cfg['pair_balance_tolerance']:return None
    marks=d['par_close'][i];j=ids[missing];q=target[missing];side=np.sign(q)
    preview=execution_price(marks[j],side,sc['impact_bps'],p['execution']['decimals'][j])
    limit=pair['entry_limits'][str(j)]
    if side*(preview-limit)>1e-10 or abs(q)*preview<p['execution']['minimum_order_value']:return None
    account=ledger if risk_ledger is None else risk_ledger
    all_qty=account.qty.copy();all_qty[ids]+=target-owned
    equity=account.equity(marks)-q*(preview-marks[j])-abs(q)*preview*sc['fee_bps']/10000
    if equity<=0 or risk_breaches(all_qty,equity,marks,d['sigma'][i],p['risk']):return None
    return dict(target=target.tolist(),entry_limits={str(j):limit},missing_asset=j,
                preview_price=float(preview),deadline=pair['completion_deadline'],
                policy='one_retry_original_limit_no_expansion',sunk_costs_charged_again=0)
