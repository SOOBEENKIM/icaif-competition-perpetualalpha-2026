"""Reviewable coherent forecast adapter. No network, credentials or order sender."""
import numpy as np
from pair_model_research.candidate import snapshot as original_snapshot
from auto_pair_selection.selection import IDS
from .scaled_coherence import project


def snapshot(data,source,received_at):
    result=original_snapshot(data,source,received_at)
    raw=np.full(45,np.nan);beta=raw.copy();unit=raw.copy();valid=np.zeros(45,bool)
    for f in result['forecasts']:
        k=f['stable_pair_index'];raw[k]=f['mu'];beta[k]=f['source_beta'];unit[k]=f['source_vol']*np.sqrt(72);valid[k]=True
    corrected,audit=project(raw,beta,IDS,valid,unit)
    for f in result['forecasts']:
        k=f['stable_pair_index'];mu=float(corrected[k]);action=int(np.sign(mu)) if abs(mu)>.0038 else 0
        f.update(model='risk_scaled_coherent_ar72',raw_mu=f['mu'],mu=mu,economic_action=action,
                 terminal_prices=(np.asarray(f['source_prices'])*[1+mu,1-mu]).tolist(),
                 formula_family='SMA_AR1_then_source_risk_scaled_cross_pair_projection')
        record=result['pair_audit'][k]
        assert record['pair_index']==k
        record.update(economic_action=action,reason='forecast_only_execution_checks_pending' if action else 'proxy_cost_gate')
    result.update(projection=audit,candidate='coherent_scaled_72',
                  execution_contract='72h, all45, source-fixed hedge, original renewable paid lifecycle, reference-price anchors; costs/funding/venue gate required separately',
                  guarantee_scope='Complete-cohort source-risk-weighted squared forecast error only; no selected-trade or PnL guarantee')
    return result
