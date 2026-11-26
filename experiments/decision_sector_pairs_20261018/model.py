"""Extend independent means and cost-sensitive gates; no pair hindsight filter."""
import numpy as np
from experiments.adaptive_residual_pairs_20261017.model import frozen_predictions
from experiments.dense_history_pairs_20261017.model import design,labels,map_state
from experiments.rolling_feature_pairs_20261017.model import rolling as rolling_mean
from experiments.decision_gate_pairs_20261017.model import rolling as rolling_gate,map_gate


def prepare(reference,symbols,pairs,spec,models):
    t=reference['market_times']
    f=design(t,reference['close'],reference['volume'],reference['trades'],symbols,pairs)
    first={k:v[:,:2] if k in ('x','beta','vol','fresh') else v[:2] if k=='pairs' else v for k,v in f.items()}
    raw=np.where(np.isfinite(f['x']).all(-1),0.,np.nan)
    raw[:,:2]=frozen_predictions(first,models,pairs[:2],24)
    target,maturity=labels(t,reference['close'],f,24)
    terminal=np.zeros_like(f['fresh']);terminal[:-24]=f['fresh'][24:]
    mean,mean_stamps,mean_records=rolling_mean(t,f,target,maturity,raw,terminal,
                                              {**spec,'candidates':['direct'],'diagnostic_variants':[]})
    action,score,available,stamps,records=rolling_gate(t,f,target,maturity,raw,terminal,spec)
    arrays=dict(times=t,target=target,maturity=maturity,unit=f['vol']*np.sqrt(24),prediction_direct=mean['direct'],
                economic_action=action,economic_available=available,decision_score=score,fit_stamps=stamps,
                mean_fit_stamps=mean_stamps,fresh=f['fresh'],terminal_fresh=terminal)
    return f,arrays,mean_records,records


def states(times,reference,features,arrays):
    raw=map_state(times,reference,features,arrays['prediction_direct'],features['pairs'],source_hours=1,lifetime_hours=1)
    gate=map_gate(times,arrays['times'],arrays['economic_action'],arrays['economic_available'],raw)
    return raw,gate,restrict_pairs(gate,3)


def restrict_pairs(state,count):
    if not 0<count<=len(state['pairs']):raise ValueError('Invalid fixed universe size')
    result={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.items()}
    result['directions'][:,count:]=0
    result['economic_available'][:,count:]=False
    result['economic_action'][:,count:]=0
    return result


def assert_original_pair_fields(new,old):
    columns={'signal_times','directions','beta','vol','mu','economic_action','economic_available'}
    for key,value in old.items():
        probe=new[key][:,:2] if key in columns else new[key][:2] if key=='pairs' else new[key]
        np.testing.assert_array_equal(probe,value,err_msg=key)
