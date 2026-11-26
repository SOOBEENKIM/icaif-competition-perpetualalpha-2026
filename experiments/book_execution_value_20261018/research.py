"""Chronological source-time execution validation; never a full-horizon ROI."""
import json,time
from pathlib import Path
import numpy as np
from .dataset import ROOT,HERE,OUT,save,sha,verify,protected
from .model import fit,predict,weights


def evaluate(model,rows,tapes):
    selected=[r for r in rows if r['tape'] in tapes]
    w=weights(selected);w/=w.sum();learned=[];constant=[];truth=[];details=[]
    for row in selected:
        a=predict(model,row['features'],row['source']);b=predict(model,row['features'],row['source'],True)
        learned.append(a);constant.append(b);truth.append(np.eye(4)[row['outcome']])
        details.append(dict(sample_id=row['sample_id'],source=row['source'],outcome=row['outcome'],
            observed_held_fraction=row['held_fraction'],observed_cost_return=row['entry_cost_return'],learned=a,constant=b))
    y=np.asarray(truth);a=np.asarray([r['probabilities'] for r in learned]);b=np.asarray([r['probabilities'] for r in constant])
    known=np.array([r['entry_cost_return'] is not None for r in selected]);balanced=np.array([r['outcome']==2 for r in selected])
    def mse(values,actual,mask):
        return float(np.sum(w[mask]*(np.asarray(values)[mask]-np.asarray(actual)[mask])**2)/w[mask].sum()) if np.any(mask) else None
    costs=np.array([r['entry_cost_return']*10000 if r['entry_cost_return'] is not None else 0 for r in selected])
    fraction=np.array([r['held_fraction'] for r in selected])
    metrics=dict(rows=len(selected),unique_source_times=len({r['source'] for r in selected}),
        brier=float(w@np.sum((a-y)**2,axis=1)),constant_brier=float(w@np.sum((b-y)**2,axis=1)),
        balanced_brier=float(w@(a[:,2]-y[:,2])**2),constant_balanced_brier=float(w@(b[:,2]-y[:,2])**2),
        cost_rmse_bps=float(np.sqrt(mse([r['extra_cost_bps_if_reconciled'] for r in learned],costs,known))),
        constant_cost_rmse_bps=float(np.sqrt(mse([r['extra_cost_bps_if_reconciled'] for r in constant],costs,known))),
        held_fraction_rmse=float(np.sqrt(mse([r['held_fraction_if_balanced'] for r in learned],fraction,balanced))),
        constant_held_fraction_rmse=float(np.sqrt(mse([r['held_fraction_if_balanced'] for r in constant],fraction,balanced))),
        unresolved_cash_excluded=int(np.sum(~known)),cash_metric_conditions_on_reconciliation=True)
    return dict(metrics=metrics,predictions=details)


def main():
    spec=json.loads((HERE/'protocol.json').read_text());data=json.loads((OUT/'labels.json').read_text());rows=data['rows']
    test=(HERE/'model_test_results.txt').read_text();assert 'Ran 5 tests' in test and test.rstrip().endswith('OK')
    sources=json.loads((OUT/'frozen.json').read_text())['sources'].copy()
    files=[Path(__file__),HERE/'model.py',HERE/'test_model.py',HERE/'model_test_results.txt',OUT/'labels.json',*sorted((OUT/'episodes').glob('*.gz'))]
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'evaluation_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    start=min(r['source'] for r in rows if r['tape'] in spec['validation_tapes'])
    initial=fit(rows,spec['training_tapes'],start,spec);assert initial is not None
    save(OUT/'initial_model.json',initial);print('INITIAL FIT',initial['rows'],initial['unique_source_times'],initial['class_counts'],flush=True)
    validation=evaluate(initial,rows,spec['validation_tapes']);save(OUT/'validation.json',validation)
    m=validation['metrics'];decision=dict(execution_model_selected=m['brier']<m['constant_brier'] and m['cost_rmse_bps']<m['constant_cost_rmse_bps'],
        metrics=m,criterion='Both weighted outcome Brier and paid-entry-cost RMSE improve over past frequency/mean. This is execution-only evidence, not PnL/Q promotion.',
        frozen_before_later=True,historical_independence=False)
    save(OUT/'selection.json',decision);selection_hash=sha(OUT/'selection.json');print('VALIDATION',decision,flush=True)
    later_start=min(r['source'] for r in rows if r['tape'] in spec['later_tapes'])
    final=fit(rows,spec['training_tapes']+spec['validation_tapes'],later_start,spec);assert final is not None
    save(OUT/'later_model.json',final);print('LATER FIT',final['rows'],final['unique_source_times'],final['class_counts'],flush=True)
    later=evaluate(final,rows,spec['later_tapes']);save(OUT/'later.json',later)
    assert sha(OUT/'selection.json')==selection_hash;verify(sources);verify(prospective)
    save(OUT/'results.json',dict(validation=validation['metrics'],later=later['metrics'],selection=decision,
        first_fit=dict(rows=initial['rows'],unique_sources=initial['unique_source_times']),later_fit=dict(rows=final['rows'],unique_sources=final['unique_source_times']),
        tests_passed=11,model_fits=6,labels_are_execution_not_full_trade_returns=True,full_return=None,financial_score=None,
        orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,protected=protected()))
    save(HERE/'work_state.json',dict(status='progress_received_book_execution_models_compared',
        next_action='Apply frozen execution scoring to unchanged retained forecast plans and source-only size alternatives, then report actual scope.',
        goal_complete=False,no_progress_streak=0,orders_sent=0,live_strategy_changed=False))
    print('DONE book model',later['metrics'],flush=True)


if __name__=='__main__':main()
