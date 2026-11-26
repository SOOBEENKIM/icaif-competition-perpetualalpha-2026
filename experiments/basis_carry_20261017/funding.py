"""Fixed pre-cutoff funding models. Results are carry-only, not trade P&L."""
import json
import numpy as np
from .collect import HERE,OUT,save,sha,ts

HOUR=3600000

def fit_ar1(values):
    x=np.array(values[:-1]);y=np.array(values[1:]);v=float(np.var(x))
    phi=float(np.clip(np.mean((x-x.mean())*(y-y.mean()))/v,0,.99)) if v>1e-20 else 0.
    intercept=float(y.mean()-phi*x.mean())
    return dict(phi=phi,intercept=intercept,rows=len(x))

def predict(model,current,horizon,kind):
    if kind=='zero':return 0.
    if kind=='persistence':return horizon*current
    total=0.;v=current
    for _ in range(horizon):v=model['intercept']+model['phi']*v;total+=v
    return total

def main():
    dest=OUT/'funding';dest.mkdir(exist_ok=True)
    if (dest/'results.json').exists():raise FileExistsError('Funding study already frozen')
    cfg=json.loads((HERE/'protocol.json').read_text());fc=cfg['funding_study'];cut=ts(fc['fit_end']);split=ts(fc['validation_end'])
    save(dest/'frozen.json',dict(source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),
          collection_manifest_sha256=sha(OUT/'raw/manifest.json'),refit_after_outcomes=False))
    observations=[];fits=[];summaries=[]
    for symbol in cfg['symbols']:
        rows=json.loads((OUT/'raw'/f'funding_{symbol}.json').read_text())['rows']
        raw_times=np.array([r['time'] for r in rows],dtype=np.int64)
        # Subsecond settlement reporting jitter is recorded, not moved backwards
        # for signal availability. Hour buckets are used only to check coverage.
        assert np.all(raw_times%HOUR<1000)
        times=raw_times//HOUR*HOUR;rates=np.array([float(r['fundingRate']) for r in rows])
        assert np.all(np.diff(times)==HOUR)
        training=rates[raw_times<cut];model=fit_ar1(training);fits.append(dict(symbol=symbol,last_fit_time=int(raw_times[raw_times<cut][-1]),**model))
        for horizon in fc['horizons_hours']:
            for phase,lo,hi in [('validation',cut,split),('later',split,int(raw_times[-1])+1)]:
                indices=[i for i in range(len(rates)-horizon) if raw_times[i]>=lo and raw_times[i+horizon]<hi and times[i]%(horizon*HOUR)==0]
                for kind in fc['models']:
                    actual=[];predicted=[];carry=[]
                    for i in indices:
                        estimate=predict(model,float(rates[i]),horizon,kind);realized=float(rates[i+1:i+horizon+1].sum())
                        # Positive rate pays shorts. Sign is selected before the
                        # future settles; a zero forecast is a cash decision.
                        side=-np.sign(estimate);income=-side*realized
                        item=dict(symbol=symbol,horizon=horizon,phase=phase,model=kind,decision_time=int(raw_times[i]),
                                  last_outcome_time=int(raw_times[i+horizon]),prediction_bps=estimate*10000,
                                  actual_sum_bps=realized*10000,side=float(side),carry_only_bps=float(income*10000))
                        observations.append(item);actual.append(realized);predicted.append(estimate);carry.append(income)
                    if not actual:continue
                    ac=np.array(actual);pr=np.array(predicted);ca=np.array(carry)
                    active=abs(pr)>1e-15
                    summaries.append(dict(symbol=symbol,horizon=horizon,phase=phase,model=kind,blocks=len(ac),
                       mae_bps=float(np.mean(abs(pr-ac))*10000),mse_bps_squared=float(np.mean((pr-ac)**2)*1e8),
                       sign_accuracy=float(np.mean(np.sign(pr[active])==np.sign(ac[active]))) if active.any() else None,
                       average_carry_only_bps=float(ca.mean()*10000),
                       blocks_exceeding_18bp_fee_only=int((ca>.0018).sum()),
                       not_price_or_spread_adjusted=True))
    save(dest/'results.json',dict(fits=fits,summaries=summaries,observations=observations,
          independent=False,orders_sent=0,units='Sum of future settled funding rates, fixed-notional approximation, not strategy profit'))
    print('funding results',len(summaries),'cells',len(observations),'model/block observations',flush=True)

if __name__=='__main__':main()
