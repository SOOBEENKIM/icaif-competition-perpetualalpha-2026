"""Forecast diagnostics and score audit, separate from executable PnL."""
import gzip,json
import numpy as np
from .study import ROOT,INPUT,save,read
from .scaled_study import SOURCE,OUT as SCALED,verify
from .scaled_coherence import project
from .pas_extension import OUT as PAS


def main():
    verify();raw=dict(np.load(INPUT));a=dict(np.load(SOURCE));b=dict(np.load(SCALED/'coherent_scaled_72.npz'))
    stats={k:dict(n=0,sources=0,absolute_error=0.,weighted_sse=0.,hits=0,selected=0,selected_net_bps_sum=0.) for k in ['adaptive','coherent_scaled']}
    max_identity_error=0.;removed=0.
    for i,t in enumerate(a['times']):
        if not 1785585600<=t<=1792065600-72*3600:continue
        j=int(np.searchsorted(raw['market_times'],t));h=j+72
        assert raw['market_times'][h]==t+72*3600
        ret=raw['close'][h]/raw['close'][j]-1
        beta=a['beta'][i];valid=a['economic_available'][i];unit=a['vol'][i]*np.sqrt(72)
        if not np.any(valid):continue
        y=np.array([(ret[x]-bt*ret[z])/(1+bt) for (x,z),bt in zip(a['pairs'],beta)])
        pred,info=project(a['mu'][i],beta,a['pairs'],valid,unit)
        old=float(np.sum(((a['mu'][i,valid]-y[valid])/unit[valid])**2))
        new=float(np.sum(((pred[valid]-y[valid])/unit[valid])**2))
        max_identity_error=max(max_identity_error,abs(old-new-info['removed_energy']));removed+=info['removed_energy']
        for name,state in [('adaptive',a),('coherent_scaled',b)]:
            mu=state['mu'][i];q=stats[name];q['n']+=int(valid.sum());q['sources']+=1
            q['absolute_error']+=float(abs(mu[valid]-y[valid]).sum())
            q['weighted_sse']+=float(np.sum(((mu[valid]-y[valid])/unit[valid])**2))
            q['hits']+=int((mu[valid]*y[valid]>0).sum())
            sel=valid&(abs(mu)>.0038);q['selected']+=int(sel.sum())
            q['selected_net_bps_sum']+=float((np.sign(mu[sel])*y[sel]*10000-38).sum())
    for q in stats.values():
        q['MAE_bps']=q.pop('absolute_error')/q['n']*10000
        q['source_risk_weighted_MSE']=q['weighted_sse']/q['n'];q['direction_accuracy']=q['hits']/q['n']
        q['selected_mean_price_return_minus_38bp']=q.pop('selected_net_bps_sum')/max(1,q['selected'])
    assert max_identity_error<1e-8
    save(ROOT/'reports/pair_value_learning_20261023/forecast_diagnostics.json',dict(metrics=stats,identity_max_abs_error=max_identity_error,removed_energy=removed,
         scope='Complete available cohort, source-fixed beta, arithmetic72h targets fully mature insideAug1-Oct15 reference window. Hourly overlapping labels, not independent trades. Cost-gate statistic is price-only minus38bp; no funding/fill or portfolio simulation.',
         independent_test=False))
    from competition_live.scoring import evaluate
    audit=[]
    for path in sorted((PAS/'runs').glob('*.json.gz')):
        r=json.loads(gzip.open(path,'rt').read());daily=[dict(time=x['time'],equity=x['equity']+50000) for x in r['snapshots'] if x['time']%86400==43200]
        last=r['snapshots'][-1];sc=evaluate(daily,latest=dict(time=last['time'],equity=last['equity']+50000),trading_days=len({f['time']//86400 for f in r['fills']}))
        audit.append(dict(run=path.stem,**sc))
    save(PAS/'score_audit.json',dict(source='https://perpetual-alpha26.github.io/leaderboard',checked_utc='2026-10-23',
         correction='Current official page distinguishes daily standings from intraday informational PnL. Originalpas84 rows contain a hypothetical intradayQ, not daily standingsQ. This audit reports completed-noonQ and separately intradayreturn; actual money PnL unchanged. Reference windows endatnoon, unchanged.',runs=audit))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
