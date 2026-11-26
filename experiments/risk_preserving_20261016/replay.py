"""Identical hourly reference execution for the incumbent and candidate.

Reuses the existing ledger/executor, with controller injection as the only
strategy difference. Reports residuals instead of pretending targets filled.
"""
import copy,gzip,json
from pathlib import Path
import numpy as np
from replay_lab.common_comparison.models import Market,HOUR
from replay_lab.common_comparison.engine import Ledger,execute_intent,risk_values,risk_breaches
from replay_lab.ema_rebuild.engine import Controller
from replay_lab.macro_development.engine import simulate as legacy_simulate
from .audit import audit
from strategy.agent import TAIL
from .policy import RiskCapController
from .prepare import HERE,OUT,ROOT,sha,save,json_default


def simulate(m,weights,retention,sigma,protocol,first,last,scenario_name,candidate=False):
    ledger=Ledger(protocol['initial'],m.symbols)
    control=RiskCapController() if candidate else Controller()
    scenario=protocol['scenarios'][scenario_name]
    pending=None;peak=ledger.initial;mdd=0.;halted=False;streak=max_streak=0
    snapshots=[];decisions=[];orders=[];daily_values={};violations=[];cap_residuals=[]
    for i in range(first,last+1):
        t=int(m.times[i]);prices=m.close[i];cov=sigma[i]
        if i>first:
            ledger.charge_funding(t,prices,m.funding[i],m.observed[i],scenario['adverse_funding_bps_hour'])
        if pending is not None:
            pending['sigma']=cov
            orders.extend(execute_intent(ledger,pending,m,i,protocol,scenario))
            if candidate:
                active=np.asarray(pending['risk_cap_active'])
                residual=np.maximum(abs(ledger.qty)-np.asarray(pending['risk_cap_qty']),0)*active
                if np.any(residual>1e-9):cap_residuals.append({'time':t,'qty':residual.tolist()})
            pending=None
        equity=ledger.reconcile(prices);peak=max(peak,equity);mdd=min(mdd,equity/peak-1)
        halted|=equity/peak-1<=-protocol['risk']['drawdown_halt']
        breach=risk_breaches(ledger.qty,equity,prices,cov,protocol['risk'])
        streak=streak+1 if breach else 0;max_streak=max(streak,max_streak)
        if breach:violations.append({'time':t,'constraints':breach})
        if t%86400==0:daily_values[t]=equity
        if i<last:
            kwargs={'retention':retention[i]} if candidate else {}
            action=control.propose(t,prices,ledger.qty,equity,weights[i],cov,m.volume[i],m.trades[i],protocol,
                                   buffer=.2,halted=halted,**kwargs)
            if action is not None:
                event=dict(decision_time=t,execution_time=t+HOUR,source_close=t,source_available=t,source_index=i,
                           source_prices=prices.tolist(),visible_equity=equity,position_before=ledger.qty.tolist(),
                           source_volume=m.volume[i].tolist(),source_trades=m.trades[i].tolist(),
                           raw_target=weights[i].tolist(),**action)
                decisions.append(event)
                pending={**event,'source_volume':m.volume[i].copy(),'source_trades':m.trades[i].copy(),'sigma':cov}
        snapshots.append(dict(time=t,equity=equity,cash=ledger.cash,qty=ledger.qty.tolist(),
            unrealized=equity-ledger.cash,fees=ledger.fees,funding=ledger.funding,realized=ledger.realized,
            **risk_values(ledger.qty,equity,prices,cov)))
    daily=[dict(time=t,return_=daily_values[t]/daily_values[t-86400]-1) for t in sorted(daily_values) if t-86400 in daily_values]
    completed=sum(f['position_after']==0 and f['position_before']!=0 for f in ledger.fills)
    metrics={'roi':equity/ledger.initial-1,'equity':equity,'net_pnl':equity-ledger.initial,
        'daily_std_sample':float(np.std([d['return_'] for d in daily],ddof=1)) if len(daily)>1 else None,
        'daily_std_population':float(np.std([d['return_'] for d in daily])) if daily else None,
        'complete_days':len(daily),'max_drawdown_magnitude':-mdd,'fees':ledger.fees,'funding':ledger.funding,
        'realized':ledger.realized,'unrealized':equity-ledger.cash,'turnover':ledger.turnover,
        'fills':len(ledger.fills),'orders':len(orders),'completed_lifecycles':completed,
        'mean_gross':float(np.mean([s['gross'] for s in snapshots])),'maximum_gross':max(s['gross'] for s in snapshots),
        'incomplete_orders':sum(o['status']!='FILLED' for o in orders),'unreached_cap_events':len(cap_residuals),
        'final_qty':ledger.qty.tolist(),'maximum_ledger_error':ledger.max_error}
    result=dict(policy='final_risk_cap_v1' if candidate else 'comparison',scenario=scenario_name,
        start=int(m.times[first]),end=int(m.times[last]),metrics=metrics,snapshots=snapshots,decisions=decisions,
        orders=orders,fills=ledger.fills,funding_events=ledger.funding_events,daily=daily,
        cap_residuals=cap_residuals,risk_breaches=violations,
        gates={'accounting':ledger.max_error<1e-6,'information_time':all(f['source_available']<=f['decision_time']<f['time'] for f in ledger.fills),
               'risk':max_streak<2,'not_halted':not halted},live_orders_sent=0,
        terminal_valuation='reference close; no invented liquidation, no historical par oracle')
    checked=audit(result,m,protocol)
    if not checked['passed']:raise AssertionError(checked)
    result['independent_accounting_audit']=checked
    return result


def main():
    if (OUT/'comparison.json').exists():raise ValueError('Do not overwrite a completed comparison')
    prepared=json.loads((OUT/'prepared.json').read_text())
    if sha(OUT/'inputs.npz')!=prepared['input_sha256'] or sha(OUT/'targets.npz')!=prepared['targets_sha256']:
        raise ValueError('Prepared inputs changed')
    plan=json.loads((HERE/'protocol.json').read_text())
    p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    with np.load(OUT/'inputs.npz') as z:
        m=Market(*(z[k].copy() for k in ('market_times','close','volume','trades','funding','observed')),tuple(p['symbols']))
    with np.load(OUT/'targets.npz') as z:targets={k:z[k].copy() for k in z.files}
    # Commit sources, periods and assumptions before any PnL comparison.
    save(OUT/'evaluation_manifest.json',{'source_hashes':{str(x.relative_to(ROOT)):sha(x) for x in HERE.glob('*.py')},
         'protocol_sha256':sha(HERE/'protocol.json'),'prepared':prepared,'historical_independence':False})
    rows=[];runs=OUT/'runs';runs.mkdir(exist_ok=True)
    for k,(start,end) in enumerate(plan['evaluation_periods_utc']):
        a=m.exact(int(np.datetime64(start,'s').astype('int64')));b=m.exact(int(np.datetime64(end,'s').astype('int64')))
        for name in plan['comparators']:
            w=targets['candidate' if name=='final_risk_cap_v1' else name]
            for scenario in ('base','stress'):
                result=simulate(m,w,targets['retention'],targets['sigma'],p,a,b,scenario,name=='final_risk_cap_v1')
                result['policy']=name
                if k==0 and name=='incumbent':
                    prior=legacy_simulate(m,{'target_weight':w},name,p,scenario,int(m.times[a]),int(m.times[b]),buffer=.2,sigmas=targets['sigma'])
                    np.testing.assert_allclose([s['equity'] for s in result['snapshots']],[s['equity'] for s in prior['snapshots']],rtol=0,atol=1e-8)
                    assert result['fills']==prior['fills']
                destination=runs/f'period{k}-{name}-{scenario}.json.gz'
                temp=destination.with_suffix('.tmp')
                with gzip.open(temp,'wt') as f:json.dump(result,f,allow_nan=False,default=json_default)
                temp.replace(destination)
                row={x:result[x] for x in ('policy','scenario','start','end','metrics','gates')};row['period']=k;rows.append(row)
                print(k,name,scenario,'ROI%',round(result['metrics']['roi']*100,6),'STD%',round(result['metrics']['daily_std_sample']*100,6),flush=True)
    comparisons=[]
    for k in range(len(plan['evaluation_periods_utc'])):
        for scenario in ('base','stress'):
            pair={r['policy']:r for r in rows if r['period']==k and r['scenario']==scenario}
            c,b=pair['final_risk_cap_v1'],pair['incumbent'];cm,bm=c['metrics'],b['metrics']
            checks={'positive_net_return':cm['roi']>0,'higher_return':cm['roi']>bm['roi'],
                'daily_std_no_higher':cm['daily_std_sample']<=bm['daily_std_sample'],
                'mdd_no_worse':cm['max_drawdown_magnitude']<=bm['max_drawdown_magnitude'],
                'engineering':all(c['gates'].values()) and all(b['gates'].values())}
            comparisons.append({'period':k,'scenario':scenario,'checks':checks,'dominates':all(checks.values()),
                'return_delta_pp':(cm['roi']-bm['roi'])*100,
                'return_vs_constant_control_pp':(cm['roi']-pair['constant_scale_control']['metrics']['roi'])*100})
    protected=json.loads((OUT/'protected_before.json').read_text())
    changed=[f for f,d in protected.items() if sha(ROOT/f)!=d]
    if changed:raise AssertionError('Production changed: '+str(changed))
    save(OUT/'comparison.json',{'rows':rows,'comparisons':comparisons,'incumbent_replay_equivalence':True,
        'all_predefined_cells_dominate':all(x['dominates'] for x in comparisons),'independent_evaluation':False,
        'decision':'HOLD_EXISTING_PENDING_PROSPECTIVE_PAR_EVIDENCE','production_files_unchanged':len(protected),
        'live_orders_sent':0,'valuation':'reference-market close, not contest oracle'})

if __name__=='__main__':main()
