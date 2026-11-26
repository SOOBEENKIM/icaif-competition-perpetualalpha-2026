"""Attribute actual replay cash and verify 72h pricing/funding propagation."""
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np

from replay_lab.common_comparison.engine import Ledger
from .research import ROOT, HERE, OUT, RELEASE, BUNDLE, load, readrun, save, sha, verify, protected


def ownership(run,data,symbols):
    whole=Ledger(0.,symbols);episode=None;episodes=[];impact=0.;episode_impact=0.
    rows=[(f['time'],0,i,'funding',f) for i,f in enumerate(run['funding_events'])]
    rows += [(f['time'],1,i,'fill',f) for i,f in enumerate(run['owner_fills']['pair'])]
    for at,_,_,kind,event in sorted(rows,key=lambda row:row[:3]):
        if kind=='funding':
            prices=np.asarray(event['prices']);rates=np.asarray(event['rates']);observed=np.asarray(event['observed'])
            whole.charge_funding(at,prices,rates,observed,0.)
            if episode is not None:episode['ledger'].charge_funding(at,prices,rates,observed,0.)
            continue
        if episode is None:
            episode=dict(start=at,assets=set(),ledger=Ledger(0.,symbols));episode_impact=0.
        asset=event['asset'];amount=event['signed_qty'];episode['assets'].add(asset)
        ix=int(np.searchsorted(data['times'],at));assert data['times'][ix]==at
        paid_impact=amount*(event['price']-data['par_close'][ix,asset])
        impact+=paid_impact;episode_impact+=paid_impact
        for ledger in [whole,episode['ledger']]:
            ledger.fill(asset,amount,event['price'],event['fee'],at)
        assert abs(whole.qty[asset]-event['position_after'])<1e-10
        if not np.any(episode['ledger'].qty):
            ledger=episode['ledger']
            episodes.append(dict(start=episode['start'],end=at,pair=[symbols[j] for j in sorted(episode['assets'])],
                closed=True,net_pnl=ledger.cash,fees=ledger.fees,funding=ledger.funding,impact=episode_impact,
                price_move_pnl=ledger.cash+ledger.fees-ledger.funding+episode_impact,
                turnover=ledger.turnover,holding_hours=(at-episode['start'])/3600))
            episode=None
    last=int(np.searchsorted(data['times'],run['end']));marks=data['par_close'][last]
    net=whole.equity(marks)
    assert abs(net-run['metrics']['pair_owned_pnl'])<1e-6,(net,run['metrics']['pair_owned_pnl'])
    if episode is not None:
        ledger=episode['ledger'];value=ledger.equity(marks)
        episodes.append(dict(start=episode['start'],end=run['end'],pair=[symbols[j] for j in sorted(episode['assets'])],
            closed=False,net_pnl=value,fees=ledger.fees,funding=ledger.funding,impact=episode_impact,
            price_move_pnl=value+ledger.fees-ledger.funding+episode_impact,
            turnover=ledger.turnover,holding_hours=(run['end']-episode['start'])/3600))
    assert abs(sum(e['net_pnl'] for e in episodes)-net)<1e-6
    assert abs(sum(e['funding'] for e in episodes)-whole.funding)<1e-6
    return dict(net_pnl=net,fees=whole.fees,funding=whole.funding,impact=impact,
                price_move_pnl=net+whole.fees-whole.funding+impact,turnover=whole.turnover,episodes=episodes)


def main():
    results=json.loads((OUT/'results.json').read_text());spec=json.loads((HERE/'protocol.json').read_text())
    symbols=json.loads((BUNDLE/'config.json').read_text())['trading']['symbols']
    paths=sorted((OUT/'runs').glob('*.json.gz'))
    controls=[RELEASE/'reproduction_final/runs'/f'{v}-{phase}-{spec["retained"]}-{sc}.json.gz'
              for v,phase in results['windows'] for sc in ['base','stress']]
    sources=json.loads((OUT/'frozen.json').read_text())['sources'].copy()
    files=[Path(__file__),OUT/'results.json',OUT/'selection.json',OUT/'books/results.json',*paths,*controls]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files});verify(sources)
    save(OUT/'audit_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    data={v:load(BUNDLE/'data'/f'{v}.npz') for v in ['reference','par','par_extended']}
    indexed={f'{r["venue"]}-{r["phase"]}-{r["arm"]}-{r["scenario"]}':r for r in results['rows']}
    rows=[];funding_checks=0
    for path in paths+controls:
        name=path.name[:-8];info=indexed[name];run=readrun(path);d=data[info['venue']]
        attribution=ownership(run,d,symbols)
        for event in run['events']:
            if event['event']!='pair_intent':continue
            horizon=72 if info['arm'].endswith('_72') else 24
            assert event['horizon']==horizon
            now=event['time'];source=event['forecast_source'];ix=int(np.searchsorted(d['times'],now))
            assert d['times'][ix]==now and 0<=now-source<3600
            ids=event['ids'];qty=np.asarray(event['target']);remaining=horizon-(now-source)/3600
            carry=-qty*d['reference_close'][ix,ids]*d['known_hourly_funding'][ix,ids]*remaining
            carry=float(np.where(carry>0,carry*.5,carry).sum());gross=float(abs(qty)@d['par_close'][ix,ids])
            assert abs(event['expected_funding_bps']-carry/gross*10000)<1e-7
            assert np.allclose(event['terminal_prices'],np.asarray(event['source_prices'])*np.array([1+event['source_price_alpha'],1-event['source_price_alpha']]),rtol=0,atol=1e-10)
            funding_checks+=1
        reasons=Counter(event.get('reason','unknown') for event in run['events'] if event['event']=='exit_requested')
        rows.append(dict(venue=info['venue'],phase=info['phase'],arm=info['arm'],scenario=info['scenario'],
            metrics=run['metrics'],pair_attribution=attribution,exit_reasons=dict(reasons)))
    forecast_diagnostics=[]
    forecasts=load(OUT/'reference_forecasts.npz')
    for phase in ['validation','comparison']:
        run=readrun(OUT/'runs'/f'reference-{phase}-{spec["candidate"]}-base.json.gz')
        valid=(forecasts['times']>=run['start'])&(forecasts['maturity']<=run['end'])&(forecasts['times']%86400==0)
        valid=valid[:,None]&forecasts['fresh']&forecasts['terminal_fresh']&np.isfinite(forecasts['prediction'])&np.isfinite(forecasts['target'])
        actual=forecasts['target']*forecasts['unit'];predicted=forecasts['prediction']
        forecast_diagnostics.append(dict(phase=phase,pair_day_observations=int(valid.sum()),
            model_rmse_bps=float(np.sqrt(np.mean((predicted[valid]-actual[valid])**2))*10000),
            zero_rmse_bps=float(np.sqrt(np.mean(actual[valid]**2))*10000),
            gate_actions={str(k):int(np.sum(valid&forecasts['economic_available']&(forecasts['economic_action']==k))) for k in [-1,0,1]},
            sampling='00UTC sources, outcomes matured by period end; overlapping 72h labels, diagnostic not account PnL'))
    verify(sources)
    save(OUT/'audit.json',dict(accounts_audited=len(rows),rows=rows,horizon_and_funding_checks=funding_checks,
        forecast_diagnostics=forecast_diagnostics,risk_breach_bars=sum(r['metrics']['risk_breach_bars'] for r in rows),
        protected=protected(),orders_sent=0,reserved_raw_read=False,historical_independence=False))
    print('AUDIT',len(rows),'cash-conserving pair attributions;',funding_checks,'horizon/funding checks',flush=True)


if __name__=='__main__':main()
