"""Same-period conditional PAS replay; venue funding is not mainnet funding."""
import gzip
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from pas_adaptive_development.study import ROOT,OUT as BASE,save,sha,native
from .study import OUT as ECONOMIC
from .anchored import AnchoredController
from .policy import EconomicController

OUT=ROOT/'reports/pas_economic_venue_20261022'
DATA=ROOT/'reports/pas_allocation_study_20261021/pas'


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    selected=json.loads((ECONOMIC/'selected.json').read_text());cfg=json.loads((DATA/'dataset.json').read_text())
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),data=str(DATA.relative_to(ROOT)),interval=cfg,
      nominee='native_carry_allocation',risk_aversion=selected['selected']['risk_aversion'],
      arms=['incumbent','size_only','native_carry_allocation','native_without_ETH'],
      hypothesis='Reference funding cannot explain PAS carry. Apply the SAME fitted weekly price predictor and cost-aware inventory allocator to actual lagged PAS funding; evaluate all ten symbols without copying a retrospectively chosen winner.',
      funding='Use only settlement records with actual millisecond timestamp < decision source minus1h. Reindex hourly; EWM spans24/72h with minimum24 known settlements; agreement/minimum magnitude. Forecast168h, full payment/half receipt. Fixed rule before replay; no PAS label fitting.',
      cadence='Original6h schedule. Source becomes available no earlier than original published signal and source+120seconds; no same-minute fills.',
      controls='All accounts50000+50000reserve, same interval and original minute executor, paid entry/exit reserve, funding and depth constraints. Base-owner only for every arm; not a full paired strategy ranking.',
      protected_ETH='native_without_ETH is the practical disjoint-market variant: zero ETH allocation from inception. It is an ablation, not a post-result replacement nominee.',
      limitations=['Historical mark/oracle/L2 queue absent; trade candles are not executable quotes.',
        'Funding cash uses prior trade close as settlement-oracle proxy.',
        'The inherited minute executor first filled the incumbent about12h later than its real first fill. Same-engine comparisons do not remove this fidelity failure.',
        'Official daily Q cannot be established from trade-close valuations. Report conditional Q separately.',
        'Only about2days, previously inspected development; no independent test or live-readiness claim.'],
      promotion='Record PnL/Q under both cost conditions against both controls. Execution-fidelity failure explicitly blocks automatic deployment regardless of returns.',
      independent=False,history_reused=True,live_changed=False,orders_sent=0,
      deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    files=[Path(__file__),Path(__file__).with_name('pas_engine.py'),Path(__file__).with_name('policy.py'),OUT/'protocol.json',
           ECONOMIC/'forecasts.npz',*DATA.rglob('*.json')]
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})


class WithoutETH(EconomicController):
    def propose(self,*args,**kwargs):
        # Input freshness blocks ETH additions; this replay begins flat.
        args=list(args);args[6]=np.array(args[6],copy=True);args[7]=np.array(args[7],copy=True)
        args[6][1]=0.;args[7][1]=0.
        result=super().propose(*args,**kwargs)
        if result is not None:assert result['target_qty'][1]==0.
        return result


def run():
    from competition_live.profile import load_profile,verify_deployment
    from pas_strategy_audit.data import load
    from .pas_engine import simulate
    p=json.loads((OUT/'protocol.json').read_text())
    for rel,h in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/rel)==h,rel
    cfg,d=load(DATA);profile=load_profile();trading=profile['trading']
    parent=json.loads((BASE/'protocol.json').read_text());history=dict(np.load(ROOT/parent['pointer']['inputs']))
    z=dict(np.load(ECONOMIC/'forecasts.npz'));base=dict(np.load(BASE/'targets.npz'))
    publications={}
    for file in (ROOT/'practice_live/updates').glob('*/signals.json'):
        row=json.loads(file.read_text());source=int(datetime.fromisoformat(row['input_completed_bar_utc']).timestamp())
        at=int(datetime.fromisoformat(row['created_at_utc']).timestamp()*1000)
        publications[source]=min(at,publications.get(source,at))
    sources=[];rates=[];events={arm:[] for arm in p['arms']};audit=[]
    funding=[json.loads((DATA/'raw'/f'pas_funding_{s}.json').read_text())['data'] for s in cfg['symbols']]
    for i,t in enumerate(z['times']):
        t=int(t)
        if not cfg['start_ms']<=t*1000<cfg['cutoff_ms'] or t%21600:continue
        if t not in publications:raise ValueError('Missing original publication timestamp')
        at=max(publications[t],(t+120)*1000)
        if at>=cfg['cutoff_ms']:continue
        row=[];last=[]
        for values in funding:
            known=[v for v in values if v['time']<(t-3600)*1000]
            if len(known)<24:row.append(0.);last.append(None);continue
            # Missing hourly settlements invalidate the rate forecast.
            if np.max(np.diff([v['time']//3600000 for v in known[-72:]]))>1:
                row.append(0.);last.append(None);continue
            x=pd.Series([float(v['fundingRate']) for v in known])
            fast=x.ewm(span=24,adjust=False,min_periods=24).mean().iloc[-1]
            slow=x.ewm(span=72,adjust=False,min_periods=24).mean().iloc[-1]
            rate=np.sign(slow)*min(abs(fast),abs(slow)) if fast*slow>0 else 0.
            row.append(float(rate)*168);last.append(known[-1]['time'])
        rates.append(row);sources.append(t);audit.append(dict(source=t,available_ms=at,last_funding_ms=last))
        for arm in p['arms']:
            w=base['size_only' if arm=='size_only' else 'incumbent'][i]
            events[arm].append(dict(time=at,source_time=t,weights=w.tolist(),sigma=base['sigma'][i].tolist(),
                               volume=history['volume'][i].tolist(),trades=history['trades'][i].tolist()))
    ids=np.searchsorted(z['times'],sources);forecast=z['forecast'][ids];rates=np.array(rates)
    save(OUT/'availability.json',audit);save(OUT/'events.json',events)
    np.savez_compressed(OUT/'venue_inputs.npz',times=sources,forecast=forecast,carry=rates)
    scenarios={'base':dict(fee_bps=9.,impact_bps=10.,participation=.01,receipt_multiplier=1.),
               'stress':dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=.5)}
    rows=[]
    for arm in p['arms']:
        for name,sc in scenarios.items():
            c=None
            if arm.startswith('native'):
                cls=WithoutETH if arm=='native_without_ETH' else EconomicController
                c=cls(sources,forecast,rates,p['risk_aversion'])
            result=simulate(cfg,d,events[arm],trading,sc,controller=c)
            result['metrics']['official_Q']=False
            row=dict(arm=arm,scenario=name,**result['metrics']);rows.append(row)
            if c is not None:result['allocation_trace']=c.trace
            with gzip.open(OUT/f'{arm}_{name}.json.gz','wt') as f:json.dump(result,f,default=native,allow_nan=False)
            save(OUT/'results.json',rows);print(json.dumps(row),flush=True)
    assert verify_deployment()['files']==p['deployment']


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
