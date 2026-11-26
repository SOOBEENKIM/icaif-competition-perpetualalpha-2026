"""One frozen candidate, equal conditions, all outcomes retained. Offline only."""
import copy
import gzip
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from adaptive_leverage.study import live_manifest
from adaptive_leverage.controller import AdaptiveController
from competition_live.profile import load_profile
from pas_performance_research.soft_timing import SoftTimingController
from pas_performance_research.extension import OUT as EXT,START,END,SPLIT,SYMBOLS
from pas_strategy_audit.data import load
from pas_counterfactual_20261022.engine import simulate
from .controller import OpportunityController
from . import model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/opportunity_sizing_20261026'
ARMS=['incumbent','old_amplifier','uniform_1p5','pas_opportunity']


def save(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
                               default=lambda x:x.tolist())+'\n')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def timestamp(s):return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp())


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Frozen experiment exists')
    primary=json.loads((EXT/'protocol.json').read_text())
    extension=ROOT/'reports/pas_cashflow_research_20261023/sealed_extension/dataset'
    p=dict(created_at=datetime.now(timezone.utc).isoformat(),primary='pas_opportunity',arms=ARMS,
        primary_dataset=str(EXT.relative_to(ROOT)),secondary_dataset=str(extension.relative_to(ROOT)),
        reference_input=primary['input'],train_start=timestamp('2026-03-18T00:00:00Z'),
        train_end=timestamp('2026-06-01T00:00:00Z'),calibration_end=timestamp('2026-07-01T00:00:00Z'),
        prediction='Pooled 24h ridge with 3h/24h/168h standardized momentum, 24h deviation, volatility ratio. Train-only standardization. Ridge1, intercept0.25. Daily labels mature strictly before split boundaries. June attenuation. Cluster mean uncertainty by entry day.',
        online_basis='Past-only hourly PAS/reference log basis, >=24 matured hourly transitions, ridge prior of zero drift, rho clipped0.8..1.02, <=72h window. Online updates during evaluation explicitly allowed by this frozen rule; future24h labels never used. Lack of basis data implies zero expected drift, not convergence.',
        funding='PAS hourly settlements known after60s. 6h/24h EWM agree, >=6 observations, confidence ramps to24. 24h projection; full payments, half receipts.',
        costs='At source time,9bp fee+10bp impact+half median last-hour active minute range per entry; exit adds5bp. Quadratic entry/exit impact coefficient10bp per estimated15min capacity. Historical L2 unavailable. No use of October26 quotes for earlier backtests.',
        capacity='Past60min PAS volume *15/60 *1% participation, in notional. Same fixed estimate in all execution stress scenarios; realized fills retain scenario-specific caps.',
        sizing='Additional exposure per asset, max3x total base target and original risk limits. Opportunity objective: expected24h net return minus mean uncertainty/costs/covariance risk, lambda5 inherited. 50 coordinate sweeps; only optional sleeve can change.',
        ownership='Track actual base and extra quantities. Attribute partial increases to base first, reductions to extra first. Extra expires24h after its entry intent, no later than24h after fill. Mandatory reductions/protected flags survive.',
        uniform_control='Fixed1.5x eligible base target, same caps, time/entry/protection/capacity/expiry rules. No return/uncertainty selection. Not matched actual exposure.',
        scope='No pair-leg resizing. Require archived pair audit to establish no active pair gate for base-only replay.',
        comparisons='Original earlier48h,later23h,continuous71h,trade_tape48h; three identical fee/capacity/funding scenarios; optional97h extension and later25h starting at12UTC if pair audit permits.',
        acceptance='Candidate net PnL and last-daily-cutoff PnL/Q exceed incumbent in primary71h base AND adverse scenarios, hard-risk/timing/accounting errors0. Report terminal liquidation reserve and control comparison. Failure is reported without retuning this experiment.',
        independent_test=False,history_reused=True,primary_period_seen_before=True,
        no_global_claim='This development comparison cannot establish optimal leverage, out-of-sample superiority, exact live fills or official ranking.',
        live_manifest=live_manifest(),orders_sent=0,live_changed=False)
    save(OUT/'protocol.json',p)
    ref=dict(np.load(ROOT/p['reference_input']))
    fit=model.train(ref,p['train_start'],p['train_end'],p['calibration_end'])
    save(OUT/'fit.json',fit)
    files=list((ROOT/'opportunity_sizing').glob('*.py'))+[OUT/'protocol.json',OUT/'fit.json',ROOT/p['reference_input']]
    for dataset in [EXT,extension]:
        files += [dataset/f for f in ['protocol.json','dataset.json','candidate_inputs.npz','events.json','pair_audit.json']]
        files += sorted((dataset/'raw').glob('pas_*.json'))
        meta=json.loads((dataset/'protocol.json').read_text())
        if 'input' in meta:files.append(ROOT/meta['input'])
    files += [ROOT/'reports/pas_counterfactual_20261022'/f for f in ['trade_tape.json','funding_tape.json']]
    files += [ROOT/'pas_counterfactual_20261022/engine.py',ROOT/'competition_live/scoring.py']
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
    print(json.dumps(fit),flush=True)


def verify():
    p=json.loads((OUT/'protocol.json').read_text())
    assert live_manifest()==p['live_manifest'],'Live sources changed during offline work'
    for file,digest in json.loads((OUT/'source_freeze.json').read_text()).items():assert sha(ROOT/file)==digest,file
    return p


def dataset(path):
    cfg,d=load(path);f=dict(np.load(path/'candidate_inputs.npz'))
    meta=json.loads((path/'protocol.json').read_text())
    reference=dict(np.load(ROOT/meta['input']))
    events=json.loads((path/'events.json').read_text())
    tape=[];funding=[];history=[[] for _ in SYMBOLS]
    for j,s in enumerate(SYMBOLS):
        for b in json.loads((path/'raw'/f'pas_candles_{s}.json').read_text())['data']:
            at=int(b['t'])+60000
            if float(b['v'])>0 and b['n']>0 and cfg['start_ms']<=at<=cfg['cutoff_ms']:
                tape.append(dict(time=at,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
        for r in json.loads((path/'raw'/f'pas_funding_{s}.json').read_text())['data']:
            funding.append(dict(time=r['time'],coin='pas:'+s,rate=float(r['fundingRate']),cash_per_unit=None))
            history[j].append((r['time']/1000,float(r['fundingRate'])))
        history[j].sort()
    return cfg,d,f,reference,events,tape,funding,history


def run():
    p=verify();profile=load_profile();fit=json.loads((OUT/'fit.json').read_text())
    rows=[]
    for label,path in [('original',EXT),('extension',ROOT/p['secondary_dataset'])]:
        audit=json.loads((path/'pair_audit.json').read_text())
        if not audit.get('can_use_base_only'):
            save(OUT/f'{label}_skipped.json',dict(reason='Active pair gate cannot be omitted',audit=audit));continue
        cfg,d,f,reference,events,tape,funding,history=dataset(path)
        mu,se=model.predict(reference,fit)
        if label=='original':
            old=ROOT/'reports/pas_counterfactual_20261022';real_tape=json.loads((old/'trade_tape.json').read_text())
            real_funding=json.loads((old/'funding_tape.json').read_text());known={(r['coin'],r['time']//3600000) for r in real_funding}
            real_funding += [r for r in funding if (r['coin'],r['time']//3600000) not in known]
            runs=[('earlier48h',START,SPLIT,tape,funding),('later23h',SPLIT,END,tape,funding),
                  ('continuous71h',START,END,tape,funding),('trade_tape48h',START,SPLIT,real_tape,real_funding)]
        else:
            end=cfg['cutoff_ms']//1000;runs=[('extension97h',START,end,tape,funding),('extension_later25h',START+72*3600,end,tape,funding)]
        scenarios=json.loads((EXT/'protocol.json').read_text())['scenarios']
        for phase,start,end,tt,ff in runs:
            for scenario,sc in scenarios.items():
                for arm in ARMS:
                    file=OUT/'runs'/f'{arm}_{phase}_{scenario}.json.gz'
                    if file.exists():
                        with gzip.open(file,'rt') as fp:result=json.load(fp)
                    else:
                        cls=AdaptiveController if arm=='old_amplifier' else SoftTimingController
                        base=cls(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],5.,
                            price_times=d['times']/1000,prices=d['close'],entry_rule='pullback_any3',age=d['age_ms']/1000)
                        c=OpportunityController(base,reference,mu,se,d,history,uniform=arm=='uniform_1p5') if arm in ['uniform_1p5','pas_opportunity'] else base
                        cc=copy.deepcopy(cfg);cc.update(start_ms=start*1000,cutoff_ms=end*1000)
                        result=simulate(cc,d,copy.deepcopy(events),tt,ff,profile['trading'],sc,controller=c)
                        if arm in ['uniform_1p5','pas_opportunity']:result['sizing_trace']=c.trace
                        file.parent.mkdir(exist_ok=True)
                        with gzip.open(file,'wt') as fp:json.dump(result,fp,allow_nan=False,default=lambda x:x.tolist())
                    if label=='original' and arm in ['incumbent','old_amplifier']:
                        saved_arm='pullback_any3' if arm=='incumbent' else 'adaptive_leverage'
                        previous=ROOT/'reports/adaptive_leverage_20261026/pas'/f'{saved_arm}_{phase}_{scenario}.json.gz'
                        with gzip.open(previous,'rt') as fp:assert json.load(fp)==result,'Baseline reproduction differs'
                    m=result['metrics'];trace=result.get('sizing_trace',[])
                    assert m['timing_pass'] and m['maximum_accounting_error']<1e-6
                    row=dict(arm=arm,phase=phase,scenario=scenario,**m,
                        end=end,score_cutoff=result['daily'][-1]['time'],
                        score_pnl=m['return_fraction']*100000 if m['return_fraction'] is not None else None,
                        extra_intent_rounds=sum(any(abs(q)>0 for q in r['extra_qty']) for r in trace),
                        actual_extra_rounds=sum(any(abs(q)>0 for q in r['extra_actual']) for r in trace),
                        extra_eligible_asset_rounds=sum(sum(r['eligible']) for r in trace))
                    rows.append(row);save(OUT/'results.json',rows)
                    print(json.dumps({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q','extra_intent_rounds','actual_extra_rounds']}),flush=True)
    verify();save(OUT/'completed.json',dict(runs=len(rows),live_unchanged=True,independent_test=False,orders_sent=0))


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
