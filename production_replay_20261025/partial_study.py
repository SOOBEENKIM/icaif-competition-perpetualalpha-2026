"""Frozen, all45 joint-PCI research with a matched retained-strategy control.

External reference history is reused DEVELOPMENT, never PAS or independent
performance. No parameter grid or later selection of the diagnostic arm.
"""
import copy,gzip,json,sys,time
from dataclasses import asdict
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from auto_pair_selection.selection import IDS,PAIRS,validate_snapshot
from competition_live.profile import BUNDLE,load_profile
from .protocol import ROOT,OUT,save,sha
from .partial_model import Spec,fit,validate,filtered,expected_return,_library

DEST=OUT/'joint_partial'
ARMS=['joint_pci_validated_72','joint_pci_unvalidated_72']


def freeze():
    if (DEST/'protocol.json').exists():raise FileExistsError('Keep frozen run')
    inp=json.loads((ROOT/'reports/universe_repair_20261022/protocol.json').read_text())['input']['inputs']
    p=load_profile();wins=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==p['candidate']]
    files=[__file__,str(ROOT/'production_replay_20261025/partial_model.py'),str(ROOT/'production_replay_20261025/partial_filter.c')]
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),spec=asdict(Spec()),input=inp,input_sha256=sha(ROOT/inp),
        primary=ARMS[0],diagnostic_only=ARMS[1],control='incumbent',all_pairs=PAIRS,
        hypothesis='Fit hedge beta jointly with permanent and transient residual noise. A permanent displacement is not a mean-reversion target. All45 pairs fit on trailing60days followed by21past days of disjoint72h forecast validation, refreshed weekly. Require conditional log-score and uncalibrated fixed-share MSE better than random-walk/zero baselines; shrink magnitude using past validation slope. No pair ID, PAS future outcome or winner-wallet feature.',
        difference_from_old='Old Oct17 partial model used OLS beta,6pairs,AIC in-fit screening and no held-past fixed-share forecast check; its later failure remains part of the trial record.',
        historical_independence=False,development_data_reused=True,
        evidence='This is not a calibrated multiple-testing-adjusted cointegration test or paper replication. Individual validation windows have few correlated observations; no overfitting-free claim.',
        forecast='Fixed source beta; normalized log-spread expectation calibrated to actual simple-return labels. Existing symmetric leg-terminal mapping is an approximation, actual fixed/delta quantities pay all engine costs. 72h horizon explicit. Paid renewal/rehedge executor retained, actual cash gains only.',
        periods=[{k:w[k] for k in ['phase','scenario','start','end']} for w in wins],
        success='Primary higher PnL and Q on all three periods in base and stress, zero timing/accounting/risk failures; outcome is development qualification only. Diagnostic may explain failure but cannot replace primary after seeing results.',
        sources=['https://github.com/matthewclegg/partialCI','https://arxiv.org/html/2609.35359v1'],
        code_hashes={str(__import__('pathlib').Path(f).relative_to(ROOT)):sha(f) for f in files},orders_sent=0,live_changed=False))


def verify():
    p=json.loads((DEST/'protocol.json').read_text())
    for path,digest in p['code_hashes'].items():assert sha(ROOT/path)==digest,path
    assert sha(ROOT/p['input'])==p['input_sha256'];return p


def formation(yx,good,spec):
    y,x=yx.T;m=fit(y[:spec.fit_hours],x[:spec.fit_hours],good[:spec.fit_hours],spec)
    if m is None:return None
    m['validation']=validate(y,x,good,m,spec);return m


def build():
    p=verify();s=Spec(**p['spec']);d=dict(np.load(ROOT/p['input'],allow_pickle=False));t=validate_snapshot(d)
    logs=np.log(d['close']);active=(d['volume']>0)&(d['trades']>0);n=len(t);shape=(n,len(IDS))
    mu={arm:np.full(shape,np.nan) for arm in ARMS};beta=np.full(shape,np.nan);vol=beta.copy();records=[]
    width=s.fit_hours+s.validation_hours;first=min(w['start'] for w in p['periods'])
    # Begin one refresh before the evaluation start; never fit from evaluation
    # labels ahead of each source. Calendar alignment is fixed Unix weeks.
    sources=np.flatnonzero((t>=(first//(s.refresh_hours*3600)-1)*(s.refresh_hours*3600))&(t%(s.refresh_hours*3600)==0))
    _library()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i in sources:
            if i<width:continue
            begin=i-width;end=min(n,i+s.refresh_hours);now=int(t[i]);started=time.monotonic()
            futures={pool.submit(formation,logs[begin:i][:,ids],active[begin:i,ids[0]]&active[begin:i,ids[1]],s):(k,ids) for k,ids in enumerate(IDS)}
            models={}
            for f in as_completed(futures):
                k,ids=futures[f];m=f.result();models[k]=m
                records.append(dict(source=now,pair_index=k,pair=list(PAIRS[k]),
                    trained_through=int(t[begin+s.fit_hours-1]),validated_through=int(t[i-1]),model=m))
                if m is None:continue
                a,b=ids;good=active[begin:end,a]&active[begin:end,b]
                state=filtered(logs[begin:end,a],logs[begin:end,b],good,m['params'])
                raw=expected_return(state[width:],m,s.horizon);bt,rho,qr,qm=m['params']
                hourly_vol=np.sqrt(qr+2*qm/(1+rho))/(1+bt)
                raw=np.clip(raw,-4*hourly_vol*np.sqrt(s.horizon),4*hourly_vol*np.sqrt(s.horizon))
                rows=np.arange(i,end);valid=active[rows,a]&active[rows,b]
                beta[rows,k]=bt;vol[rows,k]=hourly_vol
                # Same admissible hedge/half-life in both; diagnostic isolates
                # the validation gate and calibration, not a hidden universe.
                admissible=not m['beta_at_bound'] and s.half_life_min<=m['half_life']<=s.half_life_max
                if admissible:mu[ARMS[1]][rows[valid],k]=raw[valid]
                if m['validation']['eligible']:mu[ARMS[0]][rows[valid],k]=raw[valid]*m['validation']['calibration']
            save(DEST/'formations.json',sorted(records,key=lambda r:(r['source'],r['pair_index'])))
            print(datetime.fromtimestamp(now,timezone.utc).date(),'fits',sum(m is not None for m in models.values()),'validated',sum(m is not None and m['validation']['eligible'] for m in models.values()),'seconds',round(time.monotonic()-started,2),flush=True)
    for arm in ARMS:
        valid=np.isfinite(mu[arm]);np.savez_compressed(DEST/f'{arm}.npz',times=t,dense=True,pairs=np.array(IDS),
            signal_times=np.broadcast_to(t[:,None],shape),forecast_times=t,source_prices=d['close'],
            beta=beta,vol=vol,mu=mu[arm],directions=valid.astype(int),economic_available=valid,
            economic_action=np.where(valid&(abs(mu[arm])>s.cost_floor),np.sign(np.nan_to_num(mu[arm])),0).astype(int))
    verify();save(DEST/'build_complete.json',dict(at=datetime.now(timezone.utc).isoformat(),formations=len(records),orders_sent=0))


def job(arm,w):
    from pas_economic_allocation.replay import replay
    from pas_allocation_study.study import summarize
    from pas_performance_research.selected import controller
    p=verify();profile=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False));cfg=copy.deepcopy(profile['configs']['reference'])
    if arm=='incumbent':
        state=dict(np.load(BUNDLE/'models/reference_gate_state.npz'));run_arm=profile['candidate']
    else:
        state=dict(np.load(DEST/f'{arm}.npz'));st=state.pop('times');ii=np.searchsorted(st,d['times']);np.testing.assert_array_equal(st[ii],d['times'])
        state={k:(v[ii] if v.ndim and len(v)==len(st) else v) for k,v in state.items()}
        cfg['dense_pairs']=[list(x) for x in PAIRS];run_arm='first_dense_hourlyhedge_jointpci_72'
    cfg['conditional_completion']={'risk_aversion':0.}
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));zi=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times'])
    np.testing.assert_array_equal(z['times'][zi],d['times']);np.testing.assert_array_equal(f['times'][fi],d['times'])
    raw=dict(np.load(ROOT/p['input']));c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][zi],z['all_market_signed_core_v1_protected'][zi],price_times=raw['market_times'],prices=raw['close'])
    r=replay(d,state,None,cfg,profile['trading'],run_arm,w['scenario'],w['start'],w['end'],base_policy=c)
    row=summarize(r,arm,w['phase'],w['scenario']);row.update(venue='external_reference',independent_test=False,entered_pairs=sorted({tuple(e['pair']) for e in r['events'] if e['event']=='pair_intent'}))
    dest=DEST/'reference';dest.mkdir(exist_ok=True)
    with gzip.open(dest/f"{arm}_{w['phase']}_{w['scenario']}.json.gz",'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    assert row['timing_pass'] and row['accounting_error']<1e-6
    save(DEST/'rows'/f"{arm}_{w['phase']}_{w['scenario']}.json",row);return row


def evaluate():
    p=verify();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,arm,w) for w in p['periods'] for arm in ['incumbent',*ARMS]]):
            row=f.result();rows.append(row);save(DEST/'results.json',rows)
            print({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q','pair_pnl','completed_pairs','risk_breach_bars']},flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
