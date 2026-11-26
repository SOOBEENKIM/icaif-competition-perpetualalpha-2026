"""Second hypothesis wave; retain first wave's failures and live immutability."""
import copy,gzip,json,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,PRIOR,read,save,sha,verify as verify_wave1
from .relative import forecasts
from auto_pair_selection.selection import PAIRS,IDS
from competition_live.profile import BUNDLE,load_profile

OUT=ROOT/'reports/pair_model_research_20261023/wave2'
ARMS=['trend_24','trend_72','trend_168','reversion_72','regime_72']


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=verify_wave1()
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),wave=2,arms=ARMS,input=p['input'],
      input_sha256=p['input_sha256'],
      model='All45 fixed1:1 dollar hedges. Trend projects past168h log relative return to h, new entries require last24h and168h relative directions to agree. Reversion fits AR1 to the past720h fixed log price difference at00UTC, using both fresh legs, >=168 rows; admissible estimated half life6..336h. Regime uses trend if aligned, otherwise reversion. Prediction capped at4 source sigmas*sqrt(h).',
      justification='Wave1 shared nonlinear and longer-horizon price learners failed. This wave tests explicit relative trend/reversion and removes changing estimated hedge beta. It is a different relative-value model, not a claim of proven cointegration.',
      action='No named-pair rules, all45 current signals ranked by expected executable net cash with the same fee/depth/funding/risk/partial-fill engine. Same6000 paired gross,50k own capital,one active pair. Held prediction/exit remains even when trend alignment blocks new entry.',
      selection='August base/stress worst Q advantage then PnL advantage selects one BEFORE later replay; reject any candidate with hard-risk violation. If none improves, record failures and continue research.',
      history_reused=True,independent_test=False,orders_sent=0,live_changed=False))
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),Path(__file__).with_name('relative.py'),OUT/'protocol.json']})


def verify():
    verify_wave1()
    for path,digest in read(OUT/'freeze.json').items():assert sha(ROOT/path)==digest,path
    return read(OUT/'protocol.json')


def build():
    p=verify();d=dict(np.load(ROOT/p['input'],allow_pickle=False));t=d['market_times']
    for arm in p['arms']:
        kind,h=arm.rsplit('_',1);h=int(h);mu,f,entry,records=forecasts(d,h,kind)
        valid=np.isfinite(mu)&f['valid'];actions=np.where(entry&(abs(mu)>.0038),np.sign(np.nan_to_num(mu)),0).astype(int)
        np.savez_compressed(OUT/f'{arm}.npz',times=t,dense=True,pairs=np.asarray(IDS),signal_times=np.broadcast_to(t[:,None],mu.shape),
          source_prices=d['close'],beta=np.ones_like(mu),vol=f['vol'],forecast_times=t,mu=mu,directions=valid.astype(int),
          economic_available=entry,economic_action=actions)
        save(OUT/f'{arm}_fits.json',records)
        print(arm,'all45','signals',np.count_nonzero(actions),flush=True)
    verify()


def job(arm,phase,scenario):
    from pas_economic_allocation.replay import replay
    from pas_performance_research.selected import controller
    from pas_allocation_study.study import summarize
    p=read(OUT/'protocol.json');profile=load_profile()
    w=next(x for x in read(BUNDLE/'expected.json')['runs'] if x['venue']=='reference' and x['arm']==profile['candidate'] and x['phase']==phase and x['scenario']==scenario)
    dest=OUT/'reference'/f'{arm}_{phase}_{scenario}.json.gz'
    if dest.exists():
        with gzip.open(dest,'rt') as fp:r=json.load(fp)
    else:
        d=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False));s=dict(np.load(OUT/f'{arm}.npz',allow_pickle=False));t=s.pop('times');ii=np.searchsorted(t,d['times'])
        assert np.array_equal(t[ii],d['times']);s={k:(v[ii] if v.ndim and len(v)==len(t) else v) for k,v in s.items()}
        z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz',allow_pickle=False));zi=np.searchsorted(z['times'],d['times'])
        f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz',allow_pickle=False));fi=np.searchsorted(f['times'],d['times'])
        raw=dict(np.load(ROOT/p['input'],allow_pickle=False));cfg=copy.deepcopy(profile['configs']['reference'])
        cfg['dense_pairs']=[list(x) for x in PAIRS];cfg['conditional_completion']={'risk_aversion':0.}
        c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z['all_market_signed_core_v1'][zi],z['all_market_signed_core_v1_protected'][zi],price_times=raw['market_times'],prices=raw['close'])
        h=int(arm.rsplit('_',1)[1]);r=replay(d,s,None,cfg,profile['trading'],f'first_dense_hourlyhedge_relative_{h}',scenario,w['start'],w['end'],base_policy=c)
        dest.parent.mkdir(exist_ok=True)
        with gzip.open(dest,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    row=summarize(r,arm,phase,scenario);row['entered_pairs']=sorted({tuple(e['pair']) for e in r['events'] if e['event']=='pair_intent'})
    assert row['timing_pass'] and row['accounting_error']<1e-6
    row['eligible']=row['risk_breach_bars']==0
    save(OUT/'rows'/f'{arm}_{phase}_{scenario}.json',row);return row


def run_jobs(jobs):
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(job,*args) for args in jobs]):
            row=future.result();print(json.dumps({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)


def stage_a():
    p=verify();run_jobs([(a,'validation',s) for a in p['arms'] for s in ['base','stress']])
    baseline=[r for r in read(PRIOR/'reference_results.json') if r['arm']=='incumbent_fixed_two' and r['phase']=='validation'];ranking=[]
    for a in p['arms']:
        rs=[read(OUT/'rows'/f'{a}_validation_{s}.json') for s in ['base','stress']]
        ds=[(r['Q']-b['Q'],r['net_pnl']-b['net_pnl']) for r in rs for b in baseline if r['scenario']==b['scenario']]
        ranking.append(dict(arm=a,eligible=all(r['eligible'] for r in rs),min_Q_advantage=min(x[0] for x in ds),min_pnl_advantage=min(x[1] for x in ds)))
    ranking.sort(key=lambda r:(r['eligible'],r['min_Q_advantage'],r['min_pnl_advantage']),reverse=True)
    save(OUT/'selection.json',dict(selected=ranking[0]['arm'] if ranking[0]['eligible'] else None,ranking=ranking))
    print('SELECTED',ranking[0],flush=True);verify()


def stage_b():
    verify();a=read(OUT/'selection.json')['selected'];assert a is not None
    run_jobs([(a,p,s) for p in ['comparison','continuous'] for s in ['base','stress']]);verify()


if __name__=='__main__':globals()[sys.argv[1]]()
