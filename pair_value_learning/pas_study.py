"""Actual PAS tape candles + settlement timing, conditional paired replay.

This is a pair-module comparison, not official account PnL. Historical L2 and
oracle series are absent. No use of future volume to choose one leg over another.
"""
import copy,gzip,json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .study import ROOT,INPUT,read,save,sha,verify as verify_calibration
from .net_study import verify as verify_net
from auto_pair_selection.selection import PAIRS,SYMBOLS
from competition_live.profile import load_profile
from pas_strategy_audit.data import load
from replay_lab.common_comparison.models import Market,covariance

OUT=ROOT/'reports/pair_value_learning_20261023/pas'
RAW=ROOT/'reports/market_candidate_audit_20261023/raw/markets'
MINUTE=ROOT/'reports/pas_allocation_study_20261021/pas'
START=1792411200;END=1792670400  # Oct19 12UTC through Oct22 12UTC,72h.
SOURCES={
 'fixed_two_24':'reports/auto_pair_selection_20261023/states.npz',
 'adaptive_72':'reports/pair_model_research_20261023/wave9/adaptive_hedge_72.npz',
 'hierarchical_72':'reports/pair_value_learning_20261023/hierarchical_72.npz',
 'hierarchical_net_72':'reports/pair_value_learning_20261023/net_value/hierarchical_net_72.npz',
 'carry_net_24':'reports/pair_value_learning_20261023/net_value/carry_net_24.npz',
 'carry_net_72':'reports/pair_value_learning_20261023/net_value/carry_net_72.npz',
}


def freeze():
    verify_calibration();verify_net()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='carry_net_24',start=START,end=END,sources=SOURCES,
      purpose='Test PAS funding/price economics of the pairmodule over actual observed scoredphase. No directionalbase in any arm: module-only,50k sleeve+50kreserve,6000paircap. It doesnotrepresent wholeliveaccountor peerleaderboard.',
      alignment='Completed1hPAS candles. Initialstate from earlier completedminutecandles; unpricedmarkets cannotopen because knownsourcevolumezero. Candidate decisions atbarclose, independentlylimitedfills nextbar,notfuturedepth.',
      funding='Use actualsettlementrates; recognition atnextcompletedhour AFTER actualms timestamp, before nextbarfills. Cash uses completedreferenceprice atsettlementhour asoracleproxy. Prediction uses only earlieractualsettlements: half last +halfmeanlast24h,receiptsdiscount50%. No futurefundingforprediction.',
      price_anchor='Originalfixedtwo/adaptivebaseline retain theiroriginalreferenceanchor hypothesis. Newhierarchicalandcarryarms rebase returnforecasts atcurrentPAScompletedprice, preservingcurrentvenuebasis ratherthanassumingitsfullconvergence. Carryonlypricealpha explicitly0.',
      relation='Retainstablepositive-returnhedgequalification for newarms; all45considered. Originalfixedtwo baseline remains2pairs asimplemented.',
      lifecycle='Baselinefixedtwo/adaptiveoriginalrenewal; newarmsfixedsourcehorizon. Compare toreferenceablations, notclaim single-variablePASexperiment.',
      costs='base9bpfee+10bpimpact,1%minimumofsource/arrivalhourvolume;stress12+20bp,.25%,halffundingreceipts;extra nofundingreceipts scenario. Nonfill/onelegrepair/paidclose/lot/riskoriginal.',
      limitations=['No historicalL2/queueor fulloracle/markseries: conditionalreplay,notexactfillabletrades.','Hourly aggregationmaymisrepresententrytiming, unlike liveexecutionminutes.','Officialscoringusesaccountmarkvaluation; these aretradecloseproxyvaluations.','72hshortperiodpreviouslyinspected,notindependenttest; overlappingpredictionlabelsnotindependenttrades.','Peeraccountsnotrunundertheseconditions: cannotclaimhigherpeerperformance.'],
      orders_sent=0,live_changed=False,independent_test=False,history_reused=True))
    paths=[Path(__file__),Path(__file__).with_name('fixed_replay.py'),OUT/'protocol.json',INPUT,*[ROOT/p for p in SOURCES.values()],*[RAW/f'pas_{s}.json' for s in SYMBOLS],MINUTE/'dataset.json',*MINUTE.glob('raw/*.json')]
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths});print('FROZEN PASpairmodule72h',flush=True)


def verify():
    verify_calibration();verify_net()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def prepare():
    verify();raw=dict(np.load(INPUT));profile=load_profile();cfg,minute=load(MINUTE)
    t=np.arange(START,END+1,3600);ix=np.searchsorted(raw['market_times'],t);assert np.array_equal(raw['market_times'][ix],t)
    shape=(len(t),10);close=np.full(shape,np.nan);volume=np.zeros(shape);trades=np.zeros(shape);fund=np.zeros(shape);observed=np.zeros(shape,bool)
    known=np.zeros(shape);fundprice=np.zeros(shape);stamps=np.full(len(t),-1,np.int64);audit=[]
    initial=int(np.searchsorted(minute['times'],START*1000))
    close[0]=minute['close'][initial]
    # Sourcevolume is from completedprior60minutes, preserving absentmarkets.
    before=(minute['times']>=(START-3600)*1000)&(minute['times']<=START*1000)
    volume[0]=minute['volume'][before].sum(0);trades[0]=minute['active'][before].sum(0)
    for j,symbol in enumerate(SYMBOLS):
        data=read(RAW/f'pas_{symbol}.json');receipt=datetime.fromisoformat(data['candles']['received_kst']).timestamp()*1000
        for c in data['candles']['response']:
            end=(int(c['t'])//3600000+1)*3600;k=int(np.searchsorted(t,end))
            if k<len(t) and t[k]==end and int(c['T'])<receipt:
                if c['n']>0 and float(c['v'])>0:close[k,j]=float(c['c'])
                volume[k,j]=float(c['v']);trades[k,j]=c['n']
        funding=data['funding_history']['response']
        for k,now in enumerate(t):
            prior=[r for r in funding if int(r['time'])<now*1000 and int(r['time'])>=(now-24*3600)*1000]
            if prior:
                known[k,j]=.5*float(prior[-1]['fundingRate'])+.5*np.mean([float(r['fundingRate']) for r in prior])
            arrived=[r for r in funding if (now-3600)*1000<int(r['time'])<=now*1000]
            if k and arrived:
                assert len(arrived)==1
                r=arrived[0];fund[k,j]=float(r['fundingRate']);observed[k,j]=True;stamps[k]=max(stamps[k],int(r['time']))
                ref=int(np.searchsorted(raw['market_times'],int(r['time'])//3600000*3600));fundprice[k,j]=raw['close'][ref,j]
            audit.append(dict(symbol=symbol,decision_source=int(now),known_funding_last_ms=int(prior[-1]['time']) if prior else None,settled_ms=int(arrived[0]['time']) if k and arrived else None))
        last=np.nan
        for k in range(len(t)):
            if np.isfinite(close[k,j]):last=close[k,j]
            else:close[k,j]=last
    priced=np.isfinite(close)&(close>0)
    assert np.all((volume>0)<=priced)
    assert observed[1:].all(),'Missing actualsettlement: donotpretendzerofunding'
    close=np.where(priced,close,1.) # only for bookkeeping of zero unpricedinventory
    market=Market(raw['market_times'],raw['close'],raw['volume'],raw['trades'],np.zeros_like(raw['close']),np.zeros_like(raw['close'],bool),tuple(SYMBOLS))
    sigma=np.array([covariance(market,int(i),profile['trading']['risk'])[1] for i in ix])
    d=dict(times=t,par_close=close,reference_close=raw['close'][ix],par_volume=volume,par_trades=trades,
        funding=fund,funding_observed=observed,known_hourly_funding=known,funding_cutover=np.array(START-1),funding_event_ms=stamps,funding_prices=fundprice,
        first=np.zeros(shape),second=np.zeros(shape),retention=np.ones(shape),sigma=sigma)
    np.savez_compressed(OUT/'dataset.npz',**d);save(OUT/'funding_timing.json',audit)
    for arm,path in SOURCES.items():
        state=dict(np.load(ROOT/path));st=state.pop('times');ii=np.searchsorted(st,t);assert np.array_equal(st[ii],t)
        state={k:(v[ii] if v.ndim and len(v)==len(st) else v) for k,v in state.items()}
        if arm=='fixed_two_24':
            from auto_pair_selection.signals import state_for_arm
            state=state_for_arm(state,'fixed_two_rebuilt')
        if arm.startswith(('hierarchical','carry')):state['source_prices']=close.copy()
        pairs=state['pairs'];visible=np.array([priced[:,a]&priced[:,b]&(volume[:,a]>0)&(volume[:,b]>0) for a,b in pairs]).T
        state['directions']=state['directions']*visible
        np.savez_compressed(OUT/f'{arm}.npz',times=t,**state)
    verify();print('PASprepared',len(t),'hours;settlements',int(observed.sum()),flush=True)


def job(arm,scenario):
    from .fixed_replay import replay
    from pas_allocation_study.study import summarize
    d=dict(np.load(OUT/'dataset.npz'));state=dict(np.load(OUT/f'{arm}.npz'));state.pop('times');profile=load_profile()
    cfg=copy.deepcopy(profile['configs']['reference']);cfg['dense_pairs']=[list(p) for p in PAIRS];cfg['conditional_completion']={'risk_aversion':0.}
    cfg['scenarios']['no_receipts']={**cfg['scenarios']['base'],'funding_receipt_multiplier':0.}
    h=int(arm.rsplit('_',1)[1]);renewal=arm.startswith(('fixed_two','adaptive'))
    name=f'pairs_dense_hourlyhedge_individual_{h}' if renewal else f'pairs_dense_individual_{h}'
    r=replay(d,state,None,cfg,profile['trading'],name,scenario,START,END)
    assert all(f['decision_time']<f['time'] for f in r['fills'])
    assert not r['owner_fills']['base']
    row=summarize(r,arm,'PAS72h_pair_module',scenario);row.update(official_Q=False,conditional_execution=True,base_included=False)
    assert row['accounting_error']<1e-6 and row['risk_breach_bars']==0
    path=OUT/'runs'/f'{arm}_{scenario}.json.gz';path.parent.mkdir(exist_ok=True)
    with gzip.open(path,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    save(OUT/'rows'/f'{arm}_{scenario}.json',row);return row


def run():
    verify()
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(job,a,s) for a in SOURCES for s in ['base','stress','no_receipts']]
        for done in as_completed(jobs):
            r=done.result();print(json.dumps({k:r[k] for k in ['arm','scenario','net_pnl','Q','completed_pairs','funding','fees']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
