"""Predeclared executable-side anchor experiment, with three-way controls."""
import copy,gzip,json,sys,time
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
from unittest.mock import patch
from candidate_handoff.portfolio import CandidatePortfolio
from execution_revision_20261024.guarded_portfolio import GuardedPortfolio,GuardedAccount
from .quote_guard import QuotePortfolio,QuoteAccount,adopt
from .protocol import ROOT,OUT,DATA,save,sha

DEST=OUT/'quote_anchor';START=1792908000;END=1792915500;WARM=1792897500;SWITCH=1792767600


class Activated(CandidatePortfolio):
    def step_candidate(self,now,streams,packet,profile):
        if now<self.activation:return CandidatePortfolio.step_candidate(self,now,streams,packet,profile)
        if self.branch=='frozen_side_quote':
            self.__class__=QuotePortfolio;self.account.__class__=QuoteAccount;self.account.frozen_side_prices=None
        elif self.branch=='fixed_entry_1h':
            self.__class__=GuardedPortfolio;self.account.__class__=GuardedAccount;self.account.base_entry_anchor=None
        else:raise ValueError('Unknown predeclared branch')
        self.base_task=None
        return self.step_candidate(now,streams,packet,profile)


def freeze():
    assert time.time()<START and not (DEST/'protocol.json').exists()
    files=[ROOT/'production_replay_20261025'/n for n in ['quote_guard.py','quote_study.py','book_replay.py','liquidation_audit.py']]
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),start=START,end=END,warm_start=WARM,
        primary='frozen_side_quote',controls=['incumbent','fixed_entry_1h'],
        hypothesis='Marks can differ from executable bid/ask. Preserve the initial executable-side quote across a1hintent, plus the existing10bp band and any tighter owner economic limit. No later-price chasing. This may improve participation or worsen paid spreads; neither sign assumed. Same signal/risk/50kallocation, no named assets, no refit.',
        historical='Previously inspected120h development, full path and identical incumbent inventory switch. No new independent claim.',
        future='Freeze before06UTC; identical03:05-06prefix from real own inventory. Branch at06 and assess at08:05. Controls isolate1hlifetime from decision-quote anchor. No peeking/parameter revision onthisinterval; overlap withotherexperiments is disclosed.',
        acceptance='Higher own net PnL and paid terminal-exit PnL than BOTH controls in base and stress, no risk/account/time/price-limit failures. PositivecandidatePnL. Short conditional paper evidence only; no officialQ or automaticdeployment.',
        hashes={str(p.relative_to(ROOT)):sha(p) for p in files},seed_hash=sha(OUT/'forward_seed.pkl'),orders_sent=0))


def check():
    p=json.loads((DEST/'protocol.json').read_text())
    for f,h in p['hashes'].items():assert sha(ROOT/f)==h,f
    assert sha(OUT/'forward_seed.pkl')==p['seed_hash']
    return p


def historical_job(scenario,switch):
    from competition_live.profile import load_profile
    from pas_strategy_audit.data import load
    from . import engine
    check();cfg,d=load(DATA);profile=load_profile()
    source=json.loads((OUT/'packets.json').read_text());funding=json.loads((DATA/'funding.json').read_text())
    sc=json.loads((OUT/'protocol.json').read_text())['scenarios'][scenario];original=engine.build
    def build(v,ignored):
        p=original(v,'incumbent')
        if switch:
            p.__class__=Activated;p.activation=SWITCH;p.branch='frozen_side_quote'
            return p
        return adopt(p)
    with patch.object(engine,'build',build):r=engine.run(d,profile,sc,funding,source,cfg['start_ms']//1000,cfg['cutoff_ms']//1000)
    a=next(s for s in r['snapshots'] if s['time']==SWITCH);b=r['snapshots'][-1]
    row=dict(arm='frozen_side_quote',scenario=scenario,switch=switch,**r['metrics'],
             tail_pnl=b['equity']-a['equity'],fills=len(r['fills']),risk_violations=len(r['audit']['risk_violations']),
             independent_test=False,ranking_eligible=len(r['fills'])>0 and r['metrics']['completed_daily_returns']>=2)
    with gzip.open(DEST/f'historical_{scenario}_{switch}.json.gz','wt') as fp:json.dump(dict(row=row,result=r),fp,allow_nan=False,default=lambda x:x.tolist())
    return row


def historical():
    check();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(historical_job,s,w) for s in ['base','stress'] for w in [False,True]]):
            row=f.result();rows.append(row);save(DEST/'historical_results.json',rows);print(row,flush=True)


def future():
    check()
    while time.time()<END+720:
        save(DEST/'future_status.json',dict(at=time.time(),state='waiting_for_fixed_end',start=START,end=END,orders_sent=0))
        time.sleep(min(30,max(1,END+720-time.time())))
    check();dest=DEST/'future';dest.mkdir(exist_ok=True)
    assert not (dest/'started.json').exists();save(dest/'started.json',dict(at=time.time()))
    from competition_live.profile import load_profile
    from competition_live.runtime import private_load
    from .book_replay import load_books,run_book
    from .liquidation_audit import compare_at_common_end
    profile=load_profile();streams,audit=load_books((OUT/'capture').glob('*.jsonl.gz'),END)
    save(dest/'capture_manifest.json',audit)
    source=sorted([json.loads(p.read_text()) for p in (OUT/'forward_signals').glob('*.json')],key=lambda r:r['received_at'])
    funding=[]
    for s in profile['trading']['symbols']:
        funding.extend(dict(r,coin='pas:'+s) for r in json.loads((OUT/'forward_evaluation'/f'{s}_funding.json').read_text())['rows'] if WARM*1000<r['time']<=END*1000)
    seed=private_load(OUT/'forward_seed.pkl')['portfolio'];rows=[];audits=[]
    for scenario,sc in dict(base=dict(fee_bps=9.,depth_fraction=1.,latency_ms=1000,receipt_multiplier=1.),
                            stress=dict(fee_bps=12.,depth_fraction=.5,latency_ms=3000,receipt_multiplier=.5)).items():
        results={};prefix=None
        for arm in ['incumbent','fixed_entry_1h','frozen_side_quote']:
            state=copy.deepcopy(seed)
            if arm!='incumbent':state.__class__=Activated;state.activation=START;state.branch=arm
            r=run_book(streams,source,profile,sc,WARM,END,state,'incumbent',funding);r['arm']=arm
            before=[s for s in r['snapshots'] if s['time']<START]
            if prefix is None:prefix=before
            else:assert before==prefix,'Inventory changed before permitted activation'
            results[arm]=r
            with gzip.open(dest/f'{arm}_{scenario}.json.gz','wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
        for row in compare_at_common_end(streams,profile,sc,results,START,END):
            row['scenario']=scenario;row['fills']=sum(o['filled']>0 and o['arrival']>=START for o in results[row['arm']]['orders'])
            rows.append(row)
        save(dest/'results.json',rows)
    save(DEST/'future_status.json',dict(at=time.time(),state='complete',orders_sent=0))


if __name__=='__main__':globals()[sys.argv[1]]()
