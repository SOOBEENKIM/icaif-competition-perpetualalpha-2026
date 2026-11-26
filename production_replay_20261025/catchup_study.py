"""Scheduling candidate: exposed historical diagnostic plus fixed future test."""
import copy,gzip,json,sys,time
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
from unittest.mock import patch
from competition_live.profile import load_profile
from competition_live.runtime import private_load
from candidate_handoff.portfolio import CandidatePortfolio
from execution_revision_20261024.guarded_portfolio import GuardedAccount
from .protocol import ROOT,OUT,DATA,save,sha
from .catchup_portfolio import CatchupPortfolio,adopt

DEST=OUT/'catchup'
START=int(datetime(2026,10,25,5,tzinfo=timezone.utc).timestamp())
END=int(datetime(2026,10,25,8,5,tzinfo=timezone.utc).timestamp())
WARM=int(datetime(2026,10,25,3,5,tzinfo=timezone.utc).timestamp())


class ActivatedCatchup(CatchupPortfolio):
    def step_candidate(self,now,streams,packet,profile):
        if now<self.research_activation:return CandidatePortfolio.step_candidate(self,now,streams,packet,profile)
        if not getattr(self,'research_activated',False):
            self.account.__class__=GuardedAccount;self.account.base_entry_anchor=None;self.base_task=None
            self.research_activated=True
        return CatchupPortfolio.step_candidate(self,now,streams,packet,profile)


def freeze():
    assert time.time()<START
    assert not (DEST/'protocol.json').exists()
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='fresh_block_catchup',control='incumbent',
        mechanism='Once per original6hblock, first available completed-current-hour packet may trigger a normal decision, even if exactboundarypacket missing/over15minlate. Actualnow, currentPASmarks, same target logic and10bp fixed entry guard. Entry lifetime<=1hfromdecision and never beyondblockend; reduction-only remainder. A block already decided cannot decide again.',
        evidence='04UTC Oct25 source observed late by~21minutes; retrospectiveOct24 18UTCsource absent. Historical invariant tests use original failing schedule pattern, not inferred winning asset returns.',
        future_start=START,future_end=END,common_prefix_start=WARM,
        prospective='Freeze before05UTC. Replay both existing actual-inventory seed paths identically03:05-05, branch onlyat05. Fixedend08:05, compareP&Lfrom05andpaidterminaldepthreserve inbase/stress. No intraday officialQ; no model or threshold selection from this interval.',
        history='All Oct19-Oct24 plus reserved8h are now exposed development. Report all historical failures and candidate variants separately.',
        bounds='Same own50k,reserved50k,hardlimits,signals,10assets,retainedspecialist. SharedwalletteammateP&L excluded. No realorders or servicechanges.',
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'production_replay_20261025/catchup_portfolio.py',ROOT/'production_replay_20261025/catchup_study.py',ROOT/'production_replay_20261025/book_replay.py']},
        seed_hash=sha(OUT/'forward_seed.pkl'),orders_sent=0))


def check():
    p=json.loads((DEST/'protocol.json').read_text())
    for file,h in p['hashes'].items():assert sha(ROOT/file)==h,file
    assert sha(OUT/'forward_seed.pkl')==p['seed_hash'];return p


def historical_job(directory,arm,scenario):
    from pas_strategy_audit.data import load
    from . import engine
    check();directory=ROOT/directory;cfg,d=load(directory)
    packet_path=OUT/'packets.json' if directory==DATA else directory/'packets.json'
    packets=json.loads(packet_path.read_text());funding=json.loads((directory/'funding.json').read_text())
    sc=json.loads((OUT/'protocol.json').read_text())['scenarios'][scenario]
    original=engine.build
    def build(venue,ignored):
        p=original(venue,'incumbent')
        return adopt(p) if arm=='fresh_block_catchup' else p
    with patch.object(engine,'build',build):r=engine.run(d,load_profile(),sc,funding,packets,cfg['start_ms']//1000,cfg['cutoff_ms']//1000)
    label='development120h' if directory==DATA else 'exposed8h'
    row=dict(arm=arm,scenario=scenario,period=label,**r['metrics'],fills=len(r['fills']),risk_violations=len(r['audit']['risk_violations']),
        decisions=sum(e['kind'] in ['fresh_block_catchup_decision','budget_lifecycle_decision'] for e in r['events']),
        independent_test=False)
    with gzip.open(DEST/f'{label}_{arm}_{scenario}.json.gz','wt') as fp:json.dump(dict(row=row,result=r),fp,allow_nan=False,default=lambda x:x.tolist())
    return row


def historical():
    check();rows=[];directories=[str(DATA.relative_to(ROOT)),str((OUT/'reserved_eight_hours').relative_to(ROOT))]
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks=[(d,a,s) for d in directories for a in ['incumbent','fresh_block_catchup'] for s in ['base','stress']]
        for f in as_completed([pool.submit(historical_job,*t) for t in tasks]):
            r=f.result();rows.append(r);save(DEST/'historical_results.json',rows);print(r,flush=True)


def future():
    check()
    while time.time()<END+360:
        save(DEST/'future_status.json',dict(at=time.time(),state='waiting_for_fixed_future_end',start=START,end=END,orders_sent=0))
        time.sleep(min(30,max(1,END+360-time.time())))
    check();dest=DEST/'future';dest.mkdir(exist_ok=True)
    assert not (dest/'started.json').exists()
    save(dest/'started.json',dict(at=datetime.now(timezone.utc).isoformat(),orders_sent=0))
    from .book_replay import load_books,run_book
    source=sorted([json.loads(p.read_text()) for p in (OUT/'forward_signals').glob('*.json')],key=lambda r:r['received_at'])
    streams,audit=load_books((OUT/'capture').glob('*.jsonl.gz'),END);save(dest/'capture_audit.json',audit)
    profile=load_profile();funding=[]
    for symbol in profile['trading']['symbols']:
        rows=json.loads((OUT/'forward_evaluation'/f'{symbol}_funding.json').read_text())['rows']
        funding.extend(dict(r,coin='pas:'+symbol) for r in rows if WARM*1000<r['time']<=END*1000)
    seed=private_load(OUT/'forward_seed.pkl')['portfolio'];rows=[]
    for scenario,sc in dict(base=dict(fee_bps=9.,depth_fraction=1.,latency_ms=1000,receipt_multiplier=1.),stress=dict(fee_bps=12.,depth_fraction=.5,latency_ms=3000,receipt_multiplier=.5)).items():
        prefix=None
        for arm in ['incumbent','fresh_block_catchup']:
            state=copy.deepcopy(seed)
            if arm!='incumbent':state.__class__=ActivatedCatchup;state.research_activation=START
            r=run_book(streams,source,profile,sc,WARM,END,state,'incumbent',funding);r['arm']=arm
            before=[s for s in r['snapshots'] if s['time']<START]
            if prefix is None:prefix=before
            else:assert before==prefix,'Counterfactual inventories diverged before activation'
            selected=[s for s in r['snapshots'] if s['time']>=START];a,b=selected[0],selected[-1]
            row=dict(arm=arm,scenario=scenario,net_pnl=b['equity']-a['equity'],fees=b['fees']-a['fees'],funding=b['funding']-a['funding'],
                exposure_times=[a['time'],b['time']],fills=sum(x['time']>=START for x in r['fills']),
                failures=len(r['failures']),risk_violations=len(r['risk_violations']),accounting_error=r['accounting_error'],
                unknown=r['unknown'],pending_orders=r['pending_orders'],fatal=r['fatal'],official_Q=None,orders_sent=0)
            with gzip.open(dest/f'{arm}_{scenario}.json.gz','wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
            rows.append(row);save(dest/'results.json',rows);print(row,flush=True)
    save(DEST/'future_status.json',dict(at=time.time(),state='complete',orders_sent=0))

if __name__=='__main__':globals()[sys.argv[1]]()
