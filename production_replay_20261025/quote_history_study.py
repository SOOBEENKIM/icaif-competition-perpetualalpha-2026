"""First-use quote-history feature assessment, frozen before06UTC."""
import copy,gzip,json,sys,time
from datetime import datetime,timezone
from .protocol import ROOT,OUT,save,sha

DEST=OUT/'quote_history';START=1792908000;END=1792915500;WARM=1792897500


def freeze():
    assert time.time()<START and not (DEST/'protocol.json').exists()
    paths=[ROOT/'production_replay_20261025'/f for f in ['quote_history.py','quote_history_study.py','quote_guard.py','book_replay.py','liquidation_audit.py']]
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),start=START,end=END,warm_start=WARM,
        primary='quote_inputs',diagnostic='history_only',controls=['incumbent','frozen_side_quote'],
        mechanism='Keep original3hOR24h symmetric pullback rule but use actual received PAS bid/ask mid histories at source and lag endpoints. Absence of recent trades no longer implies absence of valid quote history. Every endpoint requires receivedbook<=source,serverage<=15s,contextage<=3s,consistentepoch. Missing24hhistory staysmissing; valid3hpoint may qualify under the originalORrule. No gap interpolation or mainnet substitution.',
        primary_changes='Quotehistory plus separately frozen1hdecision-side-price entry guard. Diagnostic changeshistoryonly withoriginal15min/currentmark execution. Normal6hdecisionwindow,models,all10symbols,50kcapital,protectedcuts,actualinventory andhardlimits unchanged.',
        rationale='Prior actual decision trace: SP500target rejected7times asECONOMIC_LIMIT onOct24 06UTC; multipleotherassets additions vetoed because lasttrade age>1h despite quote continuity in old PAS captures. These are observed restrictions, not proof of profitable alternative.',
        validity='First-use outcome interval06-08:05; pre06capture used only as lagged input, not a selected PnL outcome. Short conditional comparison, overlaps other preregistered tests. No claim of long independent validation or officialQ. Old captures too short for3hfeatures; do not fabricate full historical quote backtests.',
        acceptance='Primary positive aftercostPnL and terminaldepthliquidationPnL, higher than BOTHcontrols baseandstress with no accounting/risk/timestamp/missingcash failures. Diagnostic cannot replaceprimary afteroutcome.',
        hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},seed_sha=sha(OUT/'forward_seed.pkl'),orders_sent=0))


def check():
    p=json.loads((DEST/'protocol.json').read_text())
    for path,h in p['hashes'].items():assert sha(ROOT/path)==h,path
    assert sha(OUT/'forward_seed.pkl')==p['seed_sha'];return p


def future():
    check()
    while time.time()<END+1080:
        save(DEST/'status.json',dict(at=time.time(),state='waiting_for_fixed_end',orders_sent=0))
        time.sleep(min(30,max(1,END+1080-time.time())))
    check();assert not (DEST/'started.json').exists();save(DEST/'started.json',dict(at=time.time()))
    from competition_live.profile import load_profile
    from competition_live.runtime import private_load
    from .book_replay import load_books,run_book
    from .quote_history import Activated
    from .liquidation_audit import compare_at_common_end
    profile=load_profile();streams,audit=load_books((OUT/'capture').glob('*.jsonl.gz'),END)
    save(DEST/'capture_manifest.json',audit)
    source=sorted([json.loads(p.read_text()) for p in (OUT/'forward_signals').glob('*.json')],key=lambda r:r['received_at'])
    funding=[]
    for s in profile['trading']['symbols']:
        funding.extend(dict(r,coin='pas:'+s) for r in json.loads((OUT/'forward_evaluation'/f'{s}_funding.json').read_text())['rows'] if WARM*1000<r['time']<=END*1000)
    seed=private_load(OUT/'forward_seed.pkl')['portfolio'];rows=[]
    for scenario,sc in dict(base=dict(fee_bps=9.,depth_fraction=1.,latency_ms=1000,receipt_multiplier=1.),
                            stress=dict(fee_bps=12.,depth_fraction=.5,latency_ms=3000,receipt_multiplier=.5)).items():
        results={}
        for arm in ['incumbent','frozen_side_quote']:
            with gzip.open(OUT/'quote_anchor/future'/f'{arm}_{scenario}.json.gz','rt') as fp:results[arm]=json.load(fp)
        prefix=[s for s in results['incumbent']['snapshots'] if s['time']<START]
        for arm in ['history_only','quote_inputs']:
            state=copy.deepcopy(seed);state.__class__=Activated;state.activation=START;state.quote_mode=arm
            r=run_book(streams,source,profile,sc,WARM,END,state,'incumbent',funding);r['arm']=arm
            assert [s for s in r['snapshots'] if s['time']<START]==prefix,'Preactivation inventory diverged'
            results[arm]=r
            with gzip.open(DEST/f'{arm}_{scenario}.json.gz','wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
        for r in compare_at_common_end(streams,profile,sc,results,START,END):
            r['scenario']=scenario;r['fills']=sum(o['filled']>0 and o['arrival']>=START for o in results[r['arm']]['orders'])
            rows.append(r)
        save(DEST/'results.json',rows)
    save(DEST/'status.json',dict(at=time.time(),state='complete',orders_sent=0))


if __name__=='__main__':globals()[sys.argv[1]]()
