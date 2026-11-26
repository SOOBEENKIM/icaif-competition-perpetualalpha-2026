"""Single PAS writer. Default is unsigned inspection; --execute is explicit.

Run the independent producer alongside this worker. No code at import loads a
signer. Recovered unknown orders or externally altered inventory stop execution.
"""
import argparse
import fcntl
import json
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
import numpy as np

from bot.connection import signer_from_key
from bot.execution import OutcomeUnknown
from experiments.published_pair_lifecycle_20261017.signals import validate_envelope
from .candidate import build_candidate
from .execution import build_order_client,PASExecutor
from .live_account import LiveAccount
from .profile import ROOT,load_profile,verify_bundle
from .runtime import json_save,fingerprint,save_coordinator,load_coordinator
from .venue import PublicPAS,account_equity,validate_metadata,phase_state


def fresh_preflight(public,profile):
    result=public.account()
    for key,payload in {
        'meta':{'type':'meta','dex':'pas'},
        'spot_meta':{'type':'spotMeta'},
        'limits':{'type':'perpDexLimits','dex':'pas'},
        'fees':{'type':'userFees','user':profile['team_wallet']},
        'agents':{'type':'extraAgents','user':profile['team_wallet']},
    }.items():result[key]=public.post(payload)
    result['queried_at_unix']=time.time()
    validate_metadata(result,profile)
    return result


def require_initial_account(snapshot,profile):
    value=account_equity(snapshot,profile)
    if not value['funded']:raise ValueError('Waiting for actual USDP funding')
    if value['positions'] or value['open_orders']:raise ValueError('A new PAS ledger requires a flat account')
    if abs(value['equity']-profile['initial_capital'])>.01:
        raise ValueError('Actual initial capital differs from 100,000 USDP')
    return value


def reconcile_account(account,snapshot,profile):
    value=account_equity(snapshot,profile)
    if snapshot['orders'] or account.pending_orders or account.unknown:
        raise OutcomeUnknown('Unresolved orders require reconciliation; no automatic retransmission')
    rows={x['position']['coin']:x['position'] for x in snapshot['positions']['assetPositions']}
    qty=np.array([float(rows.get('pas:'+s,{}).get('szi',0)) for s in profile['trading']['symbols']])
    if not np.allclose(qty,account.physical.qty,rtol=0,atol=1e-9):
        raise OutcomeUnknown('Exchange inventory differs from the checkpoint')
    # Compare cash rather than marks sampled at different instants. Check entry
    # prices too, so an unrecorded closing/reopening cannot pass qty equality.
    for j,symbol in enumerate(profile['trading']['symbols']):
        if qty[j] and abs(float(rows['pas:'+symbol]['entryPx'])-account.physical.entry[j])*abs(qty[j])>.05:
            raise OutcomeUnknown('Exchange average entry differs from actual fill ledger')
    unrealized=sum(float(row['unrealizedPnl']) for row in rows.values())
    cash=value['equity']-unrealized
    if abs(cash-account.physical.cash)>.05:
        raise OutcomeUnknown('USDP cash differs: reconcile actual fees, funding or transfers')
    return value


def load_publication(path,profile,now):
    if not path.exists():return None,[]
    data=json.loads(path.read_text())
    if data['candidate']!=profile['candidate'] or data['profile_hash']!=fingerprint(profile):
        raise ValueError('Signal profile differs from the selected PAS candidate')
    frame=validate_envelope(data['publication'])
    if data['published_at']>now or frame['published_time']>now:
        return None,[]
    forecasts=data['forecasts']
    for f in forecasts:
        if f['pair'] not in profile['pairs'] or f['forecast_source']!=frame['source_time']:
            raise ValueError('Forecast pair/source mismatch')
        if f['available_time']<data['published_at']:
            raise ValueError('Forecast predates completed inference')
    return data['publication'],forecasts


def collect_books(profile):
    # Independent sessions avoid sharing requests.Session across threads.
    def book(symbol):
        api=PublicPAS(profile)
        try:
            value=api.post({'type':'l2Book','coin':'pas:'+symbol})
            if value['coin']!='pas:'+symbol:raise ValueError('Book market mismatch')
            return symbol,int(time.time()*1000),value
        finally:api.session.close()
    with ThreadPoolExecutor(max_workers=5) as pool:books=list(pool.map(book,profile['trading']['symbols']))
    api=PublicPAS(profile)
    try:meta,contexts=api.market();received=int(time.time()*1000)
    finally:api.session.close()
    if [m['name'] for m in meta['universe']]!=['pas:'+s for s in profile['trading']['symbols']]:
        raise ValueError('Unexpected PAS market universe')
    if meta['collateralToken']!=profile['collateral_token']:raise ValueError('Unexpected collateral')
    by_coin={m['name']:c for m,c in zip(meta['universe'],contexts)}
    return {s:dict(book=[(at,0,b)],book_times=[at],ctx=[(received,0,by_coin['pas:'+s])],
        ctx_times=[received],resets=[]) for s,at,b in books}


def history(public,kind,profile,start_ms,end_ms):
    rows=[];cursor=int(start_ms)
    while cursor<=end_ms:
        batch=public.post(dict(type=kind,user=profile['team_wallet'],startTime=cursor,endTime=int(end_ms)))
        if not isinstance(batch,list):raise ValueError('Invalid account history')
        rows.extend(batch)
        # API returns at most 500 funding / 2000 fill rows. Fail closed on a
        # saturated identical timestamp rather than silently omit an event.
        maximum=500 if kind=='userFunding' else 2000
        if len(batch)<maximum:break
        latest=max(row['time'] for row in batch)
        if latest<=cursor:raise ValueError('Saturated history timestamp requires reconciliation')
        cursor=latest # overlap; callers deduplicate identities
    return rows


class LiveRunner:
    def __init__(self,profile,directory,public,executor):
        self.profile=profile;self.directory=directory;self.public=public;self.executor=executor
        self.state_path=directory/'coordinator.pkl';self.portfolio=None
        self.last_funding=0.;self.funding_cursor=profile['start_unix']*1000
        if self.state_path.exists():
            self.portfolio=load_coordinator(self.state_path,profile,executor,self.checkpoint,self.fills,self.settlement)
            if self.portfolio.account.pending_orders or self.portfolio.account.unknown:
                raise OutcomeUnknown('Recovered unresolved order; inspect journal and exchange by cloid first')
        else:
            require_initial_account(public.account(),profile)
            account=LiveAccount(profile,executor,self.checkpoint,self.fills)
            self.portfolio=build_candidate(profile,self.settlement,account=account)['portfolio']
            self.checkpoint('initial_USDP_grant',{})

    def checkpoint(self,reason,detail):
        if self.portfolio is not None:
            save_coordinator(self.state_path,self.portfolio,self.profile,reason)

    def fills(self,oid,start_ms,expected):
        for attempt in range(6):
            rows=history(self.public,'userFillsByTime',self.profile,start_ms-2000,int(time.time()*1000))
            matching={(r['coin'],r['tid']):r for r in rows if r['oid']==oid and r['coin'].startswith('pas:')}
            result=list(matching.values())
            total=sum((Decimal(r['sz']) for r in result),Decimal(0))
            if total>=Decimal(str(expected)):return result
            if attempt<5:time.sleep(1)
        return result

    def settlement(self,at_ms,account):
        if time.time()-self.last_funding<15:return
        end=int(time.time()*1000)
        rows=history(self.public,'userFunding',self.profile,self.funding_cursor,end)
        account.apply_actual_funding(rows)
        # Allow delayed history publication and deduplicate confirmed payments.
        self.funding_cursor=max(self.profile['start_unix']*1000,end-3600000)
        self.last_funding=time.time()

    def tick(self):
        p=self.portfolio
        self.settlement(int(time.time()*1000),p.account)
        snapshot=self.public.account()
        value=reconcile_account(p.account,snapshot,self.profile)
        streams=collect_books(self.profile)
        # Do not truncate seconds: a just-received book would appear to be in
        # the future relative to a floored timestamp on nearly every poll.
        now=time.time()
        publication,forecasts=load_publication(self.directory/'signal.json',self.profile,now)
        p.step(now,streams,publication,forecasts)
        self.checkpoint('tick_complete',{})
        report=dict(at=time.time(),state='running',phase=phase_state(time.time(),self.profile),
            candidate=self.profile['candidate'],equity_before_tick=value['equity'],
            last_source=publication['data']['frame']['source_time'] if publication else None,
            qty=p.account.physical.qty.tolist(),fees=p.account.physical.fees,
            funding=p.account.physical.funding,fills=len(p.account.physical.fills),
            last_baseline_decision=p.last_decision,pair_state=p.pair.state if p.pair else None,
            orders_enabled=True,official_score=False,snapshot_time=int(snapshot['positions']['time']))
        json_save(self.directory/'status.json',report)
        # Timestamp is the actual account observation, never a fabricated exact
        # 12:00 sample. The official leaderboard remains authoritative.
        fd=os.open(self.directory/'account_observations.jsonl',os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'a') as f:f.write(json.dumps(report)+'\n')
        return report


def explicit_signer(workspace,profile,snapshot):
    settings=json.loads((workspace/'config.local.json').read_text())
    if settings.get('network')!='testnet' or settings['team_wallet'].lower()!=profile['team_wallet'].lower():
        raise ValueError('API identity does not belong to the selected team/testnet')
    if not any(a.get('address','').lower()==settings['api_wallet'].lower()
            and a.get('validUntil',0)>time.time()*1000 for a in snapshot['agents']):
        raise ValueError('API approval missing or expired')
    directory=workspace/'.secrets';info=directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700:
        raise ValueError('API key directory permissions are invalid')
    fd=os.open(directory/'api_wallet.key',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd) as f:
        info=os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
            raise ValueError('API key file permissions are invalid')
        return signer_from_key(f.read(128),settings)


def main(execute=False,once=False,workspace=None,directory=None):
    workspace=Path(workspace or ROOT).resolve();profile=load_profile();verify_bundle()
    directory=Path(directory or workspace/'.runtime/pas_live');directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    public=PublicPAS(profile)
    if not execute:
        snapshot=fresh_preflight(public,profile)
        result=dict(at=time.time(),state='read_only',phase=phase_state(time.time(),profile),
            account=account_equity(snapshot,profile),orders_sent=0,signer_loaded=False)
        json_save(directory/'inspection.json',result);print(json.dumps(result));return
    with ExitStack() as stack:
        # The previous practice daemon uses this exact lock. Hold it too so
        # practice and scored writers cannot use this API wallet concurrently.
        for path in (workspace/'practice_live/live_runtime/process.lock',directory/'process.lock',workspace/'.runtime/wallet_writer.lock'):
            path.parent.mkdir(parents=True,exist_ok=True)
            f=stack.enter_context(path.open('a+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner=None
        while not (directory/'STOP').exists():
            now=time.time();phase=phase_state(now,profile)
            if phase=='ended':
                json_save(directory/'status.json',dict(at=now,state='ended',orders_enabled=False,
                    residual_qty=runner.portfolio.account.physical.qty.tolist() if runner else None));return
            if phase=='before_start':
                json_save(directory/'status.json',dict(at=now,state='waiting_for_phase_start',orders_enabled=False))
            else:
                try:
                    if runner is None:
                        snapshot=fresh_preflight(public,profile)
                        if not account_equity(snapshot,profile)['funded']:
                            json_save(directory/'status.json',dict(at=now,state='waiting_for_USDP_grant',orders_enabled=False))
                            if once:return
                            time.sleep(15);continue
                        if not (directory/'coordinator.pkl').exists():require_initial_account(snapshot,profile)
                        signer=explicit_signer(workspace,profile,snapshot)
                        executor=PASExecutor(build_order_client(profile,signer,enable_orders=True),profile,directory/'orders.jsonl')
                        runner=LiveRunner(profile,directory,public,executor)
                    runner.tick()
                except Exception as error:
                    json_save(directory/'status.json',dict(at=time.time(),state='paused_requires_review',
                        error=str(error),orders_enabled=False))
                    raise
            if once:return
            time.sleep(15)
        json_save(directory/'status.json',dict(at=time.time(),state='stopped',orders_enabled=False,
            residual_qty=runner.portfolio.account.physical.qty.tolist() if runner else None))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--once',action='store_true')
    p.add_argument('--workspace');p.add_argument('--directory');a=p.parse_args()
    main(a.execute,a.once,a.workspace,a.directory)
