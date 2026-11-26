"""Single-writer testnet handoff; frozen candidate and independent inference.

Reuses the incumbent wallet, durable order claims, recovery, and execution code.
No transfers, mainnet, duplicate writer, retrospective round, or parameter fit.
"""
import argparse,fcntl,json,os,signal,sqlite3,subprocess,sys,time
from pathlib import Path
import numpy as np
from bot.connection import build_client,check_connection,close_client,load_settings
from practice_live.runner import Runner,COINS,SYMBOLS,HOUR,NORMAL_WINDOW_SECONDS,POLL_SECONDS,stamp_kst
from practice_live.live_store import Store
from practice_live.live_logic import identity
from practice_live.infer import protected_hashes
from practice_live.refresh import ROOT,OUT as LIVE,iso,save_json,sha
from replay_lab.common_comparison.engine import risk_breaches,risk_values
from strategy.agent import TAIL
from experiments.risk_preserving_20261016.policy import RiskCapController
from .live_signal import DIRECTORY,PUBLICATION,PROFILE,read_publication

SERVICE='icaif-par-practice-budget.service'
EPOCH='risk_cap_budget_matched_v1_20261016'

def verify_deployment():
    manifest=json.loads((DIRECTORY/'deployment_manifest.json').read_text())
    for path,digest in manifest['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen deployment file changed: '+path)
    return sha(DIRECTORY/'deployment_manifest.json')

def epoch_snapshot(account,store):
    ledger=account['ledger'];oracle=ledger['equity']+sum(account['qty'][j]*(float(account['universe'][c][1]['oraclePx'])-account['prices'][j]) for j,c in enumerate(COINS))
    return {'id':EPOCH,'started_at_utc':iso(),'profile':PROFILE,'initial_mark_equity':ledger['equity'],'initial_oracle_equity':oracle,
        'initial_external_capital':ledger['external_capital'],'cumulative_fees_at_start':ledger['fees'],'cumulative_funding_at_start':ledger['funding'],
        'actual_qty':account['qty'].tolist(),'previous_anchor':store.get('anchor'),'previous_peak':store.get('peak'),
        'previous_halted':store.get('halted',False),'previous_last_round':store.get('last_round'),
        'inherited_order_count':len(store.orders()),'inherited_fill_count':ledger['fills'],'service':SERVICE}

class BudgetRunner(Runner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.worker=None;self.worker_log=None;self.last_launch=0.;self.publication=None
        self.protocol=json.loads((TAIL/'baseline.json').read_text())['protocol'];self.signal_error=None
    def update_signal(self,now):
        try:
            frame,publication=read_publication()
            if frame.published_time>now+1 or frame.source_time>now:raise ValueError('Future signal')
            if self.inputs is not None and frame.source_time<self.inputs['source_hour']:raise ValueError('Signal rollback')
            self.publication=publication;self.inputs={'source_hour':frame.source_time,'weights':np.asarray(frame.weights),
                'retention':np.asarray(frame.retention),'sigma':np.asarray(frame.sigma),'volume':np.asarray(frame.volume),
                'trades':np.asarray(frame.trades),'input_hash':frame.source_hash,'protocol':self.protocol,'frame_hash':publication['sha256']}
            self.signal_error=None
        except (OSError,ValueError,TypeError,KeyError) as error:self.signal_error=f'{type(error).__name__}: {error}'
        fresh=self.inputs is not None and self.inputs['source_hour']==int(now//HOUR)*HOUR and self.signal_error is None
        if self.worker is not None and self.worker.poll() is not None:
            if self.worker_log:self.worker_log.close();self.worker_log=None
            self.worker=None
        if not fresh and self.worker is None and now-self.last_launch>=60:
            self.last_launch=now;self.worker_log=(DIRECTORY/'signal_worker.log').open('a')
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1'}
            self.worker=subprocess.Popen([sys.executable,'-m','experiments.performance_attribution_20261016.live_signal'],cwd=ROOT,
                stdout=self.worker_log,stderr=subprocess.STDOUT,env=env,close_fds=True)
        return fresh
    def decide(self,account,inputs,now):
        p=self.protocol;equity=account['ledger']['equity'];capital=account['ledger']['external_capital']
        peak=max(self.store.get('peak',equity)+capital-self.store.get('external_capital',0.),equity)
        halted=self.store.get('halted',False) or equity/peak-1<=-p['risk']['drawdown_halt']
        self.store.put('peak',peak);self.store.put('external_capital',capital);self.store.put('halted',halted)
        usable=inputs is not None and 0<=now-inputs['source_hour']<=6*HOUR
        fresh=usable and inputs['source_hour']==int(now//HOUR)*HOUR and self.signal_error is None
        cov=inputs['sigma'] if usable else np.zeros((10,10))
        # With stale covariance, this still checks gross/net/per-asset limits;
        # volatility risk is explicitly marked unavailable in status.
        forced=halted or bool(risk_breaches(account['qty'],equity,account['prices'],cov,p['risk']))
        hour=int(now//HOUR)*HOUR
        if not forced and (not fresh or hour%(6*HOUR) or now-hour>NORMAL_WINDOW_SECONDS):return None
        decision=int(now//60)*60 if forced else hour;key=f"round:{'risk' if forced else 'normal'}:{decision}"
        saved=self.store.get(key)
        if saved is not None:
            if saved.get('profile') not in (None,PROFILE):raise ValueError('Pending round belongs to another strategy')
            return None if saved['status']=='complete' else saved
        live_protocol=json.loads(json.dumps(p));live_protocol['execution']['decimals']=[int(account['universe'][c][0]['szDecimals']) for c in COINS]
        raw=inputs['weights'] if fresh else account['qty']*account['prices']/equity
        retention=inputs['retention'] if fresh else np.ones(10)
        volume=inputs['volume'] if usable else np.zeros(10);trades=inputs['trades'] if usable else np.zeros(10)
        action=RiskCapController().propose(decision,account['prices'],account['qty'],equity,raw,cov,volume,trades,
            live_protocol,buffer=.2,halted=halted,retention=retention)
        if action is None:return None
        plan={'key':key,'profile':PROFILE,'status':'planned','decision_time':decision,
            'deadline':min(hour+HOUR,now+60) if forced else hour+NORMAL_WINDOW_SECONDS,
            'input_hour':hour,'source_hour':inputs['source_hour'] if inputs else None,
            'input_hash':inputs['input_hash'] if inputs else None,'created_at_utc':iso(now),'raw_weights':raw.tolist(),
            'positions_before':account['qty'].tolist(),'equity':equity,'attribution':self.publication.get('attribution') if self.publication else None,**action}
        self.store.put(key,plan);return plan
    def begin_epoch(self,account):
        key='strategy_epoch:'+EPOCH
        epoch=self.store.get(key)
        if epoch is None:
            # Startup is forbidden while any prior order/round is unresolved.
            pending=[o for o in self.store.orders() if o['status'] in ('sending','unknown','responded','open')]
            if pending:raise ValueError('Unresolved orders block strategy handoff')
            rows=self.store.db.execute("select value from kv where key like 'round:%'").fetchall()
            if any(json.loads(r[0])['status']!='complete' for r in rows):raise ValueError('Pending prior round blocks handoff')
            epoch=epoch_snapshot(account,self.store);self.store.put(key,epoch);save_json(DIRECTORY/'handoff.json',epoch)
        self.store.put('active_strategy',{'profile':PROFILE,'epoch':EPOCH,'service':SERVICE})
        return epoch
    def cycle(self):
        self.recover();account=self.account();epoch=self.begin_epoch(account);now=time.time();fresh=self.update_signal(now)
        plan=self.decide(account,self.inputs,now) if self.execute else None
        if plan is not None:
            cov=self.inputs['sigma'] if self.inputs is not None and now-self.inputs['source_hour']<=6*HOUR else np.zeros((10,10))
            self.execute_round(plan,{'sigma':cov,'protocol':self.protocol});account=self.account()
        usable=self.inputs is not None and 0<=now-self.inputs['source_hour']<=6*HOUR
        ledger=account['ledger'];oracle=ledger['equity']+sum(account['qty'][j]*(float(account['universe'][c][1]['oraclePx'])-account['prices'][j]) for j,c in enumerate(COINS))
        flow=ledger['external_capital']-epoch['initial_external_capital']
        status={'updated_at_utc':iso(),'pid':os.getpid(),'mode':'live_testnet' if self.execute else 'preview','network':'testnet','dex':'par',
            'profile':PROFILE,'service':SERVICE,'state':'running' if fresh else 'monitoring_waiting_for_signal','account':ledger,
            'oracle_equity':oracle,'epoch':EPOCH,'epoch_start_utc':epoch['started_at_utc'],
            'epoch_mark_pnl':ledger['equity']-epoch['initial_mark_equity']-flow,
            'epoch_mark_return_percent':100*(ledger['equity']-epoch['initial_mark_equity']-flow)/epoch['initial_mark_equity'],
            'epoch_oracle_pnl':oracle-epoch['initial_oracle_equity']-flow,
            'epoch_oracle_return_percent':100*(oracle-epoch['initial_oracle_equity']-flow)/epoch['initial_oracle_equity'],
            'next_normal_decision_kst':stamp_kst((int(now)//(6*HOUR)+1)*(6*HOUR)),
            'data_completed_kst':stamp_kst(self.inputs['source_hour']) if self.inputs else None,
            'fresh_signal':fresh,'signal_error':self.signal_error,'volatility_risk_available':usable,
            'signal_worker_pid':self.worker.pid if self.worker is not None else None,
            'raw_target_weights':dict(zip(SYMBOLS,self.inputs['weights'])) if self.inputs else None,
            'risk':risk_values(account['qty'],ledger['equity'],account['prices'],self.inputs['sigma']) if usable else None,
            'halted':self.store.get('halted',False),'orders_claimed':len(self.store.orders()),'last_round':self.store.get('last_round'),
            'spot_usdpr_total':account['spot_usdpr_total'],'execution':'Inherited IOC executor, cross 1x, 10bp limit; same durable claims and actual fills'}
        save_json(self.directory/'status.json',status);save_json(DIRECTORY/'status.json',status)
        observation={**status,'actual_qty':account['qty'].tolist(),'mark_prices':account['prices'].tolist(),
            'oracle_prices':[float(account['universe'][c][1]['oraclePx']) for c in COINS]}
        with (DIRECTORY/'equity_observations.jsonl').open('a') as stream:stream.write(json.dumps(observation,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps({k:status[k] for k in ['updated_at_utc','profile','state','fresh_signal','signal_worker_pid','epoch_mark_pnl','next_normal_decision_kst']},allow_nan=False),flush=True)
        return status
    def close_worker(self):
        if self.worker is not None and self.worker.poll() is None:
            self.worker.terminate()
            try:self.worker.wait(timeout=20)
            except subprocess.TimeoutExpired:self.worker.kill();self.worker.wait(timeout=5)
        if self.worker_log:self.worker_log.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--once',action='store_true');args=parser.parse_args()
    if not args.execute:raise ValueError('For read-only validation use preflight, not the live state directory')
    DIRECTORY.mkdir(exist_ok=True);digest=verify_deployment()
    directory=LIVE/'live_runtime';lock=(directory/'process.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    store=Store(directory/'state.sqlite3');client=None;runner=None;stop=False
    def stopping(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    try:
        if store.get('frozen_fingerprint')!=identity(protected_hashes()):raise ValueError('Incumbent frozen fingerprint changed')
        if store.get('anchor') is None:raise ValueError('Handoff requires existing original anchor')
        if store.get('budget_deployment_digest',digest)!=digest:raise ValueError('Candidate deployment changed after activation')
        store.put('budget_deployment_digest',digest)
        settings=load_settings();client=build_client(settings,execution=True,enable_orders=True);check_connection(client,settings)
        runner=BudgetRunner(client,settings,store,directory,True);runner.stop_requested=lambda:stop or (directory/'STOP').exists()
        while not runner.stop_requested():
            try:runner.cycle()
            except Exception as error:
                status={'updated_at_utc':iso(),'pid':os.getpid(),'mode':'live_testnet','profile':PROFILE,'service':SERVICE,
                    'state':'paused_retrying','error':f'{type(error).__name__}: {error}','retry_seconds':POLL_SECONDS}
                save_json(directory/'status.json',status);save_json(DIRECTORY/'status.json',status);print(json.dumps(status),flush=True)
                if args.once:raise
            if args.once:break
            for _ in range(POLL_SECONDS):
                if runner.stop_requested():break
                time.sleep(1)
    finally:
        if runner:runner.close_worker()
        if client:close_client(client)
        store.close();lock.close()

if __name__=='__main__':main()
