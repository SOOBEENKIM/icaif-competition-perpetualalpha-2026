"""Separate signal production and read-only risk/decision supervision.

There is deliberately no exchange client. Every intent records actual supplied
inventory and remains a proposal; a live handoff must use the incumbent's
durable claim/recovery executor, never a second wallet writer.
"""
from dataclasses import dataclass,asdict
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from replay_lab.common_comparison.engine import risk_breaches,project,rounded
from .policy import RiskCapController


@dataclass
class SignalFrame:
    source_time: int
    published_time: float
    weights: list
    retention: list
    sigma: list
    volume: list
    trades: list
    source_hash: str

    def validate(self):
        if self.source_time%3600 or not np.isfinite(self.published_time) or self.source_time>self.published_time:
            raise ValueError('Invalid signal times')
        if len(self.source_hash)!=64:raise ValueError('Missing source digest')
        for name in ('weights','retention','volume','trades'):
            v=np.asarray(getattr(self,name))
            if v.shape!=(10,) or not np.isfinite(v).all():raise ValueError('Invalid '+name)
        cov=np.asarray(self.sigma)
        if cov.shape!=(10,10) or not np.isfinite(cov).all() or not np.allclose(cov,cov.T):raise ValueError('Invalid covariance')
        if np.linalg.eigvalsh(cov).min()<-1e-12:raise ValueError('Indefinite covariance')
        if np.any(np.asarray(self.retention)<0) or np.any(np.asarray(self.retention)>1):raise ValueError('Invalid cap')
        if np.any(np.asarray(self.volume)<0) or np.any(np.asarray(self.trades)<0):raise ValueError('Invalid activity')


class SignalFile:
    def __init__(self,path):self.path=Path(path)
    def publish(self,frame):
        frame.validate();data=json.dumps(asdict(frame),sort_keys=True,allow_nan=False)
        envelope={'frame':json.loads(data),'sha256':hashlib.sha256(data.encode()).hexdigest()}
        tmp=self.path.with_suffix('.tmp')
        with tmp.open('w') as f:json.dump(envelope,f);f.flush();os.fsync(f.fileno())
        tmp.replace(self.path)
    def read(self):
        obj=json.loads(self.path.read_text());data=json.dumps(obj['frame'],sort_keys=True,allow_nan=False)
        if hashlib.sha256(data.encode()).hexdigest()!=obj['sha256']:raise ValueError('Torn or altered signal publication')
        f=SignalFrame(**obj['frame']);f.validate();return f


class Supervisor:
    def __init__(self,protocol,signal_file,journal,saved=None):
        self.protocol=protocol;self.signal_file=signal_file;self.journal=Path(journal)
        saved=saved or {};self.control=RiskCapController(saved.get('control'))
        self.peak=float(saved.get('peak',protocol['initial']));self.halted=bool(saved.get('halted',False))
        self.last_tick=float(saved.get('last_tick',-1));self.last_normal=int(saved.get('last_normal',-1));self.frame=None

    def snapshot(self):
        return {'control':self.control.snapshot(),'peak':self.peak,'halted':self.halted,
                'last_tick':self.last_tick,'last_normal':self.last_normal}

    def observe(self,account,now=None):
        now=time.time() if now is None else now
        if now<=self.last_tick:raise ValueError('Non-monotone account observation')
        self.last_tick=now
        equity=float(account['equity']);qty=np.asarray(account['qty'],float);prices=np.asarray(account['prices'],float)
        if (qty.shape!=(10,) or prices.shape!=(10,) or not np.isfinite(qty).all() or not np.isfinite(prices).all()
                or np.any(prices<=0) or not np.isfinite(equity) or equity<=0 or not 0<=now-account['time']<=15):
            raise ValueError('Stale or invalid actual account')
        self.peak=max(self.peak,equity);self.halted|=equity/self.peak-1<=-self.protocol['risk']['drawdown_halt']
        error=None
        try:
            candidate=self.signal_file.read()
            if candidate.published_time>now or candidate.source_time>now:raise ValueError('Future publication')
            if self.frame and candidate.source_time<self.frame.source_time:raise ValueError('Signal rollback')
            self.frame=candidate
        except (OSError,ValueError,TypeError,KeyError) as exc:error=str(exc)
        f=self.frame;action=None;reasons=[]
        cov_valid=f is not None and 0<=now-f.source_time<=6*3600
        fresh=cov_valid and f.source_time==int(now//3600)*3600 and error is None
        breach=risk_breaches(qty,equity,prices,np.asarray(f.sigma),self.protocol['risk']) if cov_valid else []
        if self.halted or breach:
            target=np.zeros(10) if self.halted else project(qty*prices/equity,equity,np.asarray(f.sigma),self.protocol['risk'])*equity/prices
            target=np.sign(qty)*np.minimum(abs(qty),abs(target))
            target=np.array([rounded(q,d) for q,d in zip(target,self.protocol['execution']['decimals'])])
            action={'target_qty':target.tolist(),'risk_forced':True,'reduce_only':True,'decision_reason':'drawdown_halt' if self.halted else 'hard_risk'}
            reasons.append(action['decision_reason'])
        else:
            boundary=int(now//21600)*21600
            if fresh and boundary>self.last_normal and 0<=now-boundary<=900 and f.source_time==boundary:
                action=self.control.propose(boundary,prices,qty,equity,np.asarray(f.weights),np.asarray(f.sigma),
                    np.asarray(f.volume),np.asarray(f.trades),self.protocol,retention=np.asarray(f.retention))
                if action is not None:self.last_normal=boundary
                reasons.append('normal_decision')
            else:reasons.append('wait_for_fresh_signal' if not fresh else 'outside_normal_round')
        event={'at':now,'equity':equity,'actual_qty':qty.tolist(),'prices':prices.tolist(),'peak':self.peak,
            'halted':self.halted,'fresh_signal':fresh,'covariance_usable':cov_valid,'signal_error':error,
            'signal_source_time':f.source_time if f else None,'signal_hash':f.source_hash if f else None,
            'weights':f.weights if f else None,'retention':f.retention if f else None,
            'risk_breaches':breach,'reasons':reasons,'intent':action,'orders_sent':0}
        with self.journal.open('a') as out:out.write(json.dumps(event,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
        return event
