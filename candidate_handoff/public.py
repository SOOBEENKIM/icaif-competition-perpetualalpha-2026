"""Budgeted public reads for the joint-wallet executor; no exchange endpoint.

500 weight/minute and a 100-weight burst leave capacity for other local clients.
The file-backed budget is shared by all instances of this adapter. Other programs
that do not use it can still consume the IP quota; a 429 opens a shared cooldown.
"""
import fcntl,json,math,os,time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from competition_live.venue import PublicPAS as BasePublicPAS
from competition_live.profile import ROOT
from joint_wallet.dispatch import atomic_json

class PublicReadUnavailable(requests.RequestException):
    def __init__(self,message,retry_after=60):
        super().__init__(message);self.retry_after=retry_after

class ReadBudget:
    def __init__(self,path=None,*,limit=500,burst=100,clock=time.time,sleep=time.sleep):
        self.path=Path(path or ROOT/'.runtime/pas_public_budget.json');self.limit=limit;self.burst=burst
        self.clock=clock;self.sleep=sleep
    def _change(self,weight=0,cooldown=0):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            now=self.clock()
            d=json.loads(self.path.read_text()) if self.path.exists() else dict(tokens=self.burst,updated=now,events=[],blocked_until=0.)
            events=[e for e in d['events'] if e[0]>now-60];tokens=min(self.burst,d['tokens']+max(0,now-d['updated'])*self.limit/60)
            blocked=max(d['blocked_until'],now+cooldown if cooldown else 0)
            wait=max(0,blocked-now,(weight-tokens)*60/self.limit)
            used=sum(e[1] for e in events)
            if used+weight>self.limit:
                for at,w in events:
                    used-=w
                    if used+weight<=self.limit:wait=max(wait,at+60-now);break
            if wait<=0 and weight:events.append([now,weight]);tokens-=weight
            atomic_json(self.path,dict(tokens=tokens,updated=now,events=events,blocked_until=blocked))
            return wait
    def acquire(self,weight):
        while weight>0:
            part=min(weight,self.burst,self.limit)
            while True:
                wait=self._change(part)
                if wait<=0:break
                self.sleep(min(max(wait,.001),5.))
            weight-=part
    def cooldown(self,seconds):self._change(cooldown=seconds)

LIGHT={'l2Book','allMids','clearinghouseState','orderStatus','spotClearinghouseState','exchangeStatus'}
PAGED={'userFillsByTime','userFunding','fundingHistory','userNonFundingLedgerUpdates'}

class PublicPAS(BasePublicPAS):
    def __init__(self,profile,*,budget=None):
        super().__init__(profile);self.budget=budget or ReadBudget()
        # The base adapter's quick status retries bypass a common weight budget.
        self.session.mount('https://',HTTPAdapter(max_retries=0))
        self.read_count=0
    def post(self,payload):
        self.budget.acquire(2 if payload.get('type') in LIGHT else 20);self.read_count+=1
        try:
            if payload.get('type')=='candleSnapshot':
                req=payload.get('req',{})
                if (set(payload)!={'type','req'} or set(req)!={'coin','interval','startTime','endTime'}
                        or req['coin'] not in {'pas:'+s for s in self.profile['trading']['symbols']}
                        or req['interval']!='1m' or not isinstance(req['startTime'],int)
                        or not isinstance(req['endTime'],int)
                        or not 0<=req['startTime']<req['endTime']
                        or req['endTime']-req['startTime']>26*3600000):
                    raise ValueError('Only bounded PAS minute candle reads are allowed')
                response=self.session.post(self.profile['rest_url']+'/info',json=payload,timeout=15)
                response.raise_for_status();result=response.json()
            else:result=super().post(payload)
        except requests.RequestException as e:
            response=getattr(e,'response',None);status=response.status_code if response is not None else None
            if status not in (None,429,500,502,503,504):raise
            delay=60. if status==429 else 5.
            if response is not None:
                try:delay=max(delay,float(response.headers.get('Retry-After',0)))
                except (ValueError,TypeError):pass
            self.budget.cooldown(delay)
            raise PublicReadUnavailable(f'Public {payload.get("type")} unavailable (HTTP {status}); no order retry',delay) from e
        if payload.get('type') in PAGED and isinstance(result,list) and result:
            self.budget.acquire(math.ceil(len(result)/20))
        if payload.get('type')=='candleSnapshot' and isinstance(result,list) and result:
            self.budget.acquire(math.ceil(len(result)/60))
        return result

class PublicQuoteProvider:
    def __init__(self,profile,public=None):self.profile=profile;self.public=public or PublicPAS(profile)
    def __call__(self,symbol):
        if symbol not in self.profile['trading']['symbols']:raise ValueError('Market outside selected universe')
        meta,contexts=self.public.market();context_at=int(time.time()*1000)
        coin='pas:'+symbol;by_coin={m['name']:c for m,c in zip(meta['universe'],contexts)}
        if meta['collateralToken']!=self.profile['collateral_token']:raise ValueError('Unexpected collateral')
        book=self.public.post(dict(type='l2Book',coin=coin));at=int(time.time()*1000)
        if book['coin']!=coin:raise ValueError('Book market mismatch')
        return dict(book=[(at,0,book)],book_times=[at],ctx=[(context_at,0,by_coin[coin])],ctx_times=[context_at],resets=[])

def collect_books(profile,public=None):
    api=public or PublicPAS(profile);books=[]
    for s in profile['trading']['symbols']:
        b=api.post(dict(type='l2Book',coin='pas:'+s));at=int(time.time()*1000)
        if b['coin']!='pas:'+s:raise ValueError('Book market mismatch')
        books.append((s,at,b))
    meta,ctx=api.market();received=int(time.time()*1000)
    if ([m['name'] for m in meta['universe']]!=['pas:'+s for s in profile['trading']['symbols']]
            or meta['collateralToken']!=profile['collateral_token']):raise ValueError('Unexpected PAS metadata')
    return {s:dict(book=[(at,0,b)],book_times=[at],ctx=[(received,0,c)],ctx_times=[received],resets=[])
            for (s,at,b),c in zip(books,ctx)}
