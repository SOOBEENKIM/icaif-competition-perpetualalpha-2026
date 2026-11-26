"""Versioned entry policy and completed PAS prices; no signer or order client."""
import hashlib,json,time
import numpy as np
from pas_performance_research.selected import NAME,controller
from competition_live.runtime import fingerprint,json_save

BASE='budget_lifecycle'
POLICIES=(BASE,NAME)

def configured_policy(directory):
    path=directory/'execution_policy.json'
    config=json.loads(path.read_text()) if path.exists() else {'policy':BASE}
    policy=config['policy']
    if policy not in POLICIES:raise ValueError('Unsupported execution policy')
    contract_path=directory/'team_execution_contract.json'
    contract=json.loads(contract_path.read_text()) if contract_path.exists() else {}
    required=contract.get('required_execution_policy')
    if required is not None and (not path.exists() or required!=policy):
        raise ValueError('Required execution policy is missing or differs')
    from competition_live.profile import ROOT
    for relative,digest in config.get('source_sha256',{}).items():
        source=(ROOT/relative).resolve()
        if not source.is_relative_to(ROOT) or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            raise ValueError('Execution policy source changed: '+relative)
    return policy

def summarize_candles(raw,symbols,source):
    points=[source-24*3600,source-3*3600,source]
    prices=[];ages=[];closed=[];traded=[]
    for point in points:
        px=[];age=[];cs=[];ts=[]
        for symbol in symbols:
            seen={};last=None;last_active=None
            for row in raw[symbol]:
                if row['s']!='pas:'+symbol or row['i']!='1m':raise ValueError('Wrong candle market/interval')
                start=row['t'];end=row['T'];known=(start+60000)/1000
                if start%60000 or end!=start+59999:raise ValueError('Invalid candle boundaries')
                if start in seen and seen[start]!=row:raise ValueError('Conflicting candle identity')
                seen[start]=row
                if known>point:continue # never use the incomplete or future candle
                close=float(row['c']);volume=float(row['v'])
                if not np.isfinite([close,volume]).all() or close<=0 or volume<0 or row['n']<0:
                    raise ValueError('Invalid completed candle')
                if last is None or known>last[0]:last=(known,close)
                if volume>0 and row['n']>0:last_active=max(last_active or known,known)
            px.append(last[1] if last else None);cs.append(last[0] if last else None)
            age.append(point-last_active if last_active is not None else 1e20);ts.append(last_active)
        prices.append(px);ages.append(age);closed.append(cs);traded.append(ts)
    return dict(source_time=source,symbols=list(symbols),price_times=points,prices=prices,age_seconds=ages,
        close_completed_at=closed,last_active_close_at=traded,venue='pas',interval='1m')

def validate_history(history,symbols,source,published_at):
    if (history.get('source_time')!=source or history.get('symbols')!=list(symbols)
            or history.get('venue')!='pas' or history.get('interval')!='1m'
            or history.get('price_times')!=[source-24*3600,source-3*3600,source]
            or not source<=history['received_at']<=published_at):
        raise ValueError('PAS price history is not aligned/available')
    shape=(3,len(symbols));prices=np.asarray(history['prices'],float);ages=np.asarray(history['age_seconds'],float)
    closes=np.asarray(history['close_completed_at'],float);trades=np.asarray(history['last_active_close_at'],float)
    if any(a.shape!=shape for a in (prices,ages,closes,trades)):raise ValueError('Incomplete PAS history dimensions')
    if (not np.isfinite(ages).all() or np.any(ages<0) or np.isinf(prices).any()
            or np.any(np.isfinite(prices)&(prices<=0))):raise ValueError('Invalid PAS history values')
    for i,point in enumerate(history['price_times']):
        valid=np.isfinite(prices[i]);active=np.isfinite(trades[i])
        if (np.any(valid!=np.isfinite(closes[i])) or np.any(active&~valid) or np.any(closes[i,valid]>point)
                or np.any(trades[i,active]>point) or np.any(trades[i,active]>closes[i,active])
                or not np.allclose(ages[i,active],point-trades[i,active],rtol=0,atol=1e-9)
                or np.any(ages[i,~active]!=1e20)):
            raise ValueError('Future or inconsistent PAS price timestamp')
    return history

def fetch_history(public,profile,source,directory=None):
    raw={};evidence=[]
    for symbol in profile['trading']['symbols']:
        request=dict(type='candleSnapshot',req=dict(coin='pas:'+symbol,interval='1m',
            startTime=(source-26*3600)*1000,endTime=source*1000-1))
        rows=public.post(request)
        if not isinstance(rows,list):raise ValueError('Unexpected candle response')
        raw[symbol]=rows;evidence.append(dict(request=request,received_at=time.time(),response=rows))
    result=summarize_candles(raw,profile['trading']['symbols'],source)
    result.update(received_at=time.time(),raw_sha256=fingerprint(raw))
    validate_history(result,profile['trading']['symbols'],source,time.time())
    if directory is not None:json_save(directory/'venue_prices'/f'{source}.json',dict(history=result,requests=evidence))
    return result

def selected_controller(frame):
    history=frame['venue_history'];source=frame['source_time']
    # Relative clock zero preserves the existing off-cadence preview behaviour;
    # the live scheduler independently requires the real 6-hour boundary.
    return controller([0],np.array([frame['price_forecast_168h']]),
        np.array([frame['reference_funding_forecast_168h']]),np.array([frame['anchor_weights']]),
        np.array([frame['protected']],bool),price_times=np.asarray(history['price_times'])-source,
        prices=np.asarray(history['prices'],float),age=np.asarray(history['age_seconds'],float))
