"""Shared immutable minute-tape comparison for venue policy experiments."""
import gzip,json
from .data import OUT,SYMBOLS,read,save
from pas_strategy_audit.data import load

def evaluate(controller,arm,scenario,phase,start,end,directory=None):
    from pas_counterfactual_20261022.engine import simulate
    from competition_live.profile import load_profile
    directory=directory or OUT/'development';cfg,d=load(directory)
    cfg.update(start_ms=start*1000,cutoff_ms=end*1000)
    sc=read(OUT/'protocol.json')['fees'][scenario];p=load_profile()['trading']
    tape=[]
    for symbol in SYMBOLS:
        for b in read(directory/'raw'/f'pas_candles_{symbol}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and start*1000<=at<=end*1000:
                tape.append(dict(time=at,coin='pas:'+symbol,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    result=simulate(cfg,d,read(directory/'events.json'),tape,read(directory/'funding.json'),p,sc,controller=controller)
    row=dict(arm=arm,scenario=scenario,phase=phase,**result['metrics'])
    result['controller_trace']=controller.trace
    return row,result

def persist(directory,row,result):
    name=f"{row['arm']}_{row['phase']}_{row['scenario']}"
    path=directory/'runs'/f'{name}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as f:json.dump(result,f,allow_nan=False,default=lambda x:x.tolist())
    save(directory/'rows'/f'{name}.json',row)
