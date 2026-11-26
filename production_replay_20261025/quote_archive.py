"""Compact read-only adapter for the quote-history candidate.

Hourly records carry actual received book/context timestamps. Integrating this
adapter into an order worker is a separate action, never done by this module.
"""
import bisect,copy,json,math,os,tempfile
from pathlib import Path
from experiments.basis_carry_20261017.probe import asof
from .protocol import save


def make_record(streams,symbols,point,config,available_at):
    if point%3600 or available_at<point:raise ValueError('Invalid hourly cutoff/availability')
    values={}
    for symbol in symbols:
        stream=streams.get(symbol);q=asof(stream,point*1000,config) if stream is not None else None
        if q is None:values[symbol]=None;continue
        values[symbol]=dict(book=copy.deepcopy(q['book']),ctx=copy.deepcopy(q['ctx']),
                            book_received=q['received']['book'],ctx_received=q['received']['ctx'])
    return dict(schema='received_pas_hourly_quotes_v1',point=point,available_at=available_at,
                symbols=list(symbols),values=values,venue='pas')


def validate_record(record,symbols,config,now):
    if record.get('schema')!='received_pas_hourly_quotes_v1' or record.get('venue')!='pas':raise ValueError('Wrong quote archive type')
    if record['symbols']!=list(symbols) or set(record['values'])!=set(symbols):raise ValueError('Wrong quote archive universe')
    point=record['point']
    if point%3600 or not point<=record['available_at']<=now:raise ValueError('Future/unavailable quote archive')
    for symbol,r in record['values'].items():
        if r is None:continue
        if r['book'].get('coin','pas:'+symbol)!='pas:'+symbol:raise ValueError('Wrong book symbol')
        stream=dict(book=[(r['book_received'],point,r['book'])],book_times=[r['book_received']],
                    ctx=[(r['ctx_received'],point,r['ctx'])],ctx_times=[r['ctx_received']],resets=[])
        if asof(stream,point*1000,config) is None:raise ValueError('Archive was not fresh at its cutoff')
    return record


def merge_features(latest_streams,records,symbols,source,config,now):
    """Return a new stream; original current quote provider is never modified."""
    result=copy.deepcopy(latest_streams);wanted={source-86400,source-10800,source}
    seen=set()
    for record in records:
        validate_record(record,symbols,config,now)
        point=record['point']
        if point not in wanted:continue
        if point in seen:raise ValueError('Duplicate hourly quote record')
        seen.add(point)
        for symbol,r in record['values'].items():
            if r is None:continue
            stream=result.setdefault(symbol,dict(book=[],ctx=[],resets=[]))
            # Unique per-hour historical epochs. Both fields share the same
            # identity; a current mismatched epoch can never borrow the past.
            epoch=-int(point)-1
            stream['book'].append((r['book_received'],epoch,copy.deepcopy(r['book'])))
            stream['ctx'].append((r['ctx_received'],epoch,copy.deepcopy(r['ctx'])))
    for stream in result.values():
        for field in ['book','ctx']:
            stream[field].sort(key=lambda x:x[0]);stream[field+'_times']=[x[0] for x in stream[field]]
    return result


def export_capture(cutoff):
    """Research archive export only, with availability marked at actual export."""
    import time
    from competition_live.profile import load_profile
    from .book_replay import load_books
    from .protocol import ROOT,OUT
    if cutoff>time.time():raise ValueError('No future export')
    streams,_=load_books((OUT/'capture').glob('*.jsonl.gz'),cutoff)
    symbols=load_profile()['trading']['symbols']
    cfg=json.loads((ROOT/'competition_live/strategy_plan.json').read_text())['book_protocol']
    first=min(s['book_times'][0] for s in streams.values())/1000
    dest=OUT/'quote_feature_archive'
    for point in range(math.ceil(first/3600)*3600,math.floor(cutoff/3600)*3600+1,3600):
        path=dest/f'{point}.json'
        if path.exists():continue
        record=make_record(streams,symbols,point,cfg,time.time())
        validate_record(record,symbols,cfg,time.time())
        # Readers must see either the complete immutable hour or no hour.
        # link() also prevents a concurrent export replacing earlier evidence.
        dest.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w',dir=dest,prefix='.quote-',delete=False) as fp:
            temporary=Path(fp.name);json.dump(record,fp,allow_nan=False)
            fp.flush();os.fsync(fp.fileno())
        try:
            try:os.link(temporary,path)
            except FileExistsError:pass
        finally:temporary.unlink()
    return dest


def watch():
    """Bounded local feed archive. Does not read PnL or contact an exchange."""
    import time
    from .protocol import OUT
    end=1792916100
    while time.time()<end:
        hour=int((time.time()-60)//3600)*3600
        if not (OUT/'quote_feature_archive'/f'{hour}.json').exists():export_capture(time.time()-60)
        save(OUT/'quote_feature_archive_health.json',dict(at=time.time(),archived_through_hour=hour,orders_sent=0))
        time.sleep(min(30,max(1,end-time.time())))


if __name__=='__main__':watch()
