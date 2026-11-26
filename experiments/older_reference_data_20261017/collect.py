"""Download and validate public, pre-existing-reference Binance OHLCV only."""
import concurrent.futures,csv,hashlib,io,json,time,urllib.error,urllib.request,zipfile
from pathlib import Path
import numpy as np
from experiments.par_direct_forecast_20261017.data import save,sha
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent
BASE='https://data.binance.vision/data/futures/um/monthly/klines'


def months(start,end):
    value=np.datetime64(start,'M');last=np.datetime64(end,'M')
    while value<=last:
        yield str(value);value+=np.timedelta64(1,'M')


def fetch(url,limit):
    with urllib.request.urlopen(url,timeout=25) as response:body=response.read(limit+1)
    if len(body)>limit:raise ValueError('Public archive exceeds declared byte limit')
    return body


def parse_archive(body,symbol,month):
    expected=f'{symbol}-1h-{month}.csv'
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        if archive.namelist()!=[expected]:raise ValueError('Unexpected archive members')
        if archive.getinfo(expected).file_size>5_000_000:raise ValueError('Archive expands beyond declared limit')
        with archive.open(expected) as f:rows=list(csv.reader(io.TextIOWrapper(f)))
    if rows and rows[0][0].lower() in ('open_time','open time'):rows=rows[1:]
    a=np.array(rows,dtype=float)
    if a.ndim!=2 or a.shape[1]!=12 or not np.isfinite(a).all():raise ValueError('Invalid kline schema')
    # Futures normally use milliseconds. Accept documented timestamp units only
    # when exact bar duration and requested calendar month validate as well.
    units=1_000_000 if np.median(a[:,0])>=1e14 else 1000
    opens=np.rint(a[:,0]/units).astype(np.int64);available=np.rint((a[:,6]+1)/units).astype(np.int64)
    start=int(np.datetime64(month,'M').astype('datetime64[s]').astype(np.int64))
    end=int((np.datetime64(month,'M')+np.timedelta64(1,'M')).astype('datetime64[s]').astype(np.int64))
    if np.any((opens<start)|(opens>=end)) or np.any(opens%3600):raise ValueError('Wrong month or bar alignment')
    if len(set(opens))!=len(opens) or np.any(np.diff(opens)<=0):raise ValueError('Duplicate or unordered bars')
    if np.any(available-opens!=3600):raise ValueError('Invalid bar close availability')
    if np.any(a[:,1:5]<=0) or np.any(a[:,2]<a[:,[1,3,4]].max(1)) or np.any(a[:,3]>a[:,[1,2,4]].min(1)):
        raise ValueError('Invalid OHLC prices')
    if np.any(a[:,[5,7,8,9,10]]<0) or np.any(a[:,8]!=np.rint(a[:,8])):raise ValueError('Invalid volume or trades')
    return dict(open_times=opens,market_times=available,open=a[:,1],high=a[:,2],low=a[:,3],close=a[:,4],
        volume=a[:,5],quote_volume=a[:,7],trades=a[:,8].astype(np.int64)),dict(rows=len(a),first_open=int(opens[0]),last_available=int(available[-1]),
        expected_full_month_rows=(end-start)//3600,internal_missing_hours=int(np.sum(np.diff(opens)//3600-1)),timestamp_units=units)


def collect_one(task):
    symbol,month=task;name=f'{symbol}-1h-{month}.zip';url=f'{BASE}/{symbol}/1h/{name}';path=HERE/'raw'/name
    try:
        if path.exists() and path.with_suffix('.zip.CHECKSUM').exists():
            body=path.read_bytes();checksum=path.with_suffix('.zip.CHECKSUM').read_text();cached=True
        else:
            body=fetch(url,2_000_000);checksum=fetch(url+'.CHECKSUM',4096).decode();cached=False
        digest=hashlib.sha256(body).hexdigest()
        if checksum.split()[0]!=digest:raise ValueError('Official SHA256 mismatch')
        values,stats=parse_archive(body,symbol,month)
        if not cached:path.write_bytes(body);path.with_suffix('.zip.CHECKSUM').write_text(checksum)
        np.savez_compressed(HERE/'parsed'/f'{symbol}-{month}.npz',**values)
        return dict(symbol=symbol,month=month,url=url,status='ok',bytes=len(body),sha256=digest,checksum_verified=True,**stats)
    except urllib.error.HTTPError as exc:return dict(symbol=symbol,month=month,url=url,status='http_error',http_status=exc.code)
    except Exception as exc:return dict(symbol=symbol,month=month,url=url,status='error',error=type(exc).__name__+': '+str(exc))


def main():
    for name in ('raw','parsed'): (HERE/name).mkdir(exist_ok=True)
    spec=json.loads((HERE/'protocol.json').read_text());tasks=[(symbol,month) for symbol,start in spec['starts'].items() for month in months(start,spec['end_month'])]
    frozen=HERE/'frozen.json'
    if not frozen.exists():save(frozen,dict(at_unix=time.time(),collector_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'protocol.json'),protected=protected()))
    else:
        previous=json.loads(frozen.read_text());assert previous['collector_sha256']==sha(Path(__file__));assert previous['protocol_sha256']==sha(HERE/'protocol.json')
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(collect_one,tasks):
            results.append(row)
            print(row['symbol'],row['month'],row['status'],row.get('rows',row.get('http_status',row.get('error'))),flush=True)
            save(HERE/'manifest.partial.json',dict(results=results,complete=False))
    summary=[]
    for symbol in spec['starts']:
        selected=[r for r in results if r['symbol']==symbol and r['status']=='ok'];arrays=[]
        for r in selected:
            with np.load(HERE/'parsed'/f"{symbol}-{r['month']}.npz") as z:arrays.append({k:z[k].copy() for k in z.files})
        if arrays:
            data={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]};t=data['open_times']
            assert np.all(np.diff(t)>0)
            np.savez_compressed(HERE/f'{symbol}.npz',**data)
            summary.append(dict(symbol=symbol,months=len(selected),rows=len(t),first_open=int(t[0]),last_available=int(data['market_times'][-1]),
                missing_hours_after_first_observation=int(np.sum(np.diff(t)//3600-1)),parsed_sha256=sha(HERE/f'{symbol}.npz')))
    save(HERE/'manifest.json',dict(results=results,summary=summary,complete=True,protected=protected(),orders_sent=0,
        reference_only=True,par_execution_data=False,training_only_candidate_data=True,files_downloaded=sum(r['status']=='ok' for r in results)))
    print('DONE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
