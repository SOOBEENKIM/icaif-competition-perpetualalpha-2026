"""Checksum-verified public bars; no wallet or trading endpoint."""
import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'
BASE='https://data.binance.vision/data/futures/um/monthly/klines'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    def convert(value):
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,np.ndarray):return value.tolist()
        raise TypeError(type(value).__name__)
    payload=json.dumps(obj,indent=2,allow_nan=False,default=convert)+'\n'
    path=Path(p);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(payload);temp.replace(path)


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=30) as response:return response.read()
        except urllib.error.HTTPError as error:
            if error.code==404:return None
            if error.code not in (429,500,502,503,504) or attempt==2:raise
        except urllib.error.URLError:
            if attempt==2:raise
        time.sleep(attempt+1)


def parse(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names=archive.namelist()
        if len(names)!=1 or not names[0].endswith('.csv'):raise ValueError('Unexpected archive')
        rows=list(csv.reader(io.StringIO(archive.read(names[0]).decode())))
    if rows and not rows[0][0].isdigit():rows=rows[1:]
    a=np.array([[float(v) for v in row[:9]] for row in rows])
    if a.ndim!=2 or len(a)==0:raise ValueError('Empty or invalid bars')
    # Futures archives use milliseconds; detect units and require hour alignment.
    unit=1_000_000 if a[0,0]>10**14 else 1000
    starts=(a[:,0]/unit).astype(np.int64)
    ends=(a[:,6]/unit+1/unit).round().astype(np.int64)
    if not np.all(starts%3600==0) or not np.all(ends==starts+3600):raise ValueError('Bar availability alignment')
    values=a[:,[1,2,3,4,5,8]]
    o,h,l,c,v,n=values.T
    if (not np.isfinite(values).all() or np.any(np.minimum(o,c)<l) or np.any(np.maximum(o,c)>h)
            or np.any(l<=0) or np.any(v<0) or np.any(n<0) or np.any(n!=np.floor(n))):raise ValueError('Invalid OHLCV')
    if not np.all(np.diff(starts)==3600):raise ValueError('Internal missing/duplicate bar')
    return {'times':ends,**dict(zip(('open','high','low','close','volume','trades'),values.T))}


def main():
    OUT.mkdir(exist_ok=True);raw=OUT/'external';raw.mkdir(exist_ok=True)
    plan=OUT/'data_plan.json'
    if plan.exists():raise FileExistsError('Use the existing immutable receipts')
    months=[str(m) for m in np.arange(np.datetime64('2024-01','M'),np.datetime64('2026-10','M'))]
    save(plan,{'at_unix':time.time(),'months':months,'symbols':['BTC','ETH','HYPE'],
               'protocol_sha256':sha(HERE/'protocol.json'),'collector_sha256':sha(__file__)})
    def one(job):
        symbol,month=job;name=f'{symbol}USDT-1h-{month}.zip'
        url=f'{BASE}/{symbol}USDT/1h/{name}'
        blob=fetch(url)
        if blob is None:return {'symbol':symbol,'month':month,'url':url,'status':404}
        checksum=fetch(url+'.CHECKSUM')
        if checksum is None:raise ValueError('Checksum missing: '+url)
        expected=checksum.decode().split()[0]
        if hashlib.sha256(blob).hexdigest()!=expected:raise ValueError('Checksum mismatch: '+url)
        bars=parse(blob);(raw/name).write_bytes(blob);(raw/(name+'.CHECKSUM')).write_bytes(checksum)
        return {'symbol':symbol,'month':month,'url':url,'status':200,'sha256':expected,
                'first_close':int(bars['times'][0]),'last_close':int(bars['times'][-1]),'bars':len(bars['times'])}
    jobs=[(s,m) for s in ['BTC','ETH','HYPE'] for m in months]
    with ThreadPoolExecutor(max_workers=3) as pool:receipts=list(pool.map(one,jobs))
    datasets={}
    for symbol in ['BTC','ETH','HYPE']:
        present=[r for r in receipts if r['symbol']==symbol and r['status']==200]
        if not present:raise ValueError('No public bars for '+symbol)
        allbars=[parse((raw/f'{symbol}USDT-1h-{r["month"]}.zip').read_bytes()) for r in present]
        joined={k:np.concatenate([b[k] for b in allbars]) for k in allbars[0]}
        if not np.all(np.diff(joined['times'])==3600):raise ValueError('Gap across monthly archives: '+symbol)
        np.savez_compressed(raw/f'{symbol}.npz',**joined)
        datasets[symbol]={'bars':len(joined['times']),'start':int(joined['times'][0]),'end':int(joined['times'][-1]),
                          'sha256':sha(raw/f'{symbol}.npz')}
    save(OUT/'data_receipts.json',{'receipts':receipts,'datasets':datasets,'orders_sent':0})
    print(json.dumps(datasets,indent=2),flush=True)


if __name__=='__main__':main()
