"""Finite public-data validation capture. No keys, orders or service mutations."""
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import websocket
from .capture import ROOT,OUT,SYMBOLS,REFERENCES,URLS

FWD=OUT/'forward_72h'


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,separators=(',',':')))
    os.replace(temporary,path)


def prepare():
    FWD.mkdir(parents=True,exist_ok=True)
    if (FWD/'plan.json').exists():raise FileExistsError('Frozen plan already exists')
    now=time.time();start=(int(now)//60+1)*60
    protocol=json.loads((OUT/'protocol.json').read_text())
    # Preserve exact hypothesis code; later edits invalidate the registered run.
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (ROOT/'pas_venue_research').glob('*.py')}
    sources.update({str(OUT.relative_to(ROOT)/'protocol.json'):hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()})
    for path in [OUT/'raw/pas_meta.json',*(OUT/'raw').glob('main_candles_*.json')]:
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    config=dict(protocol['config'],avoid_hourly_funding_boundaries=True)
    plan=dict(created_at=now,start=start,end=start+72*3600,code_hashes=sources,
        models=protocol['models'],config=config,selection=protocol['selection'],
        scenarios=dict(base={},stress=dict(depth_fraction=.1,exit_impact_bps=20.,latency_seconds=3.)),
        reference_risk_fit_end=protocol['reference_risk_fit_end'],block_seconds=3600,
        funding='Stop new entries before the next hourly settlement so TTL+holding+exit grace ends before it. '
                'Any residual crossing a settlement makes funding unknown and invalidates headline PnL.',
        evaluation='At the fixed end, run both base and stress received-event simulations. '
                   'Any connection gap, unresolved fill, residual or unknown funding blocks adoption.',
        incumbent='Record actual published targets and actual starting inventory; do not change live execution.',
        evidence='Public quote and trade capture; counterfactual queue estimates are not exchange fills.',
        no_midrun_reselection=True,maximum_disk_bytes=3*1024**3,
        automatic_live_replacement=False,orders_sent=0,notifications='none',
        purpose='Gather new PAS evidence required before any profitability or scale claim.',
        prior_trial_overlap='The previously stopped validation_rebuild trial reserves part of this calendar '
            'window. This is a separately registered study, not a restart of that trial. Do not reuse outcomes '
            'for either trial selection or claim the overlapping intervals are independent.',
        limitations=['72h is an initial forward check, not proof of durable profitability.',
                     'Any connection gap must be disclosed; do not stitch it into a complete fill history.',
                     'No strategies are automatically activated, no capital or leverage is changed.'])
    atomic(FWD/'plan.json',plan)
    shutil.copy2(ROOT/'.runtime/pas_competition_live/status.json',FWD/'starting_live_status.json')
    shutil.copy2(ROOT/'.runtime/pas_competition_live/signal.json',FWD/'starting_publication.json')
    shutil.copy2(ROOT/'competition_live/deployment_manifest.json',FWD/'starting_live_manifest.json')
    print(json.dumps(dict(start=start,end=plan['end'],path=str(FWD))),flush=True)


def collect():
    plan=json.loads((FWD/'plan.json').read_text())
    for name,digest in plan['code_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed')
    (FWD/'raw').mkdir(exist_ok=True)
    lock=threading.Lock();stop=threading.Event();errors=[];counts={};writers={};last={};started=time.time()
    def emit(network,kind,data):
        with lock:
            if stop.is_set():return
            now=time.time();coin=data.get('coin','') if isinstance(data,dict) else ''
            key=network,kind,coin
            if kind in ('l2Book','activeAssetCtx') and now-last.get(key,0)<1:return
            last[key]=now
            hour=time.strftime('%Y%m%d_%H',time.gmtime(now))
            for name in list(writers):
                if name!=hour:writers.pop(name).close()
            if hour not in writers:writers[hour]=gzip.open(FWD/'raw'/f'{hour}.jsonl.gz','at',compresslevel=3)
            writers[hour].write(json.dumps(dict(network=network,channel=kind,received=now,data=data),separators=(',',':'))+'\n')
            counter=network+':'+kind;counts[counter]=counts.get(counter,0)+1
    def worker(network):
        coins=['pas:'+s for s in SYMBOLS] if network=='pas' else list(REFERENCES.values())
        url=URLS[network].replace('https:','wss:').replace('/info','/ws')
        while not stop.is_set() and time.time()<plan['end']:
            sock=None
            try:
                sock=websocket.create_connection(url,timeout=5)
                for coin in coins:
                    for kind in (('l2Book','activeAssetCtx','trades') if network=='pas' else ('l2Book','activeAssetCtx')):
                        sock.send(json.dumps(dict(method='subscribe',subscription=dict(type=kind,coin=coin))))
                ping=time.time();emit(network,'connection_open',{})
                while not stop.is_set() and time.time()<plan['end']:
                    try:message=sock.recv()
                    except websocket.WebSocketTimeoutException:message=None
                    if time.time()-ping>20:sock.send('{"method":"ping"}');ping=time.time()
                    if not message:continue
                    value=json.loads(message)
                    if value.get('channel') in ('l2Book','activeAssetCtx','trades'):
                        emit(network,value['channel'],value['data'])
            except Exception as error:
                row=dict(at=time.time(),network=network,error=str(error));errors.append(row)
                emit(network,'connection_gap',row)
                stop.wait(3)
            finally:
                if sock is not None:sock.close()
    threads=[threading.Thread(target=worker,args=(network,),daemon=True) for network in URLS]
    for thread in threads:thread.start()
    source=None;observation=None;last_housekeeping=0.;written=0
    try:
        while time.time()<plan['end'] and not (FWD/'STOP').exists():
            now=time.time()
            if now-last_housekeeping>=30:
                written=sum(p.stat().st_size for p in (FWD/'raw').glob('*'))
                if written>=plan['maximum_disk_bytes'] or shutil.disk_usage(FWD).free<5*1024**3:
                    raise RuntimeError('Capture disk reserve reached')
                last_housekeeping=now
            try:
                signal=json.loads((ROOT/'.runtime/pas_competition_live/signal.json').read_text())
                identity=signal['publication']['data']['publication_sha256']
                if identity!=source:
                    emit('local','strategySignal',signal);source=identity
                status=json.loads((ROOT/'.runtime/pas_competition_live/status.json').read_text())
                if status['at']!=observation:
                    emit('local','accountObservation',status);observation=status['at']
            except (OSError,ValueError,KeyError) as error:
                emit('local','signal_read_error',dict(error=str(error)))
            with lock:
                atomic(FWD/'health.json',dict(at=now,started=started,planned_start=plan['start'],
                    deadline=plan['end'],state='capturing',counts=counts,errors=errors[-20:],
                    total_connection_errors=len(errors),bytes=written,orders_sent=0,signer_loaded=False))
            stop.wait(5)
    finally:
        stop.set()
        for thread in threads:thread.join(timeout=7)
        with lock:
            for writer in writers.values():writer.close()
            atomic(FWD/'completion.json',dict(at=time.time(),planned_end=plan['end'],
                completed=time.time()>=plan['end'],counts=counts,errors=errors,orders_sent=0))
    if time.time()>=plan['end']:
        from .shadow import main as evaluate
        try:
            evaluate(FWD,'base');evaluate(FWD,'stress')
            atomic(FWD/'evaluation_status.json',dict(at=time.time(),state='evaluated',
                automatic_live_replacement=False,orders_sent=0))
        except Exception as error:
            atomic(FWD/'evaluation_status.json',dict(at=time.time(),state='evaluation_error',
                error=str(error),automatic_live_replacement=False,orders_sent=0))
            raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','collect'))
    if parser.parse_args().action=='prepare':prepare()
    else:collect()
