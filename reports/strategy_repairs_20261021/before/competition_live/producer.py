"""Independent signal worker; expensive baseline/model inference never sends orders."""
import argparse
import hashlib
import json
import os
import fcntl
import pickle
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from replay_lab.common_comparison.models import Market,covariance
from .candidate import build_candidate
from .profile import ROOT,load_profile,verify_deployment
from .runtime import json_save,atomic,private_load,fingerprint


def read_reference(workspace,profile):
    pointer=json.loads((workspace/'practice_live/latest.json').read_text())
    run=(workspace/pointer['run']).resolve()
    if not run.is_relative_to(workspace/'practice_live/updates'):raise ValueError('Invalid reference publication path')
    digest=hashlib.sha256((run/'inputs.npz').read_bytes()).hexdigest()
    if digest!=pointer['inputs_sha256']:raise ValueError('Reference input hash mismatch')
    verify=json.loads((run/'verification.json').read_text())
    if hashlib.sha256((run/'targets.npz').read_bytes()).hexdigest()!=verify['targets_sha256']:
        raise ValueError('Baseline target hash mismatch')
    with np.load(run/'inputs.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(run/'targets.npz',allow_pickle=False) as z:
        weights=z['default'].copy();times=z['times'].copy()
    if not np.array_equal(times,data['market_times']) or times[-1]!=int(time.time()//3600)*3600:
        raise ValueError('Reference input is stale')
    if not data['observed'][-1].all():raise ValueError('Incomplete reference hour')
    return data,weights,digest


def publish(adapter,data,weights,digest,profile,directory):
    now=time.time();times=data['market_times'];source=int(times[-1]);day=source//86400*86400
    def event(index):
        t=int(times[index]);prefix={k:v[:index+1].copy() for k,v in data.items()}
        market=Market(prefix['market_times'],prefix['close'],prefix['volume'],prefix['trades'],
            prefix['funding'],prefix['observed'],tuple(profile['trading']['symbols']))
        sigma=covariance(market,index,profile['trading']['risk'])[1]
        frame=dict(source_time=t,published_time=now,weights=weights[index].tolist(),retention=[1.]*10,
            sigma=sigma.tolist(),volume=data['volume'][index].tolist(),trades=data['trades'][index].tolist(),source_hash=digest)
        packed=json.dumps(frame,sort_keys=True,allow_nan=False).encode()
        envelope=dict(channel='strategySignal',received_ms=int(now*1000)+1,
            data=dict(frame=frame,publication_sha256=hashlib.sha256(packed).hexdigest(),
                incumbent=weights[index].tolist(),candidate=weights[index].tolist()))
        return envelope,prefix
    if adapter.last_source<0 or (adapter.gate_day!=day and source!=day):
        # Startup coefficients use only labels maturing before midnight. Their
        # actual availability is NOW; no historical trade is backdated.
        indices=np.flatnonzero(times==day)
        if len(indices)!=1:raise ValueError('Missing midnight model initialization history')
        envelope,prefix=event(int(indices[0]));adapter.ingest(envelope,prefix)
    envelope,prefix=event(len(times)-1)
    forecasts=adapter.ingest(envelope,prefix)
    ready=time.time()
    for f in forecasts:f['available_time']=max(f['available_time'],ready)
    json_save(directory/'signal.json',dict(published_at=ready,source_time=source,publication=envelope,
        forecasts=forecasts,candidate=profile['candidate'],profile_hash=fingerprint(profile)))


def main(workspace=None,directory=None,once=False):
    workspace=Path(workspace or ROOT).resolve();directory=Path(directory or ROOT/'.runtime/pas_competition_live')
    directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=(directory/'producer.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    profile=load_profile();verify_deployment();state=directory/'signal_state.pkl'
    if state.exists():
        saved=private_load(state)
        if saved['fingerprint']!=fingerprint(profile):raise ValueError('Signal state profile mismatch')
        adapter=saved['adapter']
    else:adapter=build_candidate(profile,lambda *a:None)['signals']
    while True:
        if time.time()>=profile['end_unix'] or (directory/'STOP').exists():return
        try:
            try:data,weights,digest=read_reference(workspace,profile)
            except (FileNotFoundError,ValueError):
                # Separate process deliberately uses the original workspace's
                # frozen baseline inference and model cache. No trading module.
                with (directory/'reference_refresh.log').open('ab') as log:
                    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
                    subprocess.run([sys.executable,'-m','practice_live'],cwd=workspace,env=env,
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                data,weights,digest=read_reference(workspace,profile)
            if int(data['market_times'][-1])>adapter.last_source:
                publish(adapter,data,weights,digest,profile,directory)
                atomic(state,pickle.dumps(dict(fingerprint=fingerprint(profile),adapter=adapter),protocol=5))
            json_save(directory/'producer_status.json',dict(at=time.time(),state='ready',source=adapter.last_source,orders_sent=0))
        except Exception as error:
            json_save(directory/'producer_status.json',dict(at=time.time(),state='error',error=str(error),orders_sent=0))
            if once:raise
        if once:return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace');p.add_argument('--directory');p.add_argument('--once',action='store_true')
    args=p.parse_args();main(args.workspace,args.directory,args.once)
