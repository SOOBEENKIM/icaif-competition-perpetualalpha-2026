"""Atomic, private checkpoints and the selected live coordinator state."""
import copy
import hashlib
import json
import os
import pickle
import stat
from pathlib import Path


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=path.with_name(path.name+'.tmp')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600)
    os.fchmod(fd,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def json_save(path,data):
    atomic(path,json.dumps(data,ensure_ascii=False,allow_nan=False,
        default=lambda v:v.tolist() if hasattr(v,'tolist') else v.item()).encode())


def fingerprint(profile):
    return hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()


def private_load(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        info=os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:
            raise ValueError('State must be a private, locally owned regular file')
        return pickle.load(f)


def save_coordinator(path,portfolio,profile,reason):
    state=copy.copy(portfolio);state.settlement=None
    account=copy.copy(portfolio.account)
    account.executor=None;account.checkpoint=None;account.collect_fills=None
    account.refresh_market=None
    state.account=account
    def detached(pair):
        if pair is None:return None
        result=copy.copy(pair);result.account=account;result.owner=account.owners['pair'];result.on_time=None
        return result
    state.pair=detached(portfolio.pair);state.sessions=[detached(x) for x in portfolio.sessions]
    atomic(path,pickle.dumps(dict(fingerprint=fingerprint(profile),reason=reason,portfolio=state),protocol=5))


def load_coordinator(path,profile,executor,checkpoint,fills,settlement):
    # Only this process's mode-600 state is accepted; never load user attachments.
    saved=private_load(path)
    if saved['fingerprint']!=fingerprint(profile):raise ValueError('Checkpoint strategy/profile mismatch')
    result=saved['portfolio'];result.settlement=settlement
    result.account.executor=executor;result.account.checkpoint=checkpoint;result.account.collect_fills=fills
    for pair in [result.pair,*result.sessions]:
        if pair is not None:
            pair.account=result.account;pair.owner=result.account.owners['pair'];pair.on_time=settlement
    return result
