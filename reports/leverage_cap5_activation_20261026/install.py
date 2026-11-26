"""Change only the margin-selection ceiling, preserving current inventory."""
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
D=ROOT/'.runtime/pas_budget_lifecycle'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic(path,data):
    tmp=path.with_name(path.name+'.cap5-install-tmp')
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def save(path,value):atomic(path,json.dumps(value,indent=2).encode())


def main():
    from candidate_handoff.locks import WriterLease
    from candidate_handoff.entry_policy import configured_policy
    from candidate_handoff.producer import validate
    from competition_live.profile import load_profile,verify_deployment
    import joint_wallet.leverage as leverage
    source='joint_wallet/leverage.py'
    expected=json.loads((OUT/'original_source_hashes.json').read_text())[source]
    assert sha(ROOT/source)==expected
    profile=load_profile();verify_deployment()
    validate(json.loads((D/'signal.json').read_text()),profile,time.time(),expected_policy='pullback_any3')
    rollback=OUT/'rollback'
    if rollback.exists():raise FileExistsError('Preserve previous installation evidence')
    with WriterLease(ROOT),(D/'producer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert configured_policy(D)=='pullback_any3'
        assert not json.loads((D/'dispatch.json').read_text()).get('pending')
        before=leverage.load_policy(D);assert before['maximum_leverage']==3
        config=json.loads((D/'execution_policy.json').read_text())
        policy=json.loads((D/'leverage_policy.json').read_text())
        pointer=json.loads((ROOT/'.runtime/active_pas_strategy.json').read_text())
        unchanged={name:sha(ROOT/name) for name in config['source_sha256'] if name!=source}
        unchanged['competition_live/config.json']=sha(ROOT/'competition_live/config.json')
        state_hash={name:sha(D/name) for name in ['coordinator.pkl','ownership.json','dispatch.json','signal.json']}
        rollback.mkdir(mode=0o700)
        shutil.copy2(ROOT/source,rollback/'leverage.py')
        for name in ['leverage_policy.json','execution_policy.json','status.json']:
            shutil.copy2(D/name,rollback/name)
        shutil.copy2(ROOT/'.runtime/active_pas_strategy.json',rollback/'active_pas_strategy.json')
        try:
            atomic(ROOT/source,(OUT/'stage'/source).read_bytes())
            policy.update(maximum_leverage=5,cap_changed_at=time.time(),cap_change_reason='Explicit user request: 5x ceiling; target sizing deferred')
            config['source_sha256'][source]=sha(ROOT/source)
            config.update(leverage_maximum=5,leverage_cap_changed_at=time.time(),leverage_cap_report=str(OUT/'report.html'))
            pointer.update(leverage_maximum=5,leverage_cap_changed_at=time.time(),leverage_cap_report=str(OUT/'report.html'))
            save(D/'leverage_policy.json',policy);save(D/'execution_policy.json',config)
            save(ROOT/'.runtime/active_pas_strategy.json',pointer)
            importlib.reload(leverage)
            after=leverage.load_policy(D)
            assert after==dict(before,maximum_leverage=5)
            assert configured_policy(D)=='pullback_any3';verify_deployment()
            assert unchanged=={name:sha(ROOT/name) for name in unchanged}
            assert state_hash=={name:sha(D/name) for name in state_hash}
            save(OUT/'install_receipt.json',dict(at=time.time(),before=before,after=after,
                source_hash=sha(ROOT/source),unchanged_strategy_hashes=unchanged,
                inventory_preserved=True,orders_sent=0,setting_updates=0,rollback=str(rollback)))
        except BaseException:
            atomic(ROOT/source,(rollback/'leverage.py').read_bytes())
            for name in ['leverage_policy.json','execution_policy.json']:
                atomic(D/name,(rollback/name).read_bytes())
            atomic(ROOT/'.runtime/active_pas_strategy.json',(rollback/'active_pas_strategy.json').read_bytes())
            raise
    print(json.dumps({'maximum_leverage':5,'target_sizing_unchanged':True,'orders_sent':0}))


if __name__=='__main__':main()
