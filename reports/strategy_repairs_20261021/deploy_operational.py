"""Install the tested operational-only patch; rollback code on failed recovery."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from competition_live.profile import verify_deployment,verify_bundle
from competition_live.runtime import private_load,atomic,json_save

OUT=Path(__file__).resolve().parent
STAGE=OUT/'release/competition_live'
UNIT='icaif-pas-scored-live.service'
NAMES=('runner.py','live_account.py','coordinator.py','runtime.py',
       'test_runtime.py','test_operational_repairs.py')


def service(*args):
    return subprocess.run(['systemctl','--user',*args],check=True,text=True,capture_output=True).stdout


def main():
    old=verify_deployment();bundle=verify_bundle()
    assert 'Ran 31 tests' in (OUT/'operational_after.txt').read_text()
    assert (OUT/'operational_after.txt').read_text().rstrip().endswith('OK')
    shadow=json.loads((OUT/'checkpoint_shadow.json').read_text())
    assert shadow['orders_sent']==0 and not shadow['signer_loaded']
    status_path=ROOT/'.runtime/pas_competition_live/status.json'
    status=json.loads(status_path.read_text())
    assert status['state']=='running' and not status['fatal'] and time.time()-status['at']<90
    state=private_load(ROOT/'.runtime/pas_competition_live/coordinator.pkl')['portfolio']
    assert not state.account.pending_orders and not state.account.unknown
    assert state.account.reserved.cash==50000 and not state.account.reserved.qty.any()
    assert service('is-active',UNIT).strip()=='active'
    backup=OUT/'deployment_backup';backup.mkdir(mode=0o700,exist_ok=False)
    for name in (*NAMES,'deployment_manifest.json'):
        src=ROOT/'competition_live'/name
        if src.exists():shutil.copy2(src,backup/name)
    for name in ('coordinator.pkl','status.json','signal.json'):
        shutil.copy2(ROOT/'.runtime/pas_competition_live'/name,backup/name)
    diff=[]
    for name in NAMES:
        before=backup/name
        diff.extend(difflib.unified_diff(
            before.read_text().splitlines(True) if before.exists() else [],
            (STAGE/name).read_text().splitlines(True),
            fromfile='before/competition_live/'+name,tofile='after/competition_live/'+name))
    (OUT/'operational.patch').write_text(''.join(diff))
    revised=dict(old,updated_at=time.time(),operational_revision='20261021-recovery-freshness-v1')
    revised['files']=dict(old['files'])
    for name in NAMES:
        revised['files']['competition_live/'+name]=hashlib.sha256((STAGE/name).read_bytes()).hexdigest()
    json_save(OUT/'release_manifest.json',revised)
    result=dict(started_at=time.time(),unit=UNIT,bundle_hash=bundle,
        candidate=old['candidate'],strategy_weights_changed=False,profile_changed=False,
        baseline_inference_changed=False,risk_candidate_activated=False)
    try:
        for name in NAMES:atomic(ROOT/'competition_live'/name,(STAGE/name).read_bytes())
        json_save(ROOT/'competition_live/deployment_manifest.json',revised)
        verify_deployment()
        result['restart_at']=time.time()
        service('restart',UNIT)
        deadline=time.time()+90
        while time.time()<deadline:
            current=json.loads(status_path.read_text())
            if (current['at']>result['restart_at'] and current['state']=='running'
                    and 'unpriced_flat_assets' in current and not current['fatal']):
                result.update(state='deployed_and_running',after=current,
                    active=service('is-active',UNIT).strip())
                json_save(OUT/'deployment_result.json',result)
                print(json.dumps({k:v for k,v in result.items() if k!='after'}))
                return
            time.sleep(2)
        raise RuntimeError('No healthy fresh status within recovery window')
    except Exception as error:
        # Roll back code only. Never replace the current fill/order checkpoint
        # with an older copy, since orders could have completed after backup.
        for name in (*NAMES,'deployment_manifest.json'):
            source=backup/name;target=ROOT/'competition_live'/name
            if source.exists():atomic(target,source.read_bytes())
            elif target.exists():target.unlink()
        service('restart',UNIT)
        result.update(state='code_rolled_back',error=str(error))
        json_save(OUT/'deployment_result.json',result)
        raise


if __name__=='__main__':
    main()
