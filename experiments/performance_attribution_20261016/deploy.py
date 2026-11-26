"""Prepare a reviewable freeze; activate only through an explicit CLI action."""
import argparse,fcntl,json,os,sqlite3,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from practice_live.refresh import ROOT,OUT as LIVE,save_json,sha,iso
from .live import SERVICE
from .live_signal import DIRECTORY,PROFILE,read_publication

OLD='icaif-par-practice.service'

def command(argv,check=True):
    r=subprocess.run(argv,text=True,capture_output=True,timeout=55)
    if check and r.returncode:raise RuntimeError('Command failed: '+str(argv)+'\n'+r.stderr)
    return r

def prepare():
    checks=(DIRECTORY/'tests.txt').read_text()
    if 'Ran 45 tests' not in checks or not checks.rstrip().endswith('OK'):raise ValueError('Regression checks not passed')
    pre=json.loads((DIRECTORY/'preflight.json').read_text())
    if not pre['passed'] or pre['orders_sent'] or pre['production_database_written']:raise ValueError('Invalid preflight')
    protected=json.loads((ROOT/'experiments/risk_preserving_20261016/artifacts/protected_before.json').read_text())
    for path,digest in protected.items():
        if sha(ROOT/path)!=digest:raise ValueError('Existing source changed: '+path)
    reserved=json.loads((ROOT/'experiments/execution_retry_20261016/forward_reservation.json').read_text())['plan']['artifacts']
    for path,digest in reserved.items():
        if sha(path)!=digest:raise ValueError('Protected trial file changed: '+path)
    files={**protected}
    for folder in [Path(__file__).parent,ROOT/'experiments/risk_preserving_20261016']:
        files.update({str(p.relative_to(ROOT)):sha(p) for p in folder.glob('*.py')})
    for name in ['artifacts/budget_frozen.json','budget_protocol.json']:
        p=Path(__file__).parent/name;files[str(p.relative_to(ROOT))]=sha(p)
    manifest={'created_at_utc':iso(),'profile':PROFILE,'sources':files,'existing_files_unchanged':len(protected),
        'reserved_files_unchanged':len(reserved),'tests_passed':45,'real_testnet_orders_authorized_by_user':True,
        'order_store':'practice_live/live_runtime/state.sqlite3','single_writer_lock':'practice_live/live_runtime/process.lock',
        'service':SERVICE,'old_service':OLD,'normal_round_kst_hours':[3,9,15,21]}
    save_json(DIRECTORY/'deployment_manifest.json',manifest)
    plan={'new_service':SERVICE,'old_service':OLD,'command':[str(ROOT/'.venv/bin/python'),'-m','experiments.performance_attribution_20261016.live','--execute'],
        'preserve_existing_positions':True,'reset_account_equity':False,'forced_rebalance_on_switch':False,
        'signal_process_independent':True,'keep_original_sources':True,'new_performance_epoch':True}
    save_json(DIRECTORY/'activation_plan.json',plan)
    print(json.dumps({'prepared':True,'manifest_sha256':sha(DIRECTORY/'deployment_manifest.json'),'tests':45}),flush=True)

def activate():
    from .live import verify_deployment
    digest=verify_deployment();pre=json.loads((DIRECTORY/'preflight.json').read_text())
    if time.time()-datetime.fromisoformat(pre['at_utc']).timestamp()>900:raise ValueError('Refresh preflight before handoff')
    f,obj=read_publication()
    if time.time()-f.source_time>3600:raise ValueError('Current-hour candidate publication required for handoff')
    if command(['systemctl','--user','is-active',SERVICE],False).stdout.strip()=='active':raise ValueError('Candidate service already active')
    if command(['systemctl','--user','is-active',OLD],False).stdout.strip()!='active':raise ValueError('Expected incumbent writer is not active')
    oldpid=command(['systemctl','--user','show',OLD,'-p','MainPID','--value']).stdout.strip()
    save_json(DIRECTORY/'before_stop_status.json',json.loads((LIVE/'live_runtime/status.json').read_text()))
    save_json(DIRECTORY/'activation_started.json',{'at_utc':iso(),'old_pid':oldpid,'deployment_sha256':digest})
    print('Stopping incumbent writer without closing positions',flush=True)
    command(['systemctl','--user','stop',OLD])
    if command(['systemctl','--user','is-active',OLD],False).stdout.strip()=='active':raise ValueError('Incumbent did not stop')
    directory=LIVE/'live_runtime'
    with (directory/'process.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=sqlite3.connect('file:'+str(directory/'state.sqlite3')+'?mode=ro',uri=True)
        destination=DIRECTORY/'state_before_switch.sqlite3'
        if destination.exists():raise FileExistsError('Do not overwrite an existing handoff backup')
        copied=sqlite3.connect(destination);source.backup(copied);copied.close();source.close();os.chmod(destination,0o600)
    args=['systemd-run','--user','--unit='+SERVICE,'--description=ICAIF par practice frozen budget candidate',
        '--property=WorkingDirectory='+str(ROOT),'--property=Restart=on-failure','--property=RestartSec=10',
        '--property=KillMode=control-group','--property=TimeoutStopSec=45',
        '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=HF_HUB_OFFLINE=1','--setenv=HF_HUB_DISABLE_TELEMETRY=1',
        str(ROOT/'.venv/bin/python'),'-m','experiments.performance_attribution_20261016.live','--execute']
    print('Starting candidate with the same exclusive lock and durable order store',flush=True)
    result=command(args)
    save_json(DIRECTORY/'activation_started.json',{'at_utc':iso(),'old_pid':oldpid,'old_service_stopped':True,
        'candidate_service_start_requested':True,'deployment_sha256':digest,'service':SERVICE,'order_store_backup_sha256':sha(destination)})
    print(result.stderr.strip(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','activate']);a=p.parse_args()
    prepare() if a.action=='prepare' else activate()
