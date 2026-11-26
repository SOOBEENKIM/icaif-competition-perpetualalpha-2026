"""Apply the user-requested margin-only PAS testnet change, then resume.

The installer rolls source/config back on failure. Neither this script nor the
installer submits orders; the existing authorized worker resumes normal work.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUNTIME=ROOT/'.runtime/pas_budget_lifecycle'
LIVE='icaif-pas-budget-live.service'
SIGNALS='icaif-pas-budget-signals.service'


def run(argv, **kwargs):
    return subprocess.run(argv,check=True,text=True,capture_output=True,timeout=45,**kwargs)


def resume(unit, module, log, restart, execute=False):
    # A stopped transient systemd unit may have been garbage-collected.
    subprocess.run(['systemctl','--user','reset-failed',unit],capture_output=True,timeout=10)
    argv=['systemd-run','--user','--unit='+unit,
          '--description=ICAIF PAS '+('margin-only automatic leverage worker' if execute else 'pullback_any3 signal producer'),
          '--property=WorkingDirectory='+str(ROOT),
          '--property=StandardOutput=append:'+str(RUNTIME/log),
          '--property=StandardError=inherit','--property=Restart='+restart]
    if restart!='no':argv+=['--property=RestartSec=30']
    argv += [str(ROOT/'.venv/bin/python'),'-m',module,'--directory',str(RUNTIME)]
    if execute:argv+=['--execute']
    return run(argv).stderr.strip()


def main():
    for unit in [LIVE,SIGNALS]:
        assert run(['systemctl','--user','is-active',unit]).stdout.strip()=='active',unit
    (OUT/'activation_started.json').write_text(json.dumps({'at':time.time(),'services':[LIVE,SIGNALS]}))
    run(['systemctl','--user','stop',LIVE,SIGNALS])
    installation_error=None
    try:
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT)
        result=run([str(ROOT/'.venv/bin/python'),str(OUT/'install.py')],cwd=ROOT,env=env)
        (OUT/'install_stdout.txt').write_text(result.stdout)
    except Exception as error:
        installation_error=error
        (OUT/'install_error.txt').write_text(str(error)+getattr(error,'stderr',''))
    # The installer guarantees originals on mutation failure. Resume the same
    # existing strategy even if installation could not proceed.
    messages=[]
    messages.append(resume(SIGNALS,'candidate_handoff.producer','producer.log','on-failure'))
    messages.append(resume(LIVE,'candidate_handoff.runner','worker.log','no',execute=True))
    (OUT/'service_start_receipt.json').write_text(json.dumps({'at':time.time(),'messages':messages,
        'installed':installation_error is None,'worker_restart_policy':'no','signals_restart_policy':'on-failure'}))
    if installation_error is not None:raise installation_error
    print(json.dumps({'installed':True,'services_started':True,'verification_pending':True}))


if __name__=='__main__':main()
