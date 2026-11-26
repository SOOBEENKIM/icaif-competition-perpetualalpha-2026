"""Change policy while keeping the CURRENT position book; never restore old fills.

Both our signal producer and writer must be stopped before running this tool.
It only changes local state, and does not start services or load a signer.
"""
import argparse,fcntl,json,shutil,time
from pathlib import Path
from . import producer
from .entry_policy import POLICIES,configured_policy
from .locks import WriterLease
from .migration import load_existing
from .runner import require_contract
from competition_live.profile import ROOT,load_profile,verify_deployment
from competition_live.runtime import json_save,save_coordinator

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--signal',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args();d=args.directory.resolve();config=json.loads(args.config.read_text())
    if config['policy'] not in POLICIES:raise ValueError('Unsupported policy')
    profile=load_profile();verify_deployment();require_contract(d,profile)
    packet=json.loads(args.signal.read_text());revision=producer.configured_revision(d)
    producer.validate(packet,profile,time.time(),expected_revision=revision,expected_policy=config['policy'])
    if packet['body']['frame']['source_time']!=int(time.time()//3600)*3600:
        raise ValueError('Switch requires current completed reference hour')
    if args.archive.exists():raise FileExistsError('Switch archive already exists')
    with WriterLease(ROOT),(d/'producer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        portfolio,digest=load_existing(d/'coordinator.pkl',profile)
        dispatch=json.loads((d/'dispatch.json').read_text())
        if dispatch.get('pending'):raise ValueError('Unresolved dispatcher order')
        previous=configured_policy(d)
        args.archive.mkdir(parents=True,mode=0o700)
        names=['coordinator.pkl','ownership.json','dispatch.json','signal.json','execution_policy.json',
            'strategy_revision.json','team_execution_contract.json','status.json']
        for name in names:
            if (d/name).exists():shutil.copy2(d/name,args.archive/name);(args.archive/name).chmod(0o600)
        contract=json.loads((d/'team_execution_contract.json').read_text());contract['required_execution_policy']=config['policy']
        # Last decision/counters and every existing fill survive. Only an
        # unsubmitted prior-policy target may be discarded.
        portfolio.base_task=None;portfolio.execution_policy=config['policy']
        portfolio.events.append(dict(kind='execution_policy_switch',time=time.time(),previous=previous,current=config['policy']))
        save_coordinator(d/'coordinator.pkl',portfolio,profile,'execution_policy_switched_orders_disabled')
        json_save(d/'execution_policy.json',dict(config,activated_at=time.time()))
        json_save(d/'team_execution_contract.json',contract);json_save(d/'signal.json',packet)
        assert configured_policy(d)==config['policy']
        json_save(d/'status.json',dict(at=time.time(),state='policy_switched_waiting_start',orders_enabled=False,
            core_revision=revision,execution_policy=config['policy'],previous_policy=previous))
        result=dict(at=time.time(),previous_policy=previous,execution_policy=config['policy'],
            prior_checkpoint_sha256=digest,last_decision=portfolio.last_decision,
            own_qty=portfolio.account.physical.qty.tolist(),teammate_qty=portfolio.account.reserved.qty.tolist(),
            own_fees=portfolio.account.physical.fees,own_funding=portfolio.account.physical.funding,
            orders_sent=0,signer_loaded=False,archive=str(args.archive.resolve()))
        json_save(args.archive/'switch_result.json',result);print(json.dumps(result))

if __name__=='__main__':main()
