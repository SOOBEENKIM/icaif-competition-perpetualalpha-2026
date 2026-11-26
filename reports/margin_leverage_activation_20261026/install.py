"""Local atomic installation after BOTH budget services have stopped.

No signer, service startup, orders, or exchange setting calls. Rollback copies
are for code/config only; never restore old holdings after resumed trading.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
D = ROOT / '.runtime/pas_budget_lifecycle'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic(path, data):
    tmp = path.with_name(path.name + '.margin-install-tmp')
    with tmp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def save(path, value): atomic(path, json.dumps(value, indent=2).encode())


def main():
    from candidate_handoff.locks import WriterLease
    from candidate_handoff.entry_policy import configured_policy
    from candidate_handoff.producer import validate
    from competition_live.profile import load_profile, verify_deployment
    from joint_wallet.leverage import DEFAULT, MODE, load_policy
    original = json.loads((OUT / 'original_source_hashes.json').read_text())
    for name, digest in original.items():
        assert sha(ROOT / name) == digest, 'Original live source changed: ' + name
    verify_deployment(); profile = load_profile()
    packet = json.loads((D / 'signal.json').read_text())
    validate(packet, profile, time.time(), expected_revision='all_market_signed_core_v1', expected_policy='pullback_any3')
    assert packet['body']['frame']['source_time'] == int(time.time() // 3600) * 3600
    rollback = OUT / 'rollback'
    if rollback.exists(): raise FileExistsError('Do not overwrite the rollback archive')
    with WriterLease(ROOT), (D / 'producer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert configured_policy(D) == 'pullback_any3'
        dispatch = json.loads((D / 'dispatch.json').read_text())
        assert not dispatch.get('pending'), 'Pending order: reconcile before installation'
        rollback.mkdir(mode=0o700)
        state_names = ['coordinator.pkl', 'ownership.json', 'dispatch.json', 'signal.json',
                       'execution_policy.json', 'strategy_revision.json', 'team_execution_contract.json', 'status.json']
        for name in state_names:
            shutil.copy2(D / name, rollback / name)
        shutil.copy2(ROOT / '.runtime/active_pas_strategy.json', rollback / 'active_pas_strategy.json')
        for name in original:
            path = rollback / 'source' / name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, path)
        unchanged = ['candidate_handoff/portfolio.py', 'candidate_handoff/producer.py',
                     'candidate_handoff/risk_priority.py', 'pas_performance_research/selected.py',
                     'pas_performance_research/soft_timing.py', 'competition_live/config.json',
                     'joint_wallet/book.py', 'joint_wallet/dispatch.py']
        old_hash = {name: sha(ROOT / name) for name in unchanged}
        config = json.loads((D / 'execution_policy.json').read_text())
        contract = json.loads((D / 'team_execution_contract.json').read_text())
        pointer = json.loads((ROOT / '.runtime/active_pas_strategy.json').read_text())
        try:
            for name in original:
                atomic(ROOT / name, (OUT / 'stage' / name).read_bytes())
            for name in [*original, 'joint_wallet/leverage.py']:
                config['source_sha256'][name] = sha(ROOT / name)
            config.update(leverage_revision=MODE, leverage_activated_at=time.time(),
                          leverage_activation_report=str(OUT / 'report.html'))
            contract['required_leverage_policy'] = MODE
            pointer.update(leverage_policy=MODE, leverage_policy_activated_at=time.time(),
                           leverage_activation_report=str(OUT / 'report.html'),
                           leverage_rollback_archive=str(rollback), leverage_scope='margin_only_no_target_resize')
            save(D / 'leverage_policy.json', dict(DEFAULT, activated_at=time.time()))
            save(D / 'team_execution_contract.json', contract)
            save(D / 'execution_policy.json', config)
            save(ROOT / '.runtime/active_pas_strategy.json', pointer)
            assert configured_policy(D) == 'pullback_any3'
            assert load_policy(D) == DEFAULT
            verify_deployment()
            assert old_hash == {name: sha(ROOT / name) for name in unchanged}
            # Never rewrite model, signals, inventory, pending tasks or fills.
            for name in ['coordinator.pkl', 'ownership.json', 'dispatch.json', 'signal.json']:
                assert sha(D / name) == sha(rollback / name), name
            save(OUT / 'install_receipt.json', dict(at=time.time(),mode=MODE,
                old_sources=original,new_sources={name:sha(ROOT/name) for name in [*original,'joint_wallet/leverage.py']},
                strategy_sources_unchanged=old_hash,orders_sent=0,setting_updates=0,
                inventory_restored=False,inventory_preserved=True,services_started=False,rollback=str(rollback)))
        except BaseException:
            for name in original: atomic(ROOT / name, (rollback / 'source' / name).read_bytes())
            for name in ['execution_policy.json', 'team_execution_contract.json']:
                atomic(D / name, (rollback / name).read_bytes())
            atomic(ROOT / '.runtime/active_pas_strategy.json', (rollback / 'active_pas_strategy.json').read_bytes())
            (D / 'leverage_policy.json').unlink(missing_ok=True)
            raise
    print(json.dumps({'installed':MODE,'orders_sent':0,'setting_updates':0,'holdings_preserved':True}))


if __name__ == '__main__': main()
