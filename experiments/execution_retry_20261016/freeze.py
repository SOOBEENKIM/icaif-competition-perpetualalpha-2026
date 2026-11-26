"""Reserve a future interval only after all evaluator checks pass."""
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from replay_lab.evaluation_control.registry import Registry, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    destination = HERE / 'forward_reservation.json'
    if destination.exists():
        raise ValueError('A reservation already exists; never silently refreeze')
    now = int(time.time())
    p = json.loads((HERE / 'protocol.json').read_text())
    protected = json.loads((ROOT / 'experiments/kronos_20261015/preserved_before.json').read_text())
    changed = [name for name, digest in protected.items() if sha(ROOT / name) != digest]
    if changed:
        raise ValueError('Protected strategy files changed: ' + str(changed))
    artifacts = {str(ROOT / name): digest for name, digest in protected.items()}
    paths = list(HERE.glob('*.py')) + [HERE / 'protocol.json', HERE / 'forward_protocol.json', HERE / 'raw/meta.json']
    paths += list((ROOT / 'practice_live').glob('*.py'))
    paths += [ROOT / name for name in ('bot/execution.py', 'bot/connection.py', 'scripts/refresh_public_wallets.py',
              'replay_lab/evaluation_control/registry.py')]
    artifacts.update({str(path): sha(path) for path in paths})
    fit = ROOT / 'replay_lab/output/tail_calibration/20261014-v1'
    regime = json.loads((fit / 'regime_fit.json').read_text())
    allocation = json.loads((fit / 'allocation_fit.json').read_text())
    latest_fit = max(regime['last_training_time'], allocation['last_training_time'])
    with np.load(HERE / 'frozen_inputs/inputs.npz') as z:
        first = int(z['market_times'][0])
    plan = {
        'evaluation_kind': 'independent_latest_period',
        'train_start': first, 'train_end': regime['cutoff'],
        'max_train_label_maturity': latest_fit, 'preprocessing_fit_last_time': latest_fit,
        'validation_start': regime['cutoff'], 'validation_end': now,
        'selection_last_time': now, 'model_knowledge_last_time': now,
        'test_start': (now // 60 + 6) * 60,
        'test_end': int(datetime.fromisoformat(p['forward_capture_end_utc']).timestamp()),
        'comparators': ['default_strategy_single_IOC', 'same_default_strategy_bounded_retry'],
        'success_rule': p['promotion_gates'], 'decision_rule': p['decision_rule'],
        'artifacts': artifacts, 'production_approved': False,
        'training_provenance': 'No model fitted in this study. Local regime/allocation fit bounds are copied from preserved fit artifacts; foundation-model historical training corpus is not claimed to be audited. Conservative model knowledge bound is the current freeze time, before the new market period.',
        'selection_provenance': 'All earlier strategy and execution results are development; no reused historical window is independent. Full current signal code and frozen model/strategy artifacts are hashed.',
        'known_sample_limit': 'This practice interval supplies at most two complete UTC daily returns; the three-day gate cannot pass.',
        'forward_rules': json.loads((HERE / 'forward_protocol.json').read_text()),
        'capture': str(HERE / 'forward_capture'), 'live_orders_sent': 0,
    }
    ident = Registry().freeze(plan)
    destination.write_text(json.dumps({'trial_id': ident, 'frozen_at': now, 'plan': plan}, indent=2))
    print(json.dumps({'trial_id': ident, 'start': plan['test_start'], 'end': plan['test_end'],
                      'frozen_artifacts': len(artifacts), 'protected_files_unchanged': len(protected)}))


if __name__ == '__main__':
    main()
