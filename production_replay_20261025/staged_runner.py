"""Prepared runner for a qualified, explicitly staged checkpoint.

Nothing is installed on import. Normal invocation without --execute remains
the existing unsigned rehearsal. A future live cutover requires its release
manifest and checkpoint to be staged together while the old writer is stopped.
"""
import argparse,json
from pathlib import Path
from competition_live.runtime import private_load
from .protocol import ROOT,OUT,sha
from .runtime_adapter import account_class,wire_candidate


def verify_release(directory):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(ROOT):raise ValueError('Release directory outside project')
    manifest=json.loads((directory/'research_release.json').read_text())
    if manifest.get('schema')!='production_replay_20261025_qualified_release_v1':raise ValueError('Wrong release schema')
    if manifest.get('source_checkpoint_is_live_reconciled') is not True:
        raise ValueError('A research seed is not a live handover checkpoint')
    if manifest.get('short_paper_gate_passed') is not True:raise ValueError('No completed candidate qualification')
    # A caller-written boolean is not qualification evidence. Require the
    # actual full audit (including slow execution and actual-ledger mismatch),
    # and bind it to the unchanged comparison artifacts.
    evidence=OUT/'qualification_review/results.json'
    if manifest.get('qualification_sha256')!=sha(evidence):
        raise ValueError('Missing or changed full qualification evidence')
    review=json.loads(evidence.read_text())
    arm=manifest.get('candidate_arm')
    matches=[g for g in review['experiments'] if g['primary']==arm]
    if len(matches)!=1 or matches[0]['verdict']['short_conditional_trial_eligible'] is not True:
        raise ValueError('Candidate did not pass the full recorded comparison')
    for relative,digest in review['source_results'].items():
        source=(OUT/relative).resolve()
        if not source.is_relative_to(OUT) or sha(source)!=digest:
            raise ValueError('Qualification result dependency changed')
    classes={'fixed_entry_1h':'GuardedPortfolio','fresh_block_catchup':'CatchupPortfolio',
             'frozen_side_quote':'QuotePortfolio','quote_inputs':'QuoteHistoryRuntime',
             'size_only_2':'BudgetPortfolio'}
    if classes.get(arm)!=manifest.get('portfolio_class'):
        raise ValueError('Qualification belongs to a different candidate')
    if not manifest.get('sources'):raise ValueError('No source provenance in release')
    for relative,digest in manifest['sources'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=digest:raise ValueError('Staged source differs: '+relative)
    state=private_load(directory/'coordinator.pkl')['portfolio']
    if type(state).__name__!=manifest['portfolio_class']:raise ValueError('Candidate/checkpoint mismatch')
    expected=account_class(state)
    if type(state.account) is not expected:raise ValueError('Checkpoint guard is missing')
    if getattr(state,'execution_revision',None)!=manifest['execution_revision']:
        raise ValueError('Unexpected execution revision')
    return manifest


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--directory',required=True)
    args,_=p.parse_known_args()
    verify_release(args.directory)
    from candidate_handoff import runner
    from joint_wallet import pas_gateway
    from competition_live import execution
    from .venue_guard import check_oi_strict
    # Applies only in this new process. Existing worker/source are untouched.
    runner.wire=wire_candidate
    pas_gateway.check_oi=check_oi_strict
    execution.check_oi=check_oi_strict
    runner.main()


if __name__=='__main__':main()
