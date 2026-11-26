"""Separate, explicit capital-risk experiment; never disguise leverage as alpha."""
import json
from datetime import datetime,timezone
from pathlib import Path
from . import anchored
from pas_adaptive_development.study import ROOT,save,sha

OUT=ROOT/'reports/pas_economic_budget_20261022'


def prepare():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=json.loads((anchored.OUT/'protocol.json').read_text())
    p.update(created_utc=datetime.now(timezone.utc).isoformat(),
        reason='The repaired original-scale anchor remains near the previous low deployment. Test a disclosed0.2% pre-gate daily sizing budget versus the inherited0.1%, still below the0.5% hard cap. Final gates reduce this desired exposure; no rescale after gates.',
        ratio=2*anchored.RATIO,
        interpretation='This changes risk/capital allocation, not predictive accuracy. Compare both old incumbent, old size-only and repaired_anchor at the SAME new budget to isolate incremental alpha. Any PnL gain must be reported with actual volatility, drawdown, turnover and average notional.',
        threshold_selection='Exactly2x the original pre-gate daily budget; no parameter sweep and no reselecting using this run outcomes. Halfway orders remain subject to original7500 per-asset and30k gross caps.',
        primary='repaired_augmented',parent='pas_economic_anchor_20261022',
        deployment=verify_deployment()['files'])
    save(OUT/'protocol.json',p)
    files=[Path(__file__),Path(anchored.__file__),Path(__file__).with_name('policy.py'),Path(__file__).with_name('replay.py'),
           OUT/'protocol.json',ROOT/p['final_risk_inputs'],ROOT/p['forecasts']]
    save(OUT/'source_freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})


def run():
    p=json.loads((OUT/'protocol.json').read_text())
    anchored.OUT=OUT;anchored.RATIO=p['ratio'];anchored.run()


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
