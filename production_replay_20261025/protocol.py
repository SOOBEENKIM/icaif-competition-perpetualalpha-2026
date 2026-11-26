"""Versioned research contract; never reads credentials or changes live state."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/production_replay_20261025'
DATA=ROOT/'reports/execution_revision_20261024/dataset'

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist()))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze():
    path=OUT/'protocol.json'
    if path.exists():raise FileExistsError(path)
    save(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Compare identical production decision, risk-priority and LiveAccount code with a strictly offline synthetic venue. No real exchange executor or signer.',
        primary='fixed_entry_1h',controls=['incumbent','refresh_entry_1h'],
        old_window=['2026-10-19T12:00:00Z','2026-10-24T12:00:00Z'],
        proposed_unopened_window=['2026-10-24T18:00:00Z','2026-10-25T02:00:00Z'],
        unopened_rule='Only primary and incumbent. First complete production-path tests and old-history development. Record input hashes and first-open time. This retrospective interval is not a preregistered prospective test; aggregate account observations may be indirectly known. No tuning on it followed by relabeling it independent.',
        exposure='Oct19-Oct24 12UTC repeatedly inspected development history. No independent-test claim.',
        execution=dict(valuation='Last completed PAS trade candle, carried forward with explicit age. NOT reconstructed mark price.',
            quote='Synthetic book around last completed PAS close; 2bp half-spread; sizes from prior minute volume times participation. No book on inactive minute.',
            arrival='Each sequential IOC consumes the next complete minute. Future high for buy/low for sell plus adverse impact, only if inside sealed production limit. Prior/future volume participation minimum; shared per-minute budget; no assumed fill on price touch.',
            funding='Settled PAS cash per unit from independently reconciled public funding records; nonzero missing cash is an error. Positive receipts haircut in stress.',
            signal='Immutable published reference frames, actual known time, completed PAS entry-filter history. No refit.',
            model_limit='No historical order queue, exact mark, book depth or official account equity reconstruction. All PnL conditional; no silent claim actual fills.'),
        scenarios={'base':dict(fee_bps=9.,impact_bps=10.,participation=.01,receipt_multiplier=1.),
                   'stress':dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=.5),
                   'no_receipts':dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=0.)},
        acceptance='Common50k sleeve+50k idle reserve; no borrowing. Positive net PnL and PnL/Q improvement in base and stress, paid terminal exit reserve, same-inventory switch. No accounting/time/risk or production-path violations. Quote-independent forward evidence required before claiming execution improvement.',
        live_changed=False,orders_sent=0,
        candidate_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'execution_revision_20261024/portfolio.py',ROOT/'execution_revision_20261024/guarded_portfolio.py']}))

if __name__=='__main__':freeze()
