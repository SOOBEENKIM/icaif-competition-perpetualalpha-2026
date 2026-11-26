import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from competition_live.profile import ROOT, verify_deployment

OUT = ROOT / "reports/all_market_alpha_20261020"
PRIOR = ROOT / "reports/isolated_repairs_20261020"
ARMS = ["incumbent", "multiscale", "pooled_ridge", "equal_ensemble"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        return json.loads(path.read_text())
    previous = json.loads((PRIOR / "protocol.json").read_text())
    spec = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        arms=ARMS, pointer=previous["pointer"], historical_independence=False,
        prior_trials="Includes all earlier project research; these historical periods are reused development evidence.",
        hypotheses=[
            "Multi-day normalized trends can adapt faster than the incumbent daily 10/60 EMA.",
            "A pooled, regularized model can learn when trend versus reversion is useful without named-asset restrictions.",
            "A fixed equal ensemble can reduce dependence on either hypothesis.",
        ],
        features=["momentum_24h", "momentum_72h", "momentum_168h",
                  "momentum_336h", "deviation_168h", "volatility_ratio",
                  "cross_asset_momentum_72h", "past_funding_24h"],
        model=dict(horizon=24, execution_delay=1, lookback_days=90,
                   half_life_days=45, sample_hours=24, refit_hours=24,
                   ridge_penalty=.1, minimum_rows=200, target_clip=5,
                   forecast_clip=2, training_maturity="strictly earlier than refit"),
        policy=dict(decision_hours=6, daily_risk_budget=.0006,
                    entry_signal=.25, exit_signal=.10, roundtrip_cost=.0038,
                    minimum_warmup_hours=672, named_asset_filter=False,
                    universe="All 10 PAS assets, subject to available observations and execution liquidity.",
                    size="Equal ex-ante risk per asset times bounded conviction; downward caps only. No forced deployment."),
        pair_overlay="Retain the exact two incumbent pair models, gate, account and execution for the controlled comparison. "
                     "The new directional model covers all 10 assets; this does NOT expand the pair-model universe.",
        selection="August PnL AND team Q must exceed incumbent under BOTH base and stress. "
                  "Zero timing/accounting/risk violations. Choose greatest worst-scenario PnL advantage, then name. "
                  "Freeze nomination before later runs. No parameter search on PAS.",
        comparison="Report September 1–October 15 and continuous August 1–October 15 for EVERY candidate, "
                   "including failed nominees. Report base, stress, completed trades, exposure and daily metrics.",
        pas="Reuse fixed Oct19 21:00–Oct20 22:44 KST conditional PAS data, never for fitting or nomination. "
            "Missing historical books limits execution validity; no unsupported sigma/Q.",
        sources=[
            dict(url="https://arxiv.org/abs/2607.00475",
                 use="Cross-asset policy comparison and turnover costs; no reproduction or transfer-of-profit claim."),
            dict(url="https://arxiv.org/abs/2302.10175",
                 use="Time-series plus cross-sectional normalized features; independent pooled ridge implementation."),
        ],
        deployment_hashes=verify_deployment()["files"], orders_sent=0,
        input_hashes=previous["source_hashes"],
    )
    path.write_text(json.dumps(spec, indent=2))
    return spec


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
