"""Read-only diagnosis of frozen historical runs; never reads forward outcomes.

Run from the repository root: .venv/bin/python reports/strategy_failure_audit_20261017/audit.py
All monetary decompositions preserve the recorded quantity path. Removing a
cost in that decomposition is attribution, not a newly executable strategy.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.cost_horizon_20261016.model import features, labels

OUT = Path(__file__).resolve().parent
ART = ROOT / "experiments/cost_horizon_20261016/artifacts"
INPUT = ROOT / "experiments/risk_preserving_20261016/artifacts/inputs.npz"
SYMBOLS = ["BTC", "ETH", "HYPE", "SP500", "EWY", "EWJ", "BRENT", "GOLD", "USDEUR", "USDJPY"]
SOURCES = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = digest(path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def stamp(s):
    return int(np.datetime64(s, "s").astype("int64"))


def scalar(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


def explain_run(path, inp):
    r = read(path)
    tt = np.array([s["time"] for s in r["snapshots"]])
    ii = np.searchsorted(inp["market_times"], tt)
    np.testing.assert_array_equal(inp["market_times"][ii], tt)
    prices = inp["close"][ii]
    q = np.array([s["qty"] for s in r["snapshots"]])
    eq = np.array([s["equity"] for s in r["snapshots"]])
    moves = q[:-1] * np.diff(prices, axis=0)
    move = moves.sum(0)
    long_move = np.where(q[:-1] > 0, moves, 0).sum(0)
    short_move = np.where(q[:-1] < 0, moves, 0).sum(0)
    execution = np.zeros(10)
    fees = np.zeros(10)
    funding = np.zeros(10)
    fill_counts = np.zeros(10, dtype=int)
    fill_notional = []
    same_direction_count = 0
    for f in r["fills"]:
        j = f["asset"]
        i = np.searchsorted(tt, f["time"])
        execution[j] += f["signed_qty"] * (f["price"] - prices[i, j])
        fees[j] += f["fee"]
        fill_counts[j] += 1
        fill_notional.append(abs(f["signed_qty"] * f["price"]))
        same_direction_count += f["position_before"] * f["position_after"] > 0
    adverse_bps = 1.0 if r["scenario"] == "stress" else 0.0
    for f in r["funding_events"]:
        notionals = np.asarray(f["qty"]) * np.asarray(f["prices"])
        funding += -notionals * np.asarray(f["rates"]) - abs(notionals) * adverse_bps / 10000
    net = move + funding - execution - fees
    assert abs(net.sum() - r["metrics"]["net_pnl"]) < 1e-5
    old = r["independent_accounting_audit"]
    for k, v in [("price_move_pnl", move.sum()), ("execution_cost", execution.sum()),
                 ("fees", fees.sum()), ("funding", funding.sum())]:
        assert abs(old[k] - v) < 1e-5
    weights = q * prices / eq[:, None]
    rows = []
    for j, s in enumerate(SYMBOLS):
        rows.append(dict(symbol=s, price_move_pnl=move[j], long_price_pnl=long_move[j],
                         short_price_pnl=short_move[j], execution_cost=execution[j], fees=fees[j],
                         funding=funding[j], net_pnl=net[j], fills=fill_counts[j],
                         long_hours=int((q[:-1, j] > 0).sum()), short_hours=int((q[:-1, j] < 0).sum()),
                         mean_abs_weight=float(abs(weights[:, j]).mean()),
                         underlying_return=float(prices[-1, j] / prices[0, j] - 1)))
    m = r["metrics"]
    return dict(path=str(path.relative_to(ROOT)), metrics=m, per_asset=rows,
                price_move_pnl=move.sum(), funding=funding.sum(), fees=fees.sum(),
                execution_cost=execution.sum(), net_pnl=net.sum(),
                pre_execution_and_fees_pnl=move.sum() + funding.sum(),
                mean_estimated_daily_vol=float(np.mean([s["daily_volatility"] for s in r["snapshots"]])),
                max_estimated_daily_vol=max(s["daily_volatility"] for s in r["snapshots"]),
                same_direction_adjustment_fills=same_direction_count,
                median_fill_notional=float(np.median(fill_notional)) if fill_notional else None,
                break_even_allin_bps_on_recorded_path=float((move.sum()+funding.sum())/m["turnover"]*10000),
                actual_allin_bps=float((execution.sum()+fees.sum())/m["turnover"]*10000),
                terminal_liquidation_not_charged=True,
                accounting_residual=float(net.sum()-m["net_pnl"]))


def forecast_stats(pred, actual):
    xx, yy = pred.ravel(), actual.ravel()
    denom = np.sum((xx-xx.mean())**2)
    return dict(n=len(xx), mean_prediction=float(xx.mean()), mean_actual=float(yy.mean()),
                prediction_std=float(xx.std()), actual_std=float(yy.std()),
                positive_prediction_fraction=float(np.mean(xx > 0)), positive_actual_fraction=float(np.mean(yy > 0)),
                mse=float(np.mean((xx-yy)**2)), zero_mse=float(np.mean(yy**2)),
                correlation=float(np.corrcoef(xx,yy)[0,1]),
                calibration_slope=float(np.sum((xx-xx.mean())*(yy-yy.mean()))/denom),
                mean_abs_prediction=float(abs(xx).mean()), sign_accuracy=float(np.mean(xx*yy > 0)))


def main():
    before = read(ART / "protocol_frozen.json")
    for category in ("active_protected_sources", "reserved_dependencies"):
        for p, h in before[category].items():
            assert digest(ROOT / p) == h, p
    SOURCES[str(INPUT.relative_to(ROOT))] = digest(INPUT)
    with np.load(INPUT) as z:
        inp = {k:z[k].copy() for k in z.files}
    pp = ART / "predictions.npz"
    SOURCES[str(pp.relative_to(ROOT))] = digest(pp)
    with np.load(pp) as z:
        pred = {k:z[k].copy() for k in z.files}
    times = inp["market_times"]
    x, vol, names = features(inp)
    result = {"scope":"frozen historical artifacts only; no forward outcomes or live mutations",
              "historical_independence":False, "fit":read(ART/"models/fit.json"),
              "risk_calibration":read(ART/"calibration.json"), "runs":{}, "forecast_diagnostics":[]}
    for phase, policies in [("validation", ["first_strategy", "second_strategy", "ridge_24", "ridge_72", "tree_24", "tree_72", "blend_24", "blend_72"]),
                            ("comparison", ["first_strategy", "second_strategy", "selected_third", "third_without_cost_optimization"])]:
        for policy in policies:
            for scenario in ("base", "stress"):
                key=f"{phase}-{policy}-{scenario}"
                result["runs"][key] = explain_run(ART/f"runs/{key}.json.gz", inp)
    for family, policy in [("diversified", "selected_third"), ("inward_band", "inward_band")]:
        for scenario in ("base", "stress"):
            key = f"comparison-{family}-{scenario}"
            result["runs"][key] = explain_run(ART/f"{family}/runs/comparison-{policy}-{scenario}.json.gz", inp)
    for h in (24, 72):
        y, maturity = labels(inp, vol, h)
        actual = y * vol * np.sqrt(h)
        for period, start, end in [("training", "2026-04-01", "2026-08-01"),
                                   ("validation", "2026-08-01", "2026-09-01"),
                                   ("comparison", "2026-09-01", "2026-10-13T12:00:00")]:
            for sampling in (6, h):
                ix = ((times >= stamp(start)) & (maturity < stamp(end)) & (times % (sampling*3600) == 0)
                      & np.isfinite(x).all((1,2)) & np.isfinite(actual).all(1))
                for model in ("ridge", "tree"):
                    p = pred[f"{model}_{h}"]
                    for j, symbol in [(None,"ALL"), *enumerate(SYMBOLS[:3])]:
                        forecast = p[ix] if j is None else p[ix,j]
                        target = actual[ix] if j is None else actual[ix,j]
                        result["forecast_diagnostics"].append(dict(period=period,horizon=h,sampling=sampling,
                            model=model,symbol=symbol,time_blocks=int(ix.sum()),**forecast_stats(forecast,target)))
    # Causal feature and label checks, independent of the stored equity results.
    cut=int(np.searchsorted(times, stamp("2026-09-01")))
    prefix={k:v[:cut].copy() for k,v in inp.items()}
    xp,vp,_ = features(prefix)
    np.testing.assert_allclose(xp,x[:cut],atol=0,rtol=0,equal_nan=True)
    for h in (24,72):
        y, maturity = labels(inp,vol,h)
        for i in [1008,2000,4000]:
            expected=(np.log(inp["close"][i+1+h,:3]/inp["close"][i+1,:3])
                      -inp["funding"][i+2:i+2+h,:3].sum(0))/(vol[i]*np.sqrt(h))
            np.testing.assert_allclose(y[i],expected,atol=0,rtol=0)
        assert result["fit"][str(h)]["last_label_time"] < stamp("2026-08-01")
    for category in ("active_protected_sources", "reserved_dependencies"):
        for p,h in before[category].items():
            assert digest(ROOT/p) == h, p
    result["verification"]={"accounting_runs":len(result["runs"]),"causal_prefix_exact":True,
                            "label_alignment_exact":True,"protected_sources_unchanged":True,"live_orders_sent":0}
    (OUT/"findings.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False,default=scalar)+"\n")
    SOURCES[str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    (OUT/"manifest.json").write_text(json.dumps(SOURCES,indent=2)+"\n")
    print(json.dumps(result["verification"]))
    for key,r in result["runs"].items():
        if key.endswith("base"):
            print(key, "PNL", round(r["net_pnl"],2),"PRICE",round(r["price_move_pnl"],2),
                  "COST",round(r["fees"]+r["execution_cost"],2),"EST_VOL",round(r["mean_estimated_daily_vol"]*100,4))
    for f in result["forecast_diagnostics"]:
        if f["model"]=="tree" and f["horizon"]==72 and f["sampling"]==72:
            print("FORECAST",json.dumps(f))


if __name__ == "__main__":
    main()
