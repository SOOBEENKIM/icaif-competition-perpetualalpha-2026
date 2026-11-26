"""Sealed PAS book comparison: capture now, evaluate once after the fixed end."""
import argparse
import copy
import gzip
import hashlib
import json
import shutil
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import numpy as np
from competition_live.profile import ROOT, load_profile, verify_deployment
from competition_live.coordinator import Portfolio
from competition_live.scoring import evaluate as score
from .tape import TapeWindow, FundingClock, raw_events
from experiments.published_pair_lifecycle_20261017.signals import validate_envelope
from replay_lab.common_comparison.models import Market, covariance
from .registry import Registry
from .model import build, publication
from .study import OUT, write, sha, native

FWD = OUT/"forward"


def atomic(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, default=native, allow_nan=False))
    temp.replace(path)


def prepare():
    FWD.mkdir(exist_ok=True)
    if (FWD/"plan.json").exists():
        raise FileExistsError("Forward experiment already frozen")
    p = load_profile()
    spec = json.loads((OUT/"protocol.json").read_text())
    model = json.loads((OUT/"model.json").read_text())
    audit = json.loads((OUT/"exposure_audit.json").read_text())
    now = time.time()
    start = (int(now-43200)//86400+1)*86400+43200
    if start < now+3600:
        start += 86400
    end = start+3*86400
    original = json.loads((ROOT/"competition_live/strategy_plan.json").read_text())
    venue = {k: copy.deepcopy(original[k]) for k in (
        "book_protocol", "direct_spec", "bias_spec", "model_latency_seconds", "step_seconds")}
    venue.update(trading_protocol=p["trading"], paper_start_unix=start, capture_end_unix=end,
                 end_liquidation_seconds=600)
    venue["book_protocol"]["pair_max_gross"] = p["configs"]["reference"]["pair_max_gross"]
    sources = {ROOT/name for name in json.loads((OUT/"source_hashes.json").read_text())}
    release = json.loads((ROOT/"reports/all_market_alpha_wave5_20261020/release_manifest.json").read_text())
    for name, expected in release["files"].items():
        path = ROOT/name
        if sha(path) != expected:
            raise ValueError("Previous candidate artifact changed: "+name)
        sources.add(path)
    # Original source tree plus imported project modules; no credentials or runtime state.
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file:
            path = Path(file).resolve()
            if (path.is_relative_to(ROOT) and path.is_file() and path.suffix == ".py"
                    and not {".venv", ".tools"}.intersection(path.parts)):
                sources.add(path)
    for path in (ROOT/"validation_rebuild").glob("*.py"):
        sources.add(path)
    for path in (ROOT/"deliverables/sfinx_pairs_candidate_20261018_final/models").glob("*.json"):
        sources.add(path)
    sources.update([OUT/"model.json", OUT/"protocol.json", ROOT/"competition_live/strategy_plan.json",
                    ROOT/"competition_live/config.json",
                    ROOT/"reports/all_market_alpha_wave5_20261020/release_manifest.json"])
    for name, digest in verify_deployment()["files"].items():
        sources.add(ROOT/name)
    artifacts = {str(path): sha(path) for path in sorted(sources)}
    plan = dict(evaluation_kind="independent_latest_period",
        train_start=spec["train_start"], train_end=spec["train_end"],
        validation_start=spec["validation_start"], validation_end=spec["validation_end"],
        test_start=start, test_end=end, max_train_label_maturity=model["last_maturity"],
        preprocessing_fit_last_time=model["feature_last_time"],
        selection_last_time=audit["latest_known_data_exposure"],
        model_knowledge_last_time=audit["latest_known_data_exposure"],
        artifacts=artifacts, candidate="fixed_fit_candidate", comparators=["incumbent"],
        success_rule=spec["success_rule"], venue_plan=venue,
        created_at=now, no_partial_outcome_inspection=True, no_reselection=True,
        refit_contract=spec["refit_contract"],
        publication_source="Copy existing public-data producer publications and causal shared pair forecasts. "
            "Auxiliary weights computed separately with the immutable fixed ridge. Common measured availability.",
        capture="PAS public l2Book and activeAssetCtx, one event per symbol/channel/second; "
                "record reconnects and receipt times. No wallet signer.",
        limitations=["Counterfactual displayed-book fills, not exchange fills or own market impact.",
                     "72 hours is an initial independent check, not evidence of long-term profitability.",
                     "Existing pair/gate online updates are shared, predeclared, and use only then-mature labels.",
                     "Any disconnect during the test invalidates the headline comparison conservatively.",
                     "Missing valuations, funding, unknown fills or terminal residuals invalidate the result."],
        orders_sent=0, automatic_live_replacement=False)
    # Persist backup before appending exposure and a single future reservation.
    registry = Registry()
    with registry.connect() as db:
        import sqlite3
        backup = sqlite3.connect(FWD/"registry_before.sqlite3")
        db.backup(backup); backup.close()
    registry.import_audit(audit)
    ident = registry.freeze(plan)
    write(FWD/"plan.json", plan)
    write(FWD/"registration.json", dict(trial_id=ident, start=start, end=end,
        independent_results_available=False, registered_before_start=True))
    print(json.dumps(dict(trial_id=ident, start=start, end=end)), flush=True)


def verify(plan):
    Registry._verify_artifacts(plan)
    verify_deployment()


def collect(plan):
    import websocket
    verify(plan)
    (FWD/"raw").mkdir(exist_ok=True)
    counts = Counter(); errors = []; lock = threading.Lock(); stop = threading.Event()
    writers = {}; last_emit = {}; latest_source = -1
    p = load_profile()
    spec = json.loads((OUT/"protocol.json").read_text())
    model = json.loads((OUT/"model.json").read_text())

    def emit(channel, data):
        with lock:
            now = time.time()
            key = (channel, data.get("coin", "signal"))
            if channel in ("l2Book", "activeAssetCtx") and now-last_emit.get(key, 0) < 1:
                return
            last_emit[key] = now
            row = dict(received_ms=int(now*1000), channel=channel, data=data)
            bucket = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%d_%H")
            for previous in list(writers):
                if previous != bucket:
                    writers.pop(previous).close()
            if bucket not in writers:
                writers[bucket] = gzip.open(FWD/"raw"/(bucket+".jsonl.gz"), "at", compresslevel=3)
            writers[bucket].write(json.dumps(row, default=native, separators=(",", ":"))+"\n")
            counts[channel] += 1

    def signals():
        nonlocal latest_source
        while not stop.wait(2) and time.time() < plan["test_end"]:
            try:
                record = json.loads((ROOT/".runtime/pas_competition_live/signal.json").read_text())
                original = record["publication"]; frame = validate_envelope(original)
                source = int(frame["source_time"])
                if source <= latest_source:
                    continue
                pointer = json.loads((ROOT/"practice_live/latest.json").read_text())
                run = (ROOT/pointer["run"]).resolve()
                if not run.is_relative_to(ROOT/"practice_live/updates"):
                    raise ValueError("Unexpected source directory")
                if sha(run/"inputs.npz") != frame["source_hash"]:
                    continue  # producer publication and pointer have not converged yet
                with np.load(run/"inputs.npz", allow_pickle=False) as z:
                    d = {k: z[k].copy() for k in z.files}
                with np.load(run/"targets.npz", allow_pickle=False) as z:
                    incumbent = z["default"].copy()
                check = json.loads((run/"verification.json").read_text())
                if sha(run/"targets.npz") != check["targets_sha256"]:
                    raise ValueError("Source targets changed")
                if int(d["market_times"][-1]) != source or source > time.time():
                    raise ValueError("Future/misaligned source")
                if any(f["available_time"] > time.time() or f["forecast_source"] != source
                       for f in record["forecasts"]):
                    raise ValueError("Unpublished shared forecast")
                market = Market(d["market_times"], d["close"], d["volume"], d["trades"],
                                d["funding"], d["observed"], tuple(p["trading"]["symbols"]))
                sigma = np.full((len(incumbent), 10, 10), np.nan)
                for i in range(168, len(incumbent)):
                    sigma[i] = covariance(market, i, p["trading"]["risk"])[1]
                weights, _ = build(d, incumbent, sigma, p["trading"], spec, spec["calibration"], model)
                verify(plan)
                packed = copy.deepcopy(original["data"])
                packed.update(candidate_weights=weights[-1].tolist(),
                              shared_forecasts=record["forecasts"], source_run=pointer["run"],
                              candidate_model_sha256=sha(OUT/"model.json"))
                # Both hypothetical arms receive the signal after candidate computation.
                emit("strategySignal", packed)
                latest_source = source
            except Exception as exc:
                errors.append(dict(at=time.time(), source="publication", error=str(exc)))

    worker = threading.Thread(target=signals, daemon=True); worker.start()
    sock = None; last_ping = last_health = 0
    try:
        while time.time() < plan["test_end"]+15 and not (FWD/"STOP").exists():
            if shutil.disk_usage(FWD).free < 5*1024**3:
                raise RuntimeError("Capture stopped: disk reserve")
            try:
                if sock is None:
                    sock = websocket.create_connection(p["ws_url"], timeout=5)
                    emit("connection", dict(state="connected"))
                    for symbol in p["trading"]["symbols"]:
                        for channel in ("l2Book", "activeAssetCtx"):
                            sock.send(json.dumps(dict(method="subscribe",
                                subscription=dict(type=channel, coin="pas:"+symbol))))
                if time.time()-last_ping > 20:
                    sock.send(json.dumps(dict(method="ping"))); last_ping = time.time()
                msg = json.loads(sock.recv())
                if msg.get("channel") in ("l2Book", "activeAssetCtx"):
                    if msg["data"]["coin"] not in {"pas:"+s for s in p["trading"]["symbols"]}:
                        raise ValueError("Unexpected venue/asset")
                    emit(msg["channel"], msg["data"])
                elif msg.get("channel") == "error":
                    raise ValueError(str(msg["data"]))
            except websocket.WebSocketTimeoutException:
                pass
            except Exception as exc:
                errors.append(dict(at=time.time(), source="websocket", error=str(exc)))
                emit("connection", dict(state="disconnected", reason=str(exc)))
                if sock:
                    sock.close()
                sock = None
                stop.wait(2)
            if time.time()-last_health >= 15:
                with lock:
                    for writer in writers.values():
                        writer.flush()
                    atomic(FWD/"health.json", dict(state="raw_capture_only", at=time.time(),
                        counts=dict(counts), latest_source=latest_source, errors=errors[-5:],
                        test_start=plan["test_start"], test_end=plan["test_end"],
                        partial_performance_inspected=False, orders_sent=0))
                last_health = time.time()
    finally:
        stop.set(); worker.join(timeout=60)
        if sock:
            sock.close()
        for writer in writers.values():
            writer.close()
    if (FWD/"STOP").exists():
        raise RuntimeError("Capture manually stopped; no automatic period shortening")


def public_funding(symbol, start, end):
    payload = dict(type="fundingHistory", coin="pas:"+symbol, startTime=start*1000, endTime=end*1000)
    request = urllib.request.Request("https://api.hyperliquid-testnet.xyz/info",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        rows = json.load(response)
    if not isinstance(rows, list) or len(rows) >= 500:
        raise ValueError("Incomplete funding response")
    write(FWD/f"funding_{symbol}.json", dict(request=payload, rows=rows))
    return rows


def outcome(plan):
    """Only Registry.evaluate may call this, after consume and end-time checks."""
    verify(plan)
    start, end = plan["test_start"], plan["test_end"]; venue = plan["venue_plan"]
    symbols = venue["trading_protocol"]["symbols"]; rates = {}; disconnects = []
    for symbol in symbols:
        for r in public_funding(symbol, start, end):
            at = int(r["time"]); bucket = at//3600000*3600000
            if r["coin"] != "pas:"+symbol or not 0 <= at-bucket < 1000:
                raise ValueError("Unexpected funding event")
            rates.setdefault(bucket, {})[symbol] = float(r["fundingRate"])
    def events():
        for event in raw_events(FWD/"raw"):
            if (event["channel"] == "connection" and event["data"]["state"] == "disconnected"
                    and start*1000 <= event["received_ms"] <= end*1000):
                disconnects.append(event["received_ms"])
            yield event
    tape = TapeWindow(events())
    arms = {}
    for scenario, sc in venue["book_protocol"]["scenarios"].items():
        for name in ("incumbent", "fixed_fit_candidate"):
            clock = FundingClock(start, tape, rates, symbols, venue["book_protocol"],
                                 sc["funding_receipt_multiplier"])
            # Identical original coordinator and unchanged pair engine in both arms.
            arms[name, scenario] = Portfolio("first_plus_decision_gate", scenario, venue, clock)
    pub = None; forecasts = []
    for now in range(start, end+1, venue["step_seconds"]):
        streams = tape.ensure(now*1000)
        for event in tape.publications(now*1000):
            validate_envelope(event)
            pub = event; forecasts = event["data"]["shared_forecasts"]
        for (name, scenario), arm in arms.items():
            transformed = publication(pub, pub["data"]["candidate_weights"]) if pub and name != "incumbent" else pub
            arm.step(now, streams, transformed, forecasts)
    rows = []
    for (name, scenario), arm in arms.items():
        snaps = arm.snapshots
        daily = [dict(time=int(s["time"]), equity=s["equity"]+50000)
                 for s in snaps if s["time"]%86400 == 43200]
        fatal = list(arm.fatal)
        if [r["time"] for r in daily] != list(range(start, end+1, 86400)):
            fatal.append("missing_daily_valuation")
        if disconnects:
            fatal.append("disconnection_during_test")
        if arm.account.unknown:
            fatal.append("unknown_fill_outcome")
        if np.any(abs(arm.account.physical.qty)>1e-10):
            fatal.append("unclosed_terminal_inventory")
        risk = venue["trading_protocol"]["risk"]
        limits = dict(gross=risk["gross_fraction"], net=risk["net_fraction"],
                      asset=risk["asset_fraction"], asset_notional=risk["asset_notional"],
                      daily_volatility=risk["daily_volatility"])
        breaches = sum(any(s[k] > limit+1e-10 for k, limit in limits.items()) for s in snaps)
        if breaches:
            fatal.append("risk_limit_breach")
        if arm.account.maximum_error >= 1e-6:
            fatal.append("owner_accounting_mismatch")
        ledger = arm.account.physical
        financial = score(daily, trading_days=len({int(f["time"])//86400 for f in ledger.fills}))
        row = dict(variant=name, scenario=scenario, valid=not fatal, fatal=sorted(set(fatal)),
            net_pnl=snaps[-1]["equity"]-50000 if snaps else None,
            fees=ledger.fees, funding=ledger.funding, fills=len(ledger.fills),
            risk_breach_snapshots=breaches, accounting_error=arm.account.maximum_error,
            **financial, daily=daily)
        rows.append(row)
        with gzip.open(FWD/f"{name}_{scenario}.json.gz", "wt") as f:
            json.dump(dict(metrics=row, snapshots=snaps, fills=ledger.fills, events=arm.events),
                      f, default=native, allow_nan=False)
    index = {(r["variant"], r["scenario"]): r for r in rows}
    passed = all(
        index["fixed_fit_candidate", s]["valid"] and index["incumbent", s]["valid"]
        and index["fixed_fit_candidate", s]["eligible"]
        and index["fixed_fit_candidate", s]["net_pnl"] > max(0, index["incumbent", s]["net_pnl"])
        and index["fixed_fit_candidate", s]["Q"] > index["incumbent", s]["Q"]
        for s in ("base", "stress"))
    result = dict(rows=rows, passed=passed, orders_sent=0, automatic_live_replacement=False,
                  no_candidate_reselection=True, short_period_evidence_only=True)
    write(FWD/"result.json", result)
    return result


def evaluate_once():
    registration = json.loads((FWD/"registration.json").read_text())
    return Registry().evaluate(registration["trial_id"], outcome)


def main():
    arg = argparse.ArgumentParser(); arg.add_argument("mode", choices=["prepare", "run", "evaluate"])
    mode = arg.parse_args().mode
    if mode == "prepare":
        prepare(); return
    if mode == "evaluate":
        evaluate_once(); return
    plan = json.loads((FWD/"plan.json").read_text())
    import fcntl
    lock = (FWD/"capture.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        collect(plan)
        evaluate_once()
        atomic(FWD/"health.json", dict(state="completed", at=time.time(), orders_sent=0,
                                       result=str(FWD/"result.json")))
    except Exception as exc:
        atomic(FWD/"health.json", dict(state="failed", at=time.time(), error=str(exc),
                                       orders_sent=0, automatic_retry_after_consumption=False))
        raise


if __name__ == "__main__":
    main()
