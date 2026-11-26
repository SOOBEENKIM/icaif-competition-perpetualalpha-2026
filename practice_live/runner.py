"""Fixed default strategy -> actual par execution. --execute enables testnet orders.

Single-writer lock, durable pre-send claims, real fills/positions, and explicit
six-hour decision windows. No model changes, transfers or mainnet orders.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import time
from zoneinfo import ZoneInfo

import numpy as np

from bot.connection import build_client, check_connection, close_client, load_settings
from bot.execution import Executor, OrderRejected, OutcomeUnknown
from replay_lab.common_comparison.engine import risk_breaches, risk_values
from replay_lab.common_comparison.models import Market, covariance
from replay_lab.ema_rebuild.engine import Controller
from strategy.agent import TAIL, profiles, read
from .infer import protected_hashes, infer
from .refresh import HOUR, OUT, ROOT, REFERENCES, iso, load_inputs, refresh, sha, save_json
from .live_logic import accounting, identity, make_ioc, phase_delta
from .live_store import Store

SYMBOLS = tuple(REFERENCES)
COINS = tuple("par:" + s for s in SYMBOLS)
NORMAL_WINDOW_SECONDS = 900
POLL_SECONDS = 30


def stamp_kst(seconds):
    return datetime.fromtimestamp(seconds, ZoneInfo("Asia/Seoul")).isoformat()


def signal_inputs(now):
    pointer = json.loads((OUT / "latest.json").read_text())
    run = ROOT / pointer["run"]
    verification = read(run / "verification.json")
    for key, path in (("inputs_sha256", run / "inputs.npz"), ("signals_sha256", run / "signals.json")):
        if sha(path) != pointer[key]:
            raise ValueError("Published data/signal hash mismatch")
    if sha(run / "targets.npz") != verification["targets_sha256"]:
        raise ValueError("Target hash mismatch")
    z = load_inputs(run / "inputs.npz")
    with np.load(run / "targets.npz", allow_pickle=False) as targets:
        times, weights = targets["times"], targets["default"]
    if not np.array_equal(times, z["market_times"]) or int(times[-1]) != int(now // HOUR) * HOUR:
        raise ValueError("Stale or misaligned strategy inputs")
    m = Market(times, z["close"], z["volume"], z["trades"], z["funding"], z["observed"], SYMBOLS)
    p = read(TAIL / "baseline.json")["protocol"]
    sigma = covariance(m, len(times)-1, p["risk"])[1]
    if not np.isfinite(weights[-1]).all() or not np.isfinite(sigma).all() or not z["observed"][-1].all():
        raise ValueError("Invalid latest strategy inputs")
    return {"hour": int(times[-1]), "weights": weights[-1], "sigma": sigma,
            "volume": z["volume"][-1], "trades": z["trades"][-1],
            "input_hash": pointer["inputs_sha256"], "run": str(run.relative_to(ROOT)), "protocol": p}


def paged(info, payload, end, limit):
    """Keep the last millisecond inclusive to avoid losing same-time events."""
    rows = {}
    for _ in range(100):
        page = info.post("/info", {**payload, "endTime": end})
        if not isinstance(page, list):
            raise ValueError("Invalid account history response")
        for row in page:
            if int(row["time"]) > end or int(row["time"]) < payload["startTime"]:
                raise ValueError("Account history outside requested interval")
            rows[identity(row)] = row
        if len(page) < limit:
            return list(rows.values())
        last = max(int(r["time"]) for r in page)
        if last <= payload["startTime"]:
            raise ValueError("Account history saturated one millisecond; refusing incomplete history")
        payload = {**payload, "startTime": last}
    raise ValueError("Account history page limit exceeded")


class Runner:
    def __init__(self, client, settings, store, directory, execute=False):
        self.client, self.info, self.settings = client, client.info, settings
        self.store, self.directory, self.execute = store, Path(directory), execute
        self.executor = Executor(client, settings, max_notional="150000", journal=self.directory / "order_events.jsonl")
        self.inputs = None
        self.stop_requested = lambda: False

    def histories(self):
        anchor = self.store.get("anchor")
        start = self.store.get("history_cursor", anchor["start_ms"])
        end = int(time.time()*1000)
        for kind, api, limit in (("fills", "userFillsByTime", 2000),
                                 ("funding", "userFunding", 500),
                                 ("transfers", "userNonFundingLedgerUpdates", 500)):
            payload = {"type": api, "user": self.settings["team_wallet"], "startTime": start}
            if kind == "fills":
                payload["aggregateByTime"] = False
            rows = paged(self.info, payload, end, limit)
            self.store.add_records(kind, [(f'{r["coin"]}:{r["tid"]}' if kind == "fills" else identity(r), r) for r in rows])
        self.store.put("history_cursor", max(anchor["start_ms"], end - 60000))

    def recover(self):
        unresolved = []
        for row in self.store.orders():
            if row["status"] not in ("sending", "unknown", "responded", "open"):
                continue
            result = self.executor.status(row["cloid"])
            if result.get("status") == "unknownOid":
                self.store.update(row["cloid"], "unknown", result)
                unresolved.append(row["cloid"])
                continue
            if result.get("status") != "order":
                raise ValueError("Unrecognized order status response")
            status = result["order"]["status"]
            order = result["order"]["order"]
            if order["coin"] != row["plan"]["coin"]:
                raise ValueError("Recovered order market mismatch")
            if status == "open":
                if self.execute:
                    # Only our persisted cloid, never cancel another user's order.
                    try:
                        self.executor.cancel(order["coin"], row["cloid"])
                    except OrderRejected:
                        pass  # Could have filled while cancelling; next poll confirms.
                unresolved.append(row["cloid"])
            elif status not in ("filled", "canceled", "rejected", "triggered") and not status.endswith(("Canceled", "Rejected")):
                raise ValueError("Unknown terminal order state")
            self.store.update(row["cloid"], status, result)
        if unresolved:
            raise OutcomeUnknown("Persisted order needs reconciliation: " + ",".join(unresolved))

    def account(self):
        started = int(time.time()*1000)
        meta, contexts = self.info.post("/info", {"type": "metaAndAssetCtxs", "dex": "par"})
        universe = {m["name"]: (m, c) for m, c in zip(meta["universe"], contexts)}
        if set(universe) != set(COINS):
            raise ValueError("par market universe changed")
        if any(universe[c][0].get("isDelisted") or universe[c][0].get("onlyIsolated") for c in COINS):
            raise ValueError("Market is delisted or requires isolated margin")
        positions = self.info.user_state(self.settings["team_wallet"], dex="par")
        spot = self.info.spot_user_state(self.settings["team_wallet"])
        orders = self.info.open_orders(self.settings["team_wallet"], dex="par")
        if orders:
            raise ValueError("Open par orders need reconciliation; no additional orders sent")
        if not -2000 <= int(time.time()*1000) - int(positions["time"]) < 15000:
            raise ValueError("Stale position snapshot")
        usdpr = [b for b in spot["balances"] if b["coin"] == "USDPR" and b["token"] == meta["collateralToken"]]
        if len(usdpr) != 1:
            raise ValueError("Missing or mismatched USDPR collateral")
        anchor = self.store.get("anchor")
        if anchor is None:
            if positions["assetPositions"]:
                raise ValueError("First start requires the confirmed flat account; cannot invent position history")
            anchor = {"start_ms": started, "initial_equity": float(usdpr[0]["total"]),
                      "team_wallet": self.settings["team_wallet"], "profile": "default"}
            if not np.isfinite(anchor["initial_equity"]) or anchor["initial_equity"] <= 0:
                raise ValueError("Invalid starting collateral")
            self.store.put("anchor", anchor)
        if anchor["team_wallet"].lower() != self.settings["team_wallet"].lower():
            raise ValueError("Runtime belongs to another wallet")
        self.histories()
        marks = {c: float(universe[c][1]["markPx"]) for c in COINS}
        if not all(np.isfinite(p) and p > 0 for p in marks.values()):
            raise ValueError("Invalid par mark price")
        fills = self.store.records("fills")
        # Signed-price accounting includes genuine fills, fees and funding.
        ledger = accounting(anchor, fills, self.store.records("funding"), self.store.records("transfers"),
                            positions["assetPositions"], marks, self.settings["team_wallet"])
        known_oids = set()
        for row in self.store.orders():
            result = row["result"] or {}
            if result.get("status") == "order":
                known_oids.add(int(result["order"]["order"]["oid"]))
            for key in ("filled", "resting"):
                if key in result:
                    known_oids.add(int(result[key]["oid"]))
        if any(int(f["oid"]) not in known_oids for f in fills):
            raise ValueError("Trade from another process/person detected; automatic execution paused")
        raw = {"received_at_utc": iso(), "positions": positions, "spot": spot, "orders": orders,
               "meta": meta, "contexts": contexts, "accounting": ledger}
        save_json(self.directory / "account_latest.json", raw)
        return {"ledger": ledger, "universe": universe, "marks": marks,
                "qty": np.array([ledger["qty"][c] for c in COINS]),
                "prices": np.array([marks[c] for c in COINS]), "spot_usdpr_total": float(usdpr[0]["total"])}

    def load_signal(self):
        now = time.time()
        hour = int(now // HOUR)*HOUR
        if self.inputs is not None and self.inputs["hour"] == hour:
            return self.inputs
        try:
            inputs = signal_inputs(now)
        except (FileNotFoundError, ValueError):
            infer(refresh())
            inputs = signal_inputs(time.time())
        self.inputs = inputs
        return inputs

    def decide(self, account, inputs, now):
        p = inputs["protocol"]
        equity = account["ledger"]["equity"]
        previous_capital = self.store.get("external_capital", 0.)
        capital = account["ledger"]["external_capital"]
        peak = max(self.store.get("peak", equity) + capital - previous_capital, equity)
        halted = self.store.get("halted", False) or equity / peak - 1 <= -p["risk"]["drawdown_halt"]
        self.store.put("peak", peak)
        self.store.put("external_capital", capital)
        self.store.put("halted", halted)
        forced = halted or bool(risk_breaches(account["qty"], equity, account["prices"], inputs["sigma"], p["risk"]))
        hour = inputs["hour"]
        if not forced and (hour % (6*HOUR) or now - hour > NORMAL_WINDOW_SECONDS):
            return None
        # Forced reductions can be reconsidered each minute after partial IOC fills.
        decision = int(now // 60)*60 if forced else hour
        key = f"round:{'risk' if forced else 'normal'}:{decision}"
        saved = self.store.get(key)
        if saved is not None:
            return None if saved["status"] == "complete" else saved
        controller = Controller()
        live_protocol = json.loads(json.dumps(p))
        live_protocol["execution"]["decimals"] = [int(account["universe"][c][0]["szDecimals"]) for c in COINS]
        result = controller.propose(decision, account["prices"], account["qty"], equity,
                    inputs["weights"], inputs["sigma"], inputs["volume"], inputs["trades"],
                    live_protocol, buffer=.2, halted=halted)
        if result is None:
            return None
        plan = {"key": key, "status": "planned", "decision_time": decision,
                "deadline": min(hour+HOUR, now+60) if forced else hour+NORMAL_WINDOW_SECONDS,
                "input_hour": hour, "input_hash": inputs["input_hash"], "created_at_utc": iso(now),
                "raw_weights": inputs["weights"].tolist(), "positions_before": account["qty"].tolist(),
                "equity": equity, **result}
        self.store.put(key, plan)
        return plan

    def execute_round(self, plan, inputs):
        for phase in ("reduce", "increase"):
            if phase == "increase" and plan["risk_forced"]:
                continue
            for j, coin in enumerate(COINS):
                if self.stop_requested():
                    return
                if time.time() > plan["deadline"] or int(time.time()//HOUR)*HOUR != plan["input_hour"]:
                    plan["expiry_reason"] = "Execution window elapsed; no stale replay"
                    break
                slot = f'{self.settings["team_wallet"].lower()}:{plan["key"]}:{coin}:{phase}'
                if any(r["slot"] == slot for r in self.store.orders()):
                    continue
                self.recover()
                account = self.account()
                delta = phase_delta(account["qty"][j], plan["target_qty"][j], phase)
                if abs(delta) < 1e-12:
                    continue
                proposed = account["qty"].copy()
                proposed[j] += delta
                if phase == "increase" and risk_breaches(proposed, account["ledger"]["equity"],
                        account["prices"], inputs["sigma"], inputs["protocol"]["risk"]):
                    continue
                active = self.info.post("/info", {"type": "activeAssetData", "user": self.settings["team_wallet"], "coin": coin})
                if active["leverage"] != {"type": "cross", "value": 1}:
                    if not self.execute:
                        raise ValueError("Execution plan requires cross 1x; preview cannot change leverage")
                    self.executor.set_leverage(coin, 1)
                    active = self.info.post("/info", {"type": "activeAssetData", "user": self.settings["team_wallet"], "coin": coin})
                    if active["leverage"] != {"type": "cross", "value": 1}:
                        raise ValueError("Cross 1x leverage was not applied")
                book = self.info.l2_snapshot(coin)
                market, context = account["universe"][coin]
                order = make_ioc(coin, delta, phase == "reduce", market, context, book, active,
                                 int(time.time()*1000), slot, inputs["protocol"]["risk"]["asset_notional"])
                if order is None:
                    continue
                if not self.execute:
                    raise ValueError("Preview mode never submits")
                if self.stop_requested():
                    return
                if time.time() > plan["deadline"] or int(time.time()//HOUR)*HOUR != plan["input_hour"]:
                    raise ValueError("Decision expired during execution preparation")
                # Reject delayed network delivery instead of trading an old intent.
                self.client.set_expires_after(min(int(plan["deadline"]*1000), int(time.time()*1000)+15000))
                # Crash here cannot lose the identity or cause a duplicate retry.
                if not self.store.claim(slot, asdict(order)):
                    continue
                try:
                    result = self.executor.submit(order)
                except OrderRejected as error:
                    self.store.update(order.cloid, "rejected", {"reason": str(error)})
                    continue
                except BaseException:
                    # The durable sending state forces recovery before any next order.
                    raise
                self.store.update(order.cloid, "responded", result)
                self.recover()
                self.account()  # Reconcile actual quantity and costs before next leg.
        plan["status"] = "complete"
        plan["completed_at_utc"] = iso()
        self.store.put(plan["key"], plan)
        self.store.put("last_round", plan)

    def cycle(self):
        self.recover()
        account = self.account()
        inputs = self.load_signal()
        # Inference can take minutes; refresh account before the decision.
        account = self.account()
        now = time.time()
        plan = self.decide(account, inputs, now) if self.execute else None
        if plan is not None:
            self.execute_round(plan, inputs)
            account = self.account()
        orders = self.store.orders()
        next_normal = (int(now)//(6*HOUR)+1)*(6*HOUR)
        status = {"updated_at_utc": iso(), "pid": os.getpid(), "mode": "live_testnet" if self.execute else "preview",
                  "network": "testnet", "dex": "par", "profile": "default", "state": "running",
                  "data_completed_kst": stamp_kst(inputs["hour"]), "next_normal_decision_kst": stamp_kst(next_normal),
                  "account": account["ledger"], "spot_usdpr_total": account["spot_usdpr_total"],
                  "raw_target_weights": dict(zip(SYMBOLS, inputs["weights"].tolist())),
                  "risk": risk_values(account["qty"], account["ledger"]["equity"], account["prices"], inputs["sigma"]),
                  "halted": self.store.get("halted", False), "orders_claimed": len(orders),
                  "last_round": self.store.get("last_round"), "stop_file": str(self.directory / "STOP"),
                  "execution": "IOC, cross 1x, 10bp limit from best quote; actual available depth and remaining target",
                  "accounting": "flat-start executed-price ledger + actual fees/funding/capital flows; marked at current par marks"}
        save_json(self.directory / "status.json", status)
        print(json.dumps({"at": iso(), "mode": status["mode"], "equity": status["account"]["equity"],
                          "fills": status["account"]["fills"], "orders": len(orders),
                          "next_normal_kst": status["next_normal_decision_kst"]}), flush=True)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Enable actual testnet par orders")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    profiles()
    preserved = protected_hashes()
    directory = OUT / ("live_runtime" if args.execute else "preview_runtime")
    directory.mkdir(exist_ok=True, mode=0o700)
    lock = (directory / "process.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
    store = Store(directory / "state.sqlite3")
    fingerprint = identity(preserved)
    if store.get("frozen_fingerprint", fingerprint) != fingerprint:
        raise ValueError("Frozen strategy differs from this runtime")
    store.put("frozen_fingerprint", fingerprint)
    settings = load_settings()
    client = build_client(settings, execution=args.execute, enable_orders=args.execute)
    check_connection(client, settings)
    runner = Runner(client, settings, store, directory, args.execute)
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    runner.stop_requested = lambda: stop or (directory / "STOP").exists()
    try:
        while not stop and not (directory / "STOP").exists():
            try:
                runner.cycle()
            except Exception as error:
                state = {"updated_at_utc": iso(), "pid": os.getpid(), "mode": "live_testnet" if args.execute else "preview",
                         "state": "paused_retrying", "error": f"{type(error).__name__}: {error}",
                         "orders_claimed": len(store.orders()), "retry_seconds": POLL_SECONDS}
                save_json(directory / "status.json", state)
                print(json.dumps(state), flush=True)
                if args.once:
                    raise
            if args.once:
                break
            for _ in range(POLL_SECONDS):
                if stop or (directory / "STOP").exists():
                    break
                time.sleep(1)
    finally:
        close_client(client)
        store.close()
        lock.close()
        if stop or (directory / "STOP").exists():
            save_json(directory / "stopped.json", {"at_utc": iso(), "positions_auto_closed": False})


if __name__ == "__main__":
    main()
