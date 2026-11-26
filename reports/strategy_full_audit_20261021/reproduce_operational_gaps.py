"""Reproduce risk-monitoring gaps using temporary files and fake services only."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from competition_live import runner
from competition_live.candidate import build_candidate
from competition_live.live_account import LiveAccount
from competition_live.profile import load_profile, START
from competition_live.runtime import save_coordinator
from competition_live.test_runtime import account_snapshot


def portfolio_with_inventory():
    profile = load_profile()
    account = LiveAccount(profile, None, lambda *args: None, lambda *args: [])
    for ledger in (account.physical, account.owners["base"]):
        ledger.fill(0, .01, 80000, 9, START)
    portfolio = build_candidate(profile, lambda *args: None, account=account)["portfolio"]
    return profile, portfolio


def restart_with_stale_signal():
    profile, portfolio = portfolio_with_inventory()
    now = START + 7 * 3600
    snapshot = account_snapshot(profile, cash="99999.28", qty=".01")
    publication = {"data": {"frame": {"source_time": START}}}
    with tempfile.TemporaryDirectory(prefix="sfinx-audit-") as temp:
        workspace = Path(temp)
        directory = workspace / ".runtime/pas_competition_live"
        directory.mkdir(parents=True)
        save_coordinator(directory / "coordinator.pkl", portfolio, profile, "offline_fixture")
        with patch.object(runner, "fresh_preflight", return_value=snapshot), \
             patch.object(runner, "load_publication", return_value=(publication, [])), \
             patch.object(runner.time, "time", return_value=now), \
             patch.object(runner, "explicit_signer", side_effect=AssertionError("No signer allowed")) as signer, \
             patch.object(runner, "LiveRunner") as live:
            runner.main(execute=True, once=True, workspace=workspace, directory=directory)
            status = json.loads((directory / "status.json").read_text())
            assert status["state"] == "waiting_for_fresh_signal"
            assert not live.called and not signer.called
            return {
                "reproduced": True, "status": status["state"],
                "checkpoint_btc": .01, "signal_age_hours": 7,
                "risk_runner_constructed": live.called, "signer_called": signer.called,
                "meaning": "Restart with inventory waits for a fresh signal before restoring the risk loop.",
            }


def unrelated_context_blocks_exit():
    profile, portfolio = portfolio_with_inventory()
    now = START + 7 * 3600
    at = now * 1000 - 5
    streams = {}
    for symbol in profile["trading"]["symbols"]:
        book = {"time": at, "levels": [[{"px": "80000", "sz": "10"}], [{"px": "80000", "sz": "10"}]]}
        streams[symbol] = {
            "book": [(at, 0, book)], "book_times": [at],
            "ctx": [(at, 0, {"markPx": "80000", "oraclePx": "80000", "funding": "0"})],
            "ctx_times": [at], "resets": [],
        }
    streams["USDJPY"]["ctx"] = []
    streams["USDJPY"]["ctx_times"] = []
    with patch.object(portfolio, "submit_baseline") as submit:
        portfolio.step(now, streams, None, [])
        assert not submit.called and portfolio.account.physical.qty[0] == .01
        assert portfolio.fatal[-1].startswith("missing_account_valuation")
        missing_result = {"submit_called": submit.called, "fatal": list(portfolio.fatal)}
    streams["USDJPY"]["ctx"] = [(at, 0, {"markPx": "80000", "oraclePx": "80000", "funding": "0"})]
    streams["USDJPY"]["ctx_times"] = [at]
    with patch.object(portfolio, "submit_baseline") as submit:
        portfolio.step(now, streams, None, [])
        assert submit.called and np.all(submit.call_args.args[1] == 0)
        return {
            "reproduced": True, "missing_unheld_context": missing_result,
            "restored_unheld_context_forced_exit_called": submit.called,
            "meaning": "One missing unheld-asset context prevents the stale-signal reduction path.",
        }


if __name__ == "__main__":
    result = {"restart_stale_signal": restart_with_stale_signal(),
              "unrelated_missing_context": unrelated_context_blocks_exit(),
              "orders_sent": 0, "all_files_temporary": True}
    (OUT / "operational_reproductions.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
