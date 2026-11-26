"""Bounded, unsigned /info capture for an independent strategy audit.

No exchange client, private keys, order methods, or live runtime writes.
"""
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PROFILE = json.loads((ROOT / "competition_live/config.json").read_text())
TEST = "https://api.hyperliquid-testnet.xyz/info"
MAIN = "https://api.hyperliquid.xyz/info"
ALLOWED = {
    "metaAndAssetCtxs", "perpDexLimits", "l2Book", "candleSnapshot",
    "userAbstraction", "clearinghouseState", "spotClearinghouseState",
    "openOrders", "userFillsByTime", "userFunding", "userNonFundingLedgerUpdates",
}


def read_public(job):
    name, endpoint, payload = job
    assert endpoint in (TEST, MAIN) and payload["type"] in ALLOWED
    begin = time.time()
    result = {"endpoint": endpoint, "request": payload, "started_unix": begin}
    try:
        response = requests.post(endpoint, json=payload, timeout=20)
        response.raise_for_status()
        result["response"] = response.json()
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["received_unix"] = time.time()
    path = OUT / "raw" / f"{name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return name, "error" not in result, round(time.time() - begin, 3)


def main():
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    now = int(time.time() * 1000)
    start = int(PROFILE["start_unix"] * 1000)
    user = PROFILE["team_wallet"]
    jobs = [
        ("pas_market", TEST, {"type": "metaAndAssetCtxs", "dex": "pas"}),
        ("main_market", MAIN, {"type": "metaAndAssetCtxs"}),
        ("xyz_market", MAIN, {"type": "metaAndAssetCtxs", "dex": "xyz"}),
        ("limits", TEST, {"type": "perpDexLimits", "dex": "pas"}),
    ]
    for kind in ["userAbstraction", "clearinghouseState", "spotClearinghouseState", "openOrders"]:
        payload = {"type": kind, "user": user}
        if kind in ("clearinghouseState", "openOrders"):
            payload["dex"] = "pas"
        jobs.append((kind, TEST, payload))
    for kind in ["userFillsByTime", "userFunding", "userNonFundingLedgerUpdates"]:
        jobs.append((kind, TEST, {"type": kind, "user": user, "startTime": start, "endTime": now}))
    for symbol in PROFILE["trading"]["symbols"]:
        jobs.append((f"book_{symbol}", TEST, {"type": "l2Book", "coin": "pas:" + symbol}))
        jobs.append((f"candles_{symbol}", TEST, {"type": "candleSnapshot", "req": {
            "coin": "pas:" + symbol, "interval": "1h",
            "startTime": now - 7 * 86400000, "endTime": now,
        }}))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(read_public, jobs))
    # Pair the single spot valuation with both surrounding position snapshots.
    results.append(read_public(("positions_after", TEST, {
        "type": "clearinghouseState", "user": user, "dex": "pas"})))
    for name in ["status.json", "signal.json", "producer_status.json", "account_latest.json"]:
        shutil.copy2(ROOT / ".runtime/pas_competition_live" / name, OUT / "raw" / ("runtime_" + name))
    shutil.copy2(ROOT / "practice_live/latest.json", OUT / "raw/reference_pointer.json")
    manifest = {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "orders_sent": 0, "signer_loaded": False, "results": results,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((OUT / "raw").glob("*.json"))},
    }
    (OUT / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"captured_at_utc": manifest["captured_at_utc"], "results": results}, indent=2))


if __name__ == "__main__":
    main()
