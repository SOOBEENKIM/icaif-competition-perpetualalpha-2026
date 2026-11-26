"""Check the local SDK and, optionally, public testnet metadata. Never trades."""

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--online", action="store_true", help="Also query public metadata for dex par."
    )
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    if sys.version_info < (3, 10):
        print("FAIL: Python 3.10 or newer is required. Activate .venv first.")
        return 1

    try:
        sdk_version = version("hyperliquid-python-sdk")
    except PackageNotFoundError:
        print("FAIL: hyperliquid-python-sdk is not installed in this environment.")
        return 1
    print(f"Hyperliquid SDK: {sdk_version}")
    if sdk_version != "0.24.0":
        print("FAIL: This project expects SDK 0.24.0, as specified by the guide.")
        return 1

    from hyperliquid.api import API
    from hyperliquid.utils.constants import TESTNET_API_URL

    print("PASS: Local environment is ready.")
    if not args.online:
        return 0

    # POST /info is read-only. No signer, private key or Exchange client is used.
    client = API(TESTNET_API_URL, timeout=15)
    try:
        meta = client.post("/info", {"type": "meta", "dex": "par"})
    except Exception as exc:
        print(f"FAIL: Testnet metadata query failed ({type(exc).__name__}).")
        return 1

    if not isinstance(meta, dict) or not isinstance(meta.get("universe"), list):
        print("FAIL: Unexpected metadata response.")
        return 1
    names = [item.get("name", "") for item in meta["universe"] if isinstance(item, dict)]
    if not names or not all(name.startswith("par:") for name in names):
        print("FAIL: The response does not identify the competition dex par.")
        return 1
    print(f"PASS: Public testnet connection; {len(names)} par markets.")
    print("Markets: " + ", ".join(names))
    print("No wallet authentication, balance checks or orders were performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
