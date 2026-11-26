"""Load local bot settings and construct testnet SDK clients in read-only mode."""

import json
import os
import re
import stat
import time
from pathlib import Path

from eth_account import Account
from eth_keys.constants import SECPK1_N
from hyperliquid.exchange import Exchange
from hyperliquid.utils.constants import TESTNET_API_URL


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config.local.json"
KEY_DIR = ROOT / ".secrets"
KEY_FILE = KEY_DIR / "api_wallet.key"


class SetupError(Exception):
    """An actionable error whose message contains no private-key material."""


def load_settings() -> dict:
    settings = json.loads(CONFIG_FILE.read_text())
    if any(settings.get(name) != value for name, value in (
        ("network", "testnet"), ("dex", "par"), ("mode", "read_only")
    )):
        raise SetupError("설정은 network=testnet, dex=par, mode=read_only여야 합니다.")
    for field in ("team_wallet", "api_wallet"):
        if not re.fullmatch(r"0x[0-9a-fA-F]{40}", str(settings.get(field, ""))):
            raise SetupError(f"설정의 {field} 공개 주소를 확인하세요.")
    if settings["team_wallet"].lower() == settings["api_wallet"].lower():
        raise SetupError("팀 지갑과 API 지갑은 서로 달라야 합니다.")
    return settings


def signer_from_key(private_key: str, settings: dict):
    match = re.fullmatch(r"(?:0[xX])?([0-9a-fA-F]{64})", private_key.strip())
    if match is None:
        raise SetupError("API 비밀키 형식이 올바르지 않습니다.")
    raw_key = bytes.fromhex(match.group(1))
    if not 0 < int.from_bytes(raw_key, "big") < SECPK1_N:
        raise SetupError("API 비밀키 값이 올바르지 않습니다.")
    signer = Account.from_key(raw_key)
    if signer.address.lower() != settings["api_wallet"].lower():
        raise SetupError("이 키는 설정에 지정된 sfinx_bot API 지갑의 키가 아닙니다.")
    return signer


def check_key_directory() -> None:
    info = KEY_DIR.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise SetupError(".secrets는 현재 사용자 소유의 일반 디렉터리이고 권한이 700이어야 합니다.")


def save_key(signer) -> None:
    KEY_DIR.mkdir(mode=0o700, exist_ok=True)
    check_key_directory()
    # Refuse existing paths, including symlinks, instead of replacing a saved key.
    fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(signer.key.hex() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        KEY_FILE.unlink(missing_ok=True)
        raise


def load_signer(settings: dict):
    check_key_directory()
    fd = os.open(KEY_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "r") as handle:
        info = os.fstat(handle.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise SetupError("API 키 파일은 현재 사용자 소유의 일반 파일이고 권한이 600이어야 합니다.")
        # Read only enough for a hex key; do not echo malformed file contents.
        return signer_from_key(handle.read(128), settings)


class ReadOnlyExchange(Exchange):
    """Configure the signing identity but block all exchange submissions."""

    def post(self, url_path, payload=None):
        if url_path != "/info":
            raise SetupError("현재 봇은 조회 모드입니다. 주문·취소·계정 변경 전송이 차단되어 있습니다.")
        return super().post(url_path, payload)


class TradingExchange(Exchange):
    """Testnet execution client; enabling order actions is explicit per process."""

    orders_enabled = False

    def post(self, url_path, payload=None):
        if self.base_url != TESTNET_API_URL:
            raise SetupError("이 클라이언트는 테스트넷에서만 사용할 수 있습니다.")
        if url_path == "/exchange":
            allowed = {"noop"}
            if self.orders_enabled:
                allowed |= {"order", "cancel", "cancelByCloid", "updateLeverage"}
            if not isinstance(payload, dict) or payload.get("action", {}).get("type") not in allowed:
                raise SetupError("이 실행 모드에서 허용되지 않은 거래 요청입니다.")
        elif url_path != "/info":
            raise SetupError("지원하지 않는 API 경로입니다.")
        return super().post(url_path, payload)


def build_client(settings: dict, signer=None, *, execution=False, enable_orders=False) -> Exchange:
    if enable_orders and not execution:
        raise SetupError("주문 활성화에는 execution=True가 필요합니다.")
    if signer is None:
        signer = load_signer(settings)
    if signer.address.lower() != settings["api_wallet"].lower():
        raise SetupError("불러온 API 지갑 주소가 설정과 다릅니다.")
    cls = TradingExchange if execution else ReadOnlyExchange
    client = cls(
        signer,
        TESTNET_API_URL,
        account_address=settings["team_wallet"],
        perp_dexs=["par"],
        timeout=15,
    )
    if execution:
        client.orders_enabled = bool(enable_orders)
    return client


def check_connection(client: Exchange, settings: dict) -> None:
    info = client.info
    team = settings["team_wallet"]
    mode = info.query_user_abstraction_state(team)
    if mode != "unifiedAccount":
        raise SetupError("팀 계정의 unifiedAccount 모드가 확인되지 않았습니다.")
    agents = info.post("/info", {"type": "extraAgents", "user": team})
    if not isinstance(agents, list):
        raise SetupError("API 지갑 승인 목록 응답을 확인할 수 없습니다.")
    approved = any(
        agent.get("address", "").lower() == settings["api_wallet"].lower()
        and int(agent.get("validUntil", 0)) > time.time() * 1000
        for agent in agents if isinstance(agent, dict)
    )
    if not approved:
        raise SetupError("이 팀 계정에서 유효한 API 지갑 승인을 확인하지 못했습니다.")
    markets = sorted(name for name in info.coin_to_asset if name.startswith("par:"))
    if not markets:
        raise SetupError("SDK에 대회 시장 par의 종목이 등록되지 않았습니다.")
    balances = info.spot_user_state(team)["balances"]
    positions = info.user_state(team, dex="par")["assetPositions"]
    orders = info.open_orders(team, dex="par")
    print(f"팀 계정: {team}")
    print(f"연결된 API 지갑: {client.wallet.address}")
    print(f"계정 모드: {mode} / API 지갑 승인: 유효")
    collateral = [item["total"] for item in balances if item.get("coin") == "USDPR"]
    print(f"USDPR 잔고 (total): {', '.join(collateral) if collateral else '잔고 항목 없음'}")
    print(f"대회 종목: {len(markets)}개 / 포지션: {len(positions)}개 / 미체결 주문: {len(orders)}개")
    mode_label = "testnet_orders" if getattr(client, "orders_enabled", False) else "read_only"
    print(f"네트워크: testnet / dex: par / 주문 모드: {mode_label}")


def close_client(client: Exchange) -> None:
    client.info.session.close()
    client.session.close()
