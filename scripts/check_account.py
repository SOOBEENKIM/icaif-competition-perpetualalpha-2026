"""Read a team's testnet account using only its public address. Never signs or trades."""

import json
import re

from hyperliquid.api import API
from hyperliquid.utils.constants import TESTNET_API_URL


def main() -> int:
    print("Hyperliquid 테스트넷 / 대회 시장 par / 조회 전용")
    print("USDPR를 받은 팀 지갑의 공개 주소를 입력하세요. 비밀키는 필요 없습니다.")
    try:
        address = input("팀 지갑 주소 (0x로 시작하는 42자리): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n조회 취소.")
        return 1
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", address):
        print("주소 형식이 맞지 않습니다. 팀 지갑의 공개 주소를 확인하세요.")
        return 1

    # All five requests are reads. No private key, signer or Exchange client exists.
    queries = {
        "계정 모드": {"type": "userAbstraction", "user": address},
        "담보 잔고": {"type": "spotClearinghouseState", "user": address},
        "대회 포지션": {"type": "clearinghouseState", "user": address, "dex": "par"},
        "대회 미체결 주문": {"type": "openOrders", "user": address, "dex": "par"},
        "API 지갑 승인 목록": {"type": "extraAgents", "user": address},
    }
    client = API(TESTNET_API_URL, timeout=15)
    results = {}
    try:
        for label, payload in queries.items():
            try:
                result = client.post("/info", payload)
            except Exception as exc:
                print(f"조회 실패: {label} ({type(exc).__name__}). 네트워크와 주소를 확인하세요.")
                return 1
            if isinstance(result, dict) and "error" in result:
                print(f"조회 실패: {label}. API가 오류 응답을 반환했습니다.")
                return 1
            results[label] = result
    finally:
        client.session.close()

    mode = results["계정 모드"]
    spot = results["담보 잔고"]
    state = results["대회 포지션"]
    orders = results["대회 미체결 주문"]
    agents = results["API 지갑 승인 목록"]
    if not (
        isinstance(mode, str)
        and isinstance(spot, dict)
        and isinstance(spot.get("balances"), list)
        and all(isinstance(item, dict) for item in spot["balances"])
        and isinstance(state, dict)
        and isinstance(state.get("assetPositions"), list)
        and isinstance(orders, list)
        and isinstance(agents, list)
    ):
        print("조회 실패: API 응답 구조가 예상과 다릅니다. 계정 상태를 판정하지 않았습니다.")
        return 1

    print(f"\n팀 지갑: {address}")
    print(f"계정 모드: {mode}")
    if mode != "unifiedAccount":
        print("확인 필요: 이 주소는 대회에서 사용하는 unifiedAccount 모드가 아닙니다.")

    balances = [item for item in spot["balances"] if item.get("coin") == "USDPR"]
    if balances:
        for balance in balances:
            print(f"USDPR 잔고 (total): {balance.get('total', '미확인')}")
        print("참고: total은 잔고이며, 새 주문에 사용할 수 있는 증거금과는 다릅니다.")
    else:
        print("USDPR 잔고 항목이 없습니다. USDPR를 받은 팀 지갑 주소인지 확인하세요.")

    positions = state["assetPositions"]
    print(f"\n대회 포지션 (par): {len(positions)}개")
    if positions:
        print(json.dumps(positions, ensure_ascii=False, indent=2))
    print(f"대회 미체결 주문 (par): {len(orders)}개")
    if orders:
        print(json.dumps(orders, ensure_ascii=False, indent=2))

    print(f"\nAPI 지갑 승인 목록: {len(agents)}개")
    if agents:
        print(json.dumps(agents, ensure_ascii=False, indent=2))
        print("브라우저 API 페이지의 지갑 주소·이름·만료 정보와 비교하세요.")
    else:
        print("승인 목록이 비어 있습니다. 브라우저의 연결 계정과 API 지갑 승인을 확인하세요.")
    print("\n조회 완료. 주문·서명·계정 설정 변경은 수행하지 않았습니다.")
    print("API 비밀키와 승인 주소의 일치 여부 및 주문 권한의 실제 동작은 아직 검증하지 않았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
