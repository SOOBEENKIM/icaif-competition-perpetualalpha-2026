"""Verify order signing locally; optionally verify one no-order signed noop online."""

import argparse
import time

from hyperliquid.utils.signing import (
    order_request_to_order_wire, order_wires_to_order_action,
    recover_agent_or_user_from_l1_action, sign_l1_action,
)
from hyperliquid.utils.types import Cloid

from .connection import SetupError, build_client, check_connection, close_client, load_settings
from .execution import Executor
from .smoke import test_order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-signature", action="store_true",
                        help="Send one signed noop using a fresh nonce; never places an order.")
    args = parser.parse_args()
    client = None
    try:
        settings = load_settings()
        # This client permits noop only; order/cancel/leverage actions remain blocked.
        client = build_client(settings, execution=True)
        check_connection(client, settings)
        order = test_order(Executor(client, settings))
        wire = order_request_to_order_wire({
            "coin": order.coin, "is_buy": order.is_buy,
            "sz": float(order.size), "limit_px": float(order.price),
            "order_type": {"limit": {"tif": order.tif}},
            "reduce_only": order.reduce_only, "cloid": Cloid.from_str(order.cloid),
        }, client.info.name_to_asset(order.coin))
        action = order_wires_to_order_action([wire])
        nonce = int(time.time() * 1000)
        signature = sign_l1_action(client.wallet, action, None, nonce, None, False)
        recovered = recover_agent_or_user_from_l1_action(action, signature, None, nonce, None, False)
        if recovered.lower() != settings["api_wallet"].lower():
            raise SetupError("로컬 주문 서명에서 복원한 API 주소가 일치하지 않습니다.")
        print("PASS: 실제 대회 주문 형식·종목 ID·API 지갑 서명·주소 복원 확인. 주문은 전송하지 않았습니다.")
        if args.verify_signature:
            result = client.noop(int(time.time() * 1000))
            if not isinstance(result, dict) or result.get("status") != "ok":
                raise SetupError("서버가 서명된 noop 요청을 승인하지 않았습니다.")
            print("PASS: 테스트넷 서버의 API 서명 인증 확인 (noop 1건).")
            print("새 nonce 하나가 사용 처리됐습니다. 주문·취소·레버리지 변경·자금 이동은 없습니다.")
        print("실제 주문 접수·체결·취소는 이 검사에 포함되지 않습니다.")
        return 0
    except SetupError as exc:
        print(f"FAIL: {exc}")
    except Exception as exc:
        print(f"FAIL: 실행 연결 검증 실패 ({type(exc).__name__}).")
    finally:
        if client is not None:
            close_client(client)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
