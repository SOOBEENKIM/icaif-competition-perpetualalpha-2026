"""Preview, then explicitly submit/cancel one small testnet post-only order."""

import argparse
import json
from dataclasses import asdict
from decimal import Decimal, ROUND_CEILING

from .connection import SetupError, build_client, check_connection, close_client, load_settings
from .execution import Executor, OrderRejected, rounded_price, number


def test_order(executor, coin="par:ETH"):
    market, context = executor.market(coin)
    book = executor.client.info.l2_snapshot(coin)
    if not book["levels"][0] or not book["levels"][1]:
        raise SetupError("양방향 호가가 없어 접수·취소 테스트 주문을 준비하지 않았습니다.")
    bid = number(book["levels"][0][0]["px"])
    mark = number(context["markPx"])
    price = rounded_price(min(bid, mark) * Decimal("0.99"), market["szDecimals"], True)
    lot = Decimal(1).scaleb(-market["szDecimals"])
    size = (Decimal("12") / price / lot).to_integral_value(rounding=ROUND_CEILING) * lot
    return executor.prepare(coin, True, size, price, tif="Alo")


def show_plan(order):
    print(json.dumps(asdict(order), ensure_ascii=False, indent=2))
    print(f"주문 금액: {order.notional} USDPR 상당 / 이번 테스트 상한: 25")
    print("계획: 해당 종목 cross 1배 설정 → post-only 매수 1건 → 즉시 해당 주문 취소·상태 확인")
    print("접수 후 취소 전 체결될 가능성은 있습니다. 체결 시 자동 청산하지 않고 결과를 보고합니다.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coin", default="par:ETH")
    parser.add_argument("--submit", action="store_true", help="Actually send the testnet action.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status-cloid", help="Read the outcome of an earlier request; never submits.")
    group.add_argument("--cancel-cloid", help="Cancel only the specified order; requires --submit.")
    args = parser.parse_args()
    client = None
    try:
        settings = load_settings()
        client = build_client(settings, execution=True, enable_orders=args.submit)
        executor = Executor(client, settings)
        if args.status_cloid:
            print(json.dumps(executor.status(args.status_cloid), ensure_ascii=False, indent=2))
            return 0
        if args.cancel_cloid:
            if not args.submit:
                raise SetupError("취소를 전송하려면 --submit을 명시해야 합니다.")
            executor.cancel(args.coin, args.cancel_cloid)
            print(json.dumps(executor.status(args.cancel_cloid), ensure_ascii=False, indent=2))
            return 0
        check_connection(client, settings)
        initial = executor.account_state()
        if initial["positions"] or initial["open_orders"]:
            raise SetupError("기존 대회 포지션 또는 주문이 있어 접수·취소 테스트를 중단했습니다.")
        order = test_order(executor, args.coin)
        show_plan(order)
        if not args.submit:
            print("PREVIEW ONLY: 주문·취소·레버리지 변경을 전송하지 않았습니다.")
            return 0

        executor.set_leverage(order.coin, 1)
        try:
            result = executor.submit(order)
            print("주문 응답:", json.dumps(result, ensure_ascii=False))
        finally:
            # Cleanup targets our unique cloid even if the submission response was lost.
            try:
                executor.cancel(order.coin, order.cloid)
                print("테스트 주문 취소 응답: success")
            except OrderRejected:
                print("취소가 접수되지 않았습니다. 이미 체결·취소됐는지 상태를 조회합니다.")
            finally:
                state = executor.status(order.cloid)
                account = executor.account_state()
                print("최종 주문 상태:", json.dumps(state, ensure_ascii=False))
                print("최종 계정 상태:", json.dumps(account, ensure_ascii=False))
                print(f"추후 확인할 주문 ID: {order.cloid}")

        terminal = state.get("order", {}).get("status")
        if terminal != "canceled" or account["positions"] or account["open_orders"]:
            raise SetupError("깨끗한 접수·취소 완료를 확인하지 못했습니다. 위 체결·포지션·주문 상태를 확인하세요.")
        print("PASS: 테스트넷 주문 접수·취소 확인 완료. 남은 대회 포지션과 주문은 없습니다.")
        return 0
    except (EOFError, KeyboardInterrupt):
        print("중단했습니다. 전송을 시작했다면 .runtime/order_events.jsonl의 cloid로 상태를 확인하세요.")
    except SetupError as exc:
        print(f"FAIL: {exc}")
    except Exception as exc:
        print(f"FAIL: 실행 확인 실패 ({type(exc).__name__}). 자동 재전송하지 않았습니다.")
    finally:
        if client is not None:
            close_client(client)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
