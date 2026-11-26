"""Limit-order preparation, submission, cancellation and reconciliation for par."""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from hyperliquid.utils.types import Cloid

from .connection import ROOT, SetupError


class OrderRejected(SetupError):
    pass


class OutcomeUnknown(SetupError):
    pass


def number(value) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite() or result <= 0:
        raise SetupError("가격과 수량은 0보다 큰 유한한 값이어야 합니다.")
    return result


def rounded_price(value: Decimal, sz_decimals: int, is_buy: bool) -> Decimal:
    # Integer prices are always valid. Fractional prices have both precision limits.
    if value == value.to_integral_value():
        return value
    places = max(0, min(6 - sz_decimals, 4 - value.adjusted()))
    tick = Decimal(1).scaleb(-places)
    return value.quantize(tick, rounding=ROUND_FLOOR if is_buy else ROUND_CEILING)


@dataclass(frozen=True)
class LimitOrder:
    coin: str
    is_buy: bool
    size: str
    price: str
    tif: str
    reduce_only: bool
    cloid: str

    @property
    def notional(self) -> Decimal:
        return Decimal(self.size) * Decimal(self.price)


def prepare_limit(coin, is_buy, size, price, *, sz_decimals, mark_price,
                  max_notional="25", tif="Alo", reduce_only=False, dex="par") -> LimitOrder:
    if (not isinstance(dex, str) or not dex or ':' in dex or dex.strip() != dex
            or not isinstance(coin, str) or not coin.startswith(dex + ':')
            or coin.count(':') != 1 or not coin.split(':')[1]
            or type(is_buy) is not bool or type(reduce_only) is not bool):
        raise SetupError("선택한 DEX 종목과 명확한 매수/매도·reduce-only 값을 지정하세요.")
    if tif not in {"Alo", "Gtc", "Ioc"} or not 0 <= sz_decimals <= 6:
        raise SetupError("지원하지 않는 주문 형식 또는 수량 정밀도입니다.")
    lot = Decimal(1).scaleb(-sz_decimals)
    quantity = number(size).quantize(lot, rounding=ROUND_FLOOR)
    limit = rounded_price(number(price), sz_decimals, is_buy)
    mark = number(mark_price)
    if quantity <= 0 or limit <= 0:
        raise SetupError("반올림 후 가격 또는 수량이 0입니다.")
    notional = quantity * limit
    if notional < Decimal("10") or notional > number(max_notional):
        raise SetupError(f"주문 금액은 10 이상 {max_notional} 이하이어야 합니다.")
    # Conservative connection-test guard; the venue's absolute band is wider.
    if abs(limit / mark - 1) > Decimal("0.05"):
        raise SetupError("현재 mark 가격에서 5% 이상 벗어난 주문은 전송하지 않습니다.")
    return LimitOrder(coin, is_buy, format(quantity, "f"), format(limit, "f"),
                      tif, reduce_only, "0x" + uuid.uuid4().hex)


def response_status(response, kind):
    if not isinstance(response, dict) or response.get("status") != "ok":
        raise OrderRejected("거래소가 요청을 거절했습니다.")
    body = response.get("response", {})
    if not isinstance(body, dict) or body.get("type") != kind:
        raise OutcomeUnknown("응답 형식을 확인할 수 없습니다. 주문 ID로 상태를 조회하세요.")
    statuses = body.get("data", {}).get("statuses")
    if not isinstance(statuses, list) or len(statuses) != 1:
        raise OutcomeUnknown("주문별 결과를 확인할 수 없습니다. 주문 ID로 상태를 조회하세요.")
    status = statuses[0]
    if isinstance(status, dict) and "error" in status:
        raise OrderRejected(f"거래소 주문 거절: {status['error']}")
    return status


class Executor:
    """One writer per API wallet. Uncertain submissions are never retried automatically."""

    def __init__(self, client, settings, *, max_notional="25", journal=None):
        self.client = client
        self.settings = settings
        self.max_notional = str(number(max_notional))
        self.journal = journal or ROOT / ".runtime" / "order_events.jsonl"

    def record(self, event, **fields):
        self.journal.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.journal, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as handle:
            handle.write(json.dumps({"time_ms": int(time.time() * 1000), "event": event, **fields}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def market(self, coin):
        data = self.client.info.post("/info", {"type": "metaAndAssetCtxs", "dex": "par"})
        for market, context in zip(data[0]["universe"], data[1]):
            if market["name"] == coin:
                return market, context
        raise SetupError("대회 시장 par에 없는 종목입니다.")

    def prepare(self, coin, is_buy, size, price, *, tif="Alo", reduce_only=False):
        market, context = self.market(coin)
        return prepare_limit(coin, is_buy, size, price, sz_decimals=market["szDecimals"],
                             mark_price=context["markPx"], max_notional=self.max_notional,
                             tif=tif, reduce_only=reduce_only)

    def _require_orders(self):
        if not getattr(self.client, "orders_enabled", False):
            raise SetupError("주문 전송은 명시적으로 활성화한 실행에서만 가능합니다.")

    def set_leverage(self, coin, leverage=1):
        self._require_orders()
        market, _ = self.market(coin)
        if type(leverage) is not int or not 1 <= leverage <= market["maxLeverage"]:
            raise SetupError("허용되지 않은 레버리지입니다.")
        result = self.client.update_leverage(leverage, coin, is_cross=True)
        if result.get("status") != "ok":
            raise OrderRejected("레버리지 설정이 거절됐습니다.")
        self.record("leverage", coin=coin, leverage=leverage)

    def submit(self, order: LimitOrder):
        self._require_orders()
        # Recheck metadata and price band immediately before sending an older plan.
        fresh = self.prepare(order.coin, order.is_buy, order.size, order.price,
                             tif=order.tif, reduce_only=order.reduce_only)
        if (fresh.size, fresh.price) != (order.size, order.price):
            raise SetupError("시장 정밀도가 달라졌습니다. 주문을 다시 준비하세요.")
        cloid = Cloid.from_str(order.cloid)
        self.record("submit_intent", order=asdict(order))
        try:
            response = self.client.order(order.coin, order.is_buy, float(order.size), float(order.price),
                                         {"limit": {"tif": order.tif}}, reduce_only=order.reduce_only,
                                         cloid=cloid)
        except Exception:
            self.record("submit_unknown", cloid=order.cloid)
            raise OutcomeUnknown(f"전송 결과가 불명확합니다. 재전송하지 말고 cloid={order.cloid}를 조회하세요.") from None
        self.record("submit_response", cloid=order.cloid, response=response)
        status = response_status(response, "order")
        if not isinstance(status, dict) or not ({"resting", "filled"} & status.keys()):
            raise OutcomeUnknown(f"접수 또는 체결을 확인하지 못했습니다. cloid={order.cloid}")
        return status

    def status(self, cloid):
        return self.client.info.query_order_by_cloid(self.settings["team_wallet"], Cloid.from_str(cloid))

    def cancel(self, coin, cloid):
        self._require_orders()
        if not coin.startswith("par:") or coin not in self.client.info.coin_to_asset:
            raise SetupError("대회 종목의 주문만 취소할 수 있습니다.")
        self.record("cancel_intent", coin=coin, cloid=cloid)
        try:
            response = self.client.cancel_by_cloid(coin, Cloid.from_str(cloid))
        except Exception:
            self.record("cancel_unknown", cloid=cloid)
            raise OutcomeUnknown(f"취소 결과가 불명확합니다. cloid={cloid}를 조회하세요.") from None
        self.record("cancel_response", cloid=cloid, response=response)
        if response_status(response, "cancel") != "success":
            raise OutcomeUnknown("취소 완료를 확인하지 못했습니다. 주문 상태를 조회하세요.")

    def account_state(self):
        team = self.settings["team_wallet"]
        return {
            "positions": self.client.info.user_state(team, dex="par")["assetPositions"],
            "open_orders": self.client.info.open_orders(team, dex="par"),
        }
