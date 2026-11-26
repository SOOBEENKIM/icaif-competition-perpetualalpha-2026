"""Compare an API private key with the approved public address, entirely offline."""

import getpass
import re
import sys
import warnings

from eth_account import Account
from eth_keys.constants import SECPK1_N


# Public address from the team's extraAgents response; this is not a secret.
EXPECTED_ADDRESS = "0xb114d0bde505681315760de9a57e084a752cc48d"


def derive_address(private_key: str) -> str:
    match = re.fullmatch(r"(?:0[xX])?([0-9a-fA-F]{64})", private_key.strip())
    if match is None:
        raise ValueError("Invalid API private-key format.")
    raw_key = bytes.fromhex(match.group(1))
    if not 0 < int.from_bytes(raw_key, "big") < SECPK1_N:
        raise ValueError("Invalid API private-key value.")
    return Account.from_key(raw_key).address


def main() -> int:
    print("API 비밀키 확인 / 로컬 계산 전용")
    print(f"비교 대상: sfinx_bot / {EXPECTED_ADDRESS}")
    print("저장해 둔 sfinx_bot의 API 비밀키를 입력하세요. 팀 지갑 비밀키는 사용하지 않습니다.")
    print("붙여넣어도 글자나 별표가 표시되지 않습니다. 붙여넣은 뒤 Enter를 누르세요.")
    if not sys.stdin.isatty():
        print("중단: 직접 연 터미널에서 실행하세요. 파이프나 파일 입력은 받지 않습니다.")
        return 1

    private_key = None
    try:
        # Abort if getpass cannot disable echo; never fall back to visible input.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            private_key = getpass.getpass("API 비밀키 (화면에 표시되지 않음): ")
        address = derive_address(private_key)
    except (EOFError, KeyboardInterrupt):
        print("\n확인 취소.")
        return 1
    except getpass.GetPassWarning:
        print("중단: 입력 숨김을 지원하는 일반 터미널에서 실행하세요.")
        return 1
    except Exception:
        # Do not print exception text or a traceback that might include key data.
        print("확인 실패: API 비밀키 형식이나 값을 확인하세요. 입력값은 출력하지 않았습니다.")
        return 1
    finally:
        private_key = None

    print(f"\n계산된 공개 주소: {address}")
    if address.lower() != EXPECTED_ADDRESS.lower():
        print("FAIL: 저장한 키가 sfinx_bot의 승인 주소와 일치하지 않습니다.")
        print("sfinx_bot을 만들 때 저장한 API 비밀키인지 확인하세요.")
        return 1

    print("PASS: API 비밀키가 sfinx_bot의 승인 주소와 일치합니다.")
    print("네트워크 요청·키 파일 저장·서명·주문은 수행하지 않았습니다.")
    print("키와 주소의 일치만 확인했습니다. 현재 승인 상태와 실제 주문 동작은 별도 확인 사항입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
