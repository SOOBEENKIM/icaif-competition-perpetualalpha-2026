"""One-time local API-key setup followed by a read-only account connection check."""

import getpass
import os
import sys
import warnings

from .connection import (
    KEY_FILE, SetupError, build_client, check_connection, close_client,
    load_settings, save_key, signer_from_key,
)


def main() -> int:
    client = None
    try:
        settings = load_settings()
        print("봇 설정: Hyperliquid 테스트넷 / par / 조회 모드")
        if os.path.lexists(KEY_FILE):
            print("이미 저장한 API 키로 연결을 확인합니다. 기존 키를 덮어쓰지 않습니다.")
        else:
            print("API 키를 .secrets/api_wallet.key에 저장합니다.")
            print("키는 암호화하지 않은 로컬 파일로 저장되며, 현재 사용자만 읽고 쓸 수 있습니다(600).")
            print(".secrets 디렉터리는 권한 700으로 만들고 Git 추적에서 제외합니다.")
            print("sfinx_bot의 API 비밀키를 입력하세요. 팀 지갑 비밀키는 사용하지 않습니다.")
            if not sys.stdin.isatty():
                raise SetupError("직접 연 터미널에서 실행하세요. 파이프나 파일 입력은 받지 않습니다.")
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                private_key = getpass.getpass("API 비밀키 (입력이 보이지 않음): ")
            try:
                signer = signer_from_key(private_key, settings)
                save_key(signer)
            finally:
                private_key = None
            print("API 키 저장 완료. 저장된 키를 다시 읽어 SDK 클라이언트에 연결합니다.")
        client = build_client(settings)
        check_connection(client, settings)
        print("PASS: 봇 설정 저장 및 계정 연결 확인 완료.")
        print("주문은 실행하지 않았습니다. 이후 python -m bot.check로 키 재입력 없이 확인할 수 있습니다.")
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\n설정 또는 확인을 중단했습니다.")
    except getpass.GetPassWarning:
        print("FAIL: 입력 숨김을 지원하는 일반 터미널에서 실행하세요.")
    except SetupError as exc:
        print(f"FAIL: {exc}")
    except Exception as exc:
        # Only print the exception class; exception messages can contain key data.
        print(f"FAIL: 설정 또는 연결 확인 실패 ({type(exc).__name__}).")
        print("연결 문제를 해결한 뒤 같은 명령을 다시 실행하세요. 저장된 키는 다시 입력하지 않습니다.")
    finally:
        if client is not None:
            close_client(client)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
