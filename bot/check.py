"""Load the saved bot configuration and API key, then query the team account."""

from .connection import SetupError, build_client, check_connection, close_client, load_settings


def main() -> int:
    client = None
    try:
        settings = load_settings()
        client = build_client(settings)
        check_connection(client, settings)
        print("PASS: 저장된 봇 설정으로 계정 연결 확인 완료. 주문은 실행하지 않았습니다.")
        return 0
    except FileNotFoundError:
        print("FAIL: 먼저 python -m bot.setup으로 봇 설정과 키 저장을 완료하세요.")
    except SetupError as exc:
        print(f"FAIL: {exc}")
    except Exception as exc:
        print(f"FAIL: 연결 확인 실패 ({type(exc).__name__}). 설정과 네트워크를 확인하세요.")
    finally:
        if client is not None:
            close_client(client)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
