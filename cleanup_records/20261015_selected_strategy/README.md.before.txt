# Perpetual Alpha · 현재 작업

작업 경로는 `/home/ssd-990/soobeenkim/icaif_competition_2026`입니다. `4way` 폴더가 아닙니다.

## 2026-10-15 전략 재구현·공통 Registry 평가 결과

최신 결과는 [전략 비교 보고서](reports/strategy_recovery_20261015/report.html)입니다.
**수익성 기준을 통과한 전략은 없습니다. 이번 작업의 실제 주문은 0건이며 공개 수집은 종료했습니다.**

현재 호가·명시적 입력·단일 USDPR 계좌를 사용하는 `bot/market_data.py`, `strategy.py`,
`portfolio.py`, `evaluation.py`를 추가했습니다. 수량의 정수 단위 처리와 시간별 펀딩 정산을
수정하고, 기존 원본 호가에서 36개 개발 비교 및 보유 추세 비교를 수행했습니다.
오라클 가격차 15분은 표시 호가 조건에서 −701.93 USDPR, 새 잔차 릿지 15분은
−921.10 USDPR였습니다. 모두 초기 자본 1,000,000 USDPR, 목표 주문액 10,000 USDPR의 모의 결과입니다.

공통 Registry에 사전 등록한 **2026-10-15 13:28:29~14:13:29 KST** 새 구간에서는
오라클 가격차·기존 고정 릿지·현금이 두 실행 조건 모두 0건 거래·0 손익이었습니다.
수익성 검증 통과가 아닙니다. SP500의 작은 양수 예상 신호 3개는 사전에 정한 2bp 펀딩
준비금 때문에 배제됐으며, 실제 펀딩 비용과 이 보수적 가정은 보고서에서 구분합니다.

15개 실행·장부 검사와 기존 Registry 검사 13개를 통과했고, 개발 모의 체결 2,645건을
원본 호가와 대조했습니다. 새 구간 6개 장부는 봉인한 소스 사본으로 동일하게 재현했습니다.
이는 실제 거래소 체결·계좌 복구·지속 수익성을 입증한 결과가 아닙니다.
[재현 안내](reports/strategy_recovery_20261015/REPRODUCE.txt) ·
[공개 자료·소스·결과 패키지](reports/strategy_recovery_20261015/evidence.zip)

## 이전 작업: 2026-10-15 현재 호가 기준 예측

최근 수정은 [현재 호가 예측 보고서](reports/directional_quotes_expanded_20261015/report.html)와
[실행·재현 안내](reports/directional_quotes_expanded_20261015/README.txt)를 먼저 확인합니다.
구현은 `replay_lab/directional_v2/`입니다. 아래의 기존 전략·학습 결과는 비교 기록으로 보존합니다.

현재 midpoint에서 5·15·30분 뒤 bid·ask·midpoint 변화를 예측하고, 현재 호가에서의 진입과
미래 반대편 호가 청산 비용을 연결했습니다. 70.85시간의 실제 호가 자료로 릿지·XGBoost·
동일 가중 앙상블을 학습했습니다. **수익성 검증을 통과한 후보는 없으며 실제 주문은 0건입니다.**
0건 거래·0 손익은 통과가 아닙니다. 실시간 선기록 결과와 모의 장부를 구분해 제공합니다.

이번 진단은 공통 평가 Registry의 사전 예약을 사용하지 않은 누락이 있습니다. 별도의 모델·
예측 시각 봉인은 남아 있으나, 공통 규약을 통과한 독립 최종평가로 표현하지 않습니다.
늦은 사용 이력 등록은 사전 예약을 대신하지 않습니다. [평가 상태 기록](reports/directional_quotes_expanded_20261015/registry_status.json)을 참고합니다.

이전 후속 공개 호가 수집 기록은 `reports/directional_quotes_expanded_20261015/next_24h_capture/`에
저장되어 있으며 해당 수집기는 중단했습니다. [수집 상태](reports/directional_quotes_expanded_20261015/next_24h_capture/status.json)의
시각·오류·`finished`를 확인합니다. 이 수집기는 모델 예측이나 주문을 실행하지 않으며,
사전 예약한 독립 평가가 아닙니다. 완료된 실험의 ZIP에는 이 진행 중인 자료를 넣지 않았습니다.

## 남겨 둔 세 전략

이 세 규칙은 비교 기준으로 고정되어 있습니다. 2026-10-07에 실제 학습 모델 네 계열을
추가했으며, 현재 화면은 **규칙 3개 + 학습 모델 4개 = 7개 버전**을 비교합니다.
학습 방법·논문 대비 변경점·기간별 결과는 [학습 모델 실험 기록](replay_lab/neural_rank/README.md)을 참고합니다.

1. **기존 규칙** — 5·10·20·30일 채널 비중 각 25%.
2. **수정판 (위험보정)** — 시장 추세 단계별 비중, 원래 예산의 85.23899299025633%.
3. **수정판 (원래투자예산)** — 같은 추세 단계별 비중, 원래 예산의 100%.

화면: <http://127.0.0.1:8765/strategy-lab/>

```bash
cd /home/ssd-990/soobeenkim/icaif_competition_2026
.venv/bin/python -m replay_lab --from-latest
```

이미 서버가 실행 중이면 명령을 다시 실행하지 않고 화면을 새로고침합니다.
다른 PC에서는 해당 PC 터미널에서 SSH 포워딩을 연결합니다.

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8765:127.0.0.1:8765 finx_sbk@166.104.16.108
```

로컬 8765 포트가 사용 중이면 기존 터널을 사용하거나 앞쪽 포트를 8766으로 바꾸고
`http://127.0.0.1:8766/strategy-lab/`를 엽니다.

- 결과 위치: `replay_lab/output/retained_strategies/`
- 세 버전 × 5개 기간 × 2개 거래 조건 = **30개 원본 거래 기록**.
- 무결성 확인: `.venv/bin/python -m replay_lab.retained verify`
- 원본 history에서 계산·개발 위험보정·주문·체결·장부를 다시 재현:
  `.venv/bin/python -m replay_lab.retained verify --replay`
- 정리 범위와 유지 사유: [현재 전략 및 파일 안내](replay_lab/RETAINED.md).

앞선 정리는 전략의 숫자나 규칙을 바꾸지 않았습니다. 당시 다른 모델의 상세 실험 결과·체크포인트·
임시 화면·캐시를 삭제하고 현재 화면에서 제거했습니다. 원본 선택 과정의 작은 요약 기록은
출처로 보존합니다. 공통 소스, 원본 데이터, 실행 환경, 지갑 설정, 진행 중인 공개 데이터
모의관측과 그 재시작 의존 파일은 보존합니다.

세 버전은 과거 가격에 대한 연구 결과이며 별도 학습 예측 모델을 사용하지 않습니다.
실제 주문이나 팀 자금 사용은 없습니다. 전체 과거 호가와 펀딩을 완전히 재현한 결과 또는
공식 대회 순위가 아닙니다.

## 추가한 학습 모델 실험

- 신경망 순위 예측, XGBoost 순위 예측, 신경망 수정판, XGBoost 초과수익 예측.
- 56개 개발 설정 중 각 계열의 후보를 앞선 기간에서 선택한 뒤 40개 상세 백테스트를 저장했습니다.
- 새 결과: `replay_lab/output/neural_rank/`, `ranktree/`, `neural_rank_refined/`, `returntree/`.
- 통합 비교: `replay_lab/output/model_challenge/`. 고정 화면 주소는 실험 기록에 있습니다.
- 저장된 학습 모델을 다시 읽어 예측과 장부를 검증:
  `.venv/bin/python -m replay_lab.neural_rank.audit`
- 비교 화면 자료 생성: `.venv/bin/python -m replay_lab.neural_rank.reporting`

이 명령은 과거 자료만 사용합니다. 새 모델을 기존 공개 데이터 관측 프로그램이나 지갑 주문에
연결하지 않았습니다. 연속 구간에서 규칙 최고 ROI를 소폭 넘긴 모델이 있으나, 최근 구간에서는
규칙 최고보다 낮고 반복 실험의 영향도 있으므로 수익성 우위가 확정된 모델로 표현하지 않습니다.

## 봇 설정 완료하기

환경이 활성화된 프로젝트 터미널에서 다음 명령을 실행합니다.

```bash
python -m bot.setup
```

처음 한 번 `sfinx_bot` API 비밀키를 숨김 입력합니다. 설정 프로그램은 다음을 수행합니다.

1. `config.local.json`에 지정된 API 지갑 주소와 키의 일치 여부를 확인합니다.
2. API 키를 `.secrets/api_wallet.key`에 저장합니다. **암호화하지 않은 로컬 파일**이며
   파일 권한은 600, 디렉터리 권한은 700입니다. `.secrets/`는 Git 추적에서 제외합니다.
   이 경로는 공유·업로드 대상에서 제외해야 합니다.
3. 저장한 키를 다시 읽어 SDK 클라이언트를 생성합니다. 팀 지갑을 `account_address`로,
   테스트넷을 API 주소로, `par`을 대회 시장으로 지정합니다.
4. API 지갑의 현재 승인·만료 상태, 통합 모드, 잔고, 포지션, 미체결 주문을 조회합니다.

**`PASS: 봇 설정 저장 및 계정 연결 확인 완료.`가 나와야 이 설정 단계가 완료된 것입니다.**
기존의 `scripts/check_*.py`는 점검 도구이며, 그 실행만으로 봇 키 설정이 저장되지는 않습니다.

설정을 저장한 뒤에는 새 터미널에서도 환경을 활성화하고 다음 명령으로 확인할 수 있습니다.

```bash
python -m bot.check
```

키를 다시 입력하지 않습니다. `bot.setup`을 다시 실행해도 기존 키를 덮어쓰지 않고
연결 확인만 수행합니다. 키 저장 후 네트워크 확인이 실패했다면 연결을 복구한 뒤 재실행합니다.

`bot.connection.build_client(load_settings())`는 이후 매매 프로그램에서 재사용할
SDK 클라이언트를 반환합니다. `client.wallet`은 승인된 API 지갑이고,
`client.account_address`는 USDPR를 받은 팀 계정입니다. 조회는 `client.info`를 사용하며,
포지션·미체결 주문 조회 시에도 `dex="par"`을 명시합니다.

현재 실행 모드는 **`read_only`**이고 `/exchange`로 보내는 모든 요청을 코드에서 차단합니다.
설정 완료는 모델·전략 구현이나 실제 주문 동작 검증의 완료를 의미하지 않습니다.
기존 점검 명령과 환경 재현 방법은 아래에 남겨둡니다.

## 환경 활성화

워크스테이션의 Bash 터미널에서 실행합니다.

```bash
cd /home/ssd-990/soobeenkim/icaif_competition_2026
source .venv/bin/activate
python --version
python scripts/check_environment.py
```

`PASS: Local environment is ready.`가 나오면 로컬 Python과 SDK 확인이 끝납니다.
새 터미널을 열 때마다 `source .venv/bin/activate`를 실행합니다.
활성화 없이 `.venv/bin/python scripts/check_environment.py`로 실행해도 됩니다.

## 주문 없이 공개 API 연결 확인

```bash
python scripts/check_environment.py --online
```

Hyperliquid **테스트넷**의 `POST /info`에 `{"type": "meta", "dex": "par"}`를
보내 대회 종목 목록만 읽습니다. POST 방식이지만 조회 API입니다.
이 스크립트는 지갑 연결, 서명, 주문, 취소, 자금 이동을 수행하지 않으며,
비밀키나 시드 구문을 입력받거나 읽지 않습니다.

이 검사는 공개 API 연결만 확인합니다. 팀 계정·승인된 API 지갑과의 연결 및
실제 주문 동작 검증은 별도 단계입니다. 팀 계정의 입금과 Unified 전환은
사용자가 이미 확인했으며, 여기서 다시 수행하지 않습니다.

## 팀 지갑 계정 조회

환경을 활성화한 상태에서 실행합니다.

```bash
python scripts/check_account.py
```

안내가 나오면 **USDPR를 받은 팀 지갑의 공개 주소**(0x로 시작하는 42자리)를
입력합니다. 조회 대상은 API 지갑 주소가 아닙니다. 입력값은 파일에 저장하지 않습니다.

이 프로그램은 테스트넷의 `/info`만 호출해 다음 내용을 표시합니다.

- 계정 모드: `unifiedAccount`인지 확인
- USDPR 잔고: `spotClearinghouseState`의 `total` 값 (주문 가능 증거금과 다름)
- 대회 시장 `par`의 포지션과 미체결 주문
- API 지갑 승인 목록: 브라우저의 주소·이름·만료 정보와 직접 비교

비밀키 입력, 서명, 주문, 계정 모드 변경은 하지 않습니다. 승인 목록 조회만으로
보관한 API 비밀키와 승인 주소의 일치 여부나 실제 주문 동작을 검증한 것은 아닙니다.
예상과 다른 모드나 빈 잔고·승인 목록이 나오면 먼저 조회한 팀 주소를 확인합니다.

## API 비밀키와 승인 주소 비교 (오프라인)

직접 연 워크스테이션 터미널에서 환경을 활성화한 뒤 실행합니다.

```bash
python scripts/check_api_key.py
```

입력 안내가 나온 **후에**, 저장한 `sfinx_bot` API 비밀키를 붙여넣고 Enter를 누릅니다.
붙여넣어도 글자나 별표가 표시되지 않는 것이 정상입니다. 팀 지갑의 비밀키는 사용하지 않습니다.
비밀키를 명령줄 인수, 채팅, 소스 코드에 적지 않습니다.

이 스크립트는 비밀키로 공개 주소를 로컬에서 계산한 뒤, 사용자가 조회한
`sfinx_bot`의 승인 주소 `0xb114d0bde505681315760de9a57e084a752cc48d`와 비교합니다.
네트워크 요청이나 파일 저장을 하지 않으며, 출력은 공개 주소와 비교 결과만 포함합니다.
입력을 숨길 수 없는 환경에서는 중단합니다.

`PASS: API 비밀키가 sfinx_bot의 승인 주소와 일치합니다.`가 나오면 키 선택이 맞습니다.
이는 봇 실행에 앞서 키를 확인하는 단계입니다. 키를 저장하고 봇에 연결하려면
위의 `python -m bot.setup`을 실행합니다. 실시간 스트림 연결, 모델·전략 구현,
주문 실행 검증은 이후 작업입니다.
API 지갑을 교체한 경우에는 먼저 `check_account.py`로 새 승인 주소를 확인하고,
`check_api_key.py`의 `EXPECTED_ADDRESS`를 새 공개 주소로 변경해야 합니다.

## 환경 재현

봇 기본 환경의 의존성 버전은 `requirements.lock`에 기록합니다.
연구 모델을 함께 사용하려면 기본 환경 구성 후 `requirements-research.txt`와
`requirements-diffusion.txt`의 패키지를 추가로 설치합니다. 아래 `uv pip sync`는
기본 lock에 없는 연구 패키지를 제거하므로, 실행했다면 연구 의존성도 다시 설치해야 합니다.

이 작업에서 사용한 uv와 관리형 Python은 `.tools/`, 다운로드 캐시는
`.cache/`, 가상환경은 `.venv/`에 있습니다. `.venv`가 `.tools/python`의
인터프리터를 참조하므로 `.tools/python`을 삭제하면 안 됩니다.
이 디렉터리들은 다른 경로로 복사하기보다 새 경로에서 다시 생성합니다.

```bash
UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" \
UV_PYTHON_BIN_DIR="$PWD/.tools/bin" \
UV_CACHE_DIR="$PWD/.cache/uv" \
.tools/uv/bin/uv venv --python 3.11 .venv

UV_CACHE_DIR="$PWD/.cache/uv" \
.tools/uv/bin/uv pip sync --python .venv/bin/python --require-hashes requirements.lock
```

가상환경에서 나가려면 `deactivate`를 실행합니다.

## 참고 문서

- [대회 Participant Guide](https://perpetual-alpha26.github.io/docs/)
- [공식 Hyperliquid Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk)
- [uv의 Python 설치 안내](https://docs.astral.sh/uv/guides/install-python/)
