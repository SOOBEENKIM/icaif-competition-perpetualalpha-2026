# 하방위험 보정판의 테스트넷 자동 운용

`practice_live.runner`가 기존 `default` 목표 비중을 실제 `par` 계좌·호가·주문에 연결한다.
모델 가중치, 전략 규칙, 보정 규모와 기존 백테스트는 그대로 보존한다. 연구 패키지의
`strategy/profiles.json`은 연구용 주문 차단 설정을 유지하며, 별도 실행기의 `--execute`가
테스트넷 주문을 명시적으로 활성화한다.

## 실행과 중지

프로젝트 루트: `/home/ssd-990/soobeenkim/icaif_competition_2026`

```bash
# 실제 조회만 한 차례: 주문·레버리지 변경 없음
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m practice_live.runner --once

# 실제 테스트넷 주문이 가능한 지속 실행
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m practice_live.runner --execute

# 설치된 사용자 서비스 상태·로그·중지
systemctl --user status icaif-par-practice.service
journalctl --user -u icaif-par-practice.service -n 30
systemctl --user stop icaif-par-practice.service
```

`practice_live/live_runtime/STOP` 파일을 만들어도 실행기가 다음 점검에서 중지된다.
중지는 새 주문을 멈추며 보유 포지션을 자동 청산하지 않는다. 재시작은 같은 SQLite 상태를
사용해야 한다. 실행 중 데이터베이스·주문 기록을 삭제하면 안 된다.

브라우저나 Rabby 로그인 세션은 필요하지 않다. 기존에 승인한 API 지갑의 키를 로컬에서
불러오며, 키를 출력하거나 이동하지 않는다. 이 컴퓨터와 사용자 서비스가 실행 중이어야 한다.

## 판단과 주문

- 확정 시간봉을 매시간 갱신하고 기존 에이전트로 계산한다. 정상 판단은 한국시간 **03·09·15·21시**.
- 새 입력 계산 시간을 포함해 정상 주문은 해당 판단 시각부터 15분 이내에만 허용한다.
  종료된 판단 시각의 주문을 뒤늦게 따라 실행하지 않는다.
- 현 시점의 `par` mark, 실제 포지션, 재구성한 순자산으로 기존 Controller의 동일한 위험 제한·20% 비중 조정 버퍼를 적용한다.
- 종목별 **cross 1배**로 설정한 뒤, 실제 호가와 거래소의 주문 가능 수량을 확인한다.
- **IOC**, 최우선 상대 호가에서 **10bp** 이내의 지정가. 보이는 호가 수량, 목표 잔여 수량,
  기존 종목별 150,000 USDPR 한도를 적용한다. 체결되지 않은 나머지는 주문장에 남기지 않는다.
- 감소 주문을 먼저 처리한다. 반대 방향 전환은 기존 포지션이 실제로 청산된 것을 확인한 뒤 진행한다.
- 정상 회차에서 종목·감소/증가 단계별로 한 번만 전송한다. 부분 체결 잔여분은 다음 정상 판단에서
  실제 포지션과 비교한다. 기존 손실 제한이나 위험 한도 초과는 분 단위로 감소 주문을 재검토한다.

기존 백테스트의 다음 시간봉 체결·거래량 참여율은 과거 체결을 근사한 시뮬레이터 가정이다.
이 실행기는 판단 후 실제 호가에 IOC를 보내고 실제 체결을 기록한다. 따라서 기존 성과 수치를
실거래 성과로 취급하지 않는다. 전략 목표 비중을 추가로 축소하거나 다른 모델로 교체하지 않는다.

## 체결·계좌 복구

`live_runtime/state.sqlite3`에는 시작 잔고, 체결·펀딩·입출금, 회차별 목표, 주문 ID와 상태를 저장한다.
주문을 보내기 **전에** 회차별 고유 ID를 디스크에 확정한다. 전송 결과가 불명확하면 해당 ID로
거래소 상태를 조회하며 재전송하지 않는다. 끝내 확인할 수 없는 주문이 있으면 신규 주문을 보류한다.
첫 시작은 확인된 무포지션 계좌에서만 가능하고, 재시작은 저장된 시작점과 실제 체결을 복구한다.

실제 포지션 수량과 체결의 순수량이 일치해야 다음 주문을 만든다. 순자산은 시작 잔고에
실제 매매가격 손익·수수료·펀딩·외부 입출금을 반영하고 현재 `par` mark로 평가한다.
입금은 매매 수익에서 제외한다. unified 계좌의 spot 잔고와 미실현손익을 중복 합산하지 않는다.
다른 프로그램의 체결이나 미체결 주문을 발견하면 자동 운용을 보류한다.

상태: `live_runtime/status.json`, 최신 계좌 원본: `account_latest.json`, 전송 기록: `order_events.jsonl`.
`paused_retrying`은 주문 조건 또는 조회 오류를 해결할 때까지 30초마다 다시 확인하는 상태다.

## 확인한 공식 문서

- [대회 참가자 가이드: 계좌·주문·체결](https://perpetual-alpha26.github.io/docs/)
- [Hyperliquid 계좌 모드](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/account-abstraction-modes)
- [주문 상태·체결 조회](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint)
- [주문 가능 수량·레버리지 조회](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals)
