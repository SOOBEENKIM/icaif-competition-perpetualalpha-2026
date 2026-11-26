# Perpetual Alpha — 선택 전략

2026-10-15: 추세＋7일 되돌림을 공통 매매 로직으로 선택했습니다. 고정 규모 보정판을 기본값으로, 원래 규모를 비교 대상으로 사용합니다.

| 프로필 | 원래 규모 대비 | 기존 정책 ID |
|---|---:|---|
| `default` | 0.6938788714042263 | `tail_calibrated` |
| `original` | 1.0 | `crypto_d7_aligned_add50` |

두 프로필은 같은 신호를 사용합니다. 69.39%는 원래 비교본의 목표 포지션 대비 비율이며 전체 계좌 자본의 69.39%를 투자한다는 뜻이 아닙니다. 선행 배율까지 포함한 계수는 각각 0.6495940879913923과 0.9361779335877263입니다.

## 실행 진입점

설정 확인: `.venv/bin/python -m strategy`

```python
from strategy import load_agent
agent = load_agent()                    # 기본 규모
comparison = load_agent("original")    # 원래 규모 비교본
agent.prepare_history(market, original_ohlcv, funding_source="mainnet_reference_settled")
intent = agent.decide(time, quantities, equity)
state = agent.snapshot()
restored = load_agent(saved=state)
```

완성된 연속 시간봉·펀딩과 계좌 상태를 호출자가 전달합니다. 프로필별 상태를 분리하며, 다른 프로필의 상태를 복원하면 거부합니다. 현재 어댑터는 과거 기준시장 자료에서 목표 포지션과 주문 의도를 계산합니다. 서명·실제 주문·상시 거래 루프는 연결하지 않았습니다.

## 재현 검증

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python -m strategy.verify --rows 5006 --output /tmp/selected-strategy-verification.json
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest replay_lab.evaluation_control.test_registry -v
```

5,006개 시점의 두 프로필 목표 포지션을 원본과 대조합니다. 새 시장 수익성 평가나 하이퍼파라미터 탐색이 아닙니다. 로컬 모델만 사용합니다. 미래 관측을 제외한 접두 구간은 `--rows 4401`로 검증할 수 있습니다.

## 유지한 파일

| 위치 | 내용 |
|---|---|
| `strategy/` | 공통 진입점·프로필·재현 검증 |
| `replay_lab/reversion_blend/` | 추세＋되돌림 공통 신호 |
| `replay_lab/tail_calibration/` | 기본 규모 배율 적용 |
| `replay_lab/output/reversion_blend/20261014-v1/` | 원래 규모의 설정·입력·예측·평가 |
| `replay_lab/output/tail_calibration/20261014-v1/` | 기본 규모의 설정·입력·평가·동결 실행 패키지 |
| 위 경로의 `runtime/pretrained/` | 필요한 Chronos 가중치와 출처 |
| `replay_lab/`의 나머지 Python 모듈 | 실제 import하는 EMA·펀딩·위험·회계 부품 |
| `history/`, 기존 캡처·펀딩 경로 | 원본 시장자료 |
| `bot/`, `scripts/check_*.py` | 계좌 설정·연결·주문 처리·공개 자료 도구 |
| `reports/wallet_leaderboard_20261015/` 등 | 타 지갑 비교 기준 |
| `replay_lab/output/evaluation_control/` | 평가 노출 이력·Registry |
| `cleanup_records/20261015_selected_strategy/` | 삭제 명세·동결 출처·정리 전후 검증 |

기존 폴더 이름에 pairs·maker 등이 남아 있어도 별도 운용 후보가 아닙니다. 해당 파일은 선택 전략이 사용하는 공용 import 의존성입니다. 불필요한 실험 진입점·학습 결과·UI는 삭제했습니다.

## 결과와 기록

- [기본 규모 원본 결과](replay_lab/output/tail_calibration/20261014-v1/test_result.json)
- [원래 규모 원본 결과](replay_lab/output/reversion_blend/20261014-v1/test_result.json)
- [정리 완료 보고서](cleanup_records/20261015_selected_strategy/report.html)
- [정리 전 성과 목록](reports/internal_strategy_inventory_20261015/report.html)

성과는 이미 검토한 과거 가격·거래량·기준시장 펀딩을 사용한 모의 결과입니다. 실제 대회 호가 체결 성과나 독립 수익성을 입증하지 않습니다. 이번 변경에서 신호·모델·규모 계수를 재학습하거나 변경하지 않았습니다. 과거 자료의 평가 노출 이력도 유지합니다.

지갑 설정·비밀키·Python 환경·원본 CSV는 삭제하지 않았습니다. `config.local.json`, `.secrets/`, `.venv/`, `.tools/`는 유지합니다.
