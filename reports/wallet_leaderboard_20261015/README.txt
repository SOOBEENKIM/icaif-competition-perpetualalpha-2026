공개 지갑 수익률 스냅샷 — 2026-10-15 14:25:07 KST

report.html: 100만 USDPR 지급 확인 지갑 33개의 수익률 순위와 지갑별 상세.
leaderboard.csv: 실제 거래 지갑 51개 및 사용자 지갑. rank가 있는 행만 본 순위.
fills.csv: 전체 53,686건의 지갑별 체결. 시장 체결 1건은 양측 지갑에 각각 기록됨.
  side B=매수, A=매도. dir=롱/숏 진입·종료·반전.
  crossed=True는 테이커, False는 메이커. closedPnl은 API 표시값, fee는 실제 수수료.
  counterparty는 coin/time/tid를 기준으로 반대편 체결과 연결한 공개 주소.
by_market.csv: 종목별 손익, 수수료, 펀딩, 보유 수량과 반올림 보정.
positions.csv / open_orders.csv: 별도 조회 시각의 포지션·레버리지와 주문.
capital_flows.csv / funding.csv: 수익에서 제외한 자금 유입·유출과 펀딩.
identity_and_exclusions.json: Keyrock 역할 확인 근거와 지갑 주소 미확인 한계, 본 순위 제외 주소.
validation.json / coverage_validation.json: 양측 체결, 포지션, 시장 캔들 체결 수 대조.
balance_reconciliation.json: W13/W26 실제 잔액과의 교차 확인.
raw/: 요청·응답과 수신 시각. 이전 저장 기록과 신규 API 조회를 합쳐 보관.
source/: 이번 조회와 계산에 사용한 소스.

수익률은 초기 USDPR 지급 후 누적치이며 지갑마다 첫 거래일이 다릅니다.
실제 체결 수량×(동일 기준시각 오라클−체결가)의 합에서 수수료를 빼고 펀딩을 더했습니다.
매도 수량은 음수입니다. 송금비용도 차감하며 입출금을 수익으로 세지 않습니다.
API 표시 closedPnl은 반올림된 진입가의 영향을 받을 수 있어 매매 가격·수량 차액과 대조합니다.
다른 자금 패턴의 계정에는 첫 USDPR 유입 이후 Modified Dietz를 별도 계산했습니다.
타 DEX·USDC 수익, 배포자 수수료 수익, 공식 위험·심사위원 점수는 이 비교에 포함하지 않습니다.
Keyrock의 구체적인 주소는 확인되지 않았습니다. 자금 조건 필터는 소유자 식별을 대신하지 않습니다.

오프라인 재계산 (프로젝트 루트):
.venv/bin/python scripts/build_public_wallet_leaderboard.py --report reports/wallet_leaderboard_20261015 --previous reports/combined_leaderboard_20261014

새 조회는 기존 디렉터리를 덮어쓰지 않고 새 --output 경로를 사용합니다.
이 작업은 공개 /info API와 공개 페이지만 읽었습니다. 주문·서명·개인키 사용은 없습니다.
