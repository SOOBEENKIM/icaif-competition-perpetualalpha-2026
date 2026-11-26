공개 지갑 수익률 리더보드 — 2026-10-16 11:11:54 KST 기준

전체 88개 지갑을 새로 조회했고, 실제 par 체결 지갑은 60개입니다.
이전과 같은 단일 100만 USDPR 지급 조건을 충족한 42개 거래 지갑을 비교합니다.
사용자 지갑: 0x125778065a578d598d88bddaa3fdd635280d0e47
사용자 순위: 31 / 42
오라클 평가 순손익: -95.025492 USDPR
오라클 평가 수익률: -0.00950255%
마크 평가 순손익: -138.813392 USDPR
마크 평가 수익률: -0.01388134%

report.html: 전체 순위, 사용자 요약, 주소 검색, 종목별 손익·최근 체결·포지션·주문.
leaderboard.csv: 전체 거래 지갑의 수익률. rank가 있는 행만 동일 자금 조건 순위.
fills.csv: 전체 지갑별 체결 58,298건과 수수료·거래 상대 주소.
실제 시장 체결은 양측 지갑을 합쳐 중복 제거하면 29,149건입니다.
by_market.csv / positions.csv / open_orders.csv: 종목별 성과·별도 조회 시각의 현재 포지션·미체결.
capital_flows.csv / funding.csv: 자금 유출입과 펀딩.
identity_and_exclusions.json: 비교 조건과 제외 주소. Keyrock 주소 자체는 미확인.
validation.json / coverage_validation.json: 양측 체결·시장 총 체결·포지션 교차 검증.
user_reconciliation.json: 사용자 체결·펀딩을 독립 실행 중인 봇 DB와 대조.
raw/: 이번 요청·응답·시각과 이전 원본을 합친 전체 이력. source/: 이번 계산 소스.

순위는 모든 지갑에 같은 기준시각의 오라클 가격을 적용한 누적 순수익률입니다.
실제 매매가격·수량에서 수수료·펀딩을 반영하고, 입출금을 수익에 포함하지 않습니다.
new_fill_count는 2026-10-15 14:25:07 KST의 이전 보고서 이후 추가된 par 체결입니다.
지갑별 운용 시작 시점은 다르며, 공식 팀·주소 대응표는 공개되지 않았습니다.
일별 오라클 평가 계좌 기록과 심사위원 점수가 없어 일별 STD·공식 종합 점수는 공란입니다.
Keyrock의 구체적인 주소를 확인하지 못했으므로 시장봇을 완전히 제외했다고 주장하지 않습니다.
시장 운영 계정을 포함할 수 있는 다른 자금 조건 지갑 18개는 별도로 제공합니다.

재계산:
.venv/bin/python reports/wallet_leaderboard_20261016/source/build_public_wallet_leaderboard.py --report reports/wallet_leaderboard_20261016 --previous reports/wallet_leaderboard_20261015

이 작업은 공개 데이터 조회와 로컬 보고서 생성만 수행했습니다.
