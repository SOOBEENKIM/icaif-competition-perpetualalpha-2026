# SFINX PAS Live Observer

읽기 전용 대회 관측 대시보드. 거래 봇·개인키·실행 서비스에 접근하지 않음.

- `app/page.jsx`: 지갑 비교, 체결, 공식 팀 순위, 시장 대응, 지갑 상세
- `lib/domain.mjs`: 입출금 보정 손익, 실제 노출, 포지션 변화, 주문 체결률
- `lib/live.mjs`: 조회 큐·API 제한 대기·자동 갱신·브라우저 IndexedDB 보존
- `app/api/info/route.ts`: 공개 Hyperliquid Testnet info API만 허용하는 프록시
- `app/api/official/route.ts`: 공식 대회 chart/leaderboard 프록시
- `public/seed.json.gz`: 2026-10-26 공개 원본 수집본. 47개 지급 지갑, 35개 거래 지갑, PAS 체결 33,997건
- `scripts/collect-public.py`: 원본 재수집. 기존 프로젝트 공개 조사본에서 시작하며 처음 수집본이 있으면 이어받음
- `tests/domain.test.mjs`: 회계·체결·과거 가격 참조·원본 대조 검사

## 실행

`npm install` 후 `npm run dev`. 최신 glibc를 요구하는 workerd 런타임이 필요함. 배포는 `.openai/hosting.json`의 기존 private Site ID를 유지함.

## 검증

`node tests/domain.test.mjs`, `node tests/team-matching.test.mjs`, `node_modules/.bin/tsc --noEmit`, `npm run build`.
2026-10-26: 10개 도메인 검사 및 브라우저의 6개 탭, 모바일 가로 넘침, 실제 Hyperliquid 읽기 조회 검증. 이 호스트의 workerd는 glibc 호환성 문제로 실행되지 않아, 로컬 브라우저 QA는 동일 앱의 임시 Node/Vite 설정으로 진행함. 배포용 설정은 그대로 유지함.

## 해석

- SFINX는 확인된 팀명. 다른 지갑은 여러 시점의 수익률을 대조한 추정 팀명과 중복/경합 표시를 붙임.
- `public/team-evidence.json`은 공개 조사본과 공식 수익률 249개 지갑 관측치의 대조 근거. `scripts/build-team-evidence.py`로 재생성하며 `lib/team-matching.mjs`가 후보를 계산함.
- 공식 표본과 최대 30분 차이를 허용하고 30분 구간당 한 번만 집계함. 후보별 관측 구간이 다르면 공통 시점의 평균 절대오차로 비교함. 동일한 0% 이력은 구분하지 못하며 복수 후보로 표시함.
- 팀명 또는 주소를 클릭하면 선택 지갑의 추정 근거 표를 볼 수 있음. 검색과 CSV에도 추정 팀명이 포함됨.
- SFINX는 팀원 포함 공유 지갑 전체 결과임.
- 지갑 순위는 조회 시각이 서로 다른 관측 수익률 순위. 공식 순위는 별도 표시.
- unifiedAccount USDP 잔고에 미실현손익을 재합산하지 않음.
- 과거 체결 레버리지와 예측 목표는 공개 자료로 알 수 없음.
- 실시간은 폴링 방식: 선택 지갑 45초마다 시도, 전체 지갑 수 분 이상 순환. 페이지를 닫으면 조회도 멈춤.
- API 이력 한도·조회 오류를 표시하며 누락 자료를 0이나 완전한 실적으로 가장하지 않음.
