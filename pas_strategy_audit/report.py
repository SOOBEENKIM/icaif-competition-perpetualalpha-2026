"""Standalone audit report with actual performance separated from conditional replays."""
import json,html,datetime,hashlib
from .data import OUT,ROOT

def table(headers,rows):
 return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def n(v,d=2):return f'{v:,.{d}f}'
def kst(ms):return datetime.datetime.fromtimestamp(ms/1000,datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')

def main():
 a=json.loads((OUT/'audit.json').read_text());rows=json.loads((OUT/'comparison.json').read_text());targets=json.loads((OUT/'target_audit.json').read_text());own=a['actual'];peer=a['peers'][0];quality=a['market_quality']
 names={'incumbent_published':'기존판 동일 조건 재생','all_markets_long_only':'10종목 확대·기존 롱 중심','all_markets_symmetric_daily':'10종목·일봉 추세 양방향','pas_intraday_signed':'PAS 15분 판단·단기 양방향','symmetric_daily_pas_veto':'일봉 양방향＋PAS 충돌 감축'}
 findings=[
 ['주력 종목 제한','EMA는 close[:, :3], 채널은 index<3, 7일 되돌림은 crypto 집합을 사용한다. 10종목 메타데이터 조회와 10종목 신호 생성은 달랐다.','모든 허용 종목에 같은 진입·청산 함수를 적용하는 후보 구현. 신호가 없는 종목은 현금 유지.'],
 ['추세의 롱 편향','선정 EMA의 상태는 0/1이며 숏 진입이 없다. 선정 채널도 롱 중심이다. 펀딩·되돌림·페어의 숏 가능성과 추세 전략 자체의 양방향성은 다르다.','EMA·채널에 대칭 숏 진입, 숏 추적 손절과 반대 교차 청산 구현. 상승/하락 합성 경로에서 10종목 모두 검사.'],
 ['느린 보유 판단','10/60일 EMA, 6시간 결정, exit_cross=False, 3ATR 청산, 20% 동일 방향 수량 버퍼와 일별 완만한 규모 변경. 하루 하락만으로 자동 숏 전환되는 설계가 아니다.','일봉 양방향 후보와 별도 PAS 15분 후보를 비교. 빠른 후보 성과는 더 나빠 자동 채택하지 않음.'],
 ['작동하지 않은 페어 오버레이','관측 구간 진입 검사가 392회 모두 decision_gate_rejected. 실제 페어 진입 0, 기존 방향성 BTC/HYPE 롱만 보유.','페어 명칭과 실제 기여를 분리 기록. 회귀 예측, 비용 임계값, 최종 action=0을 별도로 확인.'],
 ['주문 대기 중 호가 만료','공유 호가 스냅샷의 유효기간은 3초. 먼저 낸 주문의 확정 체결 대기 중 후순위 자산의 호가가 만료된다. 4초 지연을 주면 두 번째 목표 주문이 누락되는 것을 재현.','거래가 필요한 각 종목을 처리하기 직전 새 문맥·호가 조회. 이미 확정된 청산 뒤 다음 주문으로 숏 개설. 10종목 공개 호가 실조회 통과.'],
 ['위험 감축 재확대','현재 risk matching 경로에서 4% → 위험 필터 2% → 3.8%로 복원되는 예제를 재현. 현재 손실 원인이라는 주장은 아님.','추가 충돌 감축 후보는 예산 배분 후 감축하고, 다음 기본 신호가 오기 전 복원·다른 자산 재배분을 막음.'],
 ['신호 충돌','양방향 후보 HYPE에서 EMA 숏과 7일 되돌림 롱이 충돌하여 최종 목표가 작은 롱으로 남았다. 기존 “aligned” 조건과 일봉 EMA가 다른 추세 정의를 사용한다.','이미 정한 PAS 단기 추세와 반대인 목표를 에피소드 중 감축하는 후보도 추가 검증. 임의 강제 숏은 사용하지 않음.'],
 ['시장·검증 불일치','주력 방향 판단은 메인넷 과거 자료, 실제 거래는 PAS. baseline에는 PAS 괴리·보유 펀딩·왕복 비용에 대한 직접 기대수익 검사가 없다. 과거 PAS L2/오라클/마크 전체 이력도 저장되지 않았다.','이번 수집은 실제 PAS 캔들·정산 펀딩으로 재생. 전 기간의 정확한 호가 체결·공식 평가손익 재현이라고 주장하지 않음.']]
 h=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SFINX 전략·PAS 시장 재검토</title><style>body{font:16px/1.65 system-ui;color:#172635;max-width:1280px;margin:32px auto;padding:0 24px}h1,h2{line-height:1.3}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d9e1e8;padding:10px;text-align:left;vertical-align:top}th{background:#eef3f7}.box{padding:18px;background:#eef6f7;border-left:4px solid #26717a;margin:20px 0}.warn{background:#fff7e7;border-color:#ac7d26}a{color:#176581}code{background:#eef3f7}</style><h1>SFINX 제출 전략·PAS 시장 재검토</h1>',
 f'<p>고정 분석 구간: {kst(a["period"]["start_ms"])} → {kst(a["period"]["cutoff_ms"])}. 총 24시간 25분. 후보를 이 구간의 문제를 보고 만들었으므로 개발·진단 평가이며 독립 테스트로 부르지 않는다.</p>',
 '<div class="box"><b>결론:</b> 종목 제한, 추세의 롱 편향, 느린 청산 조건, 후순위 주문의 호가 만료를 실제 코드와 재현 테스트에서 확인했다. 이를 고치는 후보는 구현했다. 그러나 비교 결과는 비용까지 포함한 수익 개선을 입증하지 못했으므로 실거래 교체를 권하지 않는다.</div>',
 '<h2>실제 운용 실적</h2>',table(['관측 시각','순손익(USDP)','10만 자본 수익률','지급 수수료','수령 펀딩','체결 / 페어 진입'],[[kst(own['observation_time_ms']),n(own['pnl'],6),n(own['return_percent'],6)+'%',n(own['fees'],6),n(own['funding_received'],6),'2주문·3조각 / 0']]),
 '<p>보유: BTC 0.01899 롱, HYPE 3.52 롱. 계좌 자본 100,000 USDP 중 사용자 전략 예산은 50,000, 팀원 예약 현금은 50,000이다. 자금 부족으로 다른 종목이 막힌 상황은 아니다. 실제 포지션 총액은 약 1,800 USDP였다.</p>',
 '<h2>확인된 코드 문제와 수정 상태</h2>',table(['항목','확인한 사실','구현·검증 상태'],findings),
 '<p>실제 최초 실행 로그에도 후순위 7종목 NO_FRESH_BOOK가 있지만, 그때 해당 종목의 목표 수량은 0이었다. 따라서 실제 두 종목만 보유한 직접 원인은 신호의 제한이며, 호가 만료는 다종목 주문이 필요한 상황에서 별도로 재현한 실행 결함이다.</p>',
 '<h2>대회 시장 10종목</h2><p>수익률은 본대회 이후 각 종목에서 처음 관측된 실제 체결가격부터 마지막 실제 체결가격까지의 변화다. 종목별 첫·마지막 체결 시각이 다르며, 정확한 개장 마크 대비 수익률이 아니다.</p>',table(['종목','첫 체결 → 마지막 체결','변화','무체결 1분 비율','마지막 체결 경과(분)','수집 시 호가 간격(bp)','정산 펀딩 누적(bp)'],[[x['symbol'],n(x['first_observed_price'],5)+' → '+n(x['last_trade_price'],5),n(x['first_to_last_trade_return_percent'],3)+'%',n(x['no_trade_minutes_percent'],2)+'%',n(x['final_trade_age_minutes'],0),n(x['spread_bps'],2),n(x['settled_funding_sum_bps'],2)] for x in quality]),
 '<p>이 비교에서 9종목 하락, BRENT 상승이었다. 무체결 분봉이 많다는 사실은 호가가 없다는 뜻은 아니다. 실제 최신 호가는 10종목 모두 유효했다. 다만 무체결 종가를 현재 거래 가능한 가격으로 간주해서는 안 된다. PAS 가격·메인넷 가격·오라클·마크의 차이는 <a href="../pas_mainnet_basis_20261020/report.html">별도 가격 괴리 보고서</a>에 정리했다.</p>',
 '<h2>같은 조건의 후보 비교</h2><div class="box warn">아래는 실제 계좌 PnL이 아닌 조건부 PAS 체결 재생이다. 과거 L2와 마크 이력이 없어 1분 실제 거래가격과 거래량을 사용했다. 실제 계좌의 −14.43 USDP와 아래 기존판 −2.30 USDP를 직접 비교하여 개선됐다고 주장할 수 없다.</div>',
 '<p>공통 조건: 팀 자본 100,000, 사용자 예산 50,000, 총 목표 명목액 상한 2,000 USDP. 진입 방향에 유리한 종가 체결 대신 이후 활성 분봉의 매수 고가/매도 저가와 추가 충격을 적용했다. 기본은 편도 9bp 수수료＋10bp 충격·거래량 참여 1%, 불리한 조건은 20bp 충격·0.5% 참여다. 과거 오라클이 없어 펀딩 정산 금액은 직전 PAS 체결가격으로 근사했다. 체결이 줄어드는 조건에서는 손실도 작아질 수 있다.</p>',
 table(['후보','조건','평가 순손익','잔여 포지션 청산비용 충당 후','평균 명목액','최대낙폭','롱/숏 진입 체결 조각','감축 체결'],[[names[x['candidate']],x['scenario'],n(x['pnl'],4),n(x['pnl_after_terminal_cost_reserve'],4),n(x['mean_gross'],2),n(x['max_drawdown_percent'],5)+'%',str(x['long_entry_fills'])+' / '+str(x['short_entry_fills']),x['reduction_fills']] for x in rows]),
 '<p>일봉 양방향과 충돌 감축 후보는 기본 조건의 평가손익만 보면 개선됐으나, 청산비용 충당 후 또는 불리한 체결 조건에서는 개선이 유지되지 않았다. 단기 양방향은 과잉 전환·늦은 체결·비용으로 더 나빴다. 어떤 후보도 이 표를 근거로 수익 개선판으로 채택하지 않는다. 포지션을 평가한 마지막 거래가 오래된 경우도 있어 원본 결과에 valuation_age_minutes를 기록했다.</p>',
 '<p>완료된 일별 수익률은 1개다. 표본 표준편차를 요구하는 대회 σ·Q·Score는 아직 계산할 수 없어 null로 기록했다. 최대낙폭은 대회 σ의 대체값이 아니다. 공식 순위와 분중 PnL 그래프의 차이는 <a href="https://perpetual-alpha26.github.io/leaderboard">공식 리더보드</a> 설명을 따른다.</p>',
 '<h2>수익이 높은 공개 지갑의 실제 거래</h2>',f'<p>과거 수집했던 공개 지갑 88개를 다시 조회해 USDP 보유 47개를 확인했다. 그중 {html.escape(peer["wallet"])}의 잔액은 {kst(peer["balance_observed_ms"])}에 {n(peer["usdp_total"],6)} USDP였다. 원장은 시작 지급액 100,000 USDP 1건을 확인했다. 공식 팀명은 주소와 연결되지 않아 확정하지 않는다. 공식 일별 PnL 최고 팀은 {html.escape(a["top_pnl_daily_snapshot"]["name"])} {n(a["top_pnl_daily_snapshot"]["return"]*100,6)}%다.</p>',
 f'<p>공개 지갑: {peer["orders"]}개 주문·{peer["fill_fragments"]}개 체결 조각, 10종목. 명목 거래금액의 {n(peer["maker_notional_fraction"]*100,2)}%가 메이커, 실효 수수료 {n(peer["effective_fee_bps"],3)}bp, 현재 총 포지션 명목액 {n(peer["gross_notional"],2)} USDP. 우리 실효 수수료는 9bp, 명목액은 약 1,800이다. 수익 차이에 방향 선택뿐 아니라 거래 규모·비용 차이도 포함된다.</p>',
 table(['종목','현재 방향','실현 가격손익','미실현 가격손익','수수료','펀딩','합산 손익'],[[x['symbol'],'롱' if x['current_qty']>0 else '숏' if x['current_qty']<0 else '없음',n(x['realized_price_pnl']),n(x['unrealized_price_pnl']),n(x['fees']),n(x['funding']),n(x['net_component'])] for x in peer['assets']]),
 '<p>관측된 거래에서 각 종목은 진입 방향을 유지하면서 증감했다. 같은 종목을 계속 롱↔숏으로 뒤집어 수익을 냈다는 증거는 없다. 종목별 롱·숏 분산과 부분 청산이 확인되며, BTC 롱·SP500 롱 등 손실 종목도 있다. 내부 신호·모델은 공개 체결만으로 식별할 수 없다. 잔액·체결·포지션은 순차 조회했으므로 마크 변화로 합계와 작은 차이가 생길 수 있다.</p>',
 '<h2>검증과 전달 파일</h2><p>합성 상승/하락의 10종목 진입, 미래 자료 불변성, 기존판 전체 이력 재현, 순서 변경 불변성, 호가 지연 재현·수정, 청산 확정 뒤 숏 진입, 무체결 금지·참여율 제한, 회계 항등식, 감축 이후 재확대 금지까지 19개 테스트를 통과했다. 실제 공개 호가도 10종목 모두 확인했다. 운용 파일 20개 및 선정 모델 묶음 해시는 유지됐다.</p>',
 '<p>구현: <a href="../../all_market_strategy/signals.py">10종목 양방향 신호</a> · <a href="../../pas_strategy_audit/policy.py">PAS 단기 판단·충돌 감축</a> · <a href="../../pas_strategy_audit/execution.py">종목별 호가 갱신 실행자</a> · <a href="../../pas_strategy_audit/replay.py">조건부 체결 재생</a>.</p>',
 '<p>근거: <a href="audit.json">실제 운용·시장·상위 지갑 분석</a> · <a href="comparison.json">모든 비교 결과</a> · <a href="plan.json">사전 비교 규칙</a> · <a href="plan_followup.json">추가 충돌 감축 실험 정의</a> · <a href="target_audit.json">종목별 신호 기여</a> · <a href="tests.log">검증 로그</a> · <a href="candidate_quote_check.json">10종목 공개 호가 확인</a>.</p>',
 '<div class="box"><b>교체 판단:</b> 코드의 종목 범위와 양방향 동작 문제는 수정 후보에서 해결했다. 수익 개선은 아직 확인되지 않았다. 현재 실거래 서비스와 포지션은 교체하지 않았다. 다음 수익성 검증의 핵심은 종목별 추세 방향과 되돌림 충돌을 정리하고, 실제 PAS 실행 가격에서 수익이 남는 진입을 선별하며, 메이커 체결·미체결·취소·청산 비용을 관측 자료로 검증하는 것이다. 과거 호가가 없는데 메이커가 모두 체결됐다고 가정해 성과를 만들지 않는다.</div></html>']
 (OUT/'report.html').write_text(''.join(h))
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('report',OUT/'report.html','files',len(manifest))
if __name__=='__main__':main()
