"""Evidence report; preserves failures and distinguishes conditional simulations."""
import csv,html,json,math
from datetime import datetime,timezone
from pathlib import Path
from .study import ROOT,OUT,read,save,sha,verify

LABELS={'incumbent':'기존 신호·15분 주문','incumbent_ttl1h':'기존 신호·60분 주문+진입가격 고정',
 'carry_adverse_guard':'PAS 펀딩·역방향 가격 위험 필터','carry_only':'PAS 펀딩 단독',
 'venue_momentum':'PAS 다중 시간대 추세','peer_consensus':'46개 지갑 동일비중 방향',
 'peer_leaders':'전일 가격손익 기준 선두 집단 방향'}

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def main():
    verify();rows=[]
    for folder in ['carry_allocation','strict_ttl','venue_momentum','strict_peers']:
        for r in read(OUT/folder/'results.json'):rows.append(dict(experiment=folder,**r))
    by={(r['arm'],r['scenario'],r['phase']):r for r in rows}
    base=lambda s:by['incumbent',s,'continuous120h']
    candidate=lambda s:by['incumbent_ttl1h',s,'continuous120h']
    switch=lambda s:by['incumbent_ttl1h',s,'same_inventory_switch']
    gates=dict(positive_full_pnl_and_Q=all(candidate(s)['net_pnl']>base(s)['net_pnl'] and candidate(s)['Q']>base(s)['Q'] for s in ['base','stress','no_receipts']),
        same_inventory_pnl_improvement=all(switch(s)['net_pnl']>base(s)['net_pnl'] for s in ['base','stress','no_receipts']),
        exact_historical_live_execution=False,independent_test=False,live_promotion=False)
    missing_funding=read(OUT/'peer_missing_funding_audit.json')
    assert all(r['rate']==0. for r in missing_funding)
    for r in rows:
        assert r['timing_pass'] and r['maximum_accounting_error']<1e-6
        assert r['risk_breach_events']==0
        if r['funding_proxy_events']:
            assert r['experiment']=='strict_peers' and r['scenario']=='base' and r['phase']=='continuous120h'
            assert r['funding_proxy_events']==sum(x['arm']==r['arm'] for x in missing_funding)
    source=list((ROOT/'execution_revision_20261024').glob('*.py'))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in source}
    save(OUT/'release/PROMOTION_BLOCKED.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        candidate='one_hour_fixed_entry_guard_v1',gates=gates,
        reasons=['Same-inventory switch did not outperform in any cost scenario.',
        'Minute tape cannot reconstruct historical refreshed marks, bid/ask or order queue. Earlier5%guardresults withdrawn as promotion evidence.',
        'Exposedhistory cannot be relabelled independent after repeated candidate evaluation.'],
        implementation='execution_revision_20261024.guarded_portfolio.adopt',source_sha256=manifest,
        invalid_preview='Earlier release/adopted_coordinator.pkl and candidate_execution_policy.json are stale previews, not activation artifacts.',
        live_changed=False,orders_sent=0,signer_loaded=False))
    save(OUT/'final_validation.json',dict(gates=gates,strict_comparison_runs=len(rows),
        correct_timing_and_accounting=True,hard_risk_breaches=0,nonzero_funding_proxy_while_held=0,
        zero_rate_fallback_counter_events=len(missing_funding),
        asof_forecast_resolution=read(OUT/'forecast_asof_resolution.json'),
        tests='41 targeted tests passed, including inherited production risk/handoff paths; see tests.txt.',
        source_sha256=manifest,live_changed=False,orders_sent=0))
    fields=['experiment','arm','scenario','phase','net_pnl','return_percent','volatility','Q','score','max_drawdown','mean_notional','fees','funding','tail_pnl','tail_liquidation_adjusted_pnl','risk_breach_events','maximum_accounting_error','funding_proxy_events','final_valuation_age_seconds']
    with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    comparison=[]
    for arm in LABELS:
        a=by[arm,'base','continuous120h'];b=by[arm,'stress','continuous120h']
        comparison.append([LABELS[arm],f"{a['net_pnl']:+.4f}",f"{b['net_pnl']:+.4f}",f"{a['Q']:.6f}",f"{a['volatility']*100:.6f}%",f"{a['mean_notional']:,.2f}"])
    transitions=[]
    for s,label in [('base','기본 비용'),('stress','불리한 비용·수취 펀딩 절반'),('no_receipts','불리한 비용·수취 펀딩 0')]:
        a,b=base(s),switch(s)
        transitions.append([label,f"{a['tail_pnl']:+.6f}",f"{b['tail_pnl']:+.6f}",f"{b['tail_pnl']-a['tail_pnl']:+.6f}",f"{a['tail_liquidation_adjusted_pnl']:+.6f}",f"{b['tail_liquidation_adjusted_pnl']:+.6f}"])
    metrics=[]
    for key,label,mult in [('return_percent','100,000 기준 수익률(%)',1),('volatility','5개 일별수익률의 기간 변동성 σ(%)',100),('Q','금융점수 Q',1),('score','J=0 가정 점수 0.8Q',1),('max_drawdown','최대낙폭(%)',100),('mean_notional','평균 포지션 명목금액(USDP)',1),('fees','체결 수수료(USDP)',1),('funding','순펀딩(USDP)',1),('pnl_after_liquidation_reserve','청산비용 충당 후 PnL(USDP)',1)]:
        metrics.append([label,f'{base("base")[key]*mult:.6f}',f'{candidate("base")[key]*mult:.6f}'])
    riskratio=candidate('base')['mean_notional']/base('base')['mean_notional']
    peers=[
      ['0xdd0da0…f75f84','+2,401.69','+1,828.24','−69.48','74.56%','0.242배','BRENT 롱·다른 9종목 숏. 방향 거래와 지정가 체결이 관측됨'],
      ['0x06a336…409a11','+1,737.35','+1,460.84','−177.42','97.78%','0.310배','BTC·ETH·HYPE·EWY·GOLD 숏. 잦은 주문, 대부분 메이커'],
      ['0x7df82d…680d5b','+1,543.51','−108.63','+3,100.62','0%','5.462배','가격손익과 별개로 큰 펀딩 수취. 고노출 캐리 가능성'],
      ['0x1ba2ee…81ca4','+533.60','+37.50','+1,950.00','5%','6.902배','SP500 롱·JPY 숏 등 펀딩 수취가 수익을 지지'],
      ['SFINX 전체 지갑','+231.79','+219.20','−49.14','0%','0.131배','우리 전략과 팀원 전략의 합계. 우리 성과와 혼동 금지'],
    ]
    execution=read(OUT/'live_execution_audit.json');asof=read(OUT/'forecast_asof_resolution.json')
    body=f'''<h1>SFINX 전략 개선 실험과 교체 판정</h1>
<p class="meta">작성: {datetime.now(timezone.utc).astimezone(__import__('zoneinfo').ZoneInfo('Asia/Seoul')).isoformat()} · PAS / USDP · 실제 주문 0건 · 운용 전략 변경 없음</p>
<div class="callout"><strong>조건부 백테스트 PnL 개선 후보는 확보했지만, 교체 검증을 통과한 후보는 없다.</strong>
<p>같은 진입가격 제한에서 60분 실행 후보의 전체 PnL은 +39.1875 → +74.3439 USDP, Q는 0.106021 → 0.111069였다. 그러나 동일한 기존 보유분을 인계한 21시간 실험은 +13.5602 → +12.6874 USDP로 낮아졌다. 이 결과와 실제 호가 이력 부족을 숨기고 실전 개선 확정이라고 보고하지 않는다.</p></div>
<h2>평가 범위와 분모</h2>
<p>전체: 한국시간 2026-10-19 21:00 → 10-24 21:00(120시간). 인계: 10-24 00:00 → 21:00(21시간). 모든 후보는 우리 몫 50,000 USDP와 움직이지 않는 팀원 몫 50,000으로 평가했다. 실제 팀원의 수익은 섞지 않았다. 표의 수익률 분모는 100,000이며, 우리 몫 기준 수익률은 두 배다. 모든 숫자는 조건부 모의거래이며 실제 계좌 손익이나 공식 순위가 아니다.</p>
<p>일별 표본은 12:00 UTC(한국시간 21:00), 완료 수익률 n=5. σ=표본표준편차×√5, Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05). 따라서 σ를 일별 표준편차 자체로 부르지 않는다. 일별 표준편차는 σ/√5다. J=0 가정 점수는 0.8Q다.</p>
<h2>같은 체결 가정의 전체 비교</h2>
{table(['후보','기본 PnL(USDP)','불리한 조건 PnL','기본 Q','기본 기간 σ','평균 명목금액'],comparison)}
<p>펀딩 후보가 0인 것은 수익을 확인했다는 뜻이 아니다. 비용 문턱을 넘는 진입이 없어 거래하지 않았다. 자격 미달 후보의 계산상 Q≈0.1은 공식 순위 점수가 아니다.</p>
<h2>기존 보유분을 그대로 넘긴 교체 실험</h2>
{table(['조건','기존 21h PnL','교체 21h PnL','교체−기존','기존 청산비용 반영','교체 청산비용 반영'],transitions)}
<p>교체 이전 체결·잔고·비용·일별 표본이 완전히 같은지 검사했다. 처음부터 개선판을 쓴 경로의 좋은 포지션을 교체 시점에 공짜로 인계하지 않았다. 5% 가격 범위의 이전 탐색에서도 일별 네 인계 구간 중 개선은 1개뿐이었다. 작은 음수 차이를 통계적으로 유의한 악화라고 단정하지는 않지만, 사전에 정한 ‘같은 인계 조건에서도 개선’ 기준에는 통과하지 못했다.</p>
<h2>가장 나았던 실행 후보</h2>
<p>신호·학습모델·50,000 자본·6시간 판단 주기·총 위험 한도를 유지한다. 미체결 진입 유효시간을 15분에서 60분으로 늘린다. 다만 결정 시점 가격의 ±10bp를 고정해, 재시도 때 가격이 바뀌었다는 이유로 더 불리한 진입을 허용하지 않는다. 1시간 이후에는 실제 남은 우리 수량의 감축만 유지하고, 6시간 신호 만료나 다음 판단 때 교체한다. 숏 전환은 기존 롱 청산을 확인한 뒤 수행한다.</p>
{table(['항목','기존 15분','개선 후보 60분'],metrics)}
<p>평균 보유 명목금액은 약 {riskratio:.2f}배다. PnL 개선의 상당 부분은 더 많이 체결해 더 오래 보유한 효과다. 예측력 향상을 증명한 새 모델이라고 부를 수 없다. 변동성과 최대낙폭도 증가했다. 단순히 ‘PnL이 높고 변동성이 더 낮아졌다’고 주장하면 틀리다.</p>
<h2>실제 실행과 백테스트의 차이 발견</h2>
<p>실행 코드의 기본 신규진입 한도는 시도 시점 mark 가격 ±10bp다. 이전 분봉 실험은 결정가격 ±5%를 허용했다. 동일한 기준이 아니므로 이전 +106.9212 USDP 등의 수치를 실전 교체 근거에서 제외했다. 최종 표는 모든 후보에 결정가격 ±10bp를 동일하게 적용한 재실험이다.</p>
<p>또한 실제 기존 실행자는 매 시도마다 mark를 새로 기준으로 삼는다. 최종 연구용 엔진의 고정 결정가격 제한과도 완전히 같지 않다. 준비한 guarded_portfolio는 이 고정 가격 계약을 코드에 명시했지만, 기존 실행자의 역사적 mark·호가·대기열을 복원한 것은 아니다. 이전 unsigned gateway preflight는 이 상위 가격 필터를 거치지 않았으므로 전체 실행 통과 증거로 취급하지 않았다.</p>
<p>실제 저장 로그에서 가격 제한 차단 {len(execution['economic_limit_events'])}건을 확인했다. 이 중 10월 24일 15시 라운드 SP500 숏 차단이 7건이다. 당시 호가가 저장되어 있지 않아 이것이 잘못된 차단인지, 정상적인 비용 보호인지는 확정할 수 없다. 가격 제한을 무조건 넓히는 수정은 하지 않았다.</p>
<h2>왜 PnL이 작았는가</h2>
<ol><li>현재는 BTC·ETH 전용/롱 전용이 아니다. 우리 장부는 BTC·BRENT 롱, ETH·HYPE·EWY·EWJ·GOLD·USDEUR·USDJPY 숏이며 SP500은 0이다. 현재 미진입 이유와 종목 미평가를 구분해야 한다.</li>
<li>고정된 외부시장 추세·7일 되돌림에 Chronos와 국면 위험 보정이 적용되고, 실제 PAS 진입에는 3h 또는 24h 되돌림 조건이 추가된다. 당시 8개 종목에 위험 감축 표시가 있어 목표를 늘리기 어렵다.</li>
<li>추가 배분 모델은 기존 방향 목표가 0이고 위험 보호 대상이 아닌 슬롯을 주로 사용한다. 약한 예측과 비용 문턱 때문에 ‘한도가 남음’이 자동 투자로 이어지지 않는다. 작은 포지션을 무조건 채우면 위험 감축을 다시 확대할 수 있다.</li>
<li>6시간 판단·15분 실행창, 얇은 체결량, 최소주문금액, 가격 제한과 API 지연이 실제 수량을 줄인다. 반대로 주문창만 늘렸을 때의 추가 거래가 항상 돈을 벌지는 않았다.</li>
<li>보존된 페어 전문 모듈은 BTC/ETH·BTC/HYPE 두 관계이며 평가 기간의 진입 게이트는 모두 0이었다. 이번 개선을 45개 페어의 예측 개선이라고 포장하지 않는다.</li>
<li>EWJ 기준시장 자료의 한 시간 누락이 전체 신호 갱신을 막은 기록이 있다. 실제 미발행 시간은 재현에서도 건너뛰었다. 누락 봉을 지어내지 않았고, 뒤늦게 API 자료가 복구되며 실제 갱신도 회복했다.</li></ol>
<h2>타 지갑의 관측과 검증한 가설</h2>
<p>아래는 10월 24일 21:10~21:13 한국시간 공개 자료다. 현재 순위로 오해하지 말 것. 공식 실시간 차트에서 SFINX는 당시 6위였지만 지갑별 팀명 매핑은 공개되지 않아 비슷한 PnL로 타 지갑의 팀명을 단정하지 않았다. ‘예상 전략’은 보유·체결·펀딩에서의 추론이다.</p>
{table(['지갑','전체 PnL','실현 가격손익','펀딩','메이커 비중','실효 총노출/자산','관측에 따른 해석'],peers)}
<p>전체 PnL에는 미실현손익도 포함된다. 표시한 레버리지는 설정 레버리지가 아니라 gross notional/equity다. 설정 20배와 실제 20배 노출은 다르다. 우리 팀 전체 PnL +231.79 중 당시 우리 몫은 +69.07, 팀원 몫 +162.72였다. 우리 보유 명목금액 9,981.62는 우리 50,000의 약 19.96%다.</p>
<p>확보한 초기 지급 지갑에서 우리를 제외한 46개, PAS 체결 28,681건을 모았다. 현재 승자만 고르지 않고, 모든 지갑의 동일비중 방향과 당시 전일 가격손익(수수료 차감, 펀딩 제외)으로 선택한 양수 상위 25% 집단을 각각 재현했다. 관측에 120초 지연을 주고 같은 자본·위험·체결 제한을 적용했으나 두 방식 모두 개선하지 못했다. 타 지갑의 좋은 현재 성과만 보고 과거부터 따라샀다고 가정하지 않았다. 동일 밀리초 체결 순서는 startPosition→endPosition 연결로 검증했다.</p>
<h2>검증과 남은 한계</h2>
<ul><li>119개 실제 발행 시각과 원본 입력 접두 구간을 확인했다. 미발행 10월 24일 16시 자료는 소급 발행하지 않았다.</li>
<li>저장 당시 예측과 최신 일괄 추론의 소수점 차이는 최대 {asof['maximum_numeric_forecast_difference']:.10g}. 당시 예측으로 치환했을 때 모든 목표 비중·위험 보호 마스크가 정확히 같았다. 미래 정답으로 재학습하지 않았다.</li>
<li>최종 동일 조건 비교 {len(rows)}회 모두 시각 순서·장부 보존 검사 통과, 강제 위험 한도 초과 0. 비영 펀딩은 실제 공개 지급액에서 복원했다. 지갑 추종 두 기본 시나리오에는 대체 계산 카운터가 각각 10·5회 있지만, 전부 공식 펀딩률 0인 USDJPY 관측임을 따로 확인했다. 어떤 가격을 곱해도 지급액이 0이므로 손익 영향은 없다.</li>
<li>41개 표적 검사 통과. 실제 실행 경로에서 가격 제한, 감소 우선, 중복 방지, 미확정 체결 차단, 기존 보유 인계, 결정 시점 가격 고정, 미래 가격을 바꿔도 과거 신호가 바뀌지 않는지 검사했다.</li>
<li>기본 비용: 편도 9bp 수수료+10bp 불리한 체결, 관측 거래량 1%. 불리한 조건: 12bp+20bp, 거래량 0.25%, 수취 펀딩 절반. 추가로 수취 펀딩 0도 시험했다. 대기열·미래 호가 수량을 안다고 가정하지 않는다.</li>
<li>분봉 고가 매수·저가 매도+불리한 체결을 사용하고, 결정 뒤 완전히 끝난 다음 봉부터만 체결 가능하다. 이는 조건부 보수 모델이지 실제 호가 재현은 아니다. 최종 평가가격에도 오래 거래되지 않은 종목이 있어 공식 mark 기반 점수와 다를 수 있다.</li>
<li>119시간 신호를 갖고 여러 후보를 비교한 기록 자체가 선택 편향을 만든다. 미사용 테스트라는 이름을 붙이지 않았다. 21시간은 새 완료 일별수익률 두 개를 만들지 못하므로 독립적인 새 기간 변동성·Q를 주장하지 않는다.</li></ul>
<h2>코드와 재현 자료</h2>
<p>연구 코드는 프로젝트의 execution_revision_20261024/, 원본·결과는 reports/execution_revision_20261024/에 있다. 실제 운용 candidate_handoff 및 실행 설정은 변경하지 않았다.</p>
<ul><li><a href="comparison.csv">최종 동일조건 비교 CSV</a> · <a href="final_validation.json">검증 결과</a> · <a href="tests.txt">테스트 출력</a></li>
<li><a href="strict_ttl/results.json">60분 후보 상세</a> · <a href="strict_peers/results.json">지갑 추종 상세</a> · <a href="venue_momentum/results.json">PAS 추세 상세</a> · <a href="carry_allocation/results.json">펀딩 상세</a></li>
<li><a href="release/PROMOTION_BLOCKED.json">교체 보류 근거와 최종 코드 해시</a> · <a href="live_execution_audit.json">실제 실행 로그 집계</a> · <a href="forecast_asof_resolution.json">발행 당시 예측 대조</a></li>
<li>실행 예: .venv/bin/python -m unittest execution_revision_20261024.test_guarded_portfolio execution_revision_20261024.test_validation execution_revision_20261024.test_portfolio execution_revision_20261024.test_cashflow_overlay execution_revision_20261024.test_peer_prepare -q</li></ul>
<p>실험은 원본·결과 덮어쓰기를 막도록 설계했다. 완료한 run 명령을 그대로 재실행하면 중단하는 것이 정상이다. 복제된 연구 디렉터리와 별도 출력 경로에서 재현해야 한다. 오래된 인계 checkpoint를 실전에 복원하면 안 된다. 인계는 항상 당시 최신 장부로 다시 검증해야 한다.</p>
<p>출처: <a href="https://perpetual-alpha26.github.io/leaderboard">대회 리더보드와 점수</a>, <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding">Hyperliquid 펀딩 규정</a>, 로컬 공개 API 응답 원본과 실제 실행 로그. 펀딩은 원칙상 포지션 수량×오라클 가격×펀딩률이며, 표시 mark와 혼동하지 않았다.</p>'''
    css='body{font:16px/1.65 system-ui,sans-serif;max-width:1300px;margin:40px auto;padding:0 24px;color:#163047;background:#f7fafc}h1{font-size:30px}h2{margin-top:36px}p,li{max-width:1100px}.meta{color:#57717e}.callout{background:#fff5df;border-left:5px solid #b98212;padding:18px}.scroll{overflow:auto}table{border-collapse:collapse;min-width:760px;width:100%;background:white}th,td{text-align:left;border:1px solid #d9e4ea;padding:9px}th{background:#eaf2f5}a{color:#006997}'
    (OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>SFINX 전략 개선 검증</title><style>'+css+'</style><body>'+body+'</body></html>',encoding='utf-8')
    print(dict(rows=len(rows),gates=gates,report=str(OUT/'report.html')))

if __name__=='__main__':main()
