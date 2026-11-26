"""Report the frozen PAS replay, including unfavorable diagnostics."""
import csv,gzip,html,json
from datetime import datetime,timezone
from .prepare import ROOT,OUT,START,END
from pas_adaptive_development.study import save,sha
from competition_live.scoring import evaluate

SYMBOLS=['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY']
ARMS={'incumbent':'기존 제출판','size_only':'기존 규모 1.441배','budget_lifecycle':'개선 후보','incumbent_2x_diagnostic':'기존 규모 2배 (사후 진단)'}
SCENARIOS={'base_1pct':'관측 거래량 1% (주 비교)','stress_025pct':'0.25% + 비용·펀딩 스트레스','base_10pct':'관측 거래량 10%','optimistic_100pct':'관측 거래량 100% (낙관 가정)'}


def read(name):return json.loads((OUT/name).read_text())


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def sensitivity(rows):
    lookup={(v['coin'],v['hour']):v for v in read('settlement_oracle_proxy.json').values()}
    out=[]
    for row in rows:
        with gzip.open(OUT/f"{row['arm']}_{row['scenario']}.json.gz",'rt') as f:r=json.load(f)
        daily=[r['daily'][0]]
        for t in (START+86400,END):
            s=next(x for x in r['snapshots'] if x['time']==t*1000);adjustment=bound=0.
            for j,q in enumerate(s['qty']):
                if not q:continue
                v=lookup['pas:'+SYMBOLS[j],t//3600]
                assert t*1000<=v['time']<=t*1000+100
                adjustment+=q*(v['price']-s['prices'][j]);bound+=abs(q)*v['rounding_bound']
            daily.append(dict(time=t,equity=s['equity']+adjustment,price_rounding_bound=bound))
        out.append(dict(arm=row['arm'],scenario=row['scenario'],net_pnl=daily[-1]['equity']-100000,
            **evaluate(daily,trading_days=1),daily=daily,
            valuation='Funding-implied oracle, not official mark. Boundary settlements 4/23ms after sample. Only valuation changes; no future cashflow or decision input.'))
    save(OUT/'oracle_valuation_sensitivity.json',out)
    return out


def main():
    rows=read('results.json')+read('diagnostic_results.json');oracle=sensitivity(rows)
    p=read('protocol.json');coverage=read('tape_coverage.json');actual=read('actual_execution_reference.json')
    checks={}
    for source in ('source_freeze.json','execution_freeze.json'):
        checks[source]={name:sha(ROOT/name)==digest for name,digest in read(source).items()}
        assert all(checks[source].values())
    checks['live_original']={name:sha(ROOT/name)==digest for name,digest in p['live_original'].items()}
    assert all(checks['live_original'].values())
    tests=(OUT/'tests.txt').read_text();assert '\nOK\n' in tests
    checks['test_log_sha256']=sha(OUT/'tests.txt')
    checks['actual_trade_minute_boundary_collisions']=sum(x['time']%60000==0 for x in read('trade_tape.json'))
    checks['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    save(OUT/'verification.json',checks)
    fields=['scenario','arm','net_pnl','return_percent','volatility','Q','score','fees','funding','fills','mean_notional','max_drawdown','pnl_after_liquidation_reserve']
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    decision=dict(period_kst=p['interval_kst'],candidate='budget_lifecycle',retrained=False,
        total_capital=100000,own_budget=50000,teammate_cash_reserve=50000,
        candidate_improves_incumbent={'base_1pct':False,'stress_025pct':False,'base_10pct':True,'optimistic_100pct':True},
        prediction_benefit_beyond_scaling_established=False,
        all_pas_scenarios_extra_entries=0,all_pas_scenarios_short_entries=0,pair_gate_nonzero=0,
        replacement_superiority_established=False,
        reason='Gains versus original size require sufficient conditional fills. Candidate does not beat exact2x incumbent diagnostic in PnL or Q. New forecast/carry sleeve makes no entries in this48h interval.',
        score_is_official=False,independent_holdout=False,live_changed=False,orders_sent=0,
        limitations=['No historical L2 or exact mark stream; trade participation is assumed, not identified. Own actual trades are in the tape.',
            'Existing strategy/model references and reference-market forecasts are retained; PAS outcomes never become fit labels. Human rule selection is not erased.',
            'The168h forecast has no matured labels in this48h interval. Two daily returns do not establish stable volatility.'])
    save(OUT/'decision.json',decision)
    comparison=[]
    for scenario in SCENARIOS:
        for arm in ARMS:
            r=next(x for x in rows if x['arm']==arm and x['scenario']==scenario)
            comparison.append([SCENARIOS[scenario],ARMS[arm],f"{r['net_pnl']:+.5f}",f"{r['return_percent']:+.6f}%",f"{100*r['volatility']:.6f}%",f"{r['Q']:.9f}",f"{r['score']:.9f}",f"{100*r['max_drawdown']:.6f}%"])
    details=[]
    for row in rows:
        details.append([SCENARIOS[row['scenario']],ARMS[row['arm']],f"{row['fees']:.5f}",f"{row['funding']:+.5f}",row['fills'],f"{row['mean_notional']:.2f}",f"{row['pnl_after_liquidation_reserve']:+.5f}"])
    ortable=table(['가정','전략','오라클 평가 순손익','오라클 평가 Q'],[[SCENARIOS[r['scenario']],ARMS[r['arm']],f"{r['net_pnl']:+.5f}",f"{r['Q']:.9f}"] for r in oracle if r['scenario'] in ('base_10pct','optimistic_100pct')])
    body='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PAS 48시간 고정 전략 비교</title><style>
    body{font:16px/1.7 system-ui,sans-serif;max-width:1350px;margin:32px auto;padding:0 24px;color:#142b36;background:#f7fafc}h1,h2{line-height:1.3}h2{margin-top:36px}.summary{background:#e8f1fa;border-left:5px solid #2564a1;padding:18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{padding:9px;border:1px solid #d4dfe6;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{background:#e8eef4}code{background:#edf1f5}a{color:#115595}li{margin:8px 0}pre{white-space:pre-wrap;background:#edf1f5;padding:16px}</style>
    <h1>PAS 본대회 첫 48시간: 기존 제출판과 고정 개선 후보</h1>
    <p>한국시간 2026년 10월 19일 21:00–10월 21일 21:00. 신규 후보 <code>budget_lifecycle</code>. 재학습·실제 주문·실전 교체 없음.</p>
    <div class="summary"><b>판정: 체결을 충분히 확보한다고 가정하면 개선 후보가 기존 규모보다 더 벌었다. 그러나 새 모델의 추가 예측 이득은 확인되지 않았다.</b><br>
    사전 지정한 거래량 1% 및 스트레스 조건에서는 성과가 동일하다. 10%·100% 조건에서는 개선 후보의 PnL·Q가 기존판보다 높지만, 기존판의 목표 비중만 정확히 2배로 한 진단판보다 PnL·Q가 조금 낮다. 이번 구간만으로 ‘새 모델 덕분에 교체 우위가 입증됐다’고 판정할 수 없다.</div>
    <h2>동일 조건 비교 결과</h2><p>순손익 단위 USDP. 우리 예산 50,000 + 두 전략에 공통인 팀원 현금 50,000 = 팀 평가자본 100,000. 팀원 실제 ETH 손익은 제외해 우리 전략만 비교했다. ETH를 거래 대상에서 제외한 실험은 아니다. 초기 포지션은 모두 0. Q·변동성은 12:00 UTC 두 일간 수익률을 사용한 모의 값이다.</p>
    '''+table(['체결 가정','전략','순손익','팀 자본 수익률 R','기간 변동성 σ','Q','0.8Q (J=0)','분봉 최대낙폭'],comparison)+'''
    <p>Q = (max(R,0)+0.005) / (|R|+1.5σ+0.05). σ = 표본 일별수익률 표준편차 × √2. 보유분은 마지막 체결가격으로 평가했으므로 공식 계좌가치·공식 순위와 동일하지 않다. σ는 일별 표준편차 자체나 연율 변동성이 아니다.</p>
    <h2>무엇이 결과를 만들었나</h2><ul>
    <li>10개 종목을 입력했지만, 이 48시간에 실제로 모의 체결된 것은 두 전략 모두 BTC·HYPE 롱과 HYPE 청산이다. 개선 후보의 추가 예측 배분 진입 0건, 숏 진입 0건. 예측의 부호가 존재한다는 것과 비용을 넘겨 실제 진입한다는 것은 다르다.</li>
    <li>48개 시간별 신호에서 두 페어의 진입 판단 96개를 당시까지의 입력으로 재생했다. 모두 경제성 필터 0이므로 이 구간에는 페어 진입이 없다. 활성 페어 거래를 임의로 누락한 비교가 아니다.</li>
    <li>100% 거래량 가정에서 평균 보유 명목액은 기존 약 1,721 → 후보 약 3,150 USDP였다. 후보의 이익 증가와 함께 변동성·낙폭도 증가했다. 2배 규모 진단판은 결과를 본 뒤 원인을 분리하기 위해 정확히 한 번 추가했으며 새 후보 선정이나 파라미터 탐색으로 사용하지 않았다.</li>
    <li>후보는 기존 추세·되돌림의 위험 감축을 보존하며 기본 규모를 늘리고, 비어 있는 종목에만 168시간 예측·펀딩 기반 추가 배분을 허용한다. 이 기간에는 그 추가 부분이 주문을 만들지 못했다. PAS에서 다종목 양방향 예측력이 개선됐다는 증거로 제시할 수 없다.</li></ul>
    <h2>자료와 시간 순서</h2><ul>
    <li>49개 공개 지갑 기록에서 중복 제거한 10,426개 거래, 9,391개 종목·밀리초 배치를 복원했다. 10개 종목 모두 각 분봉 거래량과 일치하며 누락된 거래 발생 분봉은 0개다. 거래량 일치가 호가장 전체 복원을 뜻하지는 않는다.</li>
    <li>기존 제출판의 48개 실제 저장 신호와 원래 목표 비중을 대조했다. 신호는 실제 게시 시각보다 빠르게 쓰지 않았고 최소 120초 계산 지연을 적용했다. 판단 이후 거래에서만 체결을 허용했다.</li>
    <li>새 168시간 모델의 계수·정규화·보정은 그대로 사용했다. 학습·보정 정답의 마지막 성숙 시점은 10월 19일 이전이다. 기존 페어의 자동 일별 갱신 규칙만 그 시점까지 성숙한 자료로 재생했다. PAS의 19–21일 결과는 수익·체결·펀딩 평가에만 사용했다.</li>
    <li>후보 입력은 기존 외부 기준시장 시간봉과 지연된 기준시장 펀딩이다. 실현 거래·수수료·펀딩은 PAS를 사용했다. 이 후보를 PAS 펀딩으로 새로 학습했다고 설명하지 않는다.</li>
    <li>최종 위험 감축 비중의 저장 자료가 끝나는 시점 이후는 동일 공식으로 연장 계산했고, 기존 저장 구간 전체와 수치가 일치함을 확인했다. 학습이나 규칙 변경은 하지 않았다.</li>
    <li>미래 거래·가격·펀딩 값을 바꿔도 그 이전 판단·손익이 같은지 검사했다. 입력을 각 판단 시각에서 잘라 계산한 예측도 전체 자료의 해당 시각 예측과 일치했다. 원본 모델·실전 코드 해시는 그대로다.</li>
    <li>이 기간은 이전 개발에서 이미 살펴본 구간이다. 미래 자료를 코드 입력으로 넣지 않은 것과 연구자가 완전히 안 본 독립 테스트라는 주장은 별개다. 이번 결과는 역사적 개발 비교이며 독립 검증으로 표시하지 않는다.</li></ul>
    <h2>체결 가정과 실제 체결 대조</h2><p>판단 후 15분 안에 발생한 실제 거래 배치의 0.25%·1%·10%·100%까지만 수량을 배정했다. 매수는 배치 최고가, 매도는 최저가에 불리한 비용을 더했다. 기본 편도 수수료 9bp + 추가 가격 비용 10bp; 스트레스 12bp + 20bp, 수취 펀딩 50%만 인정. 주문 한도는 판단 시점 체결가격의 ±5%이다.</p>
    <p>이는 관측 거래를 이용한 조건부 시장가 체결 추정이다. 실제 실행자의 현재 호가 기반 IOC·10bp 제한과 같은 체결을 보장하지 않는다. 당시 L2 호가·대기열·다른 주문의 반응은 복원하지 못했다. 기록에는 우리의 원래 주문도 포함되어 있어 반사실 주문이 같은 유동성을 얻는다는 보장이 없다. 100%는 낙관 가정이며 ‘가능한 최고 수익’이라는 뜻이 아니다.</p>
    <p>최소 주문 10달러는 제출 요청에 적용하고 일부 체결 조각에는 강제하지 않았다. 과거 거래량이 0이었다는 이유로 미래 체결을 차단하지 않았다. 거래가 적은 시간에는 평가가격의 경과시간도 기록했다. 실제 펀딩을 포지션 수량으로 나누어 지급액을 복원했으며, 이번 보유 종목에 추정 펀딩 대체를 사용한 횟수는 0이다.</p>
    '''+f'''<p>기존 실제 우리 주문(팀원 주문 제외)의 수량·체결가·수수료·펀딩을 같은 마지막 체결가격으로 평가하면 <b>{actual['net_pnl']:+.5f} USDP</b>다. 100% 가정 기존판은 +27.67461이다. 둘의 차이는 실제 주문 수량·체결가격과 모의 가격 비용 등의 차이다. 이 실제 체결 기반 +30.11도 공식 마크가격 손익은 아니다. 최초 실제 BTC 체결은 10월 19일 21:02:03.714 KST이며 모의 체결도 그 거래 배치에서 시작한다.</p>'''+table(['체결 가정','전략','지급 수수료','순수취 펀딩','체결 배치 수','평균 보유 명목액','잔여 청산비용 적립 후 PnL'],details)+'''
    <p>잔여 청산비용은 종료 포지션 명목액 × (수수료+가격비용)으로 별도 적립한 수치다. 실제 청산 주문이 전량 체결됐다는 뜻은 아니다. ‘체결 수’는 거래 배치별 부분 체결 수이며 독립 거래 기회 수가 아니다.</p>
    <h2>평가가격 민감도</h2><p>종료 시 BTC 마지막 체결가격 83,422는 31분 경과한 값이다. 공식 마크가격 전체 자료가 없어, 추가 확인으로 펀딩 정산액에서 역산한 오라클가격으로 두 일별 평가만 바꿨다. 펀딩액 = 포지션 수량 × 오라클가격 × 펀딩률이라는 <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding">공식 펀딩 문서</a>의 관계를 이용했다. 이 오라클 역시 마크가격은 아니다. 정산 기록은 샘플 경계 4·23ms 뒤 관측값이며 거래 판단·수량·현금에는 사용하지 않았다. 결론은 동일하다.</p>
    '''+ortable+'''
    <h2>이번 테스트가 답하는 범위</h2><p><b>재학습 없이도 같은 고정 전략으로 과거 시점별 판단과 PAS 손익을 재생할 수 있다.</b> 이번 48시간에서 기존 규모 대비 조건부 이익 증가는 확인했지만, 새 모델의 추가 기여와 일관된 교체 우위는 확인하지 못했다. 168시간 뒤 가격 예측은 19일 예측의 정답이 26일에 완성되므로 21일까지만으로 그 예측 정확도를 채점할 수 없다. 48시간 운용 성과는 지금 계산할 수 있어 위에 제시했다.</p>
    <p>보유가격의 상승을 더 큰 규모로 누린 것과 더 좋은 진입·청산·종목 선택을 구분해야 한다. 이 결과로 실전 전략을 교체하거나 학습을 반복하지 않았다.</p>
    <h2>재현 자료</h2><ul>
    <li><a href="comparison.csv">전체 비교 CSV</a> · <a href="results.json">주 비교 원시 지표</a> · <a href="diagnostic_results.json">2배 규모 진단</a> · <a href="decision.json">판정 JSON</a></li>
    <li><a href="protocol.json">최초 실험 계획</a> · <a href="tape_protocol.json">첫 재생 전에 확정한 실제 거래 배치 체결 계획</a> (최초 계획의 분봉 체결 설명을 대체)</li>
    <li><a href="tape_coverage.json">종목별 거래량 대조</a> · <a href="pair_forecasts.json">48회 페어 예측</a> · <a href="component_attribution.json">후보 구성 요소 기여</a> · <a href="actual_execution_reference.json">실제 체결 대조</a></li>
    <li><a href="verification.json">해시·검증 결과</a> · <a href="tests.txt">테스트 기록</a> · <a href="oracle_valuation_sensitivity.json">오라클 평가 민감도</a></li>
    </ul><pre>코드: pas_counterfactual_20261022/
테스트: .venv/bin/python -m unittest pas_counterfactual_20261022.test_replay -v
전체 재현 확인: .venv/bin/python -m pas_counterfactual_20261022.reproduce
보고서 재생성: .venv/bin/python -m pas_counterfactual_20261022.report</pre></html>'''
    (OUT/'report.html').write_text(body)
    save(OUT/'report_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in [*(ROOT/'pas_counterfactual_20261022').glob('*.py'),OUT/'report.html',OUT/'comparison.csv',OUT/'decision.json']})
    print(json.dumps(decision,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
