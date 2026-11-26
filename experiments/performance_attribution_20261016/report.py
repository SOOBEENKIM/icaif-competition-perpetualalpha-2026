"""Publish all attempts, actual attribution, comparison limitations and checks."""
import html,json,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from experiments.risk_preserving_20261016.prepare import sha,save
from .collect import ROOT,OUT

def read(p):return json.loads(p.read_text())
def table(headers,rows):
    esc=lambda s:html.escape(str(s))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(s)+'</th>' for s in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(s)+'</td>' for s in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def main():
    test=subprocess.run([sys.executable,'-m','unittest','experiments.performance_attribution_20261016.test_model',
        'experiments.risk_preserving_20261016.test_candidate','practice_live.test_live','-v'],cwd=ROOT,capture_output=True,text=True)
    text=test.stdout+test.stderr;(OUT/'tests.txt').write_text(text)
    if test.returncode:raise ValueError(text)
    tests=int(re.search(r'Ran (\d+) tests',text).group(1))
    previous=ROOT/'experiments/risk_preserving_20261016/artifacts'
    protected=read(previous/'protected_before.json')
    for path,digest in protected.items():
        if sha(ROOT/path)!=digest:raise ValueError('Incumbent changed: '+path)
    reserved=read(ROOT/'experiments/execution_retry_20261016/forward_reservation.json')['plan']['artifacts']
    for path,digest in reserved.items():
        if sha(path)!=digest:raise ValueError('Reserved trial dependency changed: '+path)
    budget=read(OUT/'budget_results.json');economic=read(OUT/'research_results.json');actual=read(OUT/'live_attribution.json');verified=read(OUT/'agent_verification.json')
    assert verified['compared_weights_exact'] and verified['restart_exact']
    status=read(ROOT/'practice_live/live_runtime/status.json')
    live={k:status[k] for k in ['updated_at_utc','state','mode','profile','pid','next_normal_decision_kst','halted']}
    labels={'incumbent':'기존판','original_risk_cap':'앞선 위험감축판','risk_cap_budget_matched_v1':'위험감축＋고정 노출 보정판',
            'trend_only_same_scale':'되돌림을 뺀 추세 단독','calibrated_entry_v1':'예측 수익으로 추가 진입 제한'}
    def row(r):
        m=r['metrics']
        return [labels[r['policy']],f"{m['roi']*100:+.6f}%",f"{m['daily_std_sample']*100:.6f}%",f"{m['max_drawdown_magnitude']*100:.6f}%",f"{m['mean_gross']*100:.4f}%",m['fills']]
    headers=['버전','순수익률','일별 STD','최대낙폭','평균 총노출','체결 수']
    body='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>성과 원인과 추가 개선 비교</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1300px;margin:30px auto;padding:0 24px;background:#f3f6fa;color:#18243a}section{background:#fff;padding:24px;border-radius:12px;margin:18px 0}h1,h2{line-height:1.35}th,td{border-bottom:1px solid #dce3ec;padding:10px;text-align:right}th:first-child,td:first-child{text-align:left}table{width:100%;border-collapse:collapse;font-size:14px}th{background:#eef3fa}.scroll{overflow:auto}.decision{border-left:6px solid #377d60}a{color:#245eaf}code{background:#edf2f8;padding:2px 5px}</style>
<h1>성과 원인 분석과 추가 개발 비교 — 2026-10-16</h1>
<section class="decision"><h2>개발 비교에서 수익·변동성·낙폭 동시 개선 후보 확인</h2><p>기존판의 위험 신호가 발생한 종목은 줄이되, 전체 투자 규모는 미리 계산된 8월 노출 차이를 이용해 보정했습니다. 9월 1일~10월 13일 기본 비용 조건에서는 기존판보다 수익이 높고 일별 변동성과 최대낙폭이 낮습니다.</p><p><strong>새 설계는 앞선 결과를 본 뒤 추가했습니다. 같은 과거 자료를 재사용한 개발 결과이며, 독립적인 검증이나 실제 par 수익 개선 입증이 아닙니다. 현재 실거래 전략은 변경하지 않았습니다.</strong></p></section>
<section><h2>앞선 개발에서 부족했던 점</h2><ul><li>위험 감소를 구현하고 기능 검사를 통과한 것과, 사용자 목표인 거래 성과 개선을 충분히 구분하지 못했습니다.</li><li>감축판의 규모가 줄어든 상태에서 비교를 끝냈습니다. 기존판 규모를 낮춘 비교는 있었지만 감축판의 투자 규모를 동일 기준으로 보정한 비교가 없었습니다.</li><li>우리 실제 손익에서 첫 진입과 늦은 추가 매수의 기여를 먼저 분리했어야 했습니다.</li><li>다른 지갑의 누적수익과 특정 기간 변동성을 함께 제시할 때, 거래 기간·현금 보유·자료 시차 차이를 더 명확히 했어야 했습니다.</li></ul></section>
<section><h2>최종 후보의 변경 규칙</h2>'''
    body+=f'''<p>고정 배율은 <strong>{budget['multiplier']:.9f}</strong>입니다. 앞선 실험에서 8월의 목표 총노출만으로 구한 0.771683431의 역수이며, 수익이 가장 좋아지는 배수를 검색하지 않았습니다. 감축 전 배분에 이 배율을 적용한 뒤, 기존 방향별 유지 비율 0.5 / 0.75 / 1을 적용합니다.</p><p>따라서 25% 감축 종목은 기존 비중의 약 97.19%, 50% 감축 종목은 약 64.79%, 감축하지 않는 종목은 약 129.59%가 됩니다. 종목별 상한과 전체 포트폴리오 위험 한도, 주문 수량 상한을 마지막에 다시 적용합니다. 실제 실현 변동성을 정확히 맞춘다는 뜻은 아닙니다.</p></section>'''
    body+='''<section><h2>동일 기간·비용의 비교</h2><p>2026-09-01 00:00 UTC → 2026-10-13 12:00 UTC. 초기 자본 각 1,000,000, 시작 시 모두 미보유. UTC 일별 수익률 42개의 표본 표준편차(ddof=1), 비연율화. 기준시장 시간봉의 다음 종가 체결 대용치와 수량 제한을 사용했습니다.</p>'''
    for scenario in ['base','stress']:
        body+='<h3>'+('기본 비용' if scenario=='base' else '불리한 비용')+'</h3>'
        body+=table(headers,[row(r) for r in budget['rows'] if r['phase']=='historical_evaluation' and r['scenario']==scenario])
    body+='''<p>공통 편도 수수료 9bp. 기본: 가격 충격 10bp, 시간봉 거래량 참여 1%. 불리한 조건: 가격 충격 20bp, 참여 0.25%, 시간당 총노출에 불리한 펀딩 1bp 추가. 실제 정산된 기준시장 펀딩을 사용했으며, 과거 par 호가·펀딩의 정확한 재현은 아닙니다. 종료 포지션은 평가하고 가상의 청산 체결을 만들지 않았습니다.</p></section>'''
    body+='<section><h2>불리한 구간도 함께 공개</h2>'
    period={'august':'8월 (배율 산출 기간)','first_half':'9/1–9/22','second_half':'9/22–10/13'}
    body+=table(['기간','비용',*headers],[ [period[r['phase']],r['scenario'],*row(r)] for r in budget['rows'] if r['phase']!='historical_evaluation' and r['policy']!='original_risk_cap'])
    body+='''<p>8월과 9월 전반에는 기존판보다 수익이 낮습니다. 8월의 불리한 비용 조건에서는 최대낙폭도 더 큽니다. 9월 후반에는 손실이 줄지만 손실 자체가 사라지지 않았습니다. 모든 구간에서 우월한 후보라는 결론은 내리지 않습니다.</p></section>
<section><h2>폐기한 변경과 예측력</h2>'''
    body+=table(headers,[row(r) for r in economic['rows'] if r['phase']=='historical_evaluation' and r['scenario']=='base'])
    q=next(r for r in economic['forecast_quality'] if r['phase']=='historical_evaluation')
    body+=f'''<p>되돌림을 없애면 전체 수익이 낮아졌습니다. 7월까지의 완결된 자료로 보정한 24시간 예측 수익을 비용과 비교하는 후보는 9월 이후 거래를 전부 막았습니다. 이를 변동성 0의 성능 개선으로 채택하지 않았습니다.</p><p>같은 평가 구간의 42일 × 암호화폐 3종목에서 원래 Chronos 신호와 다음 체결 가능 시점부터 24시간 수익률의 상관은 {q['raw_chronos']['correlation']:.3f}, 보정 후에는 {q['calibrated']['correlation']:.3f}였습니다. 두 예측의 MSE는 모두 0% 수익 예측 기준보다 컸습니다. 약한 예측력을 필터나 새 모델 이름만으로 해결했다고 할 수 없습니다. 자산별 관측은 서로 독립이 아니며 이 비교도 재사용 자료입니다.</p></section>'''
    body+='<section><h2>실제 거래에서 확인된 진입 시점 문제</h2>'
    batches=actual['user_batches']
    body+=table(['종목','진입','수량','체결 평균가','15:30 oracle 가격손익','17:04 oracle 가격손익','수수료'],
       [[b['coin'],'첫 진입' if i<2 else '6시간 뒤 추가',f"{b['qty']:.4f}",f"{b['vwap']:.5f}",f"{b['oracle_price_pnl_at_1530']:+.4f}",f"{b['oracle_price_pnl']:+.4f}",f"{b['fees']:.4f}"] for i,b in enumerate(batches)])
    body+='''<p>15:30에는 첫 소량 진입의 가격손익이 +29.44356, 나중에 산 물량은 −178.83654였습니다. 첫 가격에서 큰 수량을 체결할 수 있었다는 증거는 없으므로, 전량을 일찍 샀다면 얻었을 가상의 수익으로 보고하지 않았습니다. 이후 가격 반등도 포함해 17:04 수치를 함께 제시합니다. 위 가격손익은 수수료·펀딩을 제외한 항목입니다.</p></section>'''
    body+='<section><h2>다른 지갑: 같은 시각 구간으로 다시 비교</h2><p>10월 16일 KST 11:11:54 → 17:04:44, oracle 평가 기준. 시작 자산 = 1M + 그 시점까지의 거래 손익. 해당 구간 외부 원장 이벤트 0, 모든 지갑 체결 수량과 현재 포지션 대조 통과. 시각이 다른 API 응답을 사용한 스냅샷 근사이며 공식 순위가 아닙니다.</p>'
    body+=table(['지갑','종료 시 누적 순손익','같은 기간 순손익','같은 기간 수익률','새 체결','종료 총노출'],
       [[('내 지갑 ' if i==0 else '')+p['address'],f"{p['current']['pnl']['oracle']:+.6f}",f"{p['same_period_pnl']:+.6f}",f"{p['same_period_return_percent']:+.8f}%",p['same_period_fills'],f"{p['current']['gross_oracle']:.2f}"] for i,p in enumerate(actual['peers'])])
    body+='''<p>이전 상위 두 지갑은 청산 후 현금 보유여서 이 구간의 수익 변화가 0이었습니다. 0x06a336…409a11은 적은 노출과 높은 메이커 비중을 사용하는 다른 거래 양상입니다. 다른 팀의 전략 내부를 공개 체결만으로 단정할 수 없습니다.</p><p><strong>기존 3일 변동성 표의 비교 한계:</strong> 사용자 지갑은 세 날 중 실제 노출이 있었던 날이 한 날이었고, 포트폴리오 관측 시차도 있습니다. 그 표의 숫자를 사용자 전략의 충분한 일별 변동성 검증으로 취급하면 안 됩니다. 이번 6시간 미만 구간에서는 일별 STD를 새로 만들지 않았습니다.</p></section>'''
    body+='<section><h2>구현과 검증 상태</h2>'
    body+=table(['검사','결과'],[['기능·회귀 검사',f'{tests}개 통과'],['신규 에이전트의 목표 비중',f"백테스트와 {verified['rows_checked']}개 시점 전부 일치"],['회계 검산','모든 실행에서 가격 손익·수수료·펀딩 재구성 통과'],['기존판 체결 재현','종전 체결 및 평가액과 정확히 일치'],['기존 코드 보호',f'{len(protected)}개 파일 해시 동일'],['봉인된 실행 실험 보호',f'{len(reserved)}개 고정 파일 해시 동일; 봉인된 중간 결과 미열람'],['실제 주문','0건; 새 후보 실행자는 연결하지 않음'],['현재 운용',f"{live['state']} / {live['profile']} / 다음 정규 판단 {live['next_normal_decision_kst']}"]])
    body+='''<p>후보 코드는 <code>experiments/performance_attribution_20261016/agent.py</code>, 모든 결과는 <code>artifacts/budget_results.json</code> 및 <code>artifacts/research_results.json</code>에 있습니다. 실제 거래와 주문·잔고 원장 인계는 아직 하지 않았습니다.</p></section>
<section><h2>판정과 다음 검증 기준</h2><p>위험감축＋고정 노출 보정판은 추가 검증할 후보로 남깁니다. 동일 자본·동일 위험 한도의 개발 비교에서는 개선했지만, 새 기간의 실제 par 체결 조건에서 확인하기 전까지 실거래 개선이 입증됐다고 판단하지 않습니다. 비용 악화 때의 손실과 8월의 열세도 해결되지 않았습니다.</p><p><a href="https://perpetual-alpha26.github.io/about">공식 평가 규정</a>은 oracle 순수익, 일별 수익률 STD, 보고서 심사 점수를 참가팀 간 정규화해 평가합니다. 여기서 사용한 동시 개선 기준은 내부 기준이며 공식 순위 계산식 자체가 아닙니다. <a href="https://arxiv.org/html/2502.04284v1">Ma &amp; Smith (2025)</a>의 비용·신호 유효기간 논의는 경제적 진입 제한의 참고자료입니다. 이번 구현은 그 논문의 수학적 정책 재현이 아닙니다.</p></section></html>'''
    dest=ROOT/'reports/performance_attribution_20261016';dest.mkdir(exist_ok=True)
    (dest/'report.html').write_text(body)
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'candidate':'risk_cap_budget_matched_v1',
        'historical_independence':False,'research_verdict':'Development improvement on main period; prospective validation pending; no live replacement',
        'budget':budget,'economic':economic,'actual':actual,'agent_verification':verified,'tests_passed':tests,'live_status':live,
        'protected_files_unchanged':len(protected),'reserved_files_unchanged':len(reserved),'live_orders_sent':0}
    save(dest/'summary.json',result)
    save(OUT/'final_manifest.json',{'sources':{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
        'report_sha256':sha(dest/'report.html'),'budget_model_sha256':sha(OUT/'budget_frozen.json'),'protected_files_unchanged':len(protected)})
    print(json.dumps({'report':str(dest/'report.html'),'tests':tests,'orders_sent':0,'live_status':live},ensure_ascii=False))

if __name__=='__main__':main()
