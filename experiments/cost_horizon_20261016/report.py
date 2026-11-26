"""Publish all trials, including failures, without touching the live strategy."""
import html,json
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from .research import ROOT,HERE,OUT,sha,save


def main():
    groups={name:json.loads((OUT/path).read_text()) for name,path in
            [('forecast','results.json'),('diversified','diversified/results.json'),('band','inward_band/results.json')]}
    labels={'first_strategy':'첫 번째 전략','second_strategy':'두 번째 전략 · 현재 운용',
            'forecast':'새 예측형 · XGBoost 72시간','diversified':'새 분산형 · 추세/캐리 혼합','band':'새 실행형 · 위험 상한 안쪽 조정'}
    comparison=[]
    for key,name in [('band','first_strategy'),('band','second_strategy'),('forecast','selected_third'),('diversified','selected_third'),('band','inward_band')]:
        for row in groups[key]['rows']:
            if row['phase']=='comparison' and row['policy']==name:
                comparison.append({'name':labels[name] if name in labels else labels[key],**row})
    protected=json.loads((OUT/'protocol_frozen.json').read_text())
    counts={}
    for category in ('active_protected_sources','reserved_dependencies'):
        for p,h in protected[category].items():
            if sha(p)!=h:raise ValueError('Existing source changed: '+p)
        counts[category]=len(protected[category])
    tests=(HERE/'all_tests.log').read_text();assert 'Ran 34 tests' in tests and tests.rstrip().endswith('OK')
    status=json.loads((ROOT/'experiments/performance_attribution_20261016/deployment/status.json').read_text())
    forward=json.loads((OUT/'forward/plan.json').read_text())
    health=json.loads((OUT/'forward/health.json').read_text()) if (OUT/'forward/health.json').exists() else {}
    kst=ZoneInfo('Asia/Seoul')
    iso=lambda t:datetime.fromtimestamp(t,kst).strftime('%Y-%m-%d %H:%M:%S KST')
    summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'candidate_variants':13,
             'backtest_runs':sum(len(g['rows']) for g in groups.values()),'tests_passed':34,
             'comparison':comparison,'historical_independence':False,'fully_dominating_candidate_found':False,
             'current_live_profile':status['profile'],'current_live_state':status['state'],'protected_unchanged':counts,
             'orders_sent_by_new_research':0,'forward_paper_start_kst':iso(forward['paper_start_unix']),
             'forward_end_kst':iso(forward['capture_end_unix']),'forward_capture_state':health.get('state'),
             'decision':'Keep the current second strategy. Forward-paper test the inward-band candidate; no automatic promotion.'}
    report=ROOT/'reports/third_strategy_research_20261016';report.mkdir(exist_ok=True)
    save(report/'summary.json',summary)
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    headers=['전략','순수익률','일별 STD','최대낙폭','체결 수','수수료 USDPR','평균 총노출']
    def render(rows):return table(headers,[[r['name'],*(f"{r['metrics'][k]*100:+.6f}%" if k=='roi' else f"{r['metrics'][k]*100:.6f}%" for k in ('roi','daily_std_sample','max_drawdown_magnitude')),r['metrics']['fills'],f"{r['metrics']['fees']:,.2f}",f"{r['metrics']['mean_gross']*100:.4f}%"] for r in rows])
    body='<h1>세 번째 전략 연구·구현 결과</h1><p class="lead">현재 두 번째 전략을 유지합니다. 13개 후보를 구현하고 104개 비교 실행과 34개 검사를 완료했지만, 수익률·변동성·최대낙폭을 함께 개선해 두 번째 전략을 대체할 후보는 확인하지 못했습니다.</p>'
    body+='<p>첫 번째·두 번째 전략 코드와 실행 원장은 그대로 보존했습니다. 새 코드에는 주문 서명·전송 기능이 없습니다. 실제 계좌는 계속 <code>'+html.escape(status['profile'])+'</code>로 운용 중이며, 새 후보는 별도의 모의 계좌에서만 비교합니다.</p>'
    body+='<h2>동일 조건의 과거 비교</h2><p>2026-09-01 00:00 UTC → 2026-10-13 12:00 UTC, 42.5일. 초기 자본 각 1,000,000, 시작 포지션 없음, 일별 수익률 42개의 표본 표준편차(ddof=1). 전체 계좌 자본 기준이며 포지션 수익률이 아닙니다. 모든 행은 이미 사용했던 기준시장 자료의 모의 결과입니다. 실제 par 호가 체결 성과나 독립 검증 결과가 아닙니다.</p>'
    body+='<h3>기본 비용</h3>'+render([r for r in comparison if r['scenario']=='base'])
    body+='<h3>불리한 비용</h3>'+render([r for r in comparison if r['scenario']=='stress'])
    body+='<p>공통 편도 수수료 9bp. 기본 가격 충격 10bp·시간봉 거래량 참여 1%; 불리한 조건은 충격 20bp·참여 0.25%·추가 불리한 펀딩 1bp/시간입니다. 관측된 기준시장 펀딩을 정산하고, 신호 다음 시간봉 종가를 체결 대용치로 사용했습니다. 실제 par 과거 호가·시장 충격·청산을 재현하지 않습니다. 종료 잔여 포지션은 평가만 하며, 가상의 청산 체결이나 청산 수수료를 만들지 않았습니다.</p>'
    body+='<h2>무엇을 구현했고 무엇이 부족했나</h2>'+table(['후보군','구현','판정'],[
        ['예측형 6개','24·72시간 순수익을 릿지/XGBoost/동일비중 앙상블로 예측. 현재 보유량·공분산·진입 비용·청산 비용 충당액·모델 간 불일치를 고려하는 볼록 최적화와 부분 조정.','8월 선택 단계에서 6개 모두 비용 차감 후 손실. 비교용 최상위 XGBoost 72시간도 9~10월 −0.148590%; 채택 불가.'],
        ['분산형 6개','10개 시장의 다중 기간 추세·확정 펀딩, 진입/반전 기준을 다르게 둔 신호 유지, 24시간 최소 유지, 암호화폐/기타시장 분산, 기존 두 번째 전략과 50:50 혼합.','선택된 혼합형은 STD·MDD가 낮아졌으나 수익률이 +0.008955%로 낮음. 수익 개선 실패.'],
        ['실행형 1개','두 번째 전략의 위험 상한이 기존 유지 구간을 무시하고 작은 감축을 요구할 때, 상한의 95%까지 한 번에 감축. 기존 상한을 초과하지 않고 이후 작은 재감축을 줄임.','137→92회 체결, STD·MDD 감소. 하지만 거래대금과 수수료가 늘고 수익은 +0.246624%→+0.237134%. 전체 우월성 미확인.']])
    body+='<p><strong>체결 횟수 감소는 수수료 감소와 다릅니다.</strong> 실행형의 수수료는 358.33→371.16 USDPR입니다. 평균 노출도 3.9254%→3.8322%로 줄었으므로 위험 감소 전부를 더 좋은 판단의 효과라고 볼 수 없습니다. 약 94.90 USDPR의 누적 수익을 덜 얻는 대신 위험 지표가 낮아진 과거 결과입니다.</p>'
    body+='<h2>검증 방법과 한계</h2><p>새 예측 모델 학습은 4월~7월이며, 미래 수익 목표가 8월 전에 모두 확정되는 행만 사용했습니다. 8월에 미리 정한 기준으로 후보를 선택한 뒤 9~10월을 비교했습니다. 다만 이 프로젝트는 이미 9~10월 결과를 여러 차례 검토했으므로 이 구간을 새 독립 테스트라고 부르지 않습니다. 예측형 실패를 본 뒤 분산형을, 그 실패를 본 뒤 실행형을 추가한 순차 개발 이력과 모든 실패 결과도 남겼습니다. 이를 13개의 독립 실험 성공확률로 해석해서는 안 됩니다.</p>'
    body+='<p>수익 목표는 다음 시간봉의 체결 대용 가격부터 24/72시간 뒤 가격과 그 사이 펀딩에 맞췄습니다. 미래를 제거한 접두 구간에서도 특징·모델 재학습·추론 결과가 동일한지, 비용 때문에 거래하지 않는 구간이 생기는지, 최적해와 독립 수치 계산이 일치하는지, 재시작 후 중복 판단이 차단되는지, 위험 감축/정지가 신호보다 우선하는지 검사했습니다. 기존 두 전략의 체결 및 순자산 경로도 원래 비교 결과와 정확히 일치했습니다.</p>'
    body+='<h2>참고한 최신 연구</h2><ul><li><a href="https://arxiv.org/html/2502.04284v1">Ma & Smith (2025), Alpha Decay and Transaction Costs</a>: 비용과 이전 포지션에 따라 거래하지 않는 구간이 달라지는 아이디어를 적용했습니다. 논문의 단일 자산 MDP와 정리 전체를 재현한 것은 아닙니다.</li><li><a href="https://arxiv.org/html/2606.00060v1">Bysik & Ślepaczuk (2026), Bitcoin Trading Under Transaction Costs</a>: 시간순 검증, 인과적 가격·거래량 특징, XGBoost와 비용 차감 성과 분리 평가를 참고했습니다. 논문의 보고 수익률을 우리 시장의 기대 수익률로 사용하지 않았습니다.</li><li><a href="https://arxiv.org/abs/2605.01176v4">Wang & Hasuike (2026), Prediction Inflation and Excessive Turnover</a>: 예측값 제한과 부분 포지션 조정을 적용했습니다. SPO 학습법 자체를 구현하거나 논문의 성능을 재현했다고 주장하지 않습니다.</li></ul><p>위 자료는 공개 arXiv 논문이며, 이번 구현은 여러 연구의 개념을 조합한 자체 실험입니다. 새로운 학술적 기여나 수익성 보장을 주장하지 않습니다.</p>'
    body+='<h2>실제 대회 호가의 독립 모의실험</h2><p>읽기 전용 서비스 <code>icaif-third-strategy-paper.service</code>가 새 공개 호가와 mark/oracle 관측을 수집합니다. 기존 예약 실험의 미공개 중간 결과는 읽지 않았습니다. 첫 번째·두 번째·실행형 후보에 각각 100만 USDPR의 가상 자본과 빈 포지션을 부여하고 동일한 시점부터 비교합니다.</p>'
    body+=table(['항목','고정 내용'],[['모의 거래 시작',summary['forward_paper_start_kst']],['평가 종료',summary['forward_end_kst']],['기본 모의 체결','공개 호가 수량 소비, IOC 1회, 편도 9bp, 1초 지연'],['불리한 모의 체결','관측 수량 50%, 편도 12bp, 3초 지연'],['누락/미체결','재사용된 같은 호가로 수량을 다시 채우지 않음. 오래된 호가·미체결·잔여 포지션을 기록. 자료 공백이나 부족한 일별 표본은 통과 처리하지 않음.'],['실제 거래 영향','실제 주문 0; 현재 두 번째 전략 유지'],['자동 교체','없음. 종료 후 결과 파일을 작성하며 계좌 전략은 변경하지 않음.']])
    body+='<p>이 실험도 실제 주문 체결이 아니라 공개 호가에 대한 반사실적 모의 체결입니다. 연습 DEX가 종료되거나 자료가 끊기면 그 기간을 성과 확인으로 인정하지 않습니다. 현재는 수집을 시작한 상태이며 미래 기간 수익이나 변동성은 아직 없습니다.</p>'
    body+='<h2>파일과 재현</h2><p><a href="../../experiments/cost_horizon_20261016/protocol.json">예측형 사전 규칙</a> · <a href="../../experiments/cost_horizon_20261016/diversified_protocol.json">분산형 규칙과 개발 이력</a> · <a href="../../experiments/cost_horizon_20261016/artifacts/inward_band/frozen.json">실행형 고정 규칙</a> · <a href="../../experiments/cost_horizon_20261016/all_tests.log">34개 검사 결과</a> · <a href="../../experiments/cost_horizon_20261016/artifacts/forward/plan.json">미래 모의실험 계획</a> · <a href="../../experiments/cost_horizon_20261016/artifacts/forward/health.json">수집 상태</a> · <a href="summary.json">결과 JSON</a></p>'
    body+='<h2>8월 후보 선택 결과 전체</h2>'
    rows=[]
    for group in ('forecast','diversified'):
        for r in groups[group]['rows']:
            if r['phase']=='validation' and r['policy'] not in ('first_strategy','second_strategy'):
                z=r['metrics'];rows.append([group,r['policy'],r['scenario'],f"{z['roi']*100:+.6f}%",f"{z['daily_std_sample']*100:.6f}%",f"{z['max_drawdown_magnitude']*100:.6f}%",z['fills']])
    body+=table(['후보군','구성','비용','순수익률','일별 STD','MDD','체결 수'],rows)
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>세 번째 전략 연구 결과</title><style>body{font:16px/1.75 system-ui,sans-serif;color:#1b2839;background:#f7f9fc;margin:0;padding:32px}main{max-width:1200px;margin:auto;background:white;padding:32px;border-radius:16px}h1{font-size:30px}h2{margin-top:36px}p{max-width:1050px}.lead{font-size:19px;font-weight:600}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #dce3ec;text-align:left}th{background:#eaf0f8}a{color:#1755a0}code{background:#edf1f6;padding:3px}</style><main>'+body+'</main></html>'
    (report/'report.html').write_text(page)
    save(report/'manifest.json',{'created_at_utc':summary['created_at_utc'],'research_sources':{str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')},
        'results':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'results.json',OUT/'diversified/results.json',OUT/'inward_band/results.json',HERE/'all_tests.log']},
        'protected_unchanged':counts})
    print(json.dumps({k:v for k,v in summary.items() if k!='comparison'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
