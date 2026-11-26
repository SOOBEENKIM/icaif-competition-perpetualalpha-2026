"""Render the execution calibration without treating probes as account returns."""
import csv
import html
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .research import HERE, ROOT, OUT
from experiments.par_direct_forecast_20261017.data import save, sha


def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    costs=json.loads((OUT/'results.json').read_text())
    audit=json.loads((OUT/'lifecycle_v2/results.json').read_text())
    target=ROOT/'reports/pair_execution_calibration_20261017';target.mkdir(exist_ok=True)
    counts=Counter();quantities=0
    for row in costs['cost_rows']:
        counts[row['status']]+=1
        if row['status']=='missing_arrival':continue
        assert row['time']+row['latency_ms']<=row['arrival']<=row['time']+row['latency_ms']+5000
        assert 0<=row['filled']<=row['requested']+1e-10
        if row['filled']:assert row['price']>0
        quantities+=1
    assert len(audit['rows'])==352 and audit['all_probes_accounted']
    for row in audit['rows']:
        assert abs(row['net_pnl']-row['price_pnl']-row['funding']+row['fees'])<1e-7
        assert row['fees']>=row['entry_fees']>0
        assert all(p['qty']==p['units']/10**p['decimals'] for p in row['positions'])
        for pos in row['positions']:
            assert all(f['time']>=row['lead_fill']['time'] for f in pos['fills'])
    # Original run sources are immutable; the superseded lifecycle source is
    # preserved by content in its own source_snapshot directory.
    for frozen in (OUT/'frozen.json',OUT/'lifecycle_v2/frozen.json'):
        for name,digest in json.loads(frozen.read_text())['sources'].items():assert sha(ROOT/name)==digest,name
    old=json.loads((OUT/'lifecycle/frozen.json').read_text())
    for name in ('lifecycle.py','lifecycle_protocol.json','test_lifecycle.py'):
        key=str((HERE/name).relative_to(ROOT))
        assert sha(OUT/'lifecycle/source_snapshot'/name)==old['sources'][key]
    cost_table=[]
    for phase in ('calibration','later_tape'):
        for notional in (500,5000,50000):
            s=next(x for x in costs['cost_summary'] if x['phase']==phase and x['notional']==notional and x['latency_ms']==1000 and x['depth_fraction']==1)
            rr=[r for r in costs['cost_rows'] if r['phase']==phase and r['notional']==notional and r['latency_ms']==1000]
            missing=sum(r['status']=='missing_arrival' for r in rr)
            z=s['actual_depth_after_zero_volume'];st=s['statuses']
            cost_table.append([phase,f'${notional:,}',s['observations'],missing,st.get('filled',0),st.get('partial',0),st.get('unfilled',0),
                f"{s['arrival_mid_cost_bps']['median']:.2f}",f"{z['full_fill']}/{z['cases']}"])
    long_table=[]
    for s in audit['summaries']:
        if s['condition'] not in ('all','tree'):continue
        long_table.append([s['scenario'],s['hours'],s['condition'],s['probes'],s['unique_quotes'],
            f"{s['net_bps_all_paths']:+.2f}",s['unknown_hedge_outcomes'],f"{s['longest_exit_delay_minutes']:.1f}"])
    coverage=[]
    for r in costs['coverage']:
        fmt=lambda t:datetime.fromtimestamp(t/1000,timezone(timedelta(hours=9))).strftime('%m-%d %H:%M:%S')
        coverage.append([r['path'],fmt(r['start_ms']),fmt(r['end_ms']),f"{r['hours']:.3f}"])
    old24=next(r for r in costs['long_summary'] if r['hours']==24 and r['exit_impact_bps']==10 and r['condition']=='all')
    new24=next(r for r in audit['summaries'] if r['hours']==24 and r['scenario']=='base' and r['condition']=='all')
    body='<h1>페어 전략: 체결 가정과 실패 경로 검증</h1>'
    body+='<p><strong>결론: 이번 후보의 순수익 개선은 확인되지 않았습니다.</strong> 실제 호가에 맞춘 체결 진단과 모든 진입 경로의 비용 원장을 보완했습니다. 실거래 주문과 운용 전략은 변경하지 않았습니다.</p>'
    body+='<h2>확인된 체결 가정의 한계</h2><p>직전 5분 거래량이 0이어도 관측된 주문장에는 체결 가능한 물량이 있었습니다. 따라서 직전 거래량의 일정 비율만 허용하는 종전 캔들 백테스트는 실제 주문장 체결 가능성을 입증하지 못합니다. 반대로 이 짧은 자료의 호가 수량을 전체 과거 기간에 복제할 수도 없습니다.</p>'
    body+=table(['구간','주문 명목액','유효 도착 관측','도착 자료 없음','가정상 전량','부분','미체결','도착 중간가 대비 비용 중앙값(bp)','직전 거래량 0 중 전량/관측'],cost_table)
    body+='<p>1초 이후 첫 유효 호가, 원래 최우선 반대 호가에서 10bp 이내의 IOC, 표시 수량 100%를 가정한 개별 탐침입니다. 실제 제출 주문이 아닙니다. 수수료는 위 호가 비용에 포함되지 않습니다. 미관측은 미체결로 집계하지 않았습니다. 반복된 호가와 양방향 탐침이므로 독립 표본이 아닙니다.</p>'
    s=next(x for x in costs['cost_summary'] if x['phase']=='calibration' and x['notional']==500 and x['latency_ms']==1000 and x['depth_fraction']==1)
    a,b=s['absolute_next_open_error_bps'],s['absolute_next_close_error_bps']
    body+=f'<p>5분 경계에서 다음 봉 거래가 있고 전량 체결 가능한 $500 탐침 {a["n"]}개에 한정하면, 실제 도착 호가 대비 다음 봉 시가의 절대 오차 중앙값은 {a["median"]:.2f}bp, 종가는 {b["median"]:.2f}bp였습니다. 시가도 정확한 1초 뒤 체결가격으로 볼 수 없습니다.</p>'
    body+='<h2>한쪽 체결·미체결·청산 지연을 포함한 재계산</h2>'
    body+=f'<p>종전 24시간 진단은 {old24["observations"]}개 경로 중 해당 시각의 청산을 모델링할 수 있었던 {old24["modeled_closed"]}개만으로 {old24["conditional_weighted_net_bps"]:+.2f}bp를 보였습니다. 새 진단은 모든 {new24["probes"]}개를 유지하며, 제한가를 적용한 부분 헤지·헤지 실패 시 축소·늦은 청산까지 모델링했을 때 {new24["net_bps_all_paths"]:+.2f}bp였습니다. 범위와 실행 규칙이 함께 달라졌으므로 이 차이를 전략 개선 효과나 통계적 인과효과로 해석하지 않습니다.</p>'
    body+=table(['비용 조건','보유시간(h)','예측 조건','경로 수','고유 주문 시점','명목액 가중 순손익(bp)','헤지 관측 불명','최대 청산 지연(분)'],long_table)
    body+='<p>all은 방향 필터 없음, tree는 이미 고정된 모델의 예측 방향과 일치하는 경우입니다. 기본 조건은 메이커 3bp·테이커 9bp·캔들 청산 충격 10bp, 스트레스는 메이커 3bp·테이커 12bp·충격 20bp·관측 호가 수량 절반·펀딩 수취 절반입니다. 청산 대기 중에도 실제 시간별 펀딩률을 적용했습니다.</p>'
    body+='<p><strong>이 표는 계좌 수익률·대회 Q가 아닙니다.</strong> 같은 주문의 체결 조각 및 같은 체결을 여러 페어에 대입한 경로가 중첩됩니다. 기본 조건의 각 보유기간에서 88개 중 67개 경로에 캔들 기반 청산 모형이 필요했습니다. 4개는 헤지 도착 자료가 없어 헤지가 미체결됐다는 가정으로 계산했습니다. 전체 기간의 실제 호가 청산을 확인한 결과가 아니며, 모두 닫혔다는 것은 모형상 결과입니다.</p>'
    body+='<h2>구현 검증과 보존</h2><p>헤지 최소 주문금액, 부분 체결, 고정 제한가, 미관측 경로, 무거래 봉, 펀딩 방향, 청산 수량 상한, 마지막 1 lot의 부동소수점 잔량을 포함한 9개 검증을 통과했습니다. 수량 원장을 정수 lot으로 바꿔 마지막 수량이 잘못 남는 문제를 수정했습니다. 최초 오류 결과는 lifecycle에 보존하고 superseded로 표시했으며, 이 보고서는 lifecycle_v2만 사용합니다.</p>'
    body+=f'<p>재계산 경로는 {len(audit["rows"])}개이며, 현금·보유수량 원장과 별도 손익 항등식의 최대 오차는 {audit["maximum_accounting_error"]:.3g} USDPR입니다. 원본 자료·예측·기존 운용 및 예약 코드의 보호 검증을 통과했습니다.</p>'
    body+='<h2>자료 범위와 다음 수정</h2>'+table(['자료','시작(KST)','종료(KST)','관측 시간(h)'],coverage)
    body+='<p>이 자료는 이미 여러 개발 실험에 사용한 과거 자료이며 독립 검증 자료가 아닙니다. 새로 봉인된 미래 자료는 열지 않았습니다. 다음에는 현재 주문 호가에서 지불하는 진입 금액을 기준으로 미래 페어 가치와 비용을 비교하도록 예측 목표와 주문 판단을 연결합니다. 거래비용만 낮추거나 이익 난 경로만 고르는 방식은 개선 근거로 사용하지 않습니다. 기존 1·2번 전략의 같은 기간·자본·위험 한도 비교도 유지합니다.</p>'
    body+='<p>출처: <a href="https://perpetual-alpha26.github.io/docs/">공식 시장·수수료·펀딩 규정</a>. 관련 관점: <a href="https://arxiv.org/abs/2502.04284">Alpha Decay and Transaction Costs (2025)</a>. 논문의 수익률을 재현했다는 주장은 없습니다.</p>'
    css='body{font-family:system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;line-height:1.65;color:#243244}h1,h2{color:#12243a}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d8e0e8;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#eff5fa}p{max-width:1100px}a{color:#1664aa}'
    (target/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 실행 검증</title><style>'+css+'</style><body>'+body+'</body></html>')
    with (target/'conditional_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(audit['summaries'][0]));w.writeheader();w.writerows(audit['summaries'])
    save(target/'summary.json',dict(cost_summary=costs['cost_summary'],conditional_summary=audit['summaries'],
        tests=9,maximum_accounting_error=audit['maximum_accounting_error'],performance_improvement_established=False,
        orders_sent=0,independent_evidence=False,account_return=None,financial_score=None,
        source_checks=True,valid_cost_rows=quantities,cost_status_counts=dict(counts),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'results.json',OUT/'lifecycle_v2/results.json')}))
    save(HERE/'work_state.json',dict(status='progress',goal_complete=False,orders_sent=0,
        this_turn='Calibrated actual arrival book costs and stale-candle errors; implemented bounded hedge IOC, all-path ledger, delayed exit model, integer-lot quantities; 352 conditional lifecycle replays and 9 tests passed. Both base and stress remain negative.',
        performance_improvement_established=False,report=str(target/'report.html'),
        next_action='Make forecasts and order decisions share the current executable entry anchor. Predict future paired value from past data, explicitly reprice the forecast at the observed quote with both leg costs and known funding; compare the same frozen candidates and incumbent baselines. Do not manufacture historical L2 or transplant volume capacity. Fixed quote-level selection must precede lead-fill outcomes.',
        data_wait_required=False,historical_independence=False,
        protected_sources_unchanged=True,superseded_run='artifacts/lifecycle',authoritative_run='artifacts/lifecycle_v2'))
    print(target/'report.html')


if __name__=='__main__':main()
