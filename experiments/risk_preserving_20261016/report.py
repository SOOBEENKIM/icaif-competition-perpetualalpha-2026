"""Publish the completed implementation checks and all predefined comparisons."""
import html,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from .prepare import HERE,OUT,ROOT,sha,save


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    tested=subprocess.run([sys.executable,'-m','unittest','experiments.risk_preserving_20261016.test_candidate','practice_live.test_live','-v'],cwd=ROOT,text=True,capture_output=True)
    (OUT/'tests.txt').write_text(tested.stdout+tested.stderr)
    if tested.returncode:raise ValueError('Tests failed; inspect artifacts/tests.txt')
    x=json.loads((OUT/'comparison.json').read_text());prepared=json.loads((OUT/'prepared.json').read_text())
    observed=json.loads((OUT/'read_only_supervision/result.json').read_text());handoff=json.loads((OUT/'handoff_check.json').read_text())
    agent=json.loads((OUT/'candidate_signal.audit.json').read_text())
    protected=json.loads((OUT/'protected_before.json').read_text())
    changed=[k for k,v in protected.items() if sha(ROOT/k)!=v]
    reserved=json.loads((ROOT/'experiments/execution_retry_20261016/forward_reservation.json').read_text())['plan']['artifacts']
    frozen_changed=[k for k,v in reserved.items() if sha(k)!=v]
    if changed or frozen_changed:raise ValueError('Existing protected artifacts changed')
    assert observed['passed'] and handoff['passed'] and agent['candidate_matches_compared_path']
    labels={'incumbent':'기존 하방위험 보정판','final_risk_cap_v1':'개발판: 최종 위험감축 상한','constant_scale_control':'기존판의 규모만 77.17%로 조정'}
    def metrics_row(row):
        m=row['metrics']
        return [labels[row['policy']],f"{m['roi']*100:+.6f}%",f"{m['daily_std_sample']*100:.6f}%",f"{m['max_drawdown_magnitude']*100:.6f}%",
            f"{m['mean_gross']*100:.4f}%",f"{m['fees']:.4f}",f"{m['funding']:+.4f}",m['fills'],m['completed_lifecycles'],m['unreached_cap_events']]
    main_headers=['버전','순수익률','일별 STD (표본)','최대낙폭 크기','평균 총노출','수수료','펀딩 순수취','개별 체결','완결 보유주기','미달 감축 목표']
    content='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>위험감축 보존 개발판 비교</title>
<style>body{font:16px/1.65 system-ui,sans-serif;background:#f5f7fa;color:#182235;max-width:1400px;margin:32px auto;padding:0 24px}section{background:white;padding:24px;margin:18px 0;border-radius:12px}h1,h2{line-height:1.35}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dce2e9;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf2f8}.scroll{overflow:auto}.decision{border-left:6px solid #cb8a19}small{color:#526174}code{background:#edf2f8;padding:2px 5px}a{color:#195caa}.ok{color:#15734a}</style>
<h1>추세＋7일 되돌림 — 위험감축 보존 개발판</h1>
<section class="decision"><h2>판정: 기존 실거래판 유지</h2><p>개발판 구현과 기능 검증은 통과했습니다. 전체 기본 조건에서 변동성과 낙폭은 줄었지만 수익률도 낮아졌으며, 불리한 비용 조건에서는 손실입니다. 사전에 정한 수익·위험 동시 개선 기준을 통과하지 못했습니다.</p><p><strong>아래 수치는 과거 기준시장 자료를 재사용한 백테스트입니다. 실제 par 체결 성과, oracle 평가 성과, 새로운 미사용 기간의 검증 결과가 아닙니다.</strong></p></section>
<section><h2>구현한 변경</h2><ul><li>기존 추세·7일 되돌림 혼합과 모델·고정 규모 계수를 유지했습니다.</li><li>기존 방향과 같은 최종 비중에 regime·Chronos 감축 비율의 평균(0.5 / 0.75 / 1)을 적용했습니다. 반대 방향의 되돌림 포지션이나 기준 추세가 없는 종목에는 해당 방향의 감축을 적용하지 않습니다.</li><li>20% 거래 생략 조건과 회전율 제한 뒤에도 감축 상한을 확인합니다. 개별 상한 적용으로 헤지가 줄어들 수 있으므로 포트폴리오 위험도 다시 점검합니다.</li><li>모델 계산은 별도 프로세스로 수행하고, 위험 감시자는 원자적으로 게시된 신호를 읽습니다. 오래된 신호로 신규 노출을 만들지 않으며, 유효한 최근 공분산과 실제 계좌로 위험 감축을 판단합니다.</li><li>판단 원장에는 실제 수량, 목표 수량, 적용 상한, 신호 시각·해시, 매매/대기 사유를 기록합니다. 추세·되돌림 혼합 후 증분·최종 위험 조정의 합도 검산합니다.</li></ul><p>이 후보는 기존 필터를 제거하는 수학적 재현이 아니라, 최종 단계에 감축 상한을 추가하는 한 가지 변경입니다. 과도한 감축과 잦은 작은 조정은 가능한 부작용으로 평가했습니다.</p></section>
<section><h2>같은 조건의 비교</h2><p>2026-09-01 00:00 UTC → 2026-10-13 12:00 UTC (KST: 9월 1일 09:00 → 10월 13일 21:00). 각 가상 계좌 1,000,000, 처음에는 모두 포지션 없음. UTC 00:00 정산·체결 반영 후 계좌 평가액으로 일별 수익률 42개, STD는 ddof=1, 비연율화입니다. 종료 포지션은 평가만 하고 가상의 청산 체결은 만들지 않았습니다.</p>'''
    for scenario,title in [('base','기본 비용 조건'),('stress','불리한 비용 조건')]:
        content+='<h3>'+title+'</h3><div class="scroll">'+table(main_headers,[metrics_row(r) for r in x['rows'] if r['period']==0 and r['scenario']==scenario])+'</div>'
    content+='''<p>기본: 편도 수수료 9bp, 가격 충격 10bp, 시간봉 거래량의 1% 한도. 스트레스: 수수료 9bp, 가격 충격 20bp, 거래량의 0.25% 한도, 총노출에 시간당 1bp 불리한 펀딩 추가. 공통으로 다음 시간봉에서 체결 가능 수량만 반영합니다. 실제 과거 par 호가를 재현한 가정은 아닙니다.</p>
<p>규모 비교본의 0.771683431 배율은 8월의 목표 총노출만으로 계산해 9월 이후 평가 전에 고정했습니다. 평가 기간의 실제 노출이나 위험이 완전히 같아지는 것은 아니므로, 이 비교만으로 독립적인 초과수익을 입증할 수 없습니다.</p></section>
<section><h2>구간별 결과를 모두 공개</h2>'''
    periods={1:'9/1–9/22 UTC',2:'9/22–10/13 12:00 UTC'}
    rows=[]
    for r in x['rows']:
        if r['period']:
            rows.append([periods[r['period']],r['scenario'],*metrics_row(r)[:4]])
    content+='<div class="scroll">'+table(['기간','비용 조건','버전','순수익률','일별 STD','최대낙폭'],rows)+'</div>'
    content+='<p>첫 구간에서는 수익 기회를 덜 가져갔고, 다음 구간에서는 손실을 줄였습니다. 전체 평균만으로 개발판 우위를 선언하지 않습니다.</p></section>'
    gates=[]
    for c in x['comparisons']:
        gates.append(['전체' if c['period']==0 else periods[c['period']],c['scenario'],
            *['통과' if c['checks'][k] else '실패' for k in ('positive_net_return','higher_return','daily_std_no_higher','mdd_no_worse','engineering')]])
    content+='<section><h2>사전 판정 기준</h2>'+table(['기간','조건','순수익 양수','기존보다 수익↑','STD 비악화','낙폭 비악화','회계·시각·위험 검사'],gates)
    content+='''<p>실거래 교체를 검증된 개선으로 권하려면 별도로 고정한 새로운 par 기간에서 체결 비용·청산 가능성까지 확인해야 합니다. 기존 실행 재시도 실험의 봉인된 미래 자료와 중간 결과는 읽지 않았습니다.</p></section>'''
    technical=[['신규 후보 및 기존 실행 회귀 테스트','30개 통과'],['기존 목표 비중 재현',f"{prepared['hours']:,}개 시간대 전부 정확히 일치"],
        ['후속 자료 추가에 따른 과거 목표 변경','새 감축 규칙의 prefix 검사 통과'],['기존 백테스트 경로 대조','동일 체결 및 시간별 평가액 일치'],
        ['별도 회계 검산','가격 변화 + 체결 비용 + 수수료 + 펀딩 재구성 통과'],
        ['신호 지연 중 실제 공개 계좌 감시',f"100초 지연 주입 중 {observed['observations_during_injected_delay']}회, 총 {observed['observations']}회 관측; 최대 간격 {observed['max_observation_gap_seconds']:.3f}초"],
        ['상태 인계 준비 검사',f"실제 체결 {handoff['matched_fills']}개 일치, 미해결 주문 {handoff['unresolved_orders']}개; 운용권 이전은 하지 않음"],
        ['기존 전략·실행 코드 보호',f'{len(protected)}개 해시 일치'],['기존 미래 비교 실험 보존',f'{len(reserved)}개 고정 파일 해시 일치'],
        ['개발판 실제 주문','0건']]
    content+='<section><h2>기능 및 운용 구조 검증</h2>'+table(['검사','결과'],technical)
    content+='''<p>100초 검사는 지연을 인위적으로 주입한 장애 시험이며, 실제 모델 추론 시간이 100초였다는 측정은 아닙니다. 실제 모델 입력에서 개발판을 산출하는 별도 신호 작업자도 실행해 비교한 목표 비중과 동일함을 확인했습니다.</p>
<p>새 감독기는 관측·판단용입니다. 실제 주문 전송은 연결하지 않았습니다. 향후 교체 시에는 기존 실행자를 정지·대조하고 같은 주문 원장의 중복 방지·불명확 체결 복구 절차를 유지하는 인계가 필요합니다.</p></section>
<section><h2>이번 결과에서 드러난 한계</h2><ul><li>기본 조건에서 개별 체결이 44 → 135회로 증가했습니다. 거래대금이 작아져 총수수료는 감소했지만, 잦은 작은 감축의 실행 부담이 생겼습니다.</li><li>감축 목표에 실제 보유가 도달하지 못한 사건이 기본 3회, 스트레스 5회 있습니다. 최소 주문금액·수량·유동성 제약 때문에 목표 생성과 실제 감축 완료를 구분해야 합니다.</li><li>일별 변동성과 낙폭 감소의 일부는 투자 규모 감소 효과입니다. 규모 비교본보다 전체 결과는 좋아도 모든 구간에서 좋지는 않았습니다.</li><li>과거 기준시장 자료는 이미 여러 차례 개발에 사용됐습니다. 이번 결과로 대회 시장의 장기 수익성이나 공식 순위를 확정하지 않습니다.</li></ul></section>
<section><h2>코드와 재현 자료</h2><p><code>experiments/risk_preserving_20261016/</code></p><ul><li><code>agent.py / policy.py</code>: 개발판과 최종 수량 상한</li><li><code>signal_worker.py / runtime.py</code>: 별도 신호 계산·위험 감시·판단 원장</li><li><code>replay.py / audit.py</code>: 공통 체결 비교와 별도 회계 검산</li><li><code>artifacts/comparison.json</code>: 모든 비교 수치</li><li><code>artifacts/runs/</code>: 기간·버전·비용 조건별 체결과 평가액</li><li><code>artifacts/tests.txt</code>: 기능 검사 전체 출력</li><li><code>protocol.json</code>: 결과를 보기 전에 정한 후보·비교 조건</li></ul>
<p><a href="https://perpetual-alpha26.github.io/about">대회 평가 규정</a>의 oracle 순수익·일별 STD·보고서 심사와 이번 개발용 기준시장 백테스트를 구분합니다. <a href="https://arxiv.org/html/2502.04284v1">Ma &amp; Smith (2025)</a>는 비용·신호 유효기간에 따른 거래 판단을 다루는 다음 단계의 참고자료이며, 이번 감축 상한이 그 논문의 재현이라는 뜻은 아닙니다.</p></section></html>'''
    report=ROOT/'reports/risk_preserving_development_20261016';report.mkdir(exist_ok=True)
    (report/'report.html').write_text(content)
    save(report/'summary.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'decision':x['decision'],
        'comparison':x,'technical_checks':technical,'independent_evaluation':False,'live_orders_sent':0})
    save(OUT/'final_source_manifest.json',{'sources':{str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')},
        'protocol_sha256':sha(HERE/'protocol.json'),'protected_files_unchanged':len(protected),
        'existing_forward_experiment_unchanged':len(reserved),'report_sha256':sha(report/'report.html')})
    print(json.dumps({'report':str(report/'report.html'),'tests_passed':30,'decision':x['decision'],'live_orders_sent':0}))

if __name__=='__main__':main()
