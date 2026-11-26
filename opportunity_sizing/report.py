"""Full results and explicit rejection; never selects a replacement candidate."""
import csv
import gzip
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from .study import OUT,ROOT,verify,save,sha


def stamp(t):return datetime.fromtimestamp(t,ZoneInfo('Asia/Seoul')).strftime('%m-%d %H:%M')
def read_run(arm,phase,scenario):
    with gzip.open(OUT/'runs'/f'{arm}_{phase}_{scenario}.json.gz','rt') as f:return json.load(f)


def main():
    protocol=verify();rows=json.loads((OUT/'results.json').read_text());assert len(rows)==72
    comparisons=[]
    for r in rows:
        if r['arm']!='pas_opportunity':continue
        base=next(x for x in rows if x['phase']==r['phase'] and x['scenario']==r['scenario'] and x['arm']=='incumbent')
        uniform=next(x for x in rows if x['phase']==r['phase'] and x['scenario']==r['scenario'] and x['arm']=='uniform_1p5')
        before=read_run('incumbent',r['phase'],r['scenario']);after=read_run('pas_opportunity',r['phase'],r['scenario'])
        identical=all(before[k]==after[k] for k in before)
        comparisons.append(dict(phase=r['phase'],scenario=r['scenario'],incumbent_pnl=base['net_pnl'],
            candidate_pnl=r['net_pnl'],uniform_pnl=uniform['net_pnl'],pnl_change=r['net_pnl']-base['net_pnl'],
            incumbent_Q=base['Q'],candidate_Q=r['Q'],candidate_volatility=r['volatility'],
            candidate_max_drawdown=r['max_drawdown'],identical_fills=before['fills']==after['fills'],
            identical_simulation=identical,terminal_time=r['end'],score_cutoff=r['score_cutoff'],
            score_cutoff_pnl=r['score_pnl'],extra_intent_rounds=r['extra_intent_rounds']))
    save(OUT/'comparison.json',comparisons)
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    funnels={}
    for phase in ['continuous71h','extension97h']:
        trace=read_run('pas_opportunity',phase,'base_1pct')['sizing_trace']
        b=np.array([r['base_qty'] for r in trace])!=0
        protected=np.array([r['protected'] for r in trace]);allowed=np.array([r['allowed'] for r in trace]);eligible=np.array([r['eligible'] for r in trace])
        funnels[phase]=dict(rounds=len(trace),asset_rounds=b.size,nonzero_base=int(b.sum()),
            nonzero_unprotected=int((b&~protected).sum()),passed_execution_gates=int((b&allowed).sum()),
            positive_cost_adjusted_opportunities=int(eligible.sum()),
            additional_target_rounds=sum(any(abs(q)>0 for q in r['extra_qty']) for r in trace))
    save(OUT/'eligibility_funnel.json',funnels)
    fit=json.loads((OUT/'fit.json').read_text())
    main_rows=[r for r in comparisons if r['phase']=='continuous71h']
    promotion=all(r['candidate_pnl']>r['incumbent_pnl'] and r['candidate_Q']>r['incumbent_Q'] for r in main_rows)
    verification=dict(runs=len(rows),comparisons=len(comparisons),primary_candidate='pas_opportunity',
        promoted=promotion,conclusion='Implementation completed; no realized behavior/performance improvement in evaluated histories.',
        all_candidate_simulations_identical_to_baseline=all(r['identical_simulation'] for r in comparisons),
        maximum_accounting_error=max(r['maximum_accounting_error'] for r in rows),
        risk_breach_events=sum(r['risk_breach_events'] for r in rows),
        timing_pass=all(r['timing_pass'] for r in rows),
        tests=dict(passed=36,command='.venv/bin/python -m unittest opportunity_sizing.test_sizing adaptive_leverage.test_policy pas_counterfactual_20261022.test_replay -v',
            when='Before experiment freeze and outcome evaluation',candidate_tests=14),
        reference_fit_rows=fit['training_rows'],reference_training_days=fit['training_days'],
        calibration_attenuation=fit['attenuation'],history_reused=True,independent_test=False,
        current_live_unchanged=True,orders_sent=0)
    assert not promotion and verification['risk_breach_events']==0 and verification['timing_pass']
    assert verification['all_candidate_simulations_identical_to_baseline']
    save(OUT/'verification.json',verification)
    table=''
    for r in comparisons:
        q='산정 불가' if r['candidate_Q'] is None else f"{r['candidate_Q']:.6f}"
        table+=f"<tr><td>{r['phase']}</td><td>{r['scenario']}</td><td>{r['incumbent_pnl']:+.5f}</td><td>{r['candidate_pnl']:+.5f}</td><td>{r['uniform_pnl']:+.5f}</td><td>{r['pnl_change']:+.5f}</td><td>{q}</td></tr>"
    primary=''
    for r in comparisons:
        if r['phase'] not in ['continuous71h','extension97h']:continue
        primary+=f"<tr><td>{r['phase']} / {r['scenario']}</td><td>{r['candidate_pnl']:+.2f}</td><td>{r['score_cutoff_pnl']:+.2f}</td><td>{stamp(r['score_cutoff'])}</td><td>{r['candidate_Q']:.6f}</td><td>{100*r['candidate_volatility']:.6f}%</td><td>{100*r['candidate_max_drawdown']:.6f}%</td></tr>"
    report=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>PAS 기회별 투자 규모 확대 — 구현 및 비교 결과</title>
<style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172033}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d8e1ec;padding:8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}.box{{padding:18px;background:#edf3fa;border-radius:10px}}a{{color:#1765a7}}small{{color:#526079}}code{{background:#edf0f6}}</style>
<h1>PAS 기회별 투자 규모 확대: 구현 완료, 성능 개선 미확인</h1>
<p class="box"><b>판정:</b> 별도 후보 코드를 구현하고 36개 검사 및 72개 백테스트를 완료함. 수정 후보는 기존판과 18개 비교 모두 손익·Q·체결·잔고 경로가 동일했음. 추가 투자 목표가 발생하지 않았으므로 <b>본대회 교체 근거를 확보하지 못했음.</b> 결과에 맞춰 조건을 낮추거나 재선택하지 않았으며 실운용은 변경하지 않음.</p>
<h2>구현한 내용</h2>
<ol><li><b>24시간 예측 모델:</b> 외부 시장 과거 자료로 새 릿지 예측기를 학습하고 별도 과거 기간에서 보정함. 예측 평균의 불확실성을 거래일 묶음으로 계산함.</li>
<li><b>PAS 가격·펀딩:</b> 과거에 관측 완료된 PAS/외부 가격 괴리의 시간 변화로 보정하고, PAS 펀딩만 추가 투자 판단에 사용함. 가격 괴리가 반드시 줄어든다고 가정하지 않음.</li>
<li><b>수량에 따른 비용·체결 여력:</b> 판단 시각 이전 PAS 거래량·분봉 범위와 동일 수수료/충격 가정으로 추가 수량·진입/청산 비용 추정. 과거 전체 호가가 없어 현재 호가를 과거에 끼워 넣지 않았음.</li>
<li><b>종목별 추가 금액:</b> 순이익·불확실성·공분산 위험·거래비용을 함께 고려해 추가 금액을 정함. 공통 배율 방식과 구분되며 기존 방향·위험 상한을 보존함.</li>
<li><b>별도 장부·회수:</b> 실제 부분체결을 기본분과 추가분으로 나누고, 감축은 추가분부터 처리함. 진입 지시 후 24시간 경과, 근거 소멸, 위험 감축 시 추가분을 줄임. 미체결을 보유로 계산하지 않음.</li></ol>
<p>새 기능은 <code>opportunity_sizing/</code>에 구현했으며 본대회 실행자는 이 모듈을 불러오지 않음. 5배 증거금 설정과 실제 보유 명목금액은 별개이고, 이 실험은 기존 총액·종목·변동성 한도를 그대로 사용함.</p>
<h2>학습·평가와 비교 조건</h2>
<p>학습: 2026-03-18부터 06-01 이전, 보정: 06-01부터 07-01 이전. 각 학습 정답은 24시간 뒤 자료까지 해당 구간 안에서 관측 완료된 경우만 사용함. 학습 {fit['training_rows']}개 자산·일 관측 / {fit['training_days']}개 시작일, 보정 {fit['calibration_rows']}개 / {fit['calibration_days']}개 시작일임. 자산별 관측을 독립 거래일로 부풀리지 않았음.</p>
<p><b>새 가격 예측의 보정 계수는 {fit['attenuation']:.1f}</b>였음. 보정 기간에서 예측과 실제 수익의 관계를 인정받지 못해 가격 예측 기여를 0으로 둔 결과임. 이 값을 사후에 올려 거래를 강제하지 않았음. PAS 괴리·펀딩 보정은 별도로 계산했으나 최종 확대 조건을 통과하지 못했음.</p>
<p>본대회 구간의 미래 24시간 정답은 모델 학습에 넣지 않았음. PAS 괴리의 온라인 보정은 각 판단 당시 이미 지난 1시간 변화만 사용하며, 평가 기간에도 이 과거 관측으로 갱신됨. 이것은 사전 고정한 온라인 알고리즘이며 완전히 고정된 PAS 모델이라고 표현하지 않음.</p>
<ul><li>비교군: 기존판 / 이전 확대 후보 / 고정 1.5배 확대 / 새 기회별 확대, 총 4개.</li>
<li>우리 자본 50,000 USDP 및 동일한 비교용 미운용 준비금 50,000 USDP. 팀원 실거래 경로를 재현한 결과가 아님.</li>
<li>6개 구간 × 3개 체결·비용 조건 × 4개 비교군 = 72회. 원래 12개 조건은 기존 저장 결과와 완전히 일치하는지 확인했음.</li>
<li>기본: 수수료 9bp·추가 충격 10bp·거래량 참여율 1%. 불리한 조건: 12bp·20bp·0.25%, 펀딩 수취 50% 또는 0%. 모든 비교군에 동일 적용.</li>
<li>연속 71시간: 10-19 21:00~10-22 20:00 KST. 확장 97시간: 10-19 21:00~10-23 22:00 KST. 나머지는 앞·뒤 구간과 실제 거래 테이프 재현이며 서로 겹침.</li>
<li>모두 이전에 사용한 개발 자료임. 새 독립 테스트 결과가 아니며, 과거 호가·대기열 부족으로 실제 체결을 완벽히 재현한 결과도 아님.</li>
<li>보존된 페어 모듈은 해당 기간 경제적 진입 신호가 0이라는 기존 감사 결과를 확인한 후에만 기본분 재현을 사용함. 활성 페어에 대한 확대 성능을 검증한 것은 아님.</li></ul>
<h2>전체 손익 비교</h2><p>단위 USDP, 수수료·펀딩 반영한 종료 평가손익. 강제 청산 비용 차감 전이며 청산 비용 준비금 적용 결과는 results.json에 따로 기록함.</p>
<table><tr><th>구간</th><th>조건</th><th>기존판</th><th>수정 후보</th><th>단순 1.5배</th><th>후보−기존</th><th>후보 Q</th></tr>{table}</table>
<p>단순 1.5배판은 97시간 기본 조건에서 +32.76239 USDP, 기존 +32.78851 USDP보다 0.02612 USDP 낮았음. 실제 추가 체결이 작아 이 차이만으로 일반적인 확대 효과를 결론내릴 수 없음.</p>
<h2>PnL·Q·변동성·낙폭의 시점 구분</h2><table><tr><th>구간 / 조건</th><th>종료 PnL</th><th>일별 기준 PnL</th><th>마지막 표본 KST</th><th>Q</th><th>기간 변동성 σ</th><th>최대낙폭</th></tr>{primary}</table>
<p><small>표의 변동성은 일별 수익률 표본표준편차 × √n인 대회 방식의 기간 σ임. 마지막 일별 표본 이후 장중 PnL은 Q에 반영하지 않음. 필요한 일별 수익률이 부족한 짧은 구간의 Q는 산정하지 않음. 모든 값은 비교용 모의 계좌 기준이며 실제 SFINX 지갑 또는 공식 순위가 아님.</small></p>
<h2>왜 확대하지 못했는지</h2>
<table><tr><th>단계</th><th>연속 71시간</th><th>확장 97시간</th></tr>
<tr><td>자산별 판단 수</td><td>{funnels['continuous71h']['asset_rounds']}</td><td>{funnels['extension97h']['asset_rounds']}</td></tr>
<tr><td>기본 목표가 0이 아님</td><td>{funnels['continuous71h']['nonzero_base']}</td><td>{funnels['extension97h']['nonzero_base']}</td></tr>
<tr><td>그중 위험 감축 보호 대상이 아님</td><td>{funnels['continuous71h']['nonzero_unprotected']}</td><td>{funnels['extension97h']['nonzero_unprotected']}</td></tr>
<tr><td>추가 진입·유동성 등 조건 통과</td><td>{funnels['continuous71h']['passed_execution_gates']}</td><td>{funnels['extension97h']['passed_execution_gates']}</td></tr>
<tr><td>비용·불확실성 차감 후 확대 조건 통과</td><td>0</td><td>0</td></tr></table>
<p>기존 보호 표시는 Chronos·시장 국면 판단이 기본 목표를 감축한 경우임. 현재 후보는 이를 최종 상한으로 보존하므로 대부분의 보유 목표를 추가로 키울 수 없음. 보호 표식을 모두 끄는 것은 이번 기능 수정과 다른 전략·위험 변경이며, 이 실험에서는 하지 않았음. 보호 여부와 별개로 새 예측 모델도 비용을 넘는 추가 기회를 만들지 못했음.</p>
<p>따라서 <b>이번 후보를 교체할 성과 근거는 없음.</b> 확인된 다음 과제는 모델이 유효한 PAS 순이익을 예측하는지와 기존 위험 감축 기준의 타당성을 각각 검증하는 것임. 단순 확대 모듈을 뒤에 붙이는 것만으로 해결됐다고 주장할 수 없음.</p>
<h2>검사와 재현</h2>
<p>36개 검사 통과: 후보 14개, 기존 확대 모듈 16개, 재현 엔진 6개. 상승/하락 대칭, 양의 기회에서 실제 확대 경로, 보호 대상 차단, 비용·불확실성, 상관 위험, 수량 정밀도, 부분체결 장부, 만료·강제감축, 미래 데이터 변경에 대한 과거 결과 불변성을 검사함. 72회 실행의 위험 한도 위반 0, 시간 순서 검사 통과, 최대 회계 오차 {verification['maximum_accounting_error']:.3g} USDP. 이것이 모든 잠재 오류가 없다는 증명은 아님.</p>
<p>실행: <code>.venv/bin/python -m opportunity_sizing.study run</code>. 기존 결과 파일과 사전 고정 소스 해시를 확인하며, 실운용 코드·설정 해시 변경은 없었음. 주문 전송 0.</p>
<p><a href="comparison.csv">전체 비교 CSV</a> · <a href="comparison.json">비교 JSON</a> · <a href="verification.json">검증 기록</a> · <a href="protocol.json">사전 고정 조건</a> · <a href="fit.json">학습·보정 결과</a> · <a href="source_freeze.json">소스·입력 해시</a></p></html>'''
    (OUT/'report.html').write_text(report)
    save(OUT/'artifact_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='artifact_manifest.json'})
    print(json.dumps(verification,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
