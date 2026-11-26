"""Preserve all attempted models and disclose the failed period explicitly."""
import gzip,html,json,importlib
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from .study import ROOT,OUT,PRIOR,read,save,sha,verify
from .candidate import snapshot


def main():
    verify()
    for wave in range(2,10):importlib.import_module(f'pair_model_research.wave{wave}').verify()
    all_rows={}
    for r in read(OUT/'wave1_failure.json')['rows']:
        all_rows[(r['arm'],r['phase'],r['scenario'])]=dict(r,wave=1)
    for wave in range(2,10):
        for path in (OUT/f'wave{wave}'/'rows').glob('*.json'):
            r=read(path);all_rows[(r['arm'],r['phase'],r['scenario'])]=dict(r,wave=wave)
    rows=list(all_rows.values());baseline=[r for r in read(PRIOR/'reference_results.json') if r['arm']=='incumbent_fixed_two']
    candidate='adaptive_hedge_72';chosen=[r for r in rows if r['arm']==candidate]
    assert len(chosen)==6
    comparisons=[]
    for r in chosen:
        b=next(x for x in baseline if x['phase']==r['phase'] and x['scenario']==r['scenario'])
        comparisons.append(dict(phase=r['phase'],scenario=r['scenario'],baseline_pnl=b['net_pnl'],candidate_pnl=r['net_pnl'],pnl_delta=r['net_pnl']-b['net_pnl'],baseline_Q=b['Q'],candidate_Q=r['Q'],Q_delta=r['Q']-b['Q'],both_improve=r['net_pnl']>b['net_pnl'] and r['Q']>b['Q']))
    protocols=[OUT/'protocol.json',*sorted(OUT.glob('wave*/protocol.json'))]
    count=sum(len(read(p)['arms']) for p in protocols)
    full=[x for x in comparisons if x['phase']=='continuous']
    summary=dict(candidate=candidate,experiment_waves=9,candidate_count=count,replay_count=len(rows),
      full_period_PnL_and_Q_improved_both_costs=all(x['both_improve'] for x in full),
      all_periods_improved=all(x['both_improve'] for x in comparisons),comparisons=comparisons,
      selection='Wave9 candidate selected using August base/stress worst Q advantage before later results; architecture was developed after previous failed waves.',
      status='FULL_PERIOD_DEVELOPMENT_GAIN_WITH_FAILED_LATER_BASE_PERIOD',
      independent_test=False,selection_bias_present=True,live_changed=False,orders_sent=0,
      test_count=39,full_PAS_paired_execution_backtest=False,
      fresh_reproduction=read(OUT/'wave9/reproduction_check.json'))
    save(OUT/'results.json',rows);save(OUT/'comparison.json',summary)
    p=read(OUT/'wave9/protocol.json');d=dict(np.load(ROOT/p['input']));t=d['market_times'];source=int(t[-1])
    snap=snapshot(d,source,source+5)
    snap['publication_timing']='Historical reconstruction: source+5s receipt and5s computation are assumptions, not an observed live receipt.'
    save(OUT/'candidate_snapshot.json',snap)
    state=dict(np.load(OUT/'wave9/adaptive_hedge_72.npz'))
    assert state['mu'].shape[1]==45 and len(snap['pair_audit'])==45
    attribution=[]
    for scenario in ['base','stress']:
        with gzip.open(OUT/f'wave9/reference/{candidate}_continuous_{scenario}.json.gz','rt') as f:run=json.load(f)
        intents=[e for e in run['events'] if e['event']=='pair_intent'];m=run['metrics']
        attribution.append(dict(scenario=scenario,metrics=m,
          pairs=dict(Counter('/'.join(e['pair']) for e in intents)),
          directions=dict(Counter(('long '+e['pair'][0]+'/short '+e['pair'][1]) if e['direction']==1 else ('short '+e['pair'][0]+'/long '+e['pair'][1]) for e in intents)),
          intent_count=len(intents),outside_original_pairs=sum(e['pair'] not in [['BTC','ETH'],['BTC','HYPE']] for e in intents)))
    save(OUT/'candidate_attribution.json',attribution)
    save(OUT/'candidate_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__).with_name('candidate.py'),Path(__file__).with_name('adaptive_hedge.py'),Path(__file__).with_name('wave9.py'),Path(__file__).with_name('horizon_evaluation.py'),OUT/'wave9/protocol.json',OUT/'wave9/adaptive_hedge_72.npz',OUT/'wave9/model_audit.json']})
    phase_names={'validation':'8/1 12UTC → 9/1 12UTC','comparison':'9/1 12UTC → 10/15 12UTC','continuous':'8/1 12UTC → 10/15 12UTC'}
    def row_html(r,label):
        vals=[phase_names[r['phase']],r['scenario'],label,f"{r['net_pnl']:+,.2f}",f"{100*r['return_fraction']:+.4f}%",f"{r['Q']:.6f}",f"{100*r['volatility']:.4f}%",f"{100*r['max_drawdown']:.4f}%",r['completed_pairs'],r['risk_breach_bars']]
        return '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in vals)+'</tr>'
    main_table=[]
    for phase in ['continuous','validation','comparison']:
        for scenario in ['base','stress']:
            for group,label in [(baseline,'기존 두 페어'),(chosen,'45개 자동 평가 후보')]:
                r=next(x for x in group if x['phase']==phase and x['scenario']==scenario);main_table.append(row_html(r,label))
    trials=[]
    for protocol in protocols:
        pp=read(protocol);wave=pp.get('wave',1)
        for arm in pp['arms']:
            cells=[wave,arm]
            for phase in ['validation','comparison','continuous']:
                for scenario in ['base','stress']:
                    r=all_rows.get((arm,phase,scenario))
                    cells.append(f"{r['net_pnl']:+.2f} / {r['Q']:.5f}"+(' ⚠위험 위반' if r['risk_breach_bars'] else '') if r else '미실행')
            trials.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in cells)+'</tr>')
    by_id={f['stable_pair_index']:f for f in snap['forecasts']};coverage=[]
    for audit in snap['pair_audit']:
        k=audit['pair_index'];f=by_id.get(k);coverage.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [k,'/'.join(audit['pair']),f"{f['source_beta']:.4f}" if f else '—',f"{f['mu']*10000:+.2f}" if f else '—',f['economic_action'] if f else 0,audit['reason']])+'</tr>')
    stamp=datetime.fromtimestamp(source,timezone(timedelta(hours=9))).isoformat()
    details=[]
    for a in attribution:
        m=a['metrics'];details.append(f"<p><b>{a['scenario']}</b>: 가격변동 {m['price_move_pnl']:+,.2f} − 수수료 {m['fees']:,.2f} − 모의 체결가격 비용 {m['execution_cost']:,.2f} ＋ 펀딩 {m['funding']:+,.2f} = 순손익 {m['net_pnl']:+,.2f}. 잔여 포지션 청산 비용 충당 후 {m['pnl_after_liquidation_reserve']:+,.2f}. 페어 의도 {a['intent_count']}건 중 기존 두 페어 밖 {a['outside_original_pairs']}건.</p><p>{html.escape(str(a['pairs']))}</p>")
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>전체45 페어 개선 실험</title>
<style>body{{max-width:1250px;margin:35px auto;padding:0 22px;font:16px/1.7 system-ui;color:#182b42;background:#fafcfe}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d6e0eb;padding:8px}}th{{background:#e7eef7}}.box{{padding:18px;background:#e8f5f0;border-left:4px solid #26745a}}.limit{{padding:18px;background:#fff2df;border-left:4px solid #a8671e}}code{{overflow-wrap:anywhere}}a{{color:#155fa5}}</style>
<h1>전체45 페어 자동 평가 — 개선 후보와 전체 실험 기록</h1>
<p class="box"><strong>같은 전체 기간·자본·위험 한도에서 기본/불리한 조건 모두 순손익과 Q가 개선된 후보를 구현했다.</strong> 8/1~10/15 기본 순손익 +529.07→+646.66, Q0.151217→0.159132. 불리한 조건 +299.83→+1,579.19, Q0.123253→0.255156. 개선 후보는 <code>adaptive_hedge_72</code>이다. 실거래에는 연결하지 않았다.</p>
<p class="limit"><strong>모든 비교 구간을 통과한 것은 아니다.</strong> 9/1부터 새로 시작한 기본 조건은 기존 +247.92/Q0.121171 대비 후보 −29.98/Q0.081170으로 악화됐다. {count}개 후보와 {len(rows)}개 재생 결과를 거친 <b>재사용 과거 자료의 개발 성과</b>다. 미래 성과나 본대회 순위 상승의 보장, 과적합이 없다는 증명이 아니다. 전체 PAS 호가·페어 체결 재생도 아직 수행하지 않았다.</p>
<h2>무엇을 고쳤는가</h2><ol>
<li>종목 이름으로 BTC/ETH와 BTC/HYPE만 허용하던 제한 대신,10개 종목의45개 비순서 조합을 모두 평가한다. 통과 페어 개수는 고정하지 않는다.</li>
<li>매일00UTC에 각 페어의 과거60일 공통 관측으로 헤지 비율을 추정한다. 양의 가격 공분산, 충분한 공통 유효 관측(80%), 지원 가능한 헤지 비율0.1~10을 검사한다. 비율을 경계값으로 억지로 자르지 않는다. β=std(log Pₐ)/std(log Pᵦ)인 대칭 major-axis 비율이며 OLS 회귀계수라고 주장하지 않는다.</li>
<li>상대가격 S=log Pₐ−β log Pᵦ에 AR(1)을 적합해 국소 평균 θ와 지속성 ρ를 추정한다.0&lt;ρ&lt;1, 추정 반감기6~336시간인 경우72시간 상대가격 변화를 계산한다. 상대 포트폴리오 수익률 근사는 μ=(θ−S)(1−ρ⁷²)/(1+β)이며, 과거168시간의 해당 헤지 변동성으로 극단 예측을 제한한다. <b>공적분이 증명된 무위험 차익거래가 아니다.</b></li>
<li>μ가 양수면 A롱/B숏, 음수면 A숏/B롱을 검토한다. 큰 예측만으로 매수하지 않고, 현재 기준가격·수수료·모의 체결 비용·관측된 펀딩·거래량 참여율·반올림·위험 한도 후 기대 순현금이 양수인 제안만 비교한다. 기존처럼 한 번에 활성 페어1개, 페어 총명목6000 한도를 사용한다.</li>
<li>보유 후에는 기존 부분 체결·균형 복구·만기·손절·비용을 지불하는 헤지 변경·최대 보유168시간 처리를 유지한다. 실패했던 조기 청산 후보의 규칙을 최종 후보에 섞지 않았다. 기본 방향 전략은 그대로다.</li></ol>
<p>최종 후보는 딥러닝이나 LLM이 아니라, 과거 가격 관계를 매일 재추정하는 통계 상대가치 모델이다. 모든45개를 평가한다는 의미는 모든45개를 거래한다는 뜻이 아니다. 자료 부족/관계 불안정/기대 수익 부족/체결 제약이면 거래하지 않는다.</p>
<h2>동일 조건 결과</h2><p>우리 운용 예산50,000＋거래하지 않는 예비50,000, 수익률 분모100,000. 팀원의 실제 동시 손익을 합친 계좌 재현은 아니다. 기준시장 가격·거래량에 가정한 실행 비용과 참여율을 적용했다. <b>PAS 실제 호가·체결 결과로 표시하지 않는다.</b> 기본: 편도 수수료9bp, 가격 비용10bp, 참여율1%. 불리:12bp/20bp/0.25%, 수취 펀딩50%. 각 경우 체결 경로와 선택 페어가 달라지므로 불리한 조건의 손익이 더 높을 수도 있다. 같은 거래에서 수수료가 이익으로 바뀐 것이 아니다.</p>
<p>Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), J=0이면 score=0.8Q. σ는12UTC 일별 수익률의 표본 표준편차×√n이며 연율 변동성이 아니다. <a href="https://perpetual-alpha26.github.io/leaderboard">공식 리더보드</a>도 일별12UTC 기준과 실시간 표시를 구분한다. 여기의 Q는 모의 계산이며 공식 순위가 아니다.</p>
<table><tr><th>기간(2026)</th><th>조건</th><th>전략</th><th>순손익</th><th>R</th><th>Q</th><th>σ</th><th>MDD</th><th>완료 페어</th><th>위험 위반 봉</th></tr>{''.join(main_table)}</table>
<p>모든 기간의 시작·종료는12UTC. 연속 실행은9/1에도 이전 포지션/실행 상태를 인계하지만,9/1 별도 실행은 현금에서 시작하므로 두 부분 구간의 수익 합이 연속 결과와 같지 않다. 이 차이가 큰 것은 시작 상태 의존성이 있다는 증거이며 실패한 별도 구간을 숨기면 안 된다. 같은 최대 한도에서도 실제 평균 보유 규모는 달라지므로 순수 예측력 향상만으로 해석하지 않는다.</p>
<h2>손익과 실제 모의 거래 조합</h2>{''.join(details)}
<h2>검증과 재현 범위</h2><p>38개 회귀 검사와 새 후보 어댑터1개 검사 통과. 미래 가격/미만기 정답 변조, 입력 구간 잘라내기,45개 ID, 페어 순서 대칭성, 제약 유지, 보유 신호 유지, 회계와 주문 시각을 점검했다. 후보의6개 재생 모두 회계 오차1e-6 미만·시각 검사 통과·위험 위반0봉이다. 저장 결과를 읽지 않고 전체 기본 구간을 다시 재생해 순손익/Q/비용을1e-10 이내에서 재현했다. <b>이는 계산 재현성 확인이며 독립 수익성 검증이 아니다.</b></p>
<p>선형/트리 공유 모델, 추세·평균회귀, 불확실성 감액, 관계 그래프, 과거 발행 예측의 조건부 보정, 조기 청산, 장기 예측 앙상블, 위험 대비 순이익 순위, 적응 헤지까지9차 실험을 했다. 각 실험은 실행 전에 코드·설정을 고정했지만 실험 방향 자체는 이전 결과를 보고 바꿨다. 동일 개발 구간을 반복 사용한 선택 편향이 남아 있다. 후보마다8월에서 미리 정한 최저 Q 개선 기준으로 대표를 고정한 후 이후 구간을 평가했다. 초기 공유 모델6개는8월 실패만 보존했다. 이전 두 차례 자동 선정 실험도 별도 보고서에 그대로 남아 있다.</p>
<h2>전체 실패·성공 기록</h2><p>칸: 순손익 / Q. 미실행 칸을0이나 성공으로 채우지 않는다. 위험 위반이 있는 후보는 제외한다.</p><table><tr><th>차수</th><th>후보</th><th>8월 기본</th><th>8월 불리</th><th>이후 기본</th><th>이후 불리</th><th>전체 기본</th><th>전체 불리</th></tr>{''.join(trials)}</table>
<h2>45개 평가 스냅샷</h2><p>자료 마지막 소스 {stamp}. β,72h 예측(bp), 원시 비용 gate 방향을 표시한다. 실제 현재 PAS 주문/체결이 아니며, 게이트 통과 후 현지 호가 검사도 필요하다.</p><table><tr><th>ID</th><th>페어</th><th>β</th><th>μ bp</th><th>방향</th><th>상태</th></tr>{''.join(coverage)}</table>
<h2>코드·원본·연결 상태</h2><p><code>pair_model_research/adaptive_hedge.py</code>:45개 모델, <code>candidate.py</code>:일별 모델/현재 소스의 읽기 전용 예측 어댑터, <code>wave9.py</code>:고정 설정과 비교, <code>horizon_evaluation.py</code>:같은 체결기 재생. 활성 실거래 파일·설정 해시는 그대로다. 어댑터와 백테스트 예측값 일치를 확인했지만, 기존24시간 게이트 검증기를 새72시간 후보에 그대로 쓰면 안 된다. 실거래 연결은 별도의 종목/모델 게이트·만기·보유 인계 통합이 필요하며 여기서 수행하지 않았다.</p>
<p><a href="comparison.json">비교 판정</a> · <a href="results.json">전체 수치</a> · <a href="candidate_attribution.json">손익/페어 분해</a> · <a href="candidate_snapshot.json">45개 판단 원장</a> · <a href="candidate_manifest.json">후보 해시</a> · <a href="wave9/protocol.json">최종 실험 설정</a> · <a href="wave9/model_audit.json">모든 일별 적합 결과</a> · <a href="wave9/reproduction_check.json">새 재생 재현</a> · <a href="tests.log">38개 검사</a> · <a href="candidate_tests.log">어댑터 검사</a></p>
<p>연구 참고: <a href="https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2026.1749337/full">2026년 동적 관계/예측 결합 연구</a>, <a href="https://www.sciencedirect.com/science/article/pii/S0957417425021153">2025년 페어 선정과 실행의 계층 분리 연구(초록)</a>. 문제 구성의 참고이며 해당 DNN/LSTM/GAT/DRL이나 논문 성과를 재현했다고 주장하지 않는다. 최종 major-axis＋AR 구조는 별도로 구현·평가했다.</p></html>'''
    (OUT/'report.html').write_text(text)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
