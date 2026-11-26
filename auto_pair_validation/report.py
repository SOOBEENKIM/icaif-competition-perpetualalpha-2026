"""Derived evidence only; no fitting, parameter choice or live changes."""
import html
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from .study import OUT,PRIOR,ROOT,read,save,sha,verify
from .policy import apply_entry_gate
from auto_pair_selection.selection import PAIRS


def main():
    p=verify();rows=read(OUT/'reference_results.json');assert len(rows)==18
    baselines=[r for r in read(PRIOR/'reference_results.json') if r['arm'] in p['baselines']]
    comparisons=[]
    for r in rows:
        b=next(x for x in baselines if x['arm']=='incumbent_fixed_two' and x['phase']==r['phase'] and x['scenario']==r['scenario'])
        comparisons.append(dict(arm=r['arm'],phase=r['phase'],scenario=r['scenario'],
          pnl_delta=r['net_pnl']-b['net_pnl'],Q_delta=r['Q']-b['Q'],
          pnl_and_Q_improved=r['net_pnl']>b['net_pnl'] and r['Q']>b['Q']))
    primary=[x for x in comparisons if x['arm']==p['primary']]
    all_better=all(x['pnl_and_Q_improved'] for x in primary)
    save(OUT/'comparison.json',dict(primary=p['primary'],primary_improved_all_conditions=all_better,
      comparisons=comparisons,orders_sent=0,live_changed=False,independent_test=False,
      decision='REFERENCE_CRITERION_PASSED_BUT_NOT_DEPLOYED' if all_better else 'DO_NOT_DEPLOY_REFERENCE_CRITERION_FAILED'))
    state=dict(np.load(OUT/'states.npz',allow_pickle=False));t=state.pop('times')
    labels=dict(np.load(OUT/'labels.npz',allow_pickle=False));history=read(OUT/'monitoring_history.json')
    latest=history[-1];signals=apply_entry_gate(state,'validated_all45')
    forecast_rows=[]
    # Descriptive forecast diagnostics on the same source rows for both models.
    for name,lo,hi in [('August',1785585600,1788264000),('later',1788264000,1792065600)]:
        mask=(t>=lo)&(labels['maturity']<=hi)&(t%86400==0)
        for k,pair in enumerate(PAIRS):
            keep=mask&np.isfinite(labels['actual'][:,k])&state['calibration_available'][:,k]
            if not keep.any():continue
            y=labels['actual'][keep,k];a=state['raw_mu'][keep,k];b=state['calibrated_mean'][keep,k]
            forecast_rows.append(dict(phase=name,pair=list(pair),rows=int(keep.sum()),
              raw_mae_bps=float(np.mean(abs(a-y))*10000),calibrated_mae_bps=float(np.mean(abs(b-y))*10000),
              zero_mae_bps=float(np.mean(abs(y))*10000)))
    save(OUT/'forecast_diagnostics.json',forecast_rows)
    pas=t>=1792411200
    entries=signals['economic_available']&(signals['economic_action']!=0)
    ii,kk=np.where(entries&pas[:,None])
    save(OUT/'pas_signal_audit.json',dict(hourly_nonzero_signals=len(ii),
      source='Frozen reference data during the PAS calendar period; NOT PAS returns or fills',
      start_utc='2026-10-19T12:00:00Z',end_source=int(t[-1]),full_PAS_backtest_completed=False,
      signals=[dict(source=int(t[i]),pair=list(PAIRS[k]),direction=int(signals['economic_action'][i,k])) for i,k in zip(ii,kk)]))
    reasons=Counter(reason for day in history if day['source']>=p['evaluation_start'] for x in day['pairs'] for reason in x['reasons'])
    latest_table=[]
    for k,r in enumerate(latest['pairs']):
        latest_table.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [
          '/'.join(r['pair']),r['rows'],r['trades'],'통과' if r['eligible'] else '보류',
          ', '.join(r['reasons']) or 'past_monitor_passed',int(signals['economic_action'][-1,k])])+'</tr>')
    table=[]
    for phase in ('continuous','validation','comparison'):
        for scenario in ('base','stress'):
            for r in [x for x in baselines+rows if x['phase']==phase and x['scenario']==scenario]:
                table.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [phase,scenario,r['arm'],
                  f"{r['net_pnl']:+,.2f}",f"{100*r['return_fraction']:+.4f}%",f"{r['Q']:.6f}",
                  f"{100*r['volatility']:.4f}%",r['completed_pairs'],f"{r['pair_pnl']:+,.2f}",r['risk_breach_bars']])+'</tr>')
    headline=('기준시장 비교 조건은 통과했지만 PAS 체결 검증과 실전 통합은 별도입니다.' if all_better else
              '자동 판단 구조는 수정했지만 기존 실거래 전략을 교체할 성과 기준은 통과하지 못했습니다.')
    stamp=datetime.fromtimestamp(int(t[-1]),timezone(timedelta(hours=9))).isoformat()
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>전체 페어 예측·검증·거래 보류 구조 비교</title><style>body{{max-width:1280px;margin:32px auto;padding:0 24px;font:16px/1.7 system-ui;background:#fafcfe;color:#193047}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:7px;border:1px solid #ccd8e5}}th{{background:#e7eff8}}.result{{padding:16px;background:#fff0df;border-left:4px solid #bd641e}}a{{color:#235a9c}}code{{overflow-wrap:anywhere}}</style>
<h1>전체45개 페어: 예측·검증·보류 구조 수정</h1><p class="result"><strong>{headline}</strong> 현재 실거래 코드·설정은 유지했고 주문은 보내지 않았습니다.</p>
<p>자료 종료 {stamp}. 과거에 이미 사용한 기준시장 개발 자료입니다. 시점별로 미래 정답을 차단했지만 연구자가 전체 구간을 처음 보는 독립 테스트는 아닙니다. 후보와 세 비교군은 새 손익을 보기 전에 고정했습니다.</p>
<h2>수정 내용</h2><ol>
<li>모든45개 조합을 동일하게 평가합니다. 기존 확장판의 수익률 상관계수·원시 헤지 비율 임계값에 의한 사전 제외를 없앴습니다. 기존 실행기가 사용하는 양수 헤지 비율 범위와 양쪽 반대 방향 주문 방식은 유지합니다. 모든 가능한 헤지 구조를 지원한다는 뜻은 아닙니다.</li>
<li>각 페어의90일 rolling ridge를 매일 재학습합니다.24시간 수익률 정답이 학습 시점 전에 확정된 자료만 사용합니다.6시간 간격 학습 표본은 서로 겹치며 독립 표본으로 계산하지 않습니다.</li>
<li>과거 모델이 실제 그 시점에 낸 예측만 모아60일 구간에서 예측의 과장을 보정합니다. 하루1회씩, 최소20개의 만기 지난 표본을 사용합니다. 새 모델로 과거 예측을 다시 써 넣지 않습니다.</li>
<li>이미 발표한 보정 예측의 최근30일 결과를 별도 감시합니다. 최소20일, 거래 신호3회, 변화 없음 예측보다 작은 오차, 비용38bp를 뺀 평균 수익이 HAC2 표준오차1배보다 클 때 신규 진입 자격을 줍니다.</li>
<li>현재 예상 수익의 크기에서 추정 불확실성을 차감해 비용38bp＋여유5bp를 넘는지 확인합니다. 마지막에는 실제 호가·양쪽 수량·펀딩·체결비용·위험 한도가 추가로 거래를 거절할 수 있습니다. 조건을 통과해도 매수·매도 체결을 보장하지 않습니다.</li>
<li>보류는 신규 진입에만 적용합니다. 보유 페어의 ID·예측·손절·만기·유료 헤지 조정은 남겨 두며 동시 활성 페어는 기존처럼1개입니다.</li></ol>
<p>표준오차 여유와 임계값은 사전 고정한 견고성 규칙입니다. 독립성이 없는 시계열과45개 비교 전체에 대한 통계적 유의성 보장이나 수익 보장이 아닙니다.38bp 과거 검증 손익은 펀딩·부분 체결·보유 연장을 포함하지 않는 진입 검사용 근사치이며 아래 계좌 재생과 구분됩니다.</p>
<h2>효과 분리</h2><ul><li>incumbent_fixed_two: 현재판의 기존 두 페어.</li><li>automatic_45: 앞선 상관·헤지 필터 확장판.</li><li>expanded_only: 기존 예측과 진입 gate를 전체45개에 적용. 새 사전 필터 없음.</li><li>calibrated_only: 예측 보정과 현재 불확실성 검사 추가, 과거 성과 감시는 생략한 진단군.</li><li>validated_all45: 예측 보정＋최근 과거 성과 감시까지 적용한 사전 지정 주 후보.</li></ul>
<h2>동일 조건 결과</h2><p>우리 예산50,000＋현금 예비50,000, 수익률 분모100,000. 기본 방향 전략·실행 비용·부분 체결·위험 한도 동일. 같은 한도라도 실제 노출은 달라집니다. 기본 방향 전략의 기존 후처리 경제성 문제도 두 군에 동일하게 남겨 이번 페어 변경과 섞지 않았습니다. Q는 모의 산식이며 공식 순위가 아닙니다.</p>
<table><tr><th>기간</th><th>조건</th><th>군</th><th>전체 순손익</th><th>수익률</th><th>Q</th><th>기간 변동성</th><th>완료 페어</th><th>페어 순손익</th><th>위험 위반</th></tr>{''.join(table)}</table>
<p>validation8/1~9/1, comparison9/1~10/15, continuous8/1~10/15, 각 경계12UTC. 기존 보고서의 기간 이름을 유지한 것이며 새 독립 검증이라는 의미는 아닙니다. 순손익은 평가손익을 포함하고 잔여 청산 비용 충당 후 손익은 원본에 별도 보존했습니다.</p>
<h2>최신 판단</h2><p>45개 중 최근 감시 조건을 통과한 페어는 {sum(x['eligible'] for x in latest['pairs'])}개, 자료 마지막 시점의 비영 신규 진입 신호는 {int(entries[-1].sum())}개입니다. 실제 주문 수가 아닙니다. 일별 감시에서 여러 보류 사유가 동시에 발생할 수 있습니다.</p>
<table><tr><th>페어</th><th>관측 일수</th><th>과거 신호 수</th><th>감시 결과</th><th>이유</th><th>최신 방향</th></tr>{''.join(latest_table)}</table>
<h2>PAS와 배포 상태</h2><p>10/19 12UTC 이후 같은 달력 구간에서 기준시장 자료로 생성된 비영 신호는 {len(ii)}개입니다. PAS 시장 전체 체결 재생은 수행하지 않았으며 이 숫자는 PAS 주문·체결·수익이 아닙니다. 새 모듈은 오프라인 연구 후보이며 실거래 신호 생산자와 실행자에는 연결하지 않았습니다.</p>
<h2>검증과 원본</h2><p>모든45개 원래 ridge 예측이 앞선 구현과 일치하고, 앞서 학습했던 기존 gate의 판단도 일치하는지 검사했습니다. 미래 가격·미만기 정답 오염 검사, 과거 예측 덮어쓰기 방지, 롱·숏·보류, 보유 신호 보존 검사 기록은 아래에 있습니다.</p>
<p><a href="protocol.json">사전 고정 규칙</a> · <a href="parity.json">기존 예측 재현</a> · <a href="comparison.json">주 후보 비교</a> · <a href="reference_results.json">18개 신규 재생 결과</a> · <a href="forecast_diagnostics.json">페어별 예측 오차</a> · <a href="monitoring_history.json">매일45개 감시 원장</a> · <a href="pas_signal_audit.json">PAS 기간 신호와 한계</a> · <a href="tests.txt">검사 기록</a></p></html>'''
    (OUT/'report.html').write_text(page)
    save(OUT/'source_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.glob('*.py')})
    save(OUT/'rejection_counts.json',dict(reasons))
    print(headline)
    for r in baselines+rows:
        if r['phase']=='continuous':print(r['arm'],r['scenario'],r['net_pnl'],r['Q'],r['completed_pairs'],r['pair_pnl'])


if __name__=='__main__':main()
