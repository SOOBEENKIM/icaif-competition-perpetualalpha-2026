"""Evidence report; no strategy selection or parameter mutation."""
import gzip
import html
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from .study import OUT, ROOT, read, save, sha, verify
from .selection import PAIRS
from .signals import state_for_arm


def main():
    p=verify();rows=read(OUT/'reference_results.json')
    assert len(rows)==18
    baseline=[r for r in rows if r['arm']=='incumbent_fixed_two']
    candidate=[r for r in rows if r['arm']=='automatic_45']
    comparisons=[]
    for r in candidate:
        b=next(x for x in baseline if x['phase']==r['phase'] and x['scenario']==r['scenario'])
        comparisons.append(dict(phase=r['phase'],scenario=r['scenario'],pnl_delta=r['net_pnl']-b['net_pnl'],
            Q_delta=r['Q']-b['Q'],pnl_improved=r['net_pnl']>b['net_pnl'],Q_improved=r['Q']>b['Q']))
    history=read(OUT/'selection_history.json');latest=history[-1]
    model=read(OUT/'models'/f"{latest['source']}.json")
    state=dict(np.load(OUT/'states.npz',allow_pickle=False));t=state.pop('times')
    auto=state_for_arm(state,'automatic_45')
    pas=(t>=1792411200)&(t<1792666800)
    nonzero=auto['economic_available']&(auto['economic_action']!=0)&auto['directions'].astype(bool)
    ii,kk=np.where(nonzero&pas[:,None])
    # Do not reuse a base-only PAS engine now that real pair signals exist.
    pas_audit=dict(start_utc='2026-10-19T12:00:00Z',end_exclusive_utc='2026-10-22T11:00:00Z',
      hourly_nonzero_pair_signals=len(ii),source='reference-model reconstruction, NOT historical PAS fills',
      examples=[dict(source=int(t[i]),pair=list(PAIRS[k]),action=int(auto['economic_action'][i,k])) for i,k in zip(ii,kk)],
      can_reuse_base_only_PAS_simulator=len(ii)==0,full_PAS_backtest_completed=False,
      limitation='No fake base-only PnL: new signals require paired depth/partial-fill replay. Reference PnL is not PAS PnL.')
    save(OUT/'pas_signal_audit.json',pas_audit)
    save(OUT/'comparison.json',dict(primary='automatic_45',comparisons=comparisons,
      all_windows_improve=all(x['pnl_improved'] and x['Q_improved'] for x in comparisons),
      independent_test=False,live_changed=False,orders_sent=0,
      decision='DO_NOT_DEPLOY: functionality verified, performance promotion criterion failed'))
    diagnosis=[]
    for scenario in ('base','stress'):
        with gzip.open(OUT/'reference'/f'automatic_45_continuous_{scenario}.json.gz','rt') as f:r=__import__('json').load(f)
        m=r['metrics'];diagnosis.append(dict(scenario=scenario,price_move_pnl=m['price_move_pnl'],fees=m['fees'],
          execution_cost=m['execution_cost'],funding=m['funding'],net_pnl=m['net_pnl'],
          completed_pairs=m['completed_pairs'],aborted_pairs=m['aborted_pairs'],
          residual_gross=m['residual_gross'],pnl_after_liquidation_reserve=m['pnl_after_liquidation_reserve']))
    save(OUT/'loss_diagnosis.json',diagnosis)
    counts=Counter(tuple(pair) for day in history for pair in day['selected_pairs'])
    manifest={str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.glob('*.py')}
    save(OUT/'candidate_source_manifest.json',manifest)
    stamp=datetime.fromtimestamp(int(t[-1]),timezone(timedelta(hours=9))).isoformat()
    summary=[]
    for r in rows:
        if r['arm']=='fixed_two_rebuilt':continue
        summary.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['phase'],r['scenario'],r['arm'],
            f"{r['net_pnl']:+,.2f}",f"{100*r['return_fraction']:+.4f}%",f"{r['Q']:.6f}",
            f"{100*r['volatility']:.4f}%",r['completed_pairs'],r['risk_breach_bars']])+'</tr>')
    explanations={
      'insufficient_fresh_history':'새 거래가 있는 공통 관측 부족',
      'insufficient_return_rows':'수익률 관측 부족',
      'weak_or_negative_return_relationship':'한 구간 이상에서 수익률 관계가 약하거나 음수',
      'hedge_beta_outside_supported_range':'헤지 비율이 기존 주문 구조의 지원 범위 밖',
      'low_reference_leg_liquidity':'기준시장 거래량 부족',
      'unstable_hedge_beta':'기간별 헤지 비율 불안정'}
    listing=[]
    for r in model['selection']['pairs']:
        k=r['pair_index'];reason='통과; 예측·현지 체결 검사는 별도' if r['selected'] else '; '.join(explanations.get(x,x) for x in r['reasons'])
        listing.append(f"<tr><td>{'/'.join(r['pair'])}</td><td>{'포함' if r['selected'] else '제외'}</td><td>{html.escape(reason)}</td><td>{counts[tuple(r['pair'])]}</td><td>{int(auto['economic_action'][-1,k])}</td></tr>")
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>45개 조합 자동 페어 선정 구현과 비교</title><style>body{{max-width:1120px;margin:36px auto;padding:0 22px;font:16px/1.75 system-ui;color:#15283d;background:#fbfcfe}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d3dfeb;padding:8px;text-align:left}}th{{background:#e8f0f8}}a{{color:#185ba3}}code{{overflow-wrap:anywhere}}.result{{background:#fff2e8;border-left:4px solid #af482d;padding:18px}}</style>
<h1>45개 조합 자동 페어 선정 — 구현·비교 결과</h1>
<p class="result"><strong>자동 선정 구조와 기존 체결기 호환 검사를 구현했다. 그러나 이 첫 선정 규칙의 성과는 기존판보다 나빠 실거래 교체를 권하지 않는다.</strong> 기존 실행 코드·프로필을 그대로 유지했으며 주문을 보내지 않았다. 45개를 본다는 사실만으로 더 좋은 페어를 고르는 것이 입증되지는 않는다.</p>
<p>자료 종료: {stamp}. 실제 현재 호가나 체결을 뜻하지 않는다. 선행 정의한 하나의 후보를 평가했으며 결과를 본 뒤 임계값을 바꿔 승자를 고르지 않았다. 과거 재사용 자료이므로 독립 테스트가 아니다.</p>
<h2>구현한 경로</h2><ol>
<li>10개 종목의 45개 비순서 조합을 고정된 ID로 모두 생성한다. 특정 종목 이름으로 후보를 배제하지 않는다.</li>
<li>매일00UTC에 지난90일을30일씩 나눠 공통 유효 관측80% 이상, 각 구간 상관계수0.2 이상, 원래 헤지 비율0.25~4, 기간 간 비율 변화2배 이하를 검사한다. 기준시장 양쪽 시간당 거래대금 중앙값4000 이상도 검사한다. 상관성·양의 베타를 보는 예측형 상대가치 전략이며 공적분 차익거래의 증명이 아니다.</li>
<li>통과한 모든 조합에 기존24시간 rolling ridge와 비용 민감 진입 gate를 적용한다. 같은90일·6시간 샘플·정답 만기 조건이다. 기본 방향 전략과 자본·위험 한도는 바꾸지 않는다. 모델이 학습에 쓰는 미래 수익률은 그 시점 전에 이미 만기가 지난 것만 허용한다.</li>
<li>실제 주문 계획기는 현재 양쪽 호가·수량·수수료·펀딩·위험 한도를 다시 검사해 기대 순현금이 큰 실행 가능한 제안을 고른다. 동시 활성 페어는 기존처럼1개다. 선정에서 빠져도 보유 페어의 예측·청산·손절·헤지 관리 자료를 삭제하지 않는다.</li></ol>
<p>현재 자료에서 형성 조건을 통과한 조합은 {len(latest['selected_pairs'])}개이다. 예를 들어 ETH/HYPE, EWY/EWJ, SP500/USDEUR도 평가 대상이며 조건에 따라 포함된다. 마지막 소스 시점의 통과 조합 중 비영 진입 gate는 {int(nonzero[-1].sum())}개다. ‘후보 선정’과 ‘당장 주문 허용’은 다르다.</p>
<h2>기존판 재현과 기능 검증</h2>
<p>기존 BTC/ETH·BTC/HYPE만 허용한 재구성 대조군은 원래 예측과 진입 판단을 재현했다. 최대 예측 차이는 약6.94×10⁻¹⁸이며,6개 기간·비용 조건의 계좌 순손익도 기존판과1e-6 이내에서 일치했다. 따라서 새 예측 모델의 차이를 자동 선정의 효과로 잘못 계산하지 않았다.</p>
<p>유효한 새 후보가 고정 두 조합 밖에서도 생성되는지, 미래 가격·미만기 정답 차단, 자료 순서/시각, 선정 취소 뒤 보유 관리, 종목 ID 일치, 중복 신호 거절을 검사했다. EWY/EWJ의 양쪽 진입부터 수수료를 지불한 만기 청산까지 기존 체결 상태기계로 합성 호가를 사용해 확인했다. 이 테스트는 실거래 성과가 아니다.</p>
<h2>동일 조건 백테스트</h2>
<p>기준시장 과거 자료, 우리 예산50,000＋현금 예비50,000, 같은 비용·위험·체결 지연·부분 체결 가정. 수익률 분모는100,000이며 σ는 일별 수익률로 계산한 기간 변동성이다. Q는 모의 산식 결과이며 공식 순위가 아니다. 현재 기본 전략의 알려진 후처리 경제성 문제도 두 군에서 동일하게 유지하여 이번 변경의 효과와 섞지 않았다. 오늘 수정된 실시간 실행 상태기계 전체의 완전 재생을 주장하지 않는다.</p>
<table><tr><th>기간</th><th>조건</th><th>전략</th><th>순손익 USDP</th><th>수익률</th><th>Q</th><th>σ</th><th>완료 페어</th><th>위험 위반 봉</th></tr>{''.join(summary)}</table>
<p>validation은8/1~9/1, comparison은9/1~10/15, continuous는8/1~10/15이다. 기간별 시작 시각은 원본 expected.json의12UTC이다. 순손익은 잔여 보유의 평가액을 포함하며 종료 비용 충당 후 결과도 원본에 보존했다.</p>
<p>기본 조건의 연속 구간에서 완료 페어가5개→28개로 늘었으나 페어 부문은 +592.53→−1,990.48로 악화됐다. 수수료도215.62→437.68로 늘었다. 전체 가격 변동 손익 자체도 음수이므로 거래비용만의 문제가 아니다. 기본 방향 부문도 공유 자본·체결 경로에 따라 달라진다. 같은 위험 한도라도 실제 노출은 같지 않다.</p>
<h2>PAS에서 확인한 것과 확인하지 않은 것</h2>
<p>본대회 시작 이후 보유 자료71시간에서 새 후보의 비영 방향 신호는59개(시간×페어)다. 이것은 주문·체결 수가 아니다. 기존 두 페어의 신호가0이어서 사용했던 기본 전략 전용 PAS 재생기로 이 후보의 PnL을 계산하면 페어 거래를 누락하므로 허용하지 않았다. <strong>확장판의 PAS 전체 손익은 아직 검증하지 않았고, 기준시장 손익을 PAS 결과로 제시하지 않았다.</strong></p>
<h2>선정 목록과 제외 이유</h2><p>맨 오른쪽은 자료 마지막 시간의 진입 방향이다.0은 보류이며, 포함 횟수는 평가한 일별 형성 구간의 횟수다. 과거의 유동성 통과가 현재 PAS 호가를 대신하지 않는다.</p>
<table><tr><th>페어</th><th>당일 형성 선정</th><th>이유</th><th>일별 포함 횟수</th><th>마지막 진입 방향</th></tr>{''.join(listing)}</table>
<h2>파일과 상태</h2>
<p>코드: <code>auto_pair_selection/selection.py</code>(45개 형성 검사), <code>signals.py</code>(시점 엄격한 모델·보유용 신호), <code>staging.py</code>(종목 매핑·주문 계획 호환), <code>study.py</code>(동일 조건 비교). 설치된 실거래 실행자는 이 새 모듈을 아직 사용하지 않는다. 실제 교체 전에는 새 신호 생산자·프로필·실행자의 통합 인계가 별도로 필요하다.</p>
<p><a href="protocol.json">사전 고정 규칙</a> · <a href="incumbent_parity.json">기존 예측 재현</a> · <a href="reference_results.json">18개 비교 결과</a> · <a href="comparison.json">교체 판단</a> · <a href="selection_history.json">선정 이력</a> · <a href="pas_signal_audit.json">PAS 구간 신호 감사</a> · <a href="loss_diagnosis.json">비용과 가격손익 분해</a> · <a href="tests.txt">테스트 기록</a></p>
</html>'''
    (OUT/'report.html').write_text(text)
    print('Report written; promotion criterion failed; no live change.')


if __name__=='__main__':main()
