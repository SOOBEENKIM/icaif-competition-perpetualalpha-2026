"""Publish every comparison, including scale controls and rejected follow-up."""
import csv,html,json
from datetime import datetime,timezone,timedelta
from .research import HERE,ROOT,OUT,sha,save


def table(headers,rows):
    esc=lambda x:html.escape(str(x))
    return '<table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table>'


def main():
    r=json.loads((OUT/'results.json').read_text());d=json.loads((OUT/'dynamic/results.json').read_text())
    dest=ROOT/'reports/q_execution_20261017';dest.mkdir(exist_ok=True)
    names={'first_strategy':'첫 번째 전략','point_x1':'운용 중 두 번째 전략','point_x4':'규모만 4배인 대조군','partial_6h_x4':'분할 조절·목표 규모 4배 후보',
        'causal_7d_budget':'최근 7일 손익 기반 규모 조절 (기각)','fixed_matched_exposure':'평균 노출을 맞춘 고정 규모 대조군'}
    def tab(rows):
        return table(['전략','순수익률 R','일별 표본 SD','기간 변동성 σ','Q','Score (J=0)','최대낙폭','평균 투자 노출'],[
            [names.get(x['policy'],x['policy']),f"{x['metrics']['roi']*100:+.6f}%",f"{x['official_formula']['daily_sample_std']*100:.6f}%",
             f"{x['official_formula']['volatility_horizon']*100:.6f}%",f"{x['official_formula']['financial_score']:.9f}",
             f"{x['official_formula']['score']:.9f}",f"{x['metrics']['max_drawdown_magnitude']*100:.6f}%",f"{x['metrics']['mean_gross']*100:.3f}%"] for x in rows])
    later=[x for x in r['rows'] if x['phase']=='later' and x['scenario']=='base']
    stressed=[x for x in r['rows'] if x['phase']=='later' and x['scenario']=='execution_stress']
    funding=[x for x in r['rows'] if x['phase']=='later' and x['scenario']=='funding_stress']
    losses=[x for x in r['rows'] if x['phase']=='later_second_half' and x['scenario']=='base']
    rejected=[x for x in d['rows'] if x['phase']=='later' and x['scenario']=='base']
    full=r['rows']+d['rows'];csvrows=[]
    for x in full:
        csvrows.append(dict(phase=x['phase'],policy=x['policy'],scenario=x['scenario'],
            **{k:x['metrics'][k] for k in ('roi','daily_std_sample','max_drawdown_magnitude','fees','turnover','fills','mean_gross')},
            **{k:x['official_formula'][k] for k in ('completed_days','volatility_horizon','financial_score','score')},
            engineering_pass=all(x['gates'].values())))
    with (dest/'comparisons.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    base=next(x for x in later if x['policy']=='point_x1');cand=next(x for x in later if x['policy']=='partial_6h_x4');scale=next(x for x in later if x['policy']=='point_x4')
    sensitivity=(.05-.005+1.5*.000140176287)/(1.5*(.000140176287+.005))
    summary=dict(created_at_kst=datetime.now(timezone(timedelta(hours=9))).isoformat(),selected=r['selection'],
        primary_comparison=later,execution_stress=stressed,adverse_funding_stress=funding,loss_subperiod=losses,
        rejected_followup=rejected,historical_runs=len(full),baseline_reproduced_exactly=r['baseline_reproduced_exactly'],
        historical_independence=False,orders_sent=0,live_strategy_changed=False,
        assessment='Higher historical Q and net return than current original-size incumbent, mostly from increased exposure. Staged execution is not uniformly superior to the same-size control. Additional dynamic budget rule failed. Prospective par validation required.',
        mean_budget_scale=4.,actual_mean_gross_candidate=cand['metrics']['mean_gross'],
        Q_improvement_relative=cand['official_formula']['financial_score']/base['official_formula']['financial_score']-1,
        fee_reduction_versus_same_budget=1-cand['metrics']['fees']/scale['metrics']['fees'],
        future_comparison=dict(start_kst='2026-10-17 21:00',end_kst='2026-10-19 21:00',release_kst='2026-10-20 09:00',
            arms=['incumbent','scale_only_control','staged_candidate'],source='Existing read-only par public collector; no sealed raw data opened'))
    save(dest/'summary.json',summary)
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>SFINX Q 기준 개선 후보</title>
<style>body{{font:16px/1.7 system-ui;max-width:1250px;margin:30px auto;padding:0 24px;background:#f7fafc;color:#172b3a}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px;margin:20px 0}}td,th{{border:1px solid #d5e1ea;padding:9px;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#e9f0f6}}.note{{padding:16px;background:#fff3d9;border-left:4px solid #b97b14}}a{{color:#1764a3}}</style>
<h1>SFINX: 공식 Q 기준의 실행·투자 규모 개선</h1><p>{summary['created_at_kst']}</p>
<p class="note"><strong>기존판보다 역사적 순수익과 Q가 높은 후보를 구현했습니다. 실제 계좌는 교체하지 않았습니다.</strong>
이후 42일 기준시장 비교에서 순수익률 +0.275782% → +0.930250%, Q 0.128502540 → 0.163502124입니다.
이는 독립 실거래 성과가 아니며, 이익 증가의 상당 부분은 투자 규모 확대에 따른 것입니다. 손실 구간의 손실과 최대낙폭도 커졌습니다.</p>
<h2>사용자 팀과 평가 기준</h2>
<p>사용자가 확인한 팀명은 SFINX입니다. 공식 Practice 최종 결과는 42개 순위 대상 팀 중 5위, PnL +0.01401763%,
기간 변동성 0.01401763%, Q 0.102088010, Score 0.081670408입니다. 공식 마감은 10월 16일 21시 KST이며 아래 역사적 비교 수치와 기간·시장·거래가 다릅니다.
<a href="https://perpetual-alpha26.github.io/leaderboard#practice">공식 리더보드</a></p>
<p>모든 새 비교는 12:00 UTC / 21:00 KST의 계좌가치로 완료된 일별 수익률을 구하고 σ = 표본 SD × √n,
Q = (max(R,0)+0.005)/(abs(R)+1.5σ+0.05), Score = 0.8Q로 평가합니다. 심사 점수는 만들어 넣지 않았습니다.</p>
<p>SFINX의 현재 값 근처에서 동일한 소수 단위 변화에 대한 수익률의 유리한 효과는 변동성 감소 효과의 약 {sensitivity:.2f}배입니다.
이는 국소 민감도이며 모든 수익·변동성 수준에서 고정된 가중치가 아닙니다. 수익률과 σ가 각각 0.01%포인트 커지면 Q는 0.102088 → 0.103560입니다.</p>
<h2>구현한 변화</h2>
<p>추세＋7일 되돌림·Chronos 보정 신호는 유지합니다. 목표 비중을 기존의 1·2·4배로 비교하되, 총노출 60%, 순노출 40%,
종목별 15% 또는 150,000 USDPR, 예상 일별 변동성 0.5%, 최대낙폭 중단 8%의 기존 제한을 그대로 적용했습니다.
4배는 거래소 레버리지 설정을 4x로 바꾼다는 뜻이 아닙니다. 자본은 모든 비교에서 100만으로 동일합니다.</p>
<p>선정된 후보는 6시간마다 실제 보유 수량과 목표 수량 사이 일반 조절분의 절반을 반영합니다.
방향 전환·전량 종료 및 위험 때문에 필요한 감축에는 지연을 적용하지 않습니다. 기존 방향별 위험 상한도 최종 수량에 다시 적용합니다.
이전 주문이 전량 체결됐다고 가정하지 않고 다음 판단에서 실제 보유 수량을 사용하며, 상태 복원 후 중복 판단을 막습니다.</p>
<p><a href="https://arxiv.org/html/2502.04284v1">Ma·Smith (2025), On the Effect of Alpha Decay and Transaction Costs on the Multi-period Optimal Trading Strategy</a>의
신호 지속성과 거래 비용을 함께 고려하는 관점을 참고했습니다. 논문의 단일자산 이진 포지션 MDP를 그대로 재현한 구현은 아닙니다.
새 예측 모델을 학습했다고 주장하지 않습니다.</p>
<h2>선정과 검증 범위</h2>
<p>8월 1일 21시~9월 1일 21시 KST에서 4개 실행 방식 × 3개 규모 × 3개 비용 조건을 비교했습니다.
기본 조건과 실행비용 악화 조건에서 기존판 대비 Q 개선의 작은 쪽을 최대화하는 규칙으로 <code>partial_6h_x4</code>를 선정한 뒤 고정했습니다.
이후 9월 1일~10월 13일 및 두 개의 21일 구간에서 첫 번째 전략·현재 전략·규모만 같은 대조군과 비교했습니다.</p>
<p>이 과거 자료는 이전 연구에서 이미 열어봤으므로 새 독립 평가라고 부르지 않습니다.
가격은 기준시장 시간봉, 주문은 다음 시간봉의 가격과 거래량 한도, 기본 왕복 비용은 수수료·가격 영향 합계 약 38bp입니다.
par의 실제 과거 호가나 실제 체결 성과를 복원한 결과가 아닙니다.</p>
<h2>이후 42일 — 기본 조건</h2>{tab(later)}
<p>새 후보의 평균 투자 노출은 14.714%로 현재 전략의 3.941%보다 큽니다. 수익 증가를 예측력 개선으로 해석할 수 없습니다.
같은 규모 대조군과 비교하면 후보의 수수료는 1,361.97 → 1,110.06으로 약 {summary['fee_reduction_versus_same_budget']*100:.1f}% 낮아졌고,
최대낙폭은 2.0066% → 1.8357%로 줄었지만, 기본 조건의 Q는 0.165993 → 0.163502로 조금 낮았습니다.</p>
<h2>실행 비용·체결 한도가 불리한 조건</h2>{tab(stressed)}
<p>수수료 9bp에 편도 가격 영향 20bp, 기준시장 거래량 참여 한도 0.25%를 적용했습니다. 이 조건에서는 분할 조절 후보가
규모만 같은 대조군보다 순수익과 Q가 높았습니다. 따라서 실행 개선은 비용 조건에 민감하며 전 구간 우위를 확보한 것은 아닙니다.</p>
<h2>큰 손실을 숨기지 않은 21일 구간</h2>{tab(losses)}
<p>9월 22일~10월 13일의 손실 구간에서 현재판 −0.322496% 대비 후보 −1.255496%였습니다.
투자 확대는 이익과 손실을 모두 확대합니다. 전체 42일의 Q가 높다는 사실만으로 손실 구간의 위험이 해소되지는 않습니다.</p>
<h2>시간당 불리한 펀딩 1bp를 추가한 스트레스</h2>{tab(funding)}
<p>이 조건은 체결 악화와 별도로 펀딩이 계속 불리하게 흐르는 극단 조건입니다. 후보의 Q가 현재 전략보다 낮아집니다.
이를 실제 par의 정상 펀딩률이라고 가정하지 않았으며, 후보가 이 위험에도 우월하다고 주장하지 않습니다.</p>
<h2>추가 동적 규모 조절 — 기각</h2>{tab(rejected)}
<p>고정 4배의 손실 확대를 확인한 뒤, 최근 7일간 기준 전략의 비용 차감 손익이 양수일 때 4배, 그 외에는 1배로 조절하는 규칙을 별도 후속 실험으로 고정했습니다.
각 판단에는 직전 시간까지의 자료만 사용하고 미래 입력 변경이 과거 판단에 영향을 주지 않는지 확인했습니다.
그러나 이 규칙의 이후 42일 Q는 0.100647로 현재판 0.128503보다 낮았습니다. 평균 노출을 맞춘 고정 규모 대조군도 이기지 못해 채택하지 않습니다.</p>
<h2>판단의 불확실성</h2>
<p>3일 블록을 사용하는 대응 부트스트랩에서 후보와 현재판의 Q 차이 95% 구간은 약 [−0.0289, +0.1517]로 0을 포함합니다.
반복해서 사용한 과거 자료의 기술적 불확실성 계산이며, 유의한 미래 우위를 입증한 검정이 아닙니다.</p>
<h2>실제 par 호가로 이어지는 모의 검증</h2>
<p>현재 실행 중인 공개 호가 수집기를 변경하지 않고, 원래 규모·4배 규모 대조군·분할 조절 후보의 별도 평가를 등록합니다.
측정 구간은 <strong>10월 17일 21시~10월 19일 21시 KST</strong>, 기존 봉인 조건을 지켜 결과 열람은 <strong>10월 20일 09시 이후</strong>입니다.
각 가상 계좌는 같은 100만에서 출발하며 실제 지갑 주문을 보내지 않습니다.</p>
<p>현재 호가 수량, 지연 후 새로 받은 호가, 부분 체결, 미체결, 실제 정산 펀딩과 mark 계좌가치를 사용합니다.
오라클로 매매됐다고 가정하지 않습니다. 21시의 표본 두 개 수익률은 통계 계산을 가능하게 할 뿐, 장기 수익성 증명이 되지 않습니다.
체결 상대의 반응이나 자신의 시장 영향을 완전히 재현할 수도 없습니다. 후보의 자동 실거래 승격은 없습니다.</p>
<p>원래 실행 구조 373개 파일과 예약 평가 의존성 345개 파일은 해시를 유지했습니다. 기존 전략은 변경하지 않았고,
첫 비교 72회와 후속 비교 24회를 모두 보존했습니다. 수익이 좋았던 표만 골라 남기지 않았습니다.</p>
<p><a href="comparisons.csv">전체 96개 비교 CSV</a> · <a href="summary.json">요약 데이터</a> ·
<a href="../../experiments/q_execution_20261017/protocol.json">최초 고정 조건</a> ·
<a href="../../experiments/q_execution_20261017/dynamic_protocol.json">기각한 후속 규칙</a> ·
<a href="../../experiments/q_execution_20261017/artifacts/selection.json">후보 선정 기록</a> ·
<a href="../../experiments/q_execution_20261017/artifacts/prospective/plan.json">새 모의 검증 등록</a></p></html>'''
    (dest/'report.html').write_text(body)
    files=[*HERE.glob('*.py'),*HERE.glob('*.json'),OUT/'results.json',OUT/'selection.json',OUT/'dynamic/results.json',dest/'summary.json',dest/'report.html',dest/'comparisons.csv']
    save(dest/'manifest.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print(dest/'report.html')


if __name__=='__main__':main()
