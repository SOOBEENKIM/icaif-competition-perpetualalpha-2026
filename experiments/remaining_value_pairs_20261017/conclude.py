"""Preserve failed holding candidates separately from the improved incumbent."""
import json,time
from pathlib import Path
from .research import ROOT,HERE,OUT,CONTROL,save,sha,verify,protected


def main():
    result=json.loads((OUT/'results.json').read_text());diag=json.loads((OUT/'diagnostics.json').read_text())
    verify(json.loads((OUT/'frozen.json').read_text())['sources'])
    verify(json.loads((OUT/'diagnostics_frozen.json').read_text())['sources'])
    comparisons=[]
    for r in diag['rows']:
        assert r['decomposition']['net_pnl']<0
        comparisons.append({k:r[k] for k in ['phase','scenario','kind','metrics','decomposition','forecast_summaries']})
    assert result['selection']['selected'] is None
    conclusion=dict(at_unix=time.time(),classification='progress_remaining_value_failed_candidates_diagnosed',
        retained_improved_candidate=CONTROL,reference_comparisons=comparisons,
        previous_observed_improvement_preserved=True,selected=None,
        finding='Both additional holding candidates lower PnL and Q relative to decisiongate in all four reference conditions. '
                'Renewal-only later base price-path contribution rises 316.5271, while fee and impact cost rises 467.2367. '
                'The holding model also makes erroneous early-close decisions; these cannot be explained by costs alone.',
        structural_finding='A positive renewal at preview prices can be vetoed because the original model issued wider price limits. '
                           'The new forecast must constrain the executable budget; rejecting the entire renewal can force '
                           'close-and-reentry costs. Investigate a separate intersection of the two models\' price limits.',
        forecast_limit=diag['distinction'],
        par='All new par account paths are unchanged in metrics; no new holding reviews or profitable completed pairs were created.',
        model_fits=sum(p['models'] for p in result['preparation']),tests_passed=10,new_account_replays=24,exact_controls=12,
        protected=protected(),orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,
        historical_independence=False,goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    report=ROOT/'reports/remaining_value_pairs_20261017/report.html'
    page=report.read_text()+'''<h2>검토 결과: 기존 개선판 유지</h2>
<p>두 새 후보는 모든 reference 기간·비용 조건에서 decisiongate보다 PnL·Q가 낮았다. 8월에서 어느 것도 승격하지 않았고 이후 결과로 다시 선정하지 않는다. 이전 개선 후보의 성과를 무효로 만드는 결과가 아니다.</p>
<p>9월 이후 기본 조건에서 연장 거부판은 +3701.1718(Q 0.128039), 조기 청산 추가판은 +3129.2758(Q 0.120328)로 기존 +3852.7636(Q 0.130237)보다 낮다. 불리한 조건은 두 후보 모두 +2692.0931로 기존 +3305.7125보다 낮다.</p>
<p>연장 거부판의 이후 기본 조건 가격 손익은 기존보다 +316.5271 늘었지만 체결 비용·수수료가 +467.2367 늘고 펀딩이 -0.8821 변해 최종 순손익은 -151.5918 감소했다. 조기 청산판은 가격 손익도 -285.8133 감소했다. 45개 보유 판단의 고정 수량 가격 변화 예측 오차는 96.9774bp로, 가격 변화 없음 94.9011bp보다 컸다. 거래에서 조건부로 선택된 겹치는 표본이므로 일반 예측력 검정이 아니다.</p>
<p>9월 22일 연장 거부는 새 예측의 예상 보유 현금가치가 양수여도 기존의 넓은 지정가 한도에서의 최악 조정 비용 때문에 발생했다. 이후 청산과 재진입이 늘었다. 이를 해결할 다음 별도 실험은 현재 예상 체결가에서 예산이 유효하면 두 모델의 주문 가격 한도를 교집합으로 좁히는 것이다. 기존 한도를 넓히거나 특정 날짜를 예외 처리하지 않는다.</p>
<p>새 모델 학습 12,668개, 신규 계좌 24개, 원래 계좌 12개 정확 재현, 검사 10개를 완료했다. par의 보유 판단은 발생하지 않아 성과가 원래와 같았으며 실제 par 수익성을 입증하지 못했다.</p>'''
    report.write_text(page)
    cross=ROOT/'experiments/pairs_development_status_20261017/comparisons/remaining_value.json';assert not cross.exists();save(cross,conclusion)
    save(HERE/'work_state.json',dict(status=conclusion['classification'],report=str(report.relative_to(ROOT)),
        retained_improved_candidate=CONTROL,new_account_replays=24,exact_controls=12,tests_passed=10,
        next_action='Separate budget-intersection candidate is implemented in remaining_budget_pairs_20261017; evaluate without changing frozen v1 or incumbent.',
        orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,goal_complete=False,no_progress_streak=0))
    files=[Path(__file__),OUT/'conclusion.json',report,cross,OUT/'diagnostics.json']
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('DONE: failed holding revisions preserved; incumbent improvement retained',flush=True)


if __name__=='__main__':main()
