"""Document adapter verification and the separately registered future window."""
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from html.parser import HTMLParser

from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.complementary_alpha_20261017.research import protected
from .forward import ROOT, HERE, OUT, ORIGINAL, ready, sha, save


def stamp(value):
    return datetime.fromtimestamp(value, timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')


def main():
    plan = json.loads((OUT/'plan.json').read_text())
    audit = json.loads((OUT/'old_book_audit.json').read_text())
    tests = (HERE/'test_results.txt').read_text()
    assert 'Ran 9 tests' in tests and tests.rstrip().endswith('OK')
    assert audit['exact_ungated_accounts'] == 34
    assert sha(ORIGINAL) == plan['original_plan_sha256']
    verify(plan['sources'])
    verify(json.loads((OUT/'old_book_frozen.json').read_text())['sources'])
    assert plan['created_at_unix'] < plan['paper_start_unix']
    try:
        ready(plan,time.time())
    except ValueError as exc:
        assert 'remain sealed' in str(exc)
    else:
        raise AssertionError('This preparation report expects the prospective window still sealed')
    valid = sum(not r['outputs']['decision_gate']['fatal'] for r in audit['records'])
    next_action = ('Preserve the retained reference-improved two-pair candidate and this sealed publication/book '
        'adapter. The future sidecar is registered but not evaluated. Next audit the full candidate artifact boundary '
        'and its declared limitations for a reproducible offline submission bundle, without retuning on the sealed '
        'window, changing live trading or treating the sidecar release as an official submission prerequisite. '
        'Evaluate this exact sidecar only after the original release; record incomplete coverage and failed exits '
        'rather than shorten the window or impute fills.')
    conclusion = dict(at_unix=time.time(), classification='progress_retained_gate_publication_adapter_verified_and_sidecar_registered',
        previous_goal_turn_classification='progress_graph_gate_ablation_completed',
        retained_improved_candidate='first_dense_hourlyhedge_decisiongate_24', tests_passed=9,
        exact_ungated_old_book_checks=34, complete_account_quote_checks=valid,
        insufficient_account_quote_checks=len(audit['records'])-valid,
        observed_new_pair_entries=audit['pair_sessions'], full_strategy_quote_pnl=None,
        historical_pnl_q_improvement_preserved=True, prospective_registered=True,
        prospective_evaluated=False, start_kst=stamp(plan['paper_start_unix']),
        end_kst=stamp(plan['capture_end_unix']), release_kst=stamp(plan['outcome_release_unix']+60),
        original_plan_unchanged=True, candidate_plan_sha256=sha(OUT/'plan.json'),
        sources=len(plan['sources']), protected=protected(), next_action=next_action,
        orders_sent=0, live_strategy_changed=False, reserved_raw_read=False,
        goal_complete=False, no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    report = ROOT/'reports/decision_gate_publication_20261018/report.html'
    report.parent.mkdir(exist_ok=False)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>개선 후보의 수신 시각 기준 모의 운용 연결</title>
<style>body{{max-width:1000px;margin:36px auto;padding:0 18px;font:16px/1.7 sans-serif;color:#18263b}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cbd5e1;padding:9px;text-align:left}}.note{{background:#eef4fc;padding:16px}}</style>
<h1>개선 후보의 예측·진입·보유·청산 연결</h1>
<p class="note">기존 두 전략보다 기준시장 백테스트 PnL과 Q가 높았던 두 페어 개선 후보를 보존하고, 수신된 신호와 실제 과거 호가를 사용하는 모의 실행 코드에 연결했다. 검사 9개를 통과했고 별도 미래 평가를 고정했다. 실제 par 수익 개선을 새로 확인한 결과는 아니다.</p>
<h2>유지한 성과</h2>
<p>2026년 9월 1일–10월 15일, 12:00 UTC 경계, 초기 1,000,000의 기준시장 백테스트다. 비용·수수료·펀딩 차감 및 미청산 평가손익을 포함한다.</p>
<table><tr><th>전략</th><th>기본 조건 PnL</th><th>Q</th><th>불리한 조건 PnL</th><th>Q</th></tr>
<tr><td>첫째</td><td>+459.53</td><td>0.09219</td><td>−457.10</td><td>0.08437</td></tr>
<tr><td>둘째</td><td>+572.88</td><td>0.09512</td><td>−492.97</td><td>0.08540</td></tr>
<tr><td>유지한 개선 후보</td><td>+3,852.76</td><td>0.13024</td><td>+3,305.71</td><td>0.12447</td></tr></table>
<p>이전에 재사용한 개발 자료의 개선이다. 실제 계좌 성과 또는 독립적인 미래 검증이라고 표시하지 않는다. <a href="../decision_gate_pairs_20261017/report.html">원래 결과</a></p>
<h2>구현과 검사</h2>
<p>일별 계수는 실제 해당 시각에 공개된 입력으로만 갱신한다. 라벨 만기가 학습 시점보다 엄격히 이전인지 검사한다. 자정 자료가 없으면 나중 자료로 당시 모델을 복원하지 않는다. 직접 수익률 예측과 SPO+ 행동 점수를 별도 필드로 보존했다. 행동 점수를 미래 가격으로 쓰지 않는다.</p>
<p>허용된 과거 자료의 시간별 접두 구간에서 직접 예측값·미래 가격·진입 행동·행동 점수가 저장된 배치 모델과 일치했다. 중복·수정·미래 자료, 누락된 자정, 학습 표본 부족도 검사했다. 추가 모델 계산은 검사 구간에서 0.1초 미만이었다. 모의 연결은 기존과 같은 5초의 별도 계산 지연을 적용한다.</p>
<p>계좌는 하나이며 양쪽 주문의 수신·체결·수수료는 별도로 처리한다. 진입 행동 필터는 신규 주문에만 적용한다. 필터가 대기하거나 없더라도 기존 보유의 직접 예측은 유지돼 연장할 수 있고, 청산 시점에는 비용을 지불해 정리한다. 합성 호가 검사는 이 상태 전이를 검증하는 용도이며 실수익 증거가 아니다.</p>
<p>기존 첫째·둘째·필터 없는 실행 경로가 원래 코드와 같은 체결·현금·수량을 만드는지 검사했다. 기존 운용 파일과 등록된 네 전략 비교 계획은 변경하지 않았다.</p>
<h2>실제 과거 par 호가 점검</h2>
<p>17개 시각 × 기본/불리한 조건에서 필터 없는 원래 코드와 새 연결 코드의 34개 결과가 정확히 같았다. 이 중 {valid}개는 계좌 평가에 필요한 10개 시장 호가가 있었고, {34-valid}개는 일부 시장만 기록된 테이프여서 평가 불가로 남겼다. 누락된 호가를 합성하지 않았다.</p>
<p>필터 없는 실제 진입 점검 24개 방향 중 23개는 전체 비용보다 예상 이익이 작았고, 1개는 최소 수량 반올림 후 페어 비율을 충족하지 못했다. 유지한 진입 필터도 24개 방향을 허용하지 않았다. 두 경우 모두 추가 진입 0건이다. 전체 24시간 청산·수익률·Q는 미측정이다. 이것은 전체 전략 성과가 아니라 과거 상태를 넣은 신규 진입 연결 검사다.</p>
<h2>별도 미래 평가 등록</h2>
<p>시작: <b>{stamp(plan['paper_start_unix'])}</b><br>종료: <b>{stamp(plan['capture_end_unix'])}</b><br>기존 원자료 공개 제한을 유지한 평가 가능 시각: <b>{stamp(plan['outcome_release_unix']+60)}</b></p>
<p>같은 시작 자본·위험 한도·수신 호가에서 첫째, 둘째, 직접 예측, 진입 선별을 더한 개선 후보를 비교한다. 시작 시각은 구현 고정 이후이며 과거부터 등록했던 것처럼 소급하지 않았다. 원래 수집을 재사용하고 추가 수집기나 실거래 주문을 시작하지 않았다. 수집 상태 메타데이터는 갱신됐지만 가격·체결·부분 성과가 담긴 예약 원자료는 읽지 않았다.</p>
<p>평가 기간은 연습 시장 par의 운영 이후를 포함할 수 있다. 시장 비활성·자료 누락·미확인 체결·청산 실패는 평가 무효 또는 잔량으로 기록하며, 유리한 시간대만 골라 기간을 바꾸지 않는다. 이 평가 일정을 대회 제출 전제 조건으로 사용하지 않는다. 완료 페어 수는 근거의 양으로 보고하며 임의의 공식 제출 제한으로 만들지 않는다.</p>
<p>유지한 후보를 실거래로 자동 교체하지 않았다. 다음 작업은 이 후보의 재현 가능한 오프라인 제출 묶음과 구현 경계를 점검하는 것이다.</p></html>'''
    report.write_text(page); HTMLParser().feed(page)
    save(HERE/'work_state.json',dict(**conclusion,report=str(report.relative_to(ROOT))))
    sources = plan['sources'].copy()
    paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.log']+[report]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths});verify(sources)
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),
        excluded='mutable runner logs',original_plan_sha256=sha(ORIGINAL)))
    print('SEALED',len(sources),'sources; future start',conclusion['start_kst'],flush=True)


if __name__=='__main__':main()
