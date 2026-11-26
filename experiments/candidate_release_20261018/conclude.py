"""Seal the reproducibility result and preserve the distinction from par profit."""
import json
from pathlib import Path
import shutil
import time
import zipfile
from html.parser import HTMLParser

from .reproduce import sha, save, verify
from experiments.conditional_completion_pairs_20261017.research import verify as verify_sources
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'
BUNDLE=ROOT/'deliverables/sfinx_pairs_candidate_20261018_final'


def main():
    manifest=json.loads((BUNDLE/'manifest.json').read_text());verify(BUNDLE,manifest)
    result=json.loads((OUT/'reproduction_final/results.json').read_text())
    assert result['exact_accounts']==42 and result['refitted_models']
    assert all(r['exact_full_account'] and r['common_engine_financial_path_exact'] for r in result['rows'])
    fits=sum(r['gate_fits'] for r in result['model_checks']);assert fits==526
    incumbent='first_dense_hourlyhedge_decisiongate_24'
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in result['rows']}
    advantages=[]
    for phase in ['validation','comparison','continuous']:
        for scenario in ['base','stress']:
            candidate=lookup['reference',phase,incumbent,scenario]
            for baseline in ['first_strategy','second_strategy']:
                control=lookup['reference',phase,baseline,scenario]
                assert candidate['net_pnl']>control['net_pnl'] and candidate['financial_score']>control['financial_score']
                advantages.append(dict(phase=phase,scenario=scenario,baseline=baseline,
                    pnl_gain=candidate['net_pnl']-control['net_pnl'],q_gain=candidate['financial_score']-control['financial_score']))
    modified_logs=[r['id'] for r in result['rows'] if not (r['common_engine_order_log_exact'] and r['common_engine_decision_log_exact'])]
    adapter=ROOT/'experiments/decision_gate_publication_20261018/artifacts'
    sources=json.loads((adapter/'conclusion_frozen.json').read_text())['sources'].copy();verify_sources(sources)
    original_plan=ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json'
    sidecar=adapter/'plan.json'
    for plan in [original_plan,sidecar]:verify_sources(json.loads(plan.read_text())['sources'])
    # Preserve failed packaging diagnostics outside the user-facing deliverables.
    failed=OUT/'failed_bundles';failed.mkdir(exist_ok=True)
    for name in ['sfinx_pairs_candidate_20261018','sfinx_pairs_candidate_20261018_v2']:
        path=ROOT/'deliverables'/name
        if path.exists():shutil.move(str(path),failed/name)
    archive=ROOT/'deliverables/sfinx_pairs_candidate_20261018.zip'
    if archive.exists():raise FileExistsError('Do not overwrite the delivered archive')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as handle:
        for name in sorted([*manifest['files'],'manifest.json']):
            handle.write(BUNDLE/name,'sfinx_pairs_candidate/'+name)
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        assert len(handle.namelist())==len(manifest['files'])+1
        import hashlib
        for name,expected in manifest['files'].items():
            assert hashlib.sha256(handle.read('sfinx_pairs_candidate/'+name)).hexdigest()==expected
    next_action=('Retain this reproduced 24h candidate and the separately sealed future sidecar. A bounded next '
        'performance hypothesis is a 72h cost-sensitive entry gate using the same two pairs, existing frozen 72h '
        'expert inputs, model form, capital and risk constraints: spreading fixed round-trip cost over a longer '
        'forecast may change qualified opportunities. First verify this exact gate/horizon combination has not '
        'already been tested, audit every horizon-dependent pricing/funding/holding term, and freeze one candidate '
        'before historical replay. Include the 24h incumbent and both native strategies; do not sweep horizons or '
        'modify the registered future study. No claim of improvement before after-cost PnL/Q results.')
    conclusion=dict(at_unix=time.time(),classification='progress_offline_refit_and_42_account_reproduction_verified',
        previous_goal_turn_classification='progress_gate_adapter_and_later_start_study_registered',
        retained_improved_candidate=incumbent,exact_archived_accounts=42,exact_common_engine_financial_paths=42,
        refitted_daily_gate_models=fits,common_engine_log_differences=modified_logs,
        reference_advantages=advantages,bundle=str(BUNDLE.relative_to(ROOT)),archive=str(archive.relative_to(ROOT)),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        requirements_audit=[
            dict(requirement='Cost-adjusted PnL and Q improvement over first and second on reused reference history',status='verified',evidence='12 positive paired advantages; 42 exact historical accounts'),
            dict(requirement='Reproduce paired learning and preserve causal label maturity',status='verified',evidence='526 daily gate refits; all mapped state fields exactly equal; strict mature labels'),
            dict(requirement='Reflect par costs, funding, separate leg fills and paid exits',status='implemented_and_tested',evidence='Separate par candle rows, received-book adapter, 9 adapter tests, prior account/lifecycle tests'),
            dict(requirement='Separate candidate choice from later evaluation and historical reuse from prospective evidence',status='verified_distinction',evidence='Archived selection preserved; original and later-start plans unchanged; future outcomes not evaluated'),
            dict(requirement='Demonstrate that the improvement transfers to actual par book execution',status='unverified',evidence='No completed profitable candidate book lifecycle; future comparison still pending'),
            dict(requirement='No unsupported profit claim or automatic live replacement',status='verified',evidence='Offline audited execution, no orders, no signer or live-service changes')],
        failed_packaging_diagnostics=['First package omitted a public-query helper import; added its source dependency.',
            'Second attempt required identical order-intention logs across different historical engine versions. Final reproduction uses each original engine and additionally checks actual financial paths under the common engine; no mismatch was silently dropped.',
            'Runtime import audit found existing torch/pandas dependencies; recorded pinned installed versions. Fresh environment installation was not tested.'],
        new_par_profit_evidence=False,independent_future_evaluated=False,protected=protected(),next_action=next_action,
        orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    report=ROOT/'reports/candidate_release_20261018/report.html';report.parent.mkdir(exist_ok=False)
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>SFINX 개선 후보 재현 결과</title>',
        '<style>body{max-width:1050px;margin:36px auto;padding:0 18px;font:16px/1.65 sans-serif;color:#18263b}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd5e1;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}pre{background:#eef4fc;padding:16px;white-space:pre-wrap}</style>',
        '<h1>개선 후보: 모델 재계산과 42개 계좌 재현</h1>',
        '<p>유지한 24시간 페어 예측·진입 선별 후보의 결과가 재현됐다. 과거 입력에서 직접 예측과 526개 일별 진입 모델을 재계산해 모든 매핑 필드가 같음을 확인했다. 작업 폴더 밖 /tmp에서 격리된 Python 경로로 실행해 원래 프로젝트의 모듈을 우연히 찾는 의존성도 제거했다.</p>',
        '<p>42개 계좌의 당시 실행 버전 기록을 모두 정확히 재현했다. 별도로 같은 후보 실행 엔진에 기존 전략을 넣었을 때도 실제 체결·계좌 평가·수수료·펀딩·일별 수익률·손익·Q가 모두 같았다. 과거 버전 사이에 달랐던 미체결 주문 의도 기록은 차이 목록에 남겼으며, 이를 전체 기록 일치로 숨기지 않았다.</p>',
        '<h2>기준시장 이후 비교 구간</h2><p>2026년 9월 1일–10월 15일, 12:00 UTC 경계, 초기 자본 1,000,000. 비용과 펀딩 차감 후 평가손익이며 미청산 손익을 포함한다. 재사용 과거 자료이고 실제 par 실거래 성과가 아니다.</p>',
        '<table><tr><th>전략 / 조건</th><th>순손익</th><th>수익률</th><th>기간 σ</th><th>Q</th></tr>']
    for arm,label in [('first_strategy','첫째'),('second_strategy','둘째'),(incumbent,'개선 후보')]:
        for scenario in ['base','stress']:
            m=lookup['reference','comparison',arm,scenario]
            page.append(f"<tr><td>{label} / {scenario}</td><td>{m['net_pnl']:+,.2f}</td><td>{m['roi']*100:+.6f}%</td><td>{m['volatility_horizon']*100:.6f}%</td><td>{m['financial_score']:.8f}</td></tr>")
    page+=['</table><p>8월과 8월–10월 연속 구간도 같은 결과로 재현됐다. 겹치는 기간을 독립적인 성공 횟수로 세지 않는다. par 및 연장 구간의 불리한 결과도 comparison.csv에 함께 포함했다.</p>',
        '<h2>실행</h2><p><a href="../../deliverables/sfinx_pairs_candidate_20261018.zip">소스·입력·재현 명령 ZIP</a> · <a href="../../deliverables/sfinx_pairs_candidate_20261018_final/README.html">설명</a> · <a href="../../experiments/candidate_release_20261018/artifacts/reproduction_final/comparison.csv">42개 결과 CSV</a></p>',
        '<pre>python reproduce.py --output results</pre><p>기본 명령이 페어 모델을 다시 학습한다. --cached-models를 붙이면 고정된 모델 상태로 계좌 재생만 한다. requirements.txt는 검증에 사용한 Python 3.11 환경의 패키지 버전을 기록한다. 별도 새 환경에서 패키지를 설치하는 절차 자체는 실행하지 않았다.</p>',
        '<p>기존 첫째·둘째의 당시 목표 비중을 고정된 비교 입력으로 사용한다. 이 묶음이 기존 Chronos 기준 모델 전체를 처음부터 재훈련한다고 주장하지 않는다. 모든 소스·입력·원래 결과의 해시와 출처를 포함했고, 네트워크 호출과 하위 프로세스 실행을 금지한 상태로 검증했다. 실거래·서명·배포 명령은 없다.</p>',
        '<h2>확인한 범위와 남은 문제</h2><p>기준시장 백테스트 개선과 재현은 확인됐다. 실제 par 호가에서 이 개선 후보가 진입부터 청산까지 수익을 냈다는 증거는 아직 없다. 단기 과거 호가의 비용 조건과 계좌 평가 가능 범위를 확인했으며, 별도 미래 평가 계획은 보존했다. 이 한계를 공식 제출 금지 조건으로 취급하지 않는다.</p>',
        '<p>다음 성능 실험은 현재 24시간 모델을 보존한 채 같은 비용 학습 방식을 72시간 예측에 적용할 근거를 점검하는 것이다. 같은 두 페어·자본·위험 한도를 유지하고, 만기·펀딩·보유 규칙을 먼저 감사한 후 하나의 후보로 고정한다. 여러 만기를 사후 탐색하거나 미래 평가 계획을 바꾸지 않는다.</p></html>']
    report.write_text('\n'.join(page));HTMLParser().feed(report.read_text())
    save(HERE/'work_state.json',dict(status=conclusion['classification'],report=str(report.relative_to(ROOT)),
        next_action=next_action,retained_improved_candidate=incumbent,goal_complete=False,no_progress_streak=0))
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.log'
           and 'failed_bundles' not in p.parts and 'reproduction' not in p.parts and 'reproduction_v2' not in p.parts]
    files += [BUNDLE/'manifest.json',archive,report,*[BUNDLE/name for name in manifest['files']]]
    sources.update({str(path.relative_to(ROOT)):sha(path) for path in files});verify_sources(sources)
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),
        excluded='mutable runner logs and preserved failed packaging diagnostics'))
    print('SEALED',len(sources),'sources;',fits,'gate refits; 42 archived and common-engine financial paths;',
          len(modified_logs),'intention-log differences; archive',archive.stat().st_size,'bytes',flush=True)


if __name__=='__main__':main()
