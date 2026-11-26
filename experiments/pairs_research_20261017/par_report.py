"""Append the public-data audit without rewriting the frozen model experiments."""
import gzip
import html
import json
from datetime import datetime, timezone, timedelta

from .research import HERE, ROOT, save, sha
from experiments.complementary_alpha_20261017.research import protected

DEST = ROOT / 'reports/pairs_research_20261017'


def table(headers, rows):
    return '<table><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in headers) + '</tr>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</table>'


def iso(t):
    return datetime.fromtimestamp(t, timezone(timedelta(hours=9))).isoformat(timespec='minutes')


def main():
    capture = sorted((HERE / 'par_history').glob('*/manifest.json'))[-1].parent
    manifest = json.loads((capture / 'manifest.json').read_text())
    plan = json.loads((HERE / 'par_replay/plan.json').read_text())
    result = json.loads((HERE / 'par_replay/results.json').read_text())
    execution = json.loads((HERE / 'par_replay/execution_capacity_audit.json').read_text())
    board = json.loads((capture / 'practice_leaderboard.json').read_text())['response']
    teams = sorted([t for t in board['teams'] if t['ranked']], key=lambda t: t['rank'])
    me = next(t for t in teams if t['name'].upper() == 'SFINX')
    assert board['scoring']['method'] == 'fractional'
    score_errors = []
    for team in teams:
        r, s = team['return'], team['volatility_horizon']
        q = (max(r, 0) + .005) / (abs(r) + 1.5 * s + .05)
        score_errors.append(abs(q - team['q']))
        assert abs(.8 * q - team['score']) < 1e-12
    assert max(score_errors) < 1e-12
    for name, digest in manifest['files'].items():
        assert sha(capture / name) == digest
    for name, digest in plan['sources'].items():
        assert sha(ROOT / name) == digest
    verification = dict(public_data_cutoff_kst=manifest['cutoff_kst'], public_market_count=len(manifest['rows']),
        public_data_fetch_failures=manifest['failures'], protected=protected(),
        candle_proxy_runs=len(result['rows']), proxy_numerical_gates_passed=result['numerical_validation_passed'],
        matched_par_execution_validation_passed=False,
        reason='Volume participation assumption does not describe observed team execution, historical L2 is absent',
        leaderboard_formula_checked_teams=len(teams), leaderboard_formula_max_error=max(score_errors),
        team_rank=me['rank'], hypothetical_jury_points_to_tie_first=400 * (teams[0]['q'] - me['q']),
        live_orders_sent=0, candidate_replaces_live=False,
        historical_source_ends_kst=iso(plan['end_unix']),
        valuation='Historical par traded close and funding-notional proxy; not official oracle equity')
    save(DEST / 'par_verification.json', verification)
    coverage = table(['종목', '시간봉 시작(KST)', '완료 시간봉', '실제 거래 있는 시간봉', '완료 5분봉', '거래 있는 5분봉', '펀딩 관측'],
        [[r['coin'], r['1h']['first_kst'][:16], r['1h']['completed'], r['1h']['with_trades'],
          r['5m']['completed'], r['5m']['with_trades'], r['funding']['observations']] for r in manifest['rows']])
    names = {'first': '기존 1차', 'second': '현재 2차', 'cointegration_ou_1440': '공적분 OU',
             'partial_stopping_1440': '영구·일시 분해 + 최적정지'}
    comparison = table(['전략', '가정', '순손익', '수익률', '평균 총노출', '체결 조각', 'Q 참고값', '잔여 청산비용 추정'],
        [[names[r['policy']], r['scenario'], f"{r['metrics']['net_pnl']:+,.4f}",
          f"{100*r['metrics']['roi']:+.8f}%", f"{100*r['metrics']['mean_gross']:.8f}%",
          r['metrics']['fills'], f"{r['score_proxy']['financial_score']:.8f}",
          f"{r['metrics']['estimated_residual_exit_cost']:.4f}"] for r in result['rows']])
    capacity = table(['종목', '실제 체결 시간대(KST)', '팀 체결 수량', '해당 시간 전체 체결량', '팀의 비중'],
        [[r['coin'], r['hour_open_kst'], r['team_volume'], r['market_volume'], f"{r['fraction']*100:.2f}%"] for r in execution['rows']])
    ranking = table(['연습 순위', '팀', 'Q', '0.8Q (J=0)', 'SFINX가 동점에 필요한 J 차이(100점 환산)'],
        [[r['rank'], r['name'], f"{r['q']:.9f}", f"{r['score']:.9f}",
          f"{400*(r['q']-me['q']):+.6f}"] for r in teams[:5]])
    # Establish why the same outputs do not demonstrate baseline equivalence.
    def run(name):
        with gzip.open(HERE / 'par_replay/runs' / f'{name}-base.json.gz', 'rt') as f:
            return json.load(f)
    first, second = run('first'), run('second')
    assert first['fills'] == second['fills']
    assert first['decisions'][0]['raw_target'] != second['decisions'][0]['raw_target']
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>par 최신 자료·체결 가정·심사 점수 확인</title>
    <style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#17212e}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{border:1px solid #d5dde5;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3f8}}.finding{{background:#fff1df;padding:18px;border-left:5px solid #d48b26}}</style>
    <h1>실제 par 자료 갱신과 비교 가능 범위</h1>
    <p>수집 기준: <b>{manifest['cutoff_kst']}</b>. 10개 종목을 공개 테스트넷 API로 조회했고 실패 요청은 없었습니다. API 원문·요청·응답 시각과 해시를 보존했습니다. 가격·체결량·펀딩을 갱신했지만, 과거 호가 수량을 복구했다는 뜻은 아닙니다.</p>
    <p class="finding"><b>추가 거래종가 비교로 전략 개선을 입증하지 못했습니다.</b> 시간당 시장 거래량의 1%를 체결 한도로 쓰니 서로 다른 1차·2차 목표 주문이 같은 소량으로 잘렸습니다. 페어 후보는 거의 거래하지 못했습니다. 실제 팀 거래는 해당 시간대 시장 거래량의 28.8~74.5%였으므로, 이 참여율 제한을 우리 실제 주문 가능량의 검증된 모델로 사용할 수 없습니다.</p>
    <h2>확보한 자료</h2>{coverage}
    <p>8월 1일부터 시간봉과 펀딩을 요청했지만, 실제 응답은 시간봉이 대체로 10월 1일부터, 펀딩은 9월 30일 03시 KST부터입니다. 완료 시간봉은 10월 17일 01:59:59, 완료 5분봉은 02:34:59까지입니다. 진행 중인 봉은 비교 입력에서 제외했습니다. 거래 수가 0인 봉은 새로운 체결 관측이 아니며 체결 가능 수량을 만들지 않았습니다.</p>
    <p>현재 API 조회로 확보한 약 16~17일은 이번 모델의 21일·60일 형성창보다 짧습니다. 아래 실험은 과거 기준시장에서 고정한 신호를 사용하며, par만으로 학습한 페어 모델이 아닙니다. recentTrades는 종목별 최근 10건 응답이며 전체 과거 체결 원장이 아닙니다. 현재 L2와 오라클 스냅샷도 과거 L2·오라클 이력을 대신하지 않습니다.</p>
    <h2>첫째·둘째·새 후보의 공통 구간 참고 비교</h2>
    <p><b>{iso(plan['start_unix'])} ~ {iso(plan['end_unix'])}</b>. 동일 초기자본 1,000,000과 위험 상한. 이 종료시각은 이미 고정된 신호가 존재하는 마지막 시간이며 오늘까지의 모델 판단을 임의로 연장하지 않았습니다. 신호·위험 공분산은 기준시장, 평가 가격·거래량·펀딩률은 실제 par 자료를 사용합니다.</p>
    <p>양쪽 leg별 1시간 지연, 수수료 9bp, 기본 불리한 가격 10bp·참여율 1%, 스트레스 20bp·0.25%를 적용했습니다. 시장 거래종가를 평가 가격으로 쓰며 펀딩 명목도 종가로 근사합니다. 공식 오라클 평가나 실제 체결 검증, 지갑 실적이 아닙니다. 미완료 당일 수익률은 R에만 반영하고 완료일 표본 표준편차에서는 제외했습니다.</p>{comparison}
    <p>기존 두 전략의 최초 목표 비중은 달랐지만 실제 모의 체결 152개가 정확히 같았습니다. 거래량 제한이 차이를 가렸기 때문입니다. 페어 후보의 기본 평균 총노출은 약 0.000005%와 0.000047%이고 스트레스에서는 체결이 없었습니다. 따라서 Q가 현금 수준 0.1에 가깝다는 것을 예측력이나 수익성 개선으로 해석하지 않습니다. 체결이 없는 후보는 순위 자격도 없습니다.</p>
    <h2>체결 가정과 실제 지갑 대조</h2>{capacity}
    <p>공개 지갑 체결 11조각·주문 4개를 해당 시간의 시장 캔들 거래량과 대조했습니다. 수량은 동일 종목 기준으로 비교했습니다. 실거래 비중이 높았다는 이유로 앞으로도 75%를 항상 체결할 수 있다고 가정하지 않습니다. 이 자료는 ‘고정 1% 제한이 실제 우리 주문을 설명하지 못한다’는 증거이며 미래 깊이를 알려주지는 않습니다.</p>
    <p>다음 실행 검증에는 주문 직전 호가의 가격·수량과 실제 부분체결·취소·청산 가능성을 연결해야 합니다. 참여율 숫자를 올려서 과거 수익을 좋게 만드는 것을 개선으로 채택하지 않습니다. 기존 실시간 전략과 봉인된 미래 실험은 그대로 유지했습니다.</p>
    <h2>Q와 J의 영향</h2>{ranking}
    <p>연습 최종 자료 생성시각: {board['generated_at']}, 평가 종료: {board['to']}. 적격 42팀의 Q와 Score를 현재 fractional 공식으로 다시 계산해 일치함을 확인했습니다. SFINX는 5위입니다.</p>
    <p><b>ΔScore = 0.8ΔQ + 0.2ΔJ</b>. 이 연습 수치에 가상의 심사 점수를 붙이면, SFINX와 1위의 Q 차이 {teams[0]['q']-me['q']:.9f}는 J 차이 {4*(teams[0]['q']-me['q']):.9f}, 즉 100점 환산 약 {400*(teams[0]['q']-me['q']):.2f}점으로 상쇄됩니다. 이는 실제 심사 점수 예측이나 본대회 순위 예측이 아닙니다.</p>
    <p>About의 심사 항목은 방법론·AI/ML 활용의 깊이·독창성·엄밀성·재현성·명확성입니다. 2~4쪽 기술 보고서를 최소 3명이 독립 심사하며 알고리즘 거래 여부는 코드 설명과도 연결됩니다. 보고서 작성은 중요하지만 구현·실험의 품질을 대체하지 않습니다. 현재 About에는 이전 min–max 설명이 남아 있어 리더보드 공식과의 문서 불일치도 남아 있습니다.</p>
    <p><a href="https://perpetual-alpha26.github.io/leaderboard#practice">공식 리더보드</a> · <a href="https://perpetual-alpha26.github.io/about">공식 심사 항목</a> · <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint">API 문서</a></p>
    <p><a href="report.html">논문·구현·64개 기준시장 실험 보고서</a> · <a href="{capture/'manifest.json'}">공개 자료 원장</a> · <a href="{HERE/'par_replay/results.json'}">8개 종가 참고 비교</a> · <a href="par_verification.json">검증 결과</a></p></html>'''
    (DEST / 'par_data_audit.html').write_text(body)
    main_report = DEST / 'report.html'
    text = main_report.read_text()
    if 'par_data_audit.html' not in text:
        link = '<p><b>추가 확인:</b> <a href="par_data_audit.html">10월 17일 par 자료 갱신·체결 가정 대조·Q/J 영향 분석</a></p>'
        text = text.replace('<h2>같은 기간 비교</h2>', link + '<h2>같은 기간 비교</h2>')
        main_report.write_text(text)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
