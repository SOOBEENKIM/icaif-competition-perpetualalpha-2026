"""Evidence ledger for failed forecast families and the new venue experiment."""
from collections import Counter
from datetime import datetime
import gzip
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from experiments.cost_horizon_20261016.research import data
from experiments.complementary_alpha_20261017.research import protected
from experiments.pairs_research_20261017.research import save, sha

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/pairs_development_20261017'


def main():
    OUT.mkdir(exist_ok=True)
    first = ROOT / 'experiments/pairs_forecast_20261017/artifacts'
    matched = ROOT / 'experiments/pairs_horizon_20261017/artifacts'
    venue = ROOT / 'experiments/paired_venue_20261017/artifacts'
    forecasts = json.loads((first / 'results.json').read_text())
    horizons = json.loads((matched / 'results.json').read_text())
    audit = json.loads((venue / 'results.json').read_text())
    plan = json.loads((venue / 'forward/plan.json').read_text())
    latest = sorted(venue.glob('snapshot_*.json'))[-1]
    snapshot = json.loads(latest.read_text())
    _, _, _, market = data()
    def table(headers, rows):
        return '<table><thead><tr>' + ''.join('<th>'+html.escape(str(h))+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    def read(path):
        with gzip.open(path, 'rt') as f: return json.load(f)
    comparisons = [read(first/'runs'/f'comparison-{name}-base.json.gz') for name in ('first','second','tree_72')]
    comparisons.append(read(matched/'runs/comparison-tree_72-base.json.gz'))
    names = ['기존 첫 번째', '기존 두 번째', '트리 72시간 예측', '트리 72시간·보유기간 일치']
    compare_table = table(['전략','순손익 USD','수익률','일별 표본표준편차','기간 변동성 σ','Q','평균 총노출'],
        [[name, f"{r['metrics']['net_pnl']:,.2f}", f"{100*r['official']['return_fraction']:.5f}%",
          f"{100*r['official']['daily_sample_std']:.5f}%", f"{100*r['official']['volatility_horizon']:.5f}%",
          f"{r['official']['financial_score']:.6f}", f"{100*r['metrics']['mean_gross']:.2f}%"] for name,r in zip(names,comparisons)])
    attr_rows, attribution = [], []
    for name, r in zip(names, comparisons):
        impact = sum(f['signed_qty']*(f['price']-market.close[market.exact(f['time']),f['asset']]) for f in r['fills'])
        mt = r['metrics']; gross = mt['net_pnl'] + mt['fees'] - mt['funding'] + impact
        record = dict(name=name, fixed_fill_price_pnl_before_impact=gross, impact=impact, fees=mt['fees'], funding=mt['funding'], net_pnl=mt['net_pnl'])
        attribution.append(record)
        attr_rows.append([name, *[f'{x:,.2f}' for x in (gross,-impact,-mt['fees'],mt['funding'],mt['net_pnl'])]])
    selection_rows = []
    for name in [r['name'] for r in forecasts['selection']['ranking']]:
        rs = [next(r for r in forecasts['rows'] if r['policy']==name and r['phase']=='selection' and r['scenario']==s) for s in ('base','execution_stress')]
        selection_rows.append([name, *[f"{r['metrics']['net_pnl']:,.2f}" for r in rs]])
    venue_rows = [[r['horizon_hours'],r['scenario'],r['evaluated'],r['positive_forecast_count'],f"{r['maximum_predicted_bps']:.2f}",'미측정'] for r in audit['summaries']]
    current_rows = [[ '/'.join(r['proposed']['coins']),r['proposed']['horizon_hours'],r['scenario'],
        f"{r['proposed']['gross_notional']:,.2f}",f"{r['proposed']['predicted_net_bps']:.2f}",f"{r['proposed']['predicted_net_pnl']:,.2f}"]
        for r in snapshot['rows'] if r['proposed']['predicted_net_bps'] > 5]
    kst = lambda value: datetime.fromtimestamp(value, ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
    checks = protected()
    for filename,digest in plan['sources'].items():
        if sha(ROOT/filename)!=digest: raise ValueError('Frozen future source changed: '+filename)
    links = [
      ('선형·비선형·GRU 분해를 참고한 논문; 테스트 기간 분포로 문턱을 정하는 부분은 차용하지 않음', 'https://onlinelibrary.wiley.com/doi/10.1002/for.70074'),
      ('알파 유효기간과 거래 비용의 무거래 구간 연구; 우리 72시간 보유 비교는 별도 실험', 'https://arxiv.org/html/2502.04284v1'),
      ('2026 펀딩·베이시스 위험 분해; 주요 DEX 실험이 합성 데이터임을 확인', 'https://link.springer.com/article/10.1007/s42521-026-00213-3'),
      ('현재 대회 리더보드의 Q·Score 정의', 'https://perpetual-alpha26.github.io/leaderboard#practice')]
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 전략 개발·검증 기록</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;line-height:1.65;color:#172735}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}td,th{{border:1px solid #ccd6de;padding:8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#eef3f7}}.verdict{{background:#fff3d5;padding:18px;border-left:5px solid #ba7600}}code{{background:#edf1f5}}a{{color:#12629b}}h2{{margin-top:36px}}</style>
<h1>페어 전략 개발·검증 기록</h1><p>작성 시각: {kst(__import__('time').time())}. 연구 주문 전송 0건. 기존 운용 코드와 이전 예약 실험은 보존.</p>
<div class="verdict"><strong>교체 판단: 아직 통과한 후보 없음.</strong> 이번 추가 작업에서 14개 예측 후보와 보유기간 일치 후보 2개를 시험했고, 총 56개 계좌 백테스트를 완료했다. 기존 두 전략보다 높은 순손익과 Q를 동시에 입증하지 못했다. 오라클·펀딩 페어는 새 공개 호가 자료로 검증을 이어간다.</div>
<h2>같은 기준시장 과거 구간의 비교</h2><p>2026-09-01 21:00부터 10-13 21:00 KST, 초기 자본 각각 1,000,000. 42개 일별 수익률. <strong>기준시장 시간봉과 가정한 체결 비용의 개발용 결과이며 실제 par 체결 성과가 아니다.</strong> 같은 과거 자료를 이미 여러 번 사용했으므로 독립적인 검증도 아니다. 순손익과 Q를 우선 비교하고 노출·위험 차이를 별도 표시했다.</p>
{compare_table}
<p>Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), σ=일별 표본표준편차×√n. 수익률은 소수가 공식 입력이다. 위 Q는 공식을 적용한 비교값이며 공식 참가팀 점수가 아니다. 첫째·둘째는 마지막에 보유잔고가 남을 수 있으므로 페어의 완전 청산과 다른 점, 예상 잔여 청산 비용을 원시 결과에 함께 남겼다.</p>
<h2>왜 실패했는가</h2>
{table(['전략','가격 손익','가정한 체결 가격 영향','수수료','펀딩','순손익'],attr_rows)}
<p>위 분해는 이미 발생한 가상 체결의 수량·시각을 고정한 회계 분해다. 거래 비용을 없앤 전략의 새 백테스트가 아니다. 트리 후보는 7~8월 가격 손익이 양수였지만 거래 비용이 더 컸고, 9~10월에는 가격 손익도 음수였다. 예측기를 복잡하게 만들거나 거래 문턱만 낮추는 것으로 해결되지 않았다.</p>
<p>72시간 예측을 6시간마다 새 신호 부호가 바뀔 때 청산하던 구조를 별도 버전에서 수정했다. 진입 후 헤지 수량을 고정하고, 계좌·스프레드 위험 점검을 유지하면서 예측 만기까지 보유했다. 주문 지연 때문에 청산이 학습 목표보다 한 시간 늦어지는 시점도 수정했다. 구현 정합성은 개선됐지만 과거 수익은 개선되지 않았다. 기존 원본과 실패 결과를 덮어쓰지 않았다.</p>
<h2>14개 예측 후보의 선정 구간 결과</h2><p>7~8월, 과거 90일 학습, 7/1·8/1·9/1·10/1 재학습. 각 학습 시점 전에 만기가 끝난 레이블만 사용했다. 릿지·트리·GRU·잔차 결합·앙상블·추세·되돌림 × 24/72시간. 스트레스는 슬리피지 증가와 거래량 한도 감소가 같이 적용되어 거래 자체가 달라질 수 있다. 일부 스트레스 수익이 더 좋은 것은 비용 증가가 유리하다는 뜻이 아니다.</p>
{table(['후보','기본 순손익','실행 스트레스 순손익'],selection_rows)}
<h2>실제 par 호가로 검증한 실행 구조</h2><p>임의의 거래량 1% 제한으로 par 체결을 복원하지 않는다. 양쪽 현재 호가를 걸어 진입·청산 가능한 공통 수량을 계산하고, 지연 뒤 새로 받은 호가에서 각각 체결한다. 잔량이 같은 비율로 체결되면 유지하고, 비율이 다르면 더 많이 체결된 쪽만 유상 감축한다. 청산 목표와 실제 청산 완료를 구분하며 잔여 포지션은 실패로 남긴다. 호가가 그대로인 동안 이미 소비한 물량이 자동으로 다시 생기지 않는다.</p>
<p>도착 전 체결 금지, 최신 호가 부재 시 미체결, 부분 체결 복구, 펀딩 부호·실보유 수량, 원장, 12:00 UTC 점수, 자료 누락 실패 판정을 포함한 12개 테스트가 통과했다. 가상 가격을 사용한 통합 테스트는 실행 검증용이며 수익성 근거가 아니다.</p>
<h2>오라클 괴리와 펀딩 페어 기회</h2><p>이미 사용했던 약 4.48시간의 공개 호가 기록에서 가격 괴리 50% 복귀와 현재 펀딩의 지속을 가정하고, 유리한 예상 펀딩은 절반만 반영했다. 미래 가격·펀딩·청산 호가가 관측되지 않아 아래는 예상 기회 진단이다. 양의 예상값을 실현수익으로 세지 않았다. 한 번의 기회가 연속된 세 시점에 보일 수 있으므로 건수도 독립 거래 횟수가 아니다.</p>
{table(['보유 시간','시나리오','유효 시점·페어 수','비용+5bp 통과','최대 예상 bp','실현수익'],venue_rows)}
<p>최신 공개 호가 확인: {kst(snapshot['at_ms']/1000)}. 당시에 문턱을 넘은 후보:</p>
{table(['페어','시간','시나리오','호가 기준 총규모 USDPR','예상 순 bp','예상 순손익 USDPR'],current_rows)}
<p>72시간 펀딩이 현재와 비슷하게 유지되어야 성립하는 추정이다. EWY와 SP500은 같은 자산이 아니므로 지역·시차·가격 위험이 남는다. 베타 헤지는 완전한 시장중립이나 무위험 차익거래가 아니다. 스트레스 조건의 문턱까지 넘지 못하는 현재 스냅샷을 교체 근거로 쓰지 않는다.</p>
<h2>이후 기간을 고정한 모의 비교</h2><p>공개 데이터 수집 서비스 <code>icaif-paired-venue-paper.service</code>를 시작했다. 비교 시작 {kst(plan['paper_start_unix'])}, 종료 {kst(plan['capture_end_unix'])}. 현재는 원시 자료만 모으며, 종료 전 중간 성과를 보고 규칙을 바꾸지 않는다. 400개 소스 의존성을 고정했다.</p>
<p>첫째·둘째·72시간 오라클/펀딩 페어를 각각 별도 100만 자본 계좌에서 기본/스트레스 조건으로 비교한다. 수수료 9/12bp, 지연 1/3초, 표시 수량 100/50%를 적용한다. 모든 계좌를 같은 종료 시점 전에 청산한다. 두 조건에서 기존 두 전략보다 양의 순손익과 Q가 높고, 일별 수익률 5개 이상·완료 페어 2회 이상·잔여 포지션 및 실행 오류 없음이 필요하다. 일주일 결과만으로 장기 수익성을 단정하거나 실제 운용을 자동 교체하지 않는다.</p>
<p>보호 확인: 기존 실거래 소스 {checks['active_protected_sources']}개, 이전 예약 의존성 {checks['reserved_dependencies']}개 및 새 실험 400개 해시 일치. 새 실험은 지갑 키나 서명 클라이언트를 사용하지 않는다.</p>
<h2>근거·원시 결과</h2><ul>'''
    body += ''.join(f'<li><a href="{url}">{html.escape(label)}</a></li>' for label,url in links)
    for label, path in [('예측 48회 결과',first/'results.json'),('보유기간 일치 8회 결과',matched/'results.json'),('실제 호가 기회 진단',venue/'results.json'),('최신 공개 원본 영수증',latest),('고정된 이후 비교 계획',venue/'forward/plan.json')]:
        body += f'<li><a href="../../{path.relative_to(ROOT)}">{label}</a></li>'
    body += '</ul></html>'
    (OUT / 'report.html').write_text(body)
    save(OUT / 'manifest.json', dict(at=__import__('time').time(), report_sha256=sha(OUT/'report.html'),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [first/'results.json',matched/'results.json',venue/'results.json',latest,venue/'forward/plan.json']},
        attribution=attribution, protected=checks, development_improvement=False, live_orders_sent=0))
    print(OUT / 'report.html')


if __name__ == '__main__': main()
