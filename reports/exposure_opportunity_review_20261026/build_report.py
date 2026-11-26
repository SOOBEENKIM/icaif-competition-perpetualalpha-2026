"""Build an evidence report, not an executable trading policy."""
import hashlib
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_leverage.study import live_manifest


def read(path): return json.loads(path.read_text())
def stamp(t): return datetime.fromtimestamp(t, ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
def dump(name, value): (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


a=read(OUT/'current_audit.json'); h=read(OUT/'historical_audit.json')
fit=read(ROOT/'reports/pas_economic_allocation_20261022/fit.json')
comparisons=read(ROOT/'reports/adaptive_leverage_20261026/comparison.json')
assert len(comparisons)==12 and all(r['pnl_change']==0 for r in comparisons)
assert h['exact_archived_reproduction'] and h['eligible_asset_rounds']==0
spec=dict(
    state='reviewed_design_not_implemented_or_deployed',
    goal='Choose incremental PAS exposure from validated net edge; improve net PnL and financial Q, rather than force capital utilization.',
    live_margin_ceiling=5,live_sizing_unchanged=True,orders_sent=0,
    first_development_scope='One additive sizing sleeve for existing eligible base targets; no direction change, no forced new instrument, no relaxation of mandatory risk cuts.',
    sequence=[
        'Keep strategy base target, protected maximum, outstanding reservations and optional extra inventory separately in the ledger.',
        'Calibrate expected executable PAS return over the proposed holding period, including the external-market return and the change in PAS basis. Do not assume current basis must converge.',
        'Forecast PAS funding from only past completed PAS settlements. Treat uncertain receipts conservatively and stress adverse payments; do not substitute reference-market funding.',
        'Estimate incremental entry/exit cost as a function of quantity, using actual fills, visible depth, partial fills, fees and latency. A current-book round trip is only a diagnostic, not a future exit forecast.',
        'Allocate extra notional per eligible asset or per complete pair, accounting for uncertainty and covariance with both base and added positions. Require positive incremental utility after lot rounding and all constraints.',
        'Order reductions and repairs first; preserve protected quantities, pair hedge ratio, shared ownership, pending orders and the 50000 USDP teammate reservation.',
        'Keep 6-hour base decisions initially. Recheck quote, remaining edge, capacity and signal validity before each existing execution attempt. Extra inventory has an explicit expiry/reduction policy and a no-trade band.',
        'Select minimum sufficient margin leverage only after the final target is fixed, respecting 5x, exchange tiers and shared-market locks.'
    ],
    proposed_primary_horizon_hours=24,
    horizon_status='Proposed 24h residual value with 6h reevaluation; not a measured optimal horizon. Baseline retains its 168h signal. Freeze one candidate horizon before validation; test overrun through exits.',
    objective='Marginal expected PAS price/funding PnL minus quantity-dependent entry/exit costs, uncertainty penalty and incremental portfolio risk; optimize only the permitted extra sleeve.',
    first_stage_caps='Retain current gross60%, net40%, asset15%/7500, predicted daily volatility0.5% and 90% sizing headroom. Any later cap relaxation is a separately disclosed risk experiment.',
    inference_constraint='A statistically weak forecast does not become strong by lowering the transaction-cost threshold or undoing attenuation.',
    comparisons=[
        'Current live policy, 50000 USDP ours plus same 50000 reserve/teammate assumptions',
        'Uniform exposure expansion control under the same hard caps, parameter fixed using validation only',
        'Opportunity-dependent expansion under the same caps',
        'Matched-exposure or matched-risk diagnostic, with matching decided on validation, to separate timing/selection gains from more risk'
    ],
    data_protocol=[
        'Timestamp availability, labels, quotes, settlement events and source venue explicitly. Past midpoint returns alone are not executable PAS profits.',
        'Train then validate then test in chronological order; include only training labels whose complete return/exit horizon matures before the next boundary.',
        'The observed 19–22 October PAS history and all previously queried outcomes remain development data. Inventory historical usage before describing any past interval as unused.',
        'Freeze candidate code, thresholds and period before one final evaluation. If outcomes prompt a revision, relabel that test as development rather than repeatedly claim independent validation.',
        'Use day/block-level uncertainty summaries for overlapping signals and correlated assets. Do not count 120 symbol decisions as 120 independent profit observations.'
    ],
    acceptance=[
        'Zero look-ahead, timing, accounting, precision, ownership and hard-risk violations; pair and mandatory-reduction behavior preserved.',
        'Net PnL and daily-12:00-UTC Q improve over the baseline on the declared primary evaluation; compare PnL and Q at the same scoring cutoff.',
        'Report terminal marked PnL, terminal liquidation reserve and full-day Q separately; Q is unavailable without enough daily samples.',
        'Report drawdown, realized volatility, exposure, turnover, funding attribution, rejected/partial fills and tail stress. Increased volatility is not an automatic rejection if Q and PnL improve within preset risk limits.',
        'Compare adverse cost/latency/funding scenarios without retuning on their test outcomes. Reject a fragile gain erased by realistic execution costs.',
        'A gain over baseline alone supports larger exposure, not proof of better opportunity timing; the uniform-size and matched-risk controls test that separate claim.',
        'Record unsigned shadow decisions and reconcile them to observed quotes and fills before considering live connection; no performance guarantee.'
    ],
    sources=[
        'https://web.stanford.edu/~boyd/papers/cvx_portfolio.html',
        'https://hyperliquid.gitbook.io/hyperliquid-docs/trading/margining',
        'https://perpetual-alpha26.github.io/leaderboard'
    ])
dump('development_spec.json',spec)

rows=[]
for x in a['assets']:
    costs=[('%.2f'%v['immediate_roundtrip_bp']) if v['immediate_roundtrip_bp'] is not None else '深度不足' for v in x['visible_book_depth']]
    rows.append(f"<tr><td>{html.escape(x['symbol'])}</td><td>{x['direction']}</td><td>{x['own_notional']:,.0f}</td><td>{x['signed_edge_bp']:.2f}</td><td>{costs[0]}</td><td>{costs[1]}</td><td>{x['pas_funding_hourly_bp']:.3f}</td></tr>")
comparison_rows=[]
for r in comparisons:
    if r['scenario']=='base_1pct':
        comparison_rows.append(f"<tr><td>{r['phase']}</td><td>{r['baseline_pnl']:+.2f}</td><td>{r['candidate_pnl']:+.2f}</td><td>{r['pnl_change']:+.2f}</td></tr>")
grosspct=100*a['own_gross_notional']/a['own_equity']
source_links=' · '.join(f'<a href="{u}">{label}</a>' for u,label in zip(spec['sources'],['Boyd 외: 수익·위험·비용을 함께 고려하는 배분','Hyperliquid 증거금·레버리지','공식 점수·실시간 PnL 구분']))
report=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>수익 기회에 따른 투자 규모 확대 — 진단과 개발안</title>
<style>body{{max-width:1100px;margin:44px auto;padding:0 24px;font:16px/1.7 system-ui;color:#172033}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{padding:9px;border:1px solid #d7dfe9;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{background:#edf3fa;padding:16px;border-radius:8px}}small{{color:#526079}}a{{color:#1767a5}}</style>
<h1>수익 기회에 따른 투자 규모 확대</h1><p>{stamp(a['at'])} 공개 호가·계좌 대조 및 기존 실험 재현</p>
<p class="note"><b>결론:</b> 현재 5배 증거금 상한보다, <b>PAS에서 추가 투자한 금액이 비용을 넘겨 벌 수 있는지 판단하는 모델과 배분 방식</b>을 먼저 보완해야 함. 기존 미연결 확대 후보는 모든 종목의 예상 이익이 비용 기준에 못 미쳐 작동하지 않았음. 이번 작업은 진단 및 개발 설계이며 실제 주문·전략 교체는 없음.</p>
<h2>1. 실제로 확인한 결과</h2>
<ul><li>우리 지분 평가자산 {a['own_equity']:,.2f} USDP, 총 보유 명목금액 {a['own_gross_notional']:,.2f} USDP: <b>{grosspct:.2f}%</b>. 팀원 지분과 합쳐 계산하지 않았음.</li>
<li>현재 확대 후보의 종목별 예상 가격변화+참조 펀딩 효과는 <b>{min(x['signed_edge_bp'] for x in a['assets']):.2f}~{max(x['signed_edge_bp'] for x in a['assets']):.2f}bp</b>. 확대 조건인 43bp를 넘는 종목은 0개임. 이 값은 모델 예상치이며 실제 미래 수익이 아님.</li>
<li>현재 12,401 USDP 수준 보유에 필요한 증거금이 부족해서 확대가 막힌 상황이 아님. 기존 총액 상한은 우리 평가자산의 60%, 계획 단계 여유비율은 90%임. 이것이 수익 극대화에 최적인 상한이라는 검증은 아님.</li>
<li>기존 12개 기간·비용조건 비교의 손익 차이는 모두 0. 주문·체결도 동일했음. 겹치는 기간과 시나리오이므로 12개의 독립 실험이 아님.</li>
<li>그중 연속 71시간 기본 조건을 새로 계측하여 완전히 재현함. 12번 판단 × 10종목 = 120개 판단에서 <b>120개 전부 비용 기준 미달</b>, 96개 위험보호, 69개 진입필터 미통과였음. 사유는 중복됨. 120개 체결 또는 독립 표본이라는 뜻이 아님.</li></ul>
<table><tr><th>기존 PAS 개발 구간·기본 조건</th><th>기존 PnL (USDP)</th><th>확대 후보 PnL</th><th>차이</th></tr>{''.join(comparison_rows)}</table>
<p><small>가정한 거래 참여율과 지연·부분체결을 사용하는 조건부 재현임. 당시 전체 호가·대기열은 없으므로 실제 수익률 재현으로 단정할 수 없음. 앞·뒤·전체 구간은 서로 겹치며 합산할 수 없음. 연속 구간 종료는 {stamp(h['end'])}, Q의 마지막 완결 일별 표본은 {stamp(h['daily_score_cutoff'])}이므로 종료 PnL과 그 Q를 같은 시각의 숫자처럼 설명하면 안 됨.</small></p>
<h2>2. 현재 PAS 비용을 직접 확인</h2>
<table><tr><th>종목</th><th>목표 방향</th><th>현재 우리 보유액(부호)</th><th>기존 모델 이익 bp<br>168h, 비용 전</th><th>1,000 USDP<br>즉시 왕복비용 bp</th><th>5,000 USDP<br>즉시 왕복비용 bp</th><th>PAS 현재<br>시간당 펀딩 bp</th></tr>{''.join(rows)}</table>
<p><small>호가 양쪽을 현재 시점에 각각 소진하는 가상 계산에 편도 9bp 수수료를 더한 값. 우리 체결의 수수료 중앙값은 {a['observed_fee_bp_median']:.5f}bp였음. 미래 청산 비용·대기열·취소·지연·실제 체결을 보장하지 않음. 현재 수치로 과거 성과를 다시 계산하지 않았음. 펀딩의 양수는 롱 지급·숏 수취 방향이며, 현재 한 시간의 금리를 일주일 내내 고정한다고 가정하지 않았음.</small></p>
<p>예를 들어 ETH는 외부 시장 기준 168시간 누적 펀딩 예측이 양수이나 PAS 현재 펀딩은 음수임. 같은 예측 구간을 직접 비교한 것은 아니지만, 외부 시장의 펀딩을 PAS 추가 투자 근거로 그대로 쓸 수 없음을 보여줌.</p>
<h2>3. 보완할 부분과 이유</h2>
<table><tr><th>현재 미연결 확대 후보</th><th>개발 방향</th></tr>
<tr><td>외부 시장 168시간 릿지 예측·펀딩으로 PAS 추가 투자를 평가함</td><td>현재 PAS 체결가격에서 출발하는 예상 청산가와 PAS 펀딩으로 추가 순이익 산정. 외부 가격 변화와 PAS 가격 괴리 변화는 분리 검증</td></tr>
<tr><td>진입 19bp + 청산 24bp를 모든 종목·수량에 동일 적용함</td><td>종목·수량·주문 방식별 비용 곡선과 청산 여력 반영. 비용 기준을 임의로 낮춰 거래를 만드는 방식은 사용하지 않음</td></tr>
<tr><td>통과한 기존 목표들을 공통 1~3배 배율로 확대함</td><td>추가 금액을 종목별로 결정. 같은 방향의 상관 위험을 합산하고, 페어는 헤지 비율을 유지한 묶음으로 배분. 3배 목표 배율과 5배 증거금 설정은 서로 다른 변수임</td></tr>
<tr><td>위험 감소 대상으로 표시된 자산은 확대 불가함</td><td>정상적인 안전 장치를 보존. 기본 목표·최종 위험 상한·추가 투자분을 따로 기록해, 왜 추가할 수 없는지 구분</td></tr>
<tr><td>외부 거래량이 양수인지 검사하나 확대 가능 금액에 PAS 현재 깊이를 쓰지 않음</td><td>현재 호가 깊이·부분체결·미체결 예약량으로 추가 수량 제한. 기존 실운용의 주문 전 검증은 유지</td></tr>
<tr><td>점예측만 사용하고, 추가분의 수명·별도 회수 규칙이 없음</td><td>예측 오차·신호 경과시간을 반영하고, 추가분의 이익 근거가 사라지면 축소. 반복 매수·매도 비용을 막는 유지 구간 설정</td></tr></table>
<p>이 확대 판단에 쓰인 별도 릿지 모델의 학습은 <b>{fit['rows']}개 자산·주간 행(종목당 6개)</b>, 보정은 {fit['calibration_rows']}개임. 보정 계수는 <b>{fit['attenuation']:.5f}</b>로 원래 예측 크기의 약 {100*fit['attenuation']:.2f}%만 사용함. 이것은 정확도 {100*fit['attenuation']:.2f}%라는 뜻이 아님. 해당 보정을 임의로 없애 수익 기회를 크게 보이게 하는 대신, 과거에 실제로 성숙한 정답으로 예측 규모와 오차를 다시 검증해야 함. 이 진단은 Chronos 전체가 학습 60행이라는 뜻도 아님.</p>
<h2>4. 첫 구현 범위</h2>
<p><b>기존 기본 목표 → 보호 상한·현재 보유분 확인 → PAS 추가 순이익 계산 → 종목별 추가 금액 결정 → 합산 위험·청산 가능 수량 확인 → 필요한 증거금 배수 선택(최대 5배)</b> 순서로 구성함.</p>
<p>첫 후보는 기존 방향을 유지하는 추가 투자 모듈 하나로 제한하고, 현재 총액·종목별·변동성 한도는 유지함. 6시간 기본 판단 주기를 유지하되 주문 직전 가격과 비용으로 다시 검사함. 추가분의 잔여 이익을 판단하는 예측 구간은 24시간을 첫 설계안으로 제안함. 24시간이 최적이라는 결과는 아직 없으며, 검증 전에 고정해야 함. 6시간마다 재평가한다고 반드시 6시간 뒤 청산하는 것은 아님.</p>
<p>기본 목표가 0인 종목의 신규 진입, 총액 한도 완화, 페어 후보 변경까지 동시에 섞지 않음. 기존 위험 감축과 청산을 추가 매수가 되돌리지 못하게 하며, 팀원 5만 USDP 및 공동 시장 장부·레버리지 설정을 보호함.</p>
<h2>5. 어떻게 개선을 판정할지</h2>
<ol><li><b>기존판 / 단순 규모 확대 / 기회별 확대</b>를 같은 기간·5만 USDP·체결·비용·위험 한도로 비교함. 규모를 더 써서 번 것인지, 좋은 시점을 골라 번 것인지 분리함.</li>
<li>학습 → 보정·선택 → 평가를 시간순으로 배치하고, 학습 정답의 보유·청산 구간이 다음 구간에 걸치지 않게 제외함. 이미 여러 번 본 10월 19~22일 자료는 개발 자료로 표기함. 과거 미사용 구간은 사용 이력부터 확인하며, 없으면 후보 고정 뒤 새 관측을 따로 평가함.</li>
<li>같은 종료 시각의 비용 차감 PnL과 12:00 UTC 표본 Q를 우선 비교하고, 실현 변동성·최대낙폭·노출금액·회전율·수수료·펀딩·부분체결·청산 비용을 함께 보고함. 변동성 증가 자체를 무조건 탈락 사유로 삼지 않음.</li>
<li>예측 오차와 비용·지연을 불리하게 잡아도 결과가 유지되는지 검사함. 미래 정보·장부·정밀도·위험 한도 오류는 0이어야 함. 종목별/시간대별 결과를 모두 남기고 좋은 구간만 고르지 않음.</li>
<li>독립 평가와 모의 주문 기록을 확인한 뒤에만 연결 여부를 판단함. PnL 개선을 무조건 보장하는 검증 방법은 없으며, 이번 작업에서 개선된 새 수익률이 나온 것으로 보고하지 않음.</li></ol>
<p>수익·위험·거래비용을 함께 최적화하는 배분 틀은 Boyd 등의 연구와 일치함. 그 연구도 예측값 자체를 만드는 문제는 별도라고 명시함. 좋은 배분 알고리즘만으로 검증되지 않은 예측력이 생기지는 않음.</p>
<p>{source_links}</p>
<p><a href="current_audit.json">현재 종목별 진단</a> · <a href="historical_audit.json">재현·120개 판단 기록</a> · <a href="development_spec.json">구체 개발·검증 규격</a> · <a href="public_snapshot.json">공개 API 원자료</a> · <a href="audit.py">진단 코드</a></p></html>'''
report=report.replace('深度不足','호가 수량 부족')
(OUT/'report.html').write_text(report)
dump('manifest.json',dict(live_manifest=live_manifest(),
    evidence={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'},
    new_strategy_implemented=False,live_changed=False,orders_sent=0))
print(str(OUT/'report.html'))
