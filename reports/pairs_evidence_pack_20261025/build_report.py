"""Offline evidence export. Reads saved results; never fits models or sends orders.

Run from the repository with /usr/bin/python3. Source experiments remain unchanged.
Figures are descriptive exports of reused history, not new independent tests.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import gzip
import hashlib
import html
import json
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TABLES = HERE / 'tables'
FIGURES = HERE / 'figures'
for p in (TABLES, FIGURES):
    p.mkdir(exist_ok=True)
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
plt.rcParams.update({'font.family': FontProperties(fname=FONT).get_name(),
                     'font.size': 10, 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 140, 'savefig.dpi': 200,
                     'svg.fonttype': 'none'})
SOURCE_HASHES = {}
SECTIONS = []
PDF = PdfPages(HERE / 'figures.pdf')
NOW = datetime.now(timezone.utc)


def source(path):
    p = ROOT / path
    SOURCE_HASHES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def read_json(path):
    return json.loads(source(path).read_text())


def read_run(path):
    with gzip.open(source(path), 'rt') as f:
        return json.load(f)


def export(frame, name):
    frame.to_csv(TABLES / f'{name}.csv', index=False, encoding='utf-8-sig')
    return frame


def table(frame):
    return frame.to_html(index=False, border=0, escape=True, na_rep='미산출',
                         float_format=lambda x: f'{x:,.6f}')


def add(title, prose, frame=None, figure=None):
    SECTIONS.append((title, prose, frame, figure))


def savefig(fig, name):
    fig.savefig(FIGURES / f'{name}.png', bbox_inches='tight')
    fig.savefig(FIGURES / f'{name}.svg', bbox_inches='tight')
    PDF.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    return name


def iso(ts):
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat()


def markdown_table(frame):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return '미산출' if pd.isna(v) else f'{v:,.6f}'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                       '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                      ['| ' + ' | '.join(fmt(v) for v in row) + ' |'
                       for row in frame.itertuples(index=False, name=None)])


# Short original summaries; no paper tables or figures are reproduced.
papers = [
    dict(id='P1', year='2026-06, arXiv v2',
         title='Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets with Deep Reinforcement Learning',
         url='https://arxiv.org/html/2606.04574v2', kind='암호화폐 페어트레이딩',
         theory='관계를 선별·순위화하고 적응형 평균과 고정 위험 규칙 위에 PPO–LSTM 실행 계층을 결합한다.',
         evidence='표 7·그림 11: 2025년 비용 차감 성과·누적수익. 표 8–9·그림 27–36: 민감도·구성요소 제거. 표 15: 블록 부트스트랩. 표 16: 분기 비교.',
         finding='동일 10배 레버리지에서 CAGR 30.40%→199.45%, Sharpe 0.4921→2.8220. MDD는 34.13%→35.27%로 증가했다.',
         limitation='Sharpe 차이 p=0.0603, 95% 구간은 0 포함. Q3에는 기준판보다 부진. 수수료·시장·레버리지가 PAS와 다르다.',
         adoption='초기 연구의 적응형 관계·위험/실행 분리 참고. 현재 코드가 PPO–LSTM이나 논문 전체를 재현한 것은 아니다.'),
    dict(id='P2', year='2026-10, ICAIF 원고/arXiv v2',
         title='Amortized Neural Filtering for Calibrated Posteriors in Statistical Arbitrage',
         url='https://arxiv.org/html/2610.25802v2', kind='관련 통계적 차익거래; 2종목 페어 전용 아님',
         theory='평균회귀 상태·속도의 불확실성을 추정하고 포지션 크기에 반영한다.',
         evidence='표 1: 비용 차감 성과. 그림 2: 동일 위험 수익곡선·확률 보정. 표 2: 예측–실현 격차. 그림 3: 상태 표현.',
         finding='KRX 평가에서 AFM/NFE Sharpe 1.45/1.49. 논문 자체가 이 작은 차이를 잡음 범위로 판단한다.',
         limitation='다른 기간에는 우위가 없고, 표본 종목 선정 편향이 있다. 격차 감소는 주로 예측 축소 효과이며 새 알파 입증이 아니다.',
         adoption='평가·불확실성 설명 참고. NFE는 현재 실거래 모델에 구현되어 있지 않다.'),
    dict(id='P3', year='2026-08, Complexity',
         title='Cost-Aware Multiwindow Ensemble Decision Support for Nonstationary Foreign Exchange Markets',
         url='https://onlinelibrary.wiley.com/doi/10.1155/cplx/1155228', kind='FX 예측·앙상블; 통화쌍을 거래하는 연구',
         theory='서로 다른 기간의 회귀 전문가를 비용 차감 효용으로 결합하고 변동성·최소 보유 규칙을 둔다.',
         evidence='표 1: 워크포워드 분할. 표 3·그림 7: 기준모형 비교. 표 4: 기간 민감도. 표 5: 제거·비용 실험. 표 6·그림 8: 가중치 해석.',
         finding='평균 Sharpe 1.085 대 ridge 0.607. 순이익은 312.3 대 306.7 pips로, 높은 Sharpe가 큰 순이익 차이를 뜻하지 않는다.',
         limitation='비용 2배에서는 −24.6 pips. 16개 currency pairs는 우리가 말하는 두 자산 롱·숏 페어 조합과 다르다.',
         adoption='비용 인지 예측·평가 참고. 현재 전략을 이 논문의 다중창 앙상블 재현이라고 부를 수 없다.'),
    dict(id='P4', year='2019, 기초 이론',
         title='A singular stochastic control approach for optimal pairs trading with proportional transaction costs',
         url='https://arxiv.org/html/1911.10450v1', kind='거래비용을 포함한 페어 최적제어',
         theory='공적분·평균회귀 가정 아래 거래 시점과 수량을 함께 최적화하고 매수·매도·대기 영역을 도출한다.',
         evidence='그림 1–3: 거래 경계. 그림 4–12: 비용 등 민감도. 표 4: 6개 주식 페어의 표본 외 손익·거래횟수.',
         finding='비용 때문에 가격 괴리가 있어도 대기하는 영역이 생긴다는 이론적 근거다.',
         limitation='특정 확률과정·효용 가정에서의 최적성이다. PAS의 관계 안정성이나 수익 보장을 증명하지 않는다.',
         adoption='비용 대비 진입 이익·대기 판단의 배경. 우리 코드는 이 HJB 최적제어 문제를 풀지 않는다.'),
    dict(id='P5', year='2017 초고 / 2020 v5, 기초 방법',
         title='Smart “Predict, then Optimize”',
         url='https://arxiv.org/html/1710.08005v5', kind='의사결정 중심 학습; 페어 실거래 논문 아님',
         theory='예측 오차 대신 예측이 유도한 의사결정 손실을 학습하고 SPO+ 볼록 대리손실을 제안한다.',
         evidence='그림 4: 합성 포트폴리오 문제의 테스트 SPO 손실 비교. 조건부 일관성 정리.',
         finding='예측 정확도와 의사결정 품질이 서로 다른 평가 대상임을 설명한다.',
         limitation='합성 최적화 손실 감소는 PAS 순손익 검증과 다르다.',
         adoption='현재 페어 신규진입의 롱/숏/대기 비용 민감 게이트에 독립적으로 유도한 3행동 손실을 사용한다.'),
    dict(id='P6', year='2026-09, arXiv v1',
         title='Geometric Identification in Predict-Then-Optimize Learning',
         url='https://arxiv.org/html/2609.33472v1', kind='의사결정 학습 이론·진단',
         theory='같은 최적 행동을 만드는 예측값이 여러 개일 수 있어, 행동 일치만으로 평균 예측을 식별할 수 없음을 설명한다.',
         evidence='그림 1–2: 식별·경계 기하. 부록 I: 포트폴리오 등의 예측오차·문제 변경 후 regret. 부록 J: 알려진 생성과정 실험.',
         finding='의사결정 점수를 예상 가격 수익률로 재사용하는 데 별도 근거가 필요하다.',
         limitation='유한표본 성능·일반적인 전이 이익을 보장하지 않는다.',
         adoption='현재 게이트 점수와 가격 수익률 예측을 분리하는 설계 근거. 수익률 재현 주장은 하지 않는다.'),
]
export(pd.DataFrame(papers), '01_papers_and_evidence')
paper_html = ''.join(
    f'<article><h3>{p["id"]}. <a href="{p["url"]}">{html.escape(p["title"])}</a></h3>'
    f'<p class="meta">{p["year"]} · {p["kind"]}</p>'
    + ''.join(f'<p><b>{label}</b> {html.escape(p[key])}</p>' for label, key in
              [('이론', 'theory'), ('표·그림', 'evidence'), ('결과', 'finding'),
               ('해석 범위', 'limitation'), ('우리 구현과의 관계', 'adoption')]) + '</article>'
    for p in papers)

profile = read_json('competition_live/config.json')
signal = read_json('.runtime/pas_budget_lifecycle/signal.json')
body = signal['body']
assert body['retained_pair_universe'] == [['BTC', 'ETH'], ['BTC', 'HYPE']]
strategy = pd.DataFrame([
    ['운용 환경', 'Hyperliquid Testnet PAS / USDP; 팀 초기 100,000, 우리 배정 50,000', 'competition_live/config.json'],
    ['전략 형태', '10종목 방향성 코어 + 비용 인지 페어 오버레이 + 주문·위험 관리', 'core_revision=' + body['core_revision']],
    ['방향성 코어', '10종목의 추세·7일 되돌림, 위험·경제성 배분. 롱·숏 판단 가능', 'universe_repair/core.py; pas_economic_allocation/model.py'],
    ['현재 페어 후보', 'BTC–ETH, BTC–HYPE; 전 45개 자동선정판은 실거래 적용 아님', 'signal.body.retained_pair_universe'],
    ['페어 예측', '고정 시점 헤지비율로 24시간 상대수익 예측; 과거 90일 가중 ridge, 일별 재적합', 'experiments/rolling_feature_pairs_20261017/model.py'],
    ['진입 게이트', 'SPO+에서 유도한 비용 민감 롱/숏/대기 분류. 점수를 예상수익으로 사용하지 않음', 'experiments/decision_gate_pairs_20261017/model.py'],
    ['PAS 연결', '현재 실행가능 가격·비용·펀딩·수량·위험으로 진입가치 재검사; 두 다리 개별 체결 및 잔여 헤지 관리', 'competition_live/candidate.py; competition_live/config.json'],
    ['진입 실행 조건', 'pullback_any3; 현재 검증 후보 quote_inputs와 구분', 'execution_policy=' + body['execution_policy']],
    ['연구와 구분', 'PPO–LSTM, NFE, 전체 45페어 자동선정의 실거래 재현이라고 기술할 수 없음', '아래 실험별 버전 표 참조'],
], columns=['구성', '현재 확인 내용', '근거'])
export(strategy, '02_strategy_scope')
identity = {k: body.get(k) for k in ['candidate', 'profile_hash', 'published_at',
                                   'core_revision', 'execution_policy', 'retained_pair_universe']}
(HERE / 'strategy_identity.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2))
add('현재 전략을 정확하게 설명하는 문장',
    '<p><b>“SFINX는 다자산 추세·되돌림 코어에, 상대수익을 예측하는 비용 인지 페어트레이딩 모듈을 결합한 전략이다. '
    '페어 예측과 롱·숏·대기 게이트를 분리하고, PAS에서 실제 체결 가능한 가격과 비용을 고려해 주문한다.”</b></p>'
    '<p>순수 시장중립 페어 전략으로 계좌 전체를 설명하면 부정확하다. 아래의 과거 실험은 현재 전체 실행 파일과 동일한 버전이 아니며, '
    '각 실험이 검증한 변경만 설명한다. 표의 소스 경로는 저장소 기준 상대 경로다.</p>', strategy)

add('페어 가설과 검증 대상',
    '<p>두 자산의 가격 관계가 안정적이라면 상대적으로 비싼 쪽을 매도하고 싼 쪽을 매수해 관계 정상화에서 이익을 기대할 수 있다. '
    '그러나 상관관계만으로 정상화가 보장되지는 않는다. 현재 구현의 직접 예측 목표는 '
    '<code>g(t,24h) = [r_A(t,24h) − β_t r_B(t,24h)] / (1 + β_t)</code>를 '
    '과거 변동성으로 표준화한 값이다. β는 판단 시점의 값으로 고정한다. '
    '가격 예측과 별개로 게이트는 학습용 대리 보상 <code>a·y − C·|a|, a∈{−1,0,1}</code>를 사용한다.</p>'
    '<p>학습용 고정 비용 대리 문제에는 실제 지연·펀딩·부분체결·보유 연장이 없다. 이 조건은 계좌 재생에서 별도로 반영한다. '
    '따라서 정확한 주장은 “이론적으로 항상 회귀한다”가 아니라 “상대수익 예측과 비용 판단이 기준 전략보다 실제 순손익을 개선했는가”이다. '
    '과거 공적분·평균회귀 아이디어를 참고했더라도 현재판에 모든 페어의 공적분이 증명됐다고 쓰지 않는다.</p>')

hist = pd.read_csv(source('reports/decision_gate_pairs_20261017/results.csv'))
hist['experiment'] = 'decision_gate_pairs_20261017'
hist['capital'] = 1_000_000
hist['independent_test'] = False
export(hist, '03_historical_all_48_results')
names = {'first_strategy': '당시 첫 전략', 'second_strategy': '당시 두 번째 전략',
         'first_dense_hourlyhedge_cash_direct_24': '직접예측 페어 추가',
         'first_dense_hourlyhedge_decisiongate_24': '비용 게이트 페어 추가'}
view = hist[(hist.venue == 'reference') & (hist.phase == 'comparison')].copy()
summary = pd.DataFrame({'전략': view.arm.map(names), '비용': view.scenario,
                        '순손익': view.net_pnl, '수익률(%)': view.roi * 100,
                        '기간변동성(%)': view.volatility_horizon * 100,
                        'MDD(%)': view.max_drawdown * 100, '저장된 Q': view.financial_score,
                        '완료 페어': view.completed_pairs})
export(summary, '04_historical_comparison_summary')
add('표 1. 기존 두 전략과 페어 구성의 비교 — 과거 연구판',
    '<p>2026-09-01 12:00~10-15 12:00 UTC, 기준시장 재생, 초기 1,000,000. '
    '같은 기간·자본·비용 시나리오의 저장 결과다. 모든 기간은 반복 사용한 개발 자료이며 PAS 실거래 또는 독립 테스트 성과가 아니다. '
    '기본 조건에서 페어 게이트 추가판은 순손익이 높지만 MDD도 크다. 스트레스 조건에서는 게이트가 없는 직접예측판과 결과가 같다. '
    '완료 페어가 기본 3건, 스트레스 1건이므로 우위를 일반화할 표본은 부족하다.</p>'
    '<p><code>base/stress</code>는 해당 실험의 사전 정의 실행 시나리오다. 비용 증가가 거래를 막아 경로도 바꾸므로 '
    '일부 stress 손익이 높아질 수 있다. 모든 실험에서 stress가 같은 수치의 비용을 뜻하지는 않는다. '
    '전체 48행에는 초기 비교 구간과 PAR 평가도 함께 보존했다.</p>', summary)

paths = {
    'first_strategy': 'experiments/dense_history_pairs_20261017/artifacts/runs/reference-comparison-first_strategy-base.json.gz',
    'second_strategy': 'experiments/dense_history_pairs_20261017/artifacts/runs/reference-comparison-second_strategy-base.json.gz',
    'first_dense_hourlyhedge_cash_direct_24': 'experiments/conditional_completion_pairs_20261017/artifacts/runs/reference-comparison-first_dense_hourlyhedge_cash_direct_24-base.json.gz',
    'first_dense_hourlyhedge_decisiongate_24': 'experiments/decision_gate_pairs_20261017/artifacts/runs/reference-comparison-first_dense_hourlyhedge_decisiongate_24-base.json.gz',
}
runs = {}
curve_rows = []
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
for arm, path in paths.items():
    run = read_run(path)
    runs[arm] = run
    sn = pd.DataFrame(run['snapshots'])
    equity = sn.equity.to_numpy()
    assert abs(equity[-1] - equity[0] - float(view[(view.arm == arm) & (view.scenario == 'base')].net_pnl.iloc[0])) < 1e-6
    ret = 100 * (equity / equity[0] - 1)
    dd = 100 * (equity / np.maximum.accumulate(equity) - 1)
    t = pd.to_datetime(sn.time, unit='s', utc=True)
    axes[0].plot(t, ret, label=names[arm], lw=1.6)
    axes[1].plot(t, dd, label=names[arm], lw=1.2)
    for ts, eq, rr, draw in zip(sn.time, equity, ret, dd):
        curve_rows.append(dict(arm=arm, time_utc=iso(ts), equity=eq, return_pct=rr, drawdown_pct=draw))
axes[0].set_title('과거 연구판: 수익의 크기와 손실 경로를 함께 비교')
axes[0].set_ylabel('누적 수익률 (%)'); axes[0].legend(ncol=2, fontsize=9)
axes[1].set_ylabel('고점 대비 낙폭 (%)'); axes[1].set_xlabel('UTC · 기준시장 · 1M 자본 · 기본 비용 · 반복 사용한 개발 기간')
for ax in axes: ax.grid(alpha=.2)
figure = savefig(fig, '01_equity_and_drawdown')
export(pd.DataFrame(curve_rows), '05_equity_drawdown_points')
add('그림 1. 누적 수익률과 낙폭',
    '<p>마지막 수익률 한 숫자 대신 전체 경로를 표시했다. 이 그림만으로 현재 운용판의 성과나 독립적 예측력을 주장할 수 없다.</p>', figure=figure)

diag = read_json('experiments/decision_gate_pairs_20261017/artifacts/diagnostics.json')
attribution = []
episodes = []
for row in diag['account_attribution']:
    tot = row['totals']
    assert abs(tot['price_move'] - tot['execution_cost'] - tot['fees'] + tot['funding'] - tot['net_pnl']) < 1e-6
    attribution.append(dict(phase=row['phase'], model=row['model'], scenario=row['scenario'],
                            base_owned_pnl=row['base_owned_pnl'], **tot))
    for e in row['episodes']:
        episodes.append(dict(phase=row['phase'], model=row['model'], scenario=row['scenario'],
                             pair='–'.join(e['pair']), start_utc=iso(e['time']), end_utc=iso(e['end']),
                             net_pnl=e['net_pnl'], price_move=e['price_move'], fees=e['fees'],
                             execution_cost=e['execution_cost'], funding=e['funding'],
                             closed=e['closed'], status=e['status']))
att = export(pd.DataFrame(attribution), '06_pair_pnl_attribution')
export(pd.DataFrame(episodes), '07_pair_episodes')
pairbase = att[(att.phase == 'comparison') & (att.scenario == 'base')]
fig, ax = plt.subplots(figsize=(10, 4.7), constrained_layout=True)
xs = np.arange(5)
for j, (_, r) in enumerate(pairbase.iterrows()):
    vals = [r.price_move, -r.execution_cost, -r.fees, r.funding, r.net_pnl]
    bars = ax.bar(xs + (j-.5)*.36, vals, width=.36, label={'direct':'직접예측 페어', 'gate':'게이트 페어'}[r.model])
    ax.bar_label(bars, fmt='%.0f', fontsize=9, padding=3)
ax.axhline(0, color='#333', lw=.7); ax.set_xticks(xs, ['가격 변화', '체결 비용', '수수료', '펀딩', '페어 순손익'])
ax.set_ylabel('손익 (초기 자본 1M)'); ax.legend(); ax.grid(axis='y', alpha=.2)
ax.set_title('과거 페어 모듈 기여: 가격 변화에서 비용을 뺀 뒤 남은 이익')
figure = savefig(fig, '02_pair_cost_attribution')
add('그림 2. 페어 모듈이 실제로 보탠 순손익',
    '<p>전체 계좌의 방향성 코어 이익과 페어 기여를 구분했다. 같은 과거 비교에서 페어 순손익은 '
    '직접예측판 +2,978.02, 게이트판 +3,407.79였다. 게이트판의 체결 비용과 수수료도 더 컸다. '
    '이 금액은 현재 PAS 지갑의 손익이 아니다. 에피소드 CSV는 종료·미종료를 모두 보존하며, '
    '에피소드 수와 엔진의 완료 페어 집계는 정의가 달라 혼용하지 않는다.</p>', figure=figure)

auto = pd.DataFrame(read_json('reports/auto_pair_selection_20261023/reference_results.json'))
auto['experiment'] = 'auto_pair_selection_20261023'
auto['own_capital'] = 50_000
auto['team_capital_with_idle_reserve'] = 100_000
auto['independent_test'] = False
export(auto, '08_auto45_all_18_results')
auto_protocol = read_json('reports/auto_pair_selection_20261023/protocol.json')
auto_view = auto[(auto.phase == 'continuous') & (auto.arm != 'fixed_two_rebuilt')].copy()
auto_summary = pd.DataFrame({'전략': auto_view.arm.map({'incumbent_fixed_two':'10/23 기준판: 고정 2페어','automatic_45':'45페어 자동선정 후보'}),
                             '비용': auto_view.scenario, '순손익': auto_view.net_pnl,
                             '청산비용 반영 손익': auto_view.terminal_pnl,
                             '수익률(팀 100k 기준 %)': auto_view.return_fraction*100,
                             '기간변동성(%)': auto_view.volatility*100,
                             '저장된 Q': auto_view.Q, '완료 페어': auto_view.completed_pairs})
export(auto_summary, '09_auto45_continuous_summary')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for j, arm in enumerate(['incumbent_fixed_two','automatic_45']):
    rows = auto[(auto.arm == arm) & (auto.phase == 'continuous')].set_index('scenario').loc[['base','stress']]
    label = '10/23 고정 2페어 기준판' if j == 0 else '45페어 자동선정 후보'
    for ax, metric in zip(axes, ['net_pnl', 'Q']):
        bars = ax.bar(np.arange(2)+(j-.5)*.36, rows[metric], width=.36, label=label)
        ax.bar_label(bars, fmt='%.2f' if metric == 'net_pnl' else '%.3f', fontsize=9, padding=3)
        ax.set_xticks([0,1], ['기본', '스트레스']); ax.axhline(0, color='#333', lw=.7); ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('순손익'); axes[1].set_ylabel('저장된 Q'); axes[0].legend(fontsize=9)
fig.suptitle('45개 관계를 평가하는 기능과 수익성 개선은 별도 검증 대상')
figure = savefig(fig, '03_auto45_comparison')
add('표 2·그림 3. 전체 45개 페어로 확장한 연구 결과',
    '<p>10/23 저장 비교, 2026-08-01~10-15의 연속 기준시장 구간. 우리 배정 50k + 유휴 50k를 포함한 팀 100k 계좌. '
    '수익률의 분모는 100k다. 같은 10종목 코어와 실행·위험 조건에서 페어 선정만 바꾼 비교이며, '
    '기준판을 재구현한 통제군도 원 결과와 일치했다. 비교 표의 8월·후속·연속 구간은 겹치므로 세 독립 테스트로 셀 수 없다.</p>'
    '<p>자동선정 후보는 이 비교에서 수익성이 나빠졌고 실거래에 배포하지 않았다. '
    '이는 “45페어 접근은 불가능하다”는 증명이 아니라 해당 선별·예측·실행 조합의 부정적 결과다.</p>', auto_summary, figure)

# Read pre-existing forecast/label pairs. No training, tuning or threshold selection.
forecast_path = source('experiments/decision_gate_pairs_20261017/artifacts/reference_issued_forecasts.npz')
pair_order = read_json('experiments/hourly_refresh_pairs_20261017/model_protocol.json')['pairs']
with np.load(forecast_path) as f:
    start = 1788264000
    end = 1792065600
    grid = (f['times'] >= start) & (f['maturity'] <= end) & ((f['times']-start) % 86400 == 0)
    forecast_rows = []
    for k, pair in enumerate(pair_order):
        valid = grid & f['fresh'][:,k] & f['terminal_fresh'][:,k]
        valid &= np.isfinite(f['prediction_direct'][:,k]) & np.isfinite(f['target'][:,k]) & (f['fit_stamps'][:,k] <= f['times'])
        for i in np.flatnonzero(valid):
            forecast_rows.append(dict(pair='–'.join(pair), time_utc=iso(f['times'][i]),
                                      maturity_utc=iso(f['maturity'][i]), fit_utc=iso(f['fit_stamps'][i,k]),
                                      predicted_bp=float(f['prediction_direct'][i,k]*1e4),
                                      actual_bp=float(f['target'][i,k]*f['unit'][i,k]*1e4)))
fc = export(pd.DataFrame(forecast_rows), '10_prediction_vs_realization_points')
fcstats = []
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for ax, (pair, g) in zip(axes, fc.groupby('pair', sort=False)):
    pred, actual = g.predicted_bp.to_numpy(), g.actual_bp.to_numpy()
    fcstats.append(dict(pair=pair, n=len(g), MAE_bp=np.mean(abs(pred-actual)), zero_MAE_bp=np.mean(abs(actual)),
                        RMSE_bp=np.sqrt(np.mean((pred-actual)**2)), zero_RMSE_bp=np.sqrt(np.mean(actual**2)),
                        forecast_bias_bp=np.mean(pred-actual), directional_accuracy=np.mean(np.sign(pred)==np.sign(actual))))
    lim = max(np.max(abs(pred)),np.max(abs(actual)))*1.08
    ax.scatter(pred, actual, s=25, alpha=.7)
    ax.plot([-lim,lim],[-lim,lim], '--', color='#888', label='예측=실현')
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.axhline(0,color='#aaa',lw=.5); ax.axvline(0,color='#aaa',lw=.5)
    ax.set_title(f'{pair} · n={len(g)}'); ax.set_xlabel('24시간 상대수익 예측 (bp)'); ax.set_ylabel('24시간 실제 상대수익 (bp)')
    ax.grid(alpha=.15)
fig.suptitle('저장 예측의 보정 진단 · 동일 페어 내 겹치지 않는 24시간 평가 간격')
figure = savefig(fig, '04_prediction_calibration')
fcstats = export(pd.DataFrame(fcstats), '11_prediction_error_summary')
add('표 3·그림 4. 예측력이 실제로 있었는가',
    '<p>과거에 저장된 직접 수익률 예측과 만기가 지난 정답을 재집계했다. 9/1 12:00 UTC부터 24시간 간격을 고정하여 '
    '동일 페어 안의 목표 구간 겹침을 줄였다. 두 페어는 BTC를 공유하므로 독립 표본으로 합산하지 않는다. '
    '게이트 점수는 예상수익이 아니므로 이 그래프의 예측값에 사용하지 않았다.</p>'
    '<p><b>이 집계에서는 두 페어 모두 0 예측보다 MAE와 RMSE가 낮지 않았다.</b> '
    'BTC–ETH MAE는 38.36bp 대 38.33bp, BTC–HYPE는 77.99bp 대 72.89bp였다. '
    '따라서 과거 계좌 손익의 개선만으로 모든 시점의 가격 예측력이 개선됐다고 결론 내릴 수 없다. '
    '모든 시점 예측과 비용을 넘는 일부 거래의 선택 성과도 서로 다른 대상이다. 이 그림은 기준시장 상대수익 예측 진단이며, '
    '현재 PAS 호가에서의 수익·게이트 효과·새로운 독립 테스트를 입증하지 않는다.</p>', fcstats, figure)

# Descriptive regime report: labels are realized market moves, never trading inputs.
refpath = source('experiments/risk_preserving_20261016/artifacts/inputs.npz')
with np.load(refpath) as d:
    bt = dict(zip(d['market_times'].astype(int), d['close'][:, profile['trading']['symbols'].index('BTC')]))
regime_rows = []
daily_rows = []
for arm, run in runs.items():
    for x in run['daily']:
        t = int(x['time'])
        if t not in bt or t-86400 not in bt:
            continue
        br = bt[t]/bt[t-86400]-1
        daily_rows.append(dict(arm=arm, time_utc=iso(t), btc_return=br,
                               strategy_return=x['return_'],
                               regime='BTC 상승일' if br>0 else 'BTC 하락·보합일'))
daily = export(pd.DataFrame(daily_rows), '12_daily_market_context')
for (arm, regime), g in daily.groupby(['arm','regime'], sort=False):
    regime_rows.append({'전략': names[arm], '사후 구간': regime, '일수': len(g),
                        '평균 일수익(bp)': g.strategy_return.mean()*1e4,
                        '양의 일수익 비율(%)': (g.strategy_return>0).mean()*100})
regimes = export(pd.DataFrame(regime_rows), '13_descriptive_market_regimes')
fig, ax = plt.subplots(figsize=(10,5),constrained_layout=True)
for arm in ['first_strategy','first_dense_hourlyhedge_decisiongate_24']:
    g=daily[daily.arm==arm]
    ax.scatter(g.btc_return*100,g.strategy_return*1e4,label=names[arm],alpha=.65,s=25)
ax.axhline(0,color='#777',lw=.7);ax.axvline(0,color='#777',lw=.7)
ax.set_xlabel('같은 날 기준시장 BTC 수익률 (%)');ax.set_ylabel('전략 일수익률 (bp)')
ax.set_title('시장 상승에만 의존했는가: 사후 시장상황별 진단');ax.legend();ax.grid(alpha=.2)
figure=savefig(fig,'05_market_context')
add('표 4·그림 5. 시장 상황에 맞았는가',
    '<p>동일 44일 구간을 실제 BTC 상승일과 하락·보합일로 나누고 모든 날짜를 포함했다. '
    '이 구분은 사후 설명용이며 거래 알고리즘의 입력이나 미래를 안다는 가정이 아니다. '
    '<b>게이트 페어 추가판도 BTC 상승일 평균 +9.85bp, 하락·보합일 평균 −8.08bp였다.</b> '
    '이 과거 혼합 전략은 하락일의 손실을 제거하지 못했다. 상승일·하락일 평균만으로 시장중립성 또는 여러 레짐에서의 우수성을 증명할 수 없다. '
    '관계 붕괴·유동성·변동성 조건별 검증은 별도로 필요하며, 이 팩에서는 없는 결과를 채우지 않았다.</p>',regimes,figure)

owner = read_json('reports/top10_live_review_20261024/sfinx_owner_split.json')
owners = []
for r in owner['owners']:
    assert abs(r['realized']+r['unrealized']-r['fees']+r['funding']-r['pnl']) < 1e-6
    owners.append({**{k:v for k,v in r.items() if k!='quantities'}, 'snapshot_kst':owner['position_snapshot_kst'],
                  'pnl_div_50000_pct':r['pnl']/r['capital']*100})
ownframe = export(pd.DataFrame(owners),'14_live_owner_snapshot')
ours = next(r for r in owner['owners'] if r['owner']=='ours')
fig,ax=plt.subplots(figsize=(10,4.4),constrained_layout=True)
vals=[ours['realized'],ours['unrealized'],-ours['fees'],ours['funding'],ours['pnl']]
bars=ax.bar(['실현 가격손익','미실현 가격손익','수수료','펀딩','순손익'],vals,
            color=['#376595' if x>=0 else '#b4504f' for x in vals])
ax.bar_label(bars,fmt='%.2f',padding=3);ax.axhline(0,color='#555',lw=.7)
ax.set_ylabel('USDP');ax.grid(axis='y',alpha=.2)
ax.set_title('우리 내부 장부의 실거래 손익 · 2026-10-24 21:11 KST 저장 시점')
figure=savefig(fig,'06_actual_owner_pnl')
add('표 5·그림 6. PAS 실거래 기록 — 과거 저장 시점',
    '<p><b>현재 실시간 잔고가 아니다.</b> 10/24 21:11:23 KST에 저장한 내부 소유권 장부다. '
    '우리 배정 손익은 +69.074609 USDP, 50,000 기준 +0.138149%다. 팀원 손익은 별도 행에 있다. '
    '이 장부는 포지션 시점과 체결 소유권 기록을 대조했으며 잔고·수량 오차도 원본에 저장되어 있다.</p>'
    '<p>우리 행에도 방향성 코어와 페어, 이전 변경분의 효과가 섞여 있으므로 전액을 현재 페어 모듈의 성과라고 주장하지 않는다. '
    '표의 수익률은 배정 자본으로 나눈 장부 비율이며, 공식 팀 리더보드의 현금흐름 보정 R 또는 현재 등수가 아니다.</p>',ownframe,figure)

qual=read_json('reports/production_replay_20261025/qualification_review/results.json')
e=next(x for x in qual['experiments'] if x['primary']=='quote_inputs')
qrows=[]
for phase, comp in e['comparisons'].items():
    for r in comp['rows']:
        if r['arm'] not in ['incumbent','quote_inputs']:continue
        qrows.append(dict(phase=phase,scenario=r['scenario'],arm=r['arm'],
                          net_pnl=r['net_pnl'],net_after_terminal_exit=r['net_after_terminal_exit'],
                          price_pnl=r['price_pnl'],fees=r['fees'],funding=r['funding'],valid=r['valid'],
                          start_utc=iso(e['start']),end_utc=iso(e['end'])))
qf=export(pd.DataFrame(qrows),'15_completed_short_replay')
fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
combos=[('realistic_phase0','base'),('realistic_phase0','stress'),('realistic_phase45','base'),('realistic_phase45','stress')]
for j,arm in enumerate(['incumbent','quote_inputs']):
    rows=[qf[(qf.phase==p)&(qf.scenario==s)&(qf.arm==arm)].iloc[0] for p,s in combos]
    for ax,key in zip(axes,['net_pnl','net_after_terminal_exit']):
        ax.bar(np.arange(4)+(j-.5)*.36,[r[key] for r in rows],width=.36,label=arm)
        ax.set_xticks(range(4),['0초/기본','0초/불리','45초/기본','45초/불리'],rotation=20)
        ax.axhline(0,color='#555',lw=.7);ax.set_ylabel('USDP');ax.grid(axis='y',alpha=.2)
axes[0].set_title('평가가격 기준 손익');axes[1].set_title('종료 시 청산비용까지 반영');axes[0].legend()
fig.suptitle('10/25 15:00–17:05 KST: 더 나은 상대 성과와 수익성 통과는 다르다')
figure=savefig(fig,'07_completed_short_replay')
add('표 6·그림 7. 실행 후보의 이미 종료된 짧은 PAS 재생',
    '<p>quote_inputs는 동일 페어 예측을 유지하고 진입 판단에 사용하는 가격 이력·실행 조건을 바꾸는 후보다. '
    '10/25 15:00–17:05 KST 구간에서 평가가격 손익은 기존판보다 높았지만, 종료 청산비용 반영 후에는 모두 손실이었다. '
    '관측 공백도 있어 당시 교체 기준을 통과하지 못했다. 이 표는 실제 주문을 보낸 결과가 아니다.</p>'
    '<p>이 구간은 여러 후보의 개발·선정에 사용되었으므로 독립 검증이라고 부를 수 없다. '
    '같은 경로의 0/45초 및 기본/스트레스 설정도 네 독립 시장 표본이 아니다. '
    '그 이후 고정 검증 구간의 결과는 별도 보고서가 완료된 뒤에만 사용할 수 있다.</p>',qf,figure)

protocol=read_json('reports/quote_history_replication_20261025/protocol.json')
pending=dict(primary=protocol['primary'],start_utc=iso(protocol['start']),end_utc=iso(protocol['end']),
             report_generated_utc=NOW.isoformat(),outcomes_opened_by_this_builder=False)
(HERE/'prospective_scope.json').write_text(json.dumps(pending,ensure_ascii=False,indent=2))
add('진행 중인 고정 검증의 취급',
    '<p>별도로 고정해 진행하던 구간은 10/25 18:00~10/26 00:05 KST다. '
    '이 보고서 작성기는 그 구간의 원자료·중간 손익·검증 결과를 열지 않는다. '
    '완료 후 감사 결과가 있는 별도 보고서를 확인해야 한다. 논문 팩 생성 때문에 새 전략 탐색·학습·주문·운용 교체를 실행하지 않았다.</p>')

claims=pd.DataFrame([
    ['페어 구성으로 과거 순손익 개선','표 1·그림 1–2','관측됨, 과거 개발 기간에 한정','동일 자본의 독립 시계열 구간에서 반복 확인'],
    ['게이트가 항상 성능 개선','표 1의 stress 동일 결과','지원하지 않음','후속 독립 비교 및 비용·지연 민감도'],
    ['시장 방향과 무관한 수익','표 4·그림 5','44일의 설명용 진단만 있음','고정 기준의 시장·관계 레짐별 평가, 위험 노출 비교'],
    ['현재 PAS 페어 모듈의 알파','표 5·그림 6','전체 우리 장부만으로 식별 불가','실거래 모듈별 소유권 손익과 동시 기준 전략'],
    ['전체 45페어 후보가 우수','표 2·그림 3','해당 비교에서 반대 결과','새 선별 규칙은 별도 사전 고정 및 독립 평가'],
    ['예측오차 감소가 거래이익 개선','표 3·그림 4 및 실제 비용 분해','이 진단에서는 두 페어 MAE/RMSE 모두 0 예측보다 나쁘거나 비슷함','실행가능 호가 기준 예측·실현 순이익 보정'],
    ['논문 PPO–LSTM/NFE 재현','현재 전략 구성표','구현하지 않음','재현을 주장하려면 해당 모델·프로토콜·결과의 별도 대조'],
    ['현재판의 통계적으로 유의한 우위','전체 자료','아직 주장할 수 없음','충분한 독립 기간·블록 기반 구간추정·선택 이력 보정'],
],columns=['검토할 주장','현재 증거','가능한 판단','추가로 필요한 증거'])
export(claims,'16_claims_and_evidence')
add('발표·모델 페이퍼에서 주장할 수 있는 범위',
    '<p>좋은 수익률 표만 남기지 않고 비용 실험·부정적 결과·모듈 제거 비교를 함께 보존했다. '
    '현재 결과로는 “비용을 고려한 페어 모듈을 구현했고 특정 과거 구간에서 개선을 관측했다”까지 설명할 수 있다. '
    '“모든 시장에서 우월하다”, “전 45페어를 실거래한다”, “논문 모델을 그대로 재현했다”는 설명은 이 자료와 맞지 않는다.</p>',claims)

metrics=pd.DataFrame([
    ['순손익','실현 가격손익 + 미실현 가격손익 − 수수료 + 순펀딩; 모의실험에서는 체결 가격에 반영된 비용을 중복 차감하지 않음'],
    ['R','공식 계좌 현금흐름 보정 일수익을 연결한 누적 수익률. 단순 배정 자본 대비 비율과 구분'],
    ['일별 표준편차 s','완료된 일수익 n개의 표본 표준편차(ddof=1). 1일만으로 산출 불가'],
    ['기간 변동성 σ','대회 설명 기준 s×sqrt(n); 연율 변동성과 다름. 결측일 수익은 공식 보정 규칙 적용'],
    ['Q','(max(R,0)+0.005)/(abs(R)+1.5×σ+0.05). R과 σ는 백분율 숫자가 아닌 소수'],
    ['총점','0.8×Q+0.2×J. J 미평가 시 0이며 로컬 백테스트로 공식 등수·심사 점수를 정할 수 없음'],
    ['MDD','자산곡선의 이전 고점 대비 최대 하락률. 일별 σ와 다른 위험 지표'],
    ['짧은 구간','몇 시간 재생의 수익을 연율화한 Sharpe/CAGR, 공식 Q·σ로 과장하지 않음'],
],columns=['지표','정의·주의'])
export(metrics,'17_metric_definitions')
add('지표·재현 방법',
    '<p>공식 순위는 12:00 UTC 일별 샘플을 쓰며 실시간 Top 10 그래프와 다를 수 있다. '
    '<a href="https://perpetual-alpha26.github.io/leaderboard">공식 리더보드 설명</a>. '
    'Q 수식은 이 프로젝트에 제공된 대회 공지 기준이며, 표의 Q는 저장된 실험 계산값이다. '
    '현재 공식 점수를 재조회하거나 모든 과거 Q를 새로 인증한 것은 아니다.</p>'
    '<p>실행: <code>/usr/bin/python3 reports/pairs_evidence_pack_20261025/build_report.py</code>. '
    '저장된 계좌·예측 기록만 읽어 표와 그림을 만든다. 모델을 재학습하거나 백테스트 경로를 새로 선택하지 않는다. '
    'CSV 전체 행·그림 원자료·소스 SHA-256은 함께 저장했다. HTML은 오프라인 열람 가능하고 PDF에는 7개 그림이 들어 있다.</p>',metrics)

PDF.close()
style='''body{font:16px/1.75 system-ui,sans-serif;color:#17283b;max-width:1250px;margin:40px auto;padding:0 24px;background:#fbfcfe}h1{font-size:30px;line-height:1.3}h2{font-size:23px;margin-top:48px;border-bottom:2px solid #dbe5ef;padding-bottom:8px}h3{font-size:18px}p{max-width:1080px}article{padding:10px 22px;margin:18px 0;border:1px solid #d4dfeb;border-radius:10px;background:white}article p{margin:9px 0}.meta{color:#52677f;font-size:14px}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{border:1px solid #dce3ec;padding:9px;text-align:left;vertical-align:top}th{background:#edf3f9}.tablewrap{overflow-x:auto}img{width:100%;max-width:1120px;border:1px solid #ddd;background:white}a{color:#086cb7}code{font-size:13px;background:#edf1f6;padding:2px 5px}nav{padding:20px;background:#edf3f9}li{margin:5px 0}@media print{body{margin:0;font-size:11px}h2{break-before:page}article,img,table{break-inside:avoid}nav{display:none}}'''
title='SFINX 페어트레이딩: 논문 근거와 실제 결과 자료집'
intro=('<p>기준일: '+NOW.astimezone(timezone(timedelta(hours=9))).isoformat(timespec='seconds')+
       '. 논문 원문에서 확인한 이론·평가 방식과 저장된 프로젝트 결과를 연결한다. '
       '<b>새로운 성능 개선 실험이나 현재 순위 보고서가 아니다.</b></p>')
out=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
     f'<title>{title}</title><style>{style}</style><h1>{title}</h1>',intro,
     '<nav><a href="figures.pdf">그림 PDF</a> · <a href="README.md">설명 Markdown</a> · <a href="manifest.json">출처·해시</a> · <a href="tables/01_papers_and_evidence.csv">논문 근거 CSV</a></nav>',
     '<h2>논문은 무엇으로 성능을 검증했는가</h2>',paper_html]
md=['# '+title,'', '기준일: '+NOW.isoformat(),'',
    '논문 원문과 기존 저장 결과의 정리. 새로운 독립 성능 실험 또는 현재 실시간 순위가 아니다.', '',
    '## 논문별 근거','']
for p in papers:
    md.extend([f'### {p["id"]}. [{p["title"]}]({p["url"]})', '',f'{p["year"]} · {p["kind"]}', ''])
    for key in ['theory','evidence','finding','limitation','adoption']:
        md.extend([p[key], ''])
from html.parser import HTMLParser
class Plain(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data)
    def handle_endtag(self,tag):
        if tag=='p':self.parts.append('\n\n')
for heading,prose,frame,figure in SECTIONS:
    out.extend([f'<h2>{heading}</h2>',prose])
    md.extend(['## '+heading,''])
    p=Plain();p.feed(prose);md.extend([''.join(p.parts).strip(),''])
    if frame is not None:
        out.append('<div class="tablewrap">'+table(frame)+'</div>')
        md.extend([markdown_table(frame),''])
    if figure:
        out.append(f'<figure><img src="figures/{figure}.png" alt="{heading}"><figcaption>원자료는 tables/ CSV에 포함. PNG·SVG·통합 PDF 제공.</figcaption></figure>')
        md.extend([f'![{heading}](figures/{figure}.png)',''])
out.append('</html>')
(HERE/'report.html').write_text('\n'.join(out))
(HERE/'README.md').write_text('\n'.join(md))
source('reports/pairs_evidence_pack_20261025/build_report.py')
manifest=dict(generated_utc=NOW.isoformat(),sources_sha256=SOURCE_HASHES,
              external_sources=[dict(id=p['id'],url=p['url'],version=p['year']) for p in papers],
              orders_sent=0,live_changed=False,model_fit=False,new_strategy_search=False,
              prospective_outcomes_read=False,independent_test_claim=False,
              notes=['All plots are derived from saved historical records; no paper images copied.',
                     'Source histories were reused. Comparison table version/capital/venue must remain attached.',
                     'Rebuilding needs the existing repository source files; the ZIP can be viewed standalone.'])
(HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
validation=dict(historical_rows=len(hist),auto45_rows=len(auto),curve_points=len(curve_rows),
                equity_endpoints_match=True,pair_cost_identity=True,owner_pnl_identity=True,
                forecast_rows=len(fc),forecast_summary=fcstats.to_dict('records'),
                output_tables=len(list(TABLES.glob('*.csv'))),output_figures=len(list(FIGURES.glob('*.png'))),
                orders_sent=0,live_changed=False)
(HERE/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
archive=HERE/'SFINX_pairs_evidence_20261025.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p!=archive and '__pycache__' not in p.parts:
            z.write(p,p.relative_to(HERE))
print(json.dumps(validation,ensure_ascii=False,indent=2))
print('Saved',HERE/'report.html')
