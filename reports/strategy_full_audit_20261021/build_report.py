"""Build a standalone Korean audit from preserved evidence."""
import datetime as dt
import hashlib
import html
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(name):
    return json.loads((OUT / name).read_text())


def code(path, line=None):
    target = str(ROOT / path)
    return f'<a href="{target}"><code>{html.escape(path)}{":" + str(line) if line else ""}</code></a>'


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{x}</th>" for x in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows) + \
        "</tbody></table></div>"


def main():
    data = read("analysis.json")
    account = data["account"]
    trace = read("signal_trace.json")
    runtime = read("runtime_summary.json")
    latest = trace["rows"][-1]
    history = read("../isolated_repairs_20261020/reference_results.json")
    board = read("raw/official_leaderboard.json")
    official = next(t for t in board["response"]["teams"] if t["name"] == "SFINX")
    market_rows = []
    for m in data["markets"]:
        market_rows.append([
            m["symbol"], f'{m["reference_168h_pct"]:+.2f}%', f'{m["reference_24h_pct"]:+.2f}%',
            f'{m["reference_6h_pct"]:+.2f}%', f'{m["pas_mark_24h_pct"]:+.2f}%',
            f'{m["spread_bps"]:.2f}', f'{m["funding_bps_hour"]:+.3f}',
            f'{m["volume_24h_usdp"]:,.0f}', f'{m["raw_weight"]*100:.3f}%',
        ])
    liquidity_rows = [[m["symbol"],
        f'{m["bid_depth_10bp_from_best_usdp"]:,.0f}', f'{m["ask_depth_10bp_from_best_usdp"]:,.0f}',
        f'{m["instant_roundtrip_cost_bps"]["1000"]:.2f}',
        f'{m["instant_roundtrip_cost_bps"]["5000"]:.2f}',
        f'{m["basis_mark_bps"]:+.2f}', f'{m["mainnet_funding_bps_hour"]:+.3f}',
        f'{m["pas_zero_trade_hours_since_start"]}/{m["pas_completed_hours_since_start"]}'
        ] for m in data["markets"]]
    stages = [
        ("추세·캐리 원 신호", "trend_carry"),
        ("JM 필터", "jm_filtered"),
        ("Chronos sticky 필터", "chronos_filtered"),
        ("두 필터 평균", "mean_filtered"),
        ("기준 위험 재정규화", "risk_normalized"),
        ("0.1% 위험 목표·3/7일 규모 완화", "risk_sized"),
        ("뒤에 추가하는 되돌림 기여", "reversion_added"),
        ("혼합 후", "combined"),
        ("고정 배율 적용한 최종 목표", "final"),
    ]
    stage_rows = [[label] + [f"{latest['stages'][key][j]*100:.6f}%" for j in range(3)]
                  for label, key in stages]
    history_rows = []
    labels = {"incumbent":"현재판", "short_only":"숏 추가", "risk_only":"감축 보존",
              "risk_short_exit_all":"전 종목·숏·위험 변경", "size_only":"고정 규모 보정 제거"}
    for phase, phase_name in [("continuous","8/1~10/15"),("comparison","9/1~10/15")]:
        for variant, name in labels.items():
            r = next(x for x in history if x["variant"] == variant and x["phase"] == phase and x["scenario"] == "base")
            history_rows.append([phase_name,name,f'{r["net_pnl"]:+,.2f}',f'{r["base_pnl"]:+,.2f}',
                f'{r["pair_pnl"]:+,.2f}',f'{r["Q"]:.5f}',f'{r["volatility"]*100:.4f}%'])
    sections = f"""
<header><p class="eyebrow">SFINX · 실제 운용판 독립 재점검 · 2026-10-21 KST</p>
<h1>모델은 복잡하지만, 현재 운용은 작은 BTC 추세 롱과 HYPE 되돌림 롱이다.</h1>
<p class="lead">핵심 한계는 PAS에서 검증된 비용 차감 수익 근거의 부족, 최종 비중까지 일관되지 않은 위험 필터,
좁은 기회 범위와 느린 조정, 그리고 장애 시 포지션 관리의 빈틈이다.
현재 계좌가 작은 플러스라는 사실만으로 예측력이 검증되지는 않는다.</p>
<p>시장·계좌 기준: <b>2026-10-21 01:20:28 KST</b>. 기준시장 시간봉은 01:00까지의 완결 자료.
서비스와 런타임은 그 이후 별도로 읽었다. 시각이 다른 관측값을 동일 시점으로 합치지 않았다.</p></header>

<section><h2>1. 실제로 무엇을 운용 중인가</h2>
<p><code>first_dense_hourlyhedge_decisiongate_24</code>를 Linux의
<code>competition_live.producer</code>와 <code>competition_live.runner --execute</code>가 실행한다.
테스트넷 PAS·USDP, 팀 지갑 <code>0x125778065a578d598D88bddaA3FdD635280d0E47</code>.
팀 자본 100,000 중 이 전략에 50,000, 팀원 예약에 50,000이다. 모델 자체를 거래소에 업로드한 구조가 아니다.</p>
<p>이번 감사는 첨부 문서의 결론을 그대로 채택하지 않고 실제 import 경로, 코드, 원장, 공개 API, 저장된 목표를 대조했다.
첨부 문서의 과거 hold나 후속 작업 지시는 현재 사용자 요청의 별도 지시로 취급하지 않았다.
브라우저에는 테스트넷 화면 열기를 요청했으며 로그인 완료·화면 직접 열람은 확인하지 못했다.
계좌와 시장 수치는 인증 없는 공개 API로 검증했다.</p>
{table(["항목","01:20:28 KST 관측"],[
["USDP 총자산",f'{account["equity"]:,.6f}'],
["순손익 / 팀 수익률",f'{account["pnl"]:+.6f} / {account["return_pct"]:+.6f}%'],
["가격 미실현손익 / 가격 실현손익",f'{account["unrealized"]:+.6f} / 0'],
["펀딩 수령 / 지급 수수료",f'{account["funding"]:+.6f} / −{account["fees"]:.6f}'],
["보유","BTC +0.01899 · HYPE +3.52"],
["명목 노출",f'{account["gross"]:,.2f} USDP · 팀의 {account["gross_team_pct"]:.4f}% · 운용 배정액의 {account["gross_sleeve_pct"]:.4f}%'],
["거래","매수 주문 2개 / 체결 조각 3개 / 청산 0 / 페어 진입 0 / 미체결 0"],
["런타임 표본 손익 범위",f'{runtime["minimum_team_pnl"]:+.2f} ~ {runtime["maximum_team_pnl"]:+.2f} USDP'],
])}
<p><b>1.343250 + 4.222996 − 1.656233 = 3.910013 USDP.</b> 공개 계좌 총액과 오차 약 6×10⁻¹³으로 맞는다.
통합 계좌의 USDP total에는 평가손익이 들어 있으므로 미실현손익을 다시 더하지 않았다.
현재 플러스에는 펀딩의 기여가 가격손익보다 크다. 이 진입을 의도된 PAS 캐리 모델의 성과로 해석할 수도 없다:
기본 캐리 입력은 mainnet 펀딩이다.</p>
<p>현금 여유를 나타내는 <code>(total−hold)/total</code>은 98.1585%다. 이는 별도 주식 현금계정 같은 분류가 아니라
현재 cross 1x와 통합 계좌에서의 담보 미사용 비율이다.</p></section>

<section><h2>2. 현재 시장: 일주일 약세, 단기 반등, 종목별 차이</h2>
<p>BTC·ETH·HYPE는 7일 전 대비 각각 −0.98%, −2.97%, −1.27%이지만, 최근 완결 24시간은
+0.80%, +1.97%, +1.47%다. PAS 현재 mark의 24시간 등락도 세 종목 모두 플러스다.
EWY는 24시간 −2.46%이지만 최근 6시간 +2.31% 반등했다.
따라서 “지금 전 종목이 계속 내려가니 즉시 숏”이라는 결론은 자료가 지지하지 않는다.
여기서 약세는 측정한 구간의 수익률을 뜻하며 장기 하락장 확정이나 다음 방향 예측이 아니다.</p>
{table(["종목","기준 7일","기준 24h","기준 6h","PAS 24h","스프레드 bp","PAS 펀딩 bp/h","PAS 24h 거래액","전략 목표"],market_rows)}
<p>기준시장 수익률은 같은 01:00 종료 시간봉끼리 비교했다. PAS 24시간은 API의 <code>markPx/prevDayPx−1</code>.
두 열은 가격 종류와 종료시각이 다르다. 목표는 50,000 운용 sleeve에 대한 비중이다.</p></section>

<section><h2>3. 유동성: 현재 규모의 주된 제약은 아니지만, 비용과 지속성은 제약이다</h2>
<p>BTC의 best bid에서 아래 10bp 이내 매수 대기액은 약 23,456, best ask에서 위 10bp 이내 매도 대기액은
약 65,880 USDP다. HYPE도 각각 약 57,932와 30,871이다.
현재 보유 1,842를 “호가가 없어서”라고 설명할 근거는 없다.
다만 수량 확대 시 이 호가가 그대로 남는다고 가정할 수 없고, 시간대별 반복 관측도 필요하다.</p>
{table(["종목","매수 대기액 ≤10bp","매도 대기액 ≤10bp","1천 왕복 bp","5천 왕복 bp","PAS/main mark 괴리 bp","main 펀딩 bp/h","무거래 시간/완결 시간"],liquidity_rows)}
<p>왕복 비용은 같은 호가를 매수·매도 방향으로 소진한 평균가격 차이에 taker 9bp×2를 더한 정적 미리보기다.
실제 진입·청산 거래가 아니며 지연, 호가 취소, 미래 가격 변화, 미래 펀딩은 제외한다.
깊이 10bp는 각 방향 best quote 기준이지 중간가격 기준이 아니다. 무거래 시간은 본대회 시작 후 완결된 28시간 중 거래수 0인 시간이다.
이는 현재 호가가 없다는 뜻은 아니다.</p>
<p><b>거래 수수료만 왕복 18bp.</b> 1천 USDP 왕복은 BTC 약 23.02bp, ETH 22.74bp, HYPE 26.20bp이며
5천에서는 각각 28.27, 26.39, 29.71bp다. 예측 edge가 10~20bp라면 단순 방향 적중만으로 부족하다.
maker 주문으로 바꾸는 안도 체결률·대기열·역선택을 포함해 비교해야 하며 무료 개선으로 간주하면 안 된다.</p>
<p>PAS BTC·ETH·HYPE의 당시 펀딩은 약 −0.70~−0.77bp/h인데 mainnet은 각각 +0.125bp/h다.
롱의 지급·수령 방향까지 다르다. 이 차이와 basis 변동은 mainnet 수익률 공분산만으로 충분히 표현되지 않는다.</p></section>

<section><h2>4. 처음부터 추적한 신호: HYPE 롱의 실제 출처와 필터 상쇄</h2>
<p>저장된 Chronos 예측·JM 지표를 입력으로 원 신호, 필터, 위험 정규화, 규모 배분, 되돌림, 고정 배율을 재계산했다.
<b>5,187개 시간 행 전체의 최종 목표가 저장값과 정확히 일치했다(최대 오차 0).</b>
Chronos를 새로 학습하거나 새로운 성과 백테스트를 수행한 것은 아니다.</p>
{table(["단계 · 10/20 21:00 KST","BTC 비중","ETH 비중","HYPE 비중"],stage_rows)}
<p><b>HYPE는 추세·캐리 단계에서 0이고, 최종 위험 필터 뒤의 7일 되돌림에서 추가된다.</b>
JM의 HYPE 상승상태 지표는 약 4.19×10⁻⁸로 하락 상태에 가깝지만, 이 필터는 뒤에서 추가되는 되돌림에 다시 적용되지 않는다.
이 수치는 확률 보정된 ‘상승 확률’이 아니라 이진 상태를 지수 평활한 지표다.
되돌림에는 별도의 장기 EMA 방향 일치 조건이 있으므로 무조건 역추세 매수는 아니지만, 두 국면 판단을 조정하는 최종 규칙은 없다.</p>
<p>BTC에는 Chronos sticky 감축이 남아 0.4871%→0.2436%가 되지만, JM과 평균하고 위험 정규화·목표 위험 배분을 거치며
최종 목표는 3.1667%가 된다. 서로 다른 단계의 단위를 혼동하면 안 된다.
핵심은 <b>감축 신호가 최종 절대 노출의 같은 비율 감축을 보장하지 않는다</b>는 점이다.
<code>targets</code>는 다시 목표 변동성/예상 변동성으로 배율을 계산하므로, 일정 양의 배율을 곱한 입력은 다시 같은 위험으로 확대될 수 있다.
이는 명시된 위험 목표 정책이며 단순 산술 오류는 아니다.</p>
<p>고정 tail 배율 0.693879는 실시간 급락 감지기가 아니다. Chronos도 현재 단독 방향 주문 모델이 아니라 기존 신호 감축 필터다.
현재 active policy는 <code>v10_frozen</code>이며 파일에 저장된 learned ensemble 계수가 최종 활성 배분이라는 뜻은 아니다.</p>
<p>근거: {code("replay_lab/v10_revision/core.py",69)},
{code("replay_lab/portfolio_rebuild_v10/lifecycle_allocation.py",15)},
{code("replay_lab/reversion_blend/agent.py",13)},
{code("replay_lab/reversion_blend/model.py",45)},
<a href="signal_trace.json">정확 재구성 결과</a>.</p></section>

<section><h2>5. 왜 많이 안 사고, 왜 안 바꾸는가</h2>
<p>현재 raw 목표 총비중은 3.8410%, 약 1,920.66 USDP이며 마지막 risk projection이 줄인 값은 정확히 0이다.
01:20 공분산 기준 목표 일별 위험은 운용 sleeve의 약 0.06451%로 후단 상한 0.5%보다 훨씬 작다.
앞단 목표 0.1%, 3/7일 규모 완화, 고정 배율, 제한된 활성 종목의 결과다.</p>
<p>보유와 현재 목표 수량의 차이는 BTC 약 1.32%, HYPE 약 17.25%로 20% 동일 방향 거래 생략 기준 안이다.
실제로 시작부터 다섯 번의 6시간 경계를 처리했지만 외부 주문 packet은 첫 진입 하나뿐이다.
자본 50% 배정은 배정액 50%를 항상 투자한다는 뜻이 아니다. 예약 자본은 합의된 제약으로 존중해야 한다.</p>
<p>다만 <b>이 작은 위험 목표가 현재 팀 단위 Q 점수와 PAS 비용에서 적절한지 입증한 자료는 부족하다.</b>
비중을 단순히 키우면 수익과 손실, 비용, 슬리피지가 함께 커진다.
고정 가격 경로에서 R과 σ가 k배 선형 확대된다는 제한된 가정하에
Q=(kR+0.005)/(k(R+1.5σ)+0.05)는 R&gt;σ/6일 때 k와 함께 증가한다.
이 산술은 낮은 변동성 자체가 항상 높은 Q라는 직관이 틀릴 수 있음을 보여줄 뿐, 실전 확대의 검증은 아니다.</p>
<p>EMA 10/60일과 6시간 판단, <code>exit_cross=False</code>, ATR 청산은 짧은 급락에 즉각 반응하는 설계가 아니다.
숏 실행 자체는 지원하지만 기본 EMA 선택은 롱 중심이다. 별도 숏·전 종목 후보의 성과는 아래처럼 일관되지 않았다.
기본 포트폴리오에는 PAS의 종목별 가격 급변을 직접 판단하는 독립적인 장중 손절 계약이 없고, 전략 청산은 기준시장 신호·전체 계좌 위험에 의존한다.
현재 노출에서 두 롱이 10% 하락해도 팀 손실 약 184 USDP이므로, 운용 sleeve 8% 최대낙폭 중단은 빠른 개별 손절을 대신하지 못한다.</p>
<p>또한 {code("competition_live/coordinator.py",92)}의 정상 리밸런싱 창은 경계 후 15분이다.
해당 창을 신호 생산 지연으로 놓치면 같은 방향 조정·전략 청산을 다음 경계까지 놓칠 수 있다.
현재 창 누락을 관측한 것은 아니며, 지금 producer 지연은 약 2분이다.</p></section>

<section><h2>6. 페어 모델이 쉬는 이유: 필터만 느슨하게 해서는 해결되지 않는다</h2>
<p>런타임 기록의 114회 검사 시점, 두 페어·양 방향 456개 검사 모두
<code>decision_gate_rejected</code>였다. 456개 독립 매매 기회가 있었음을 뜻하지 않는다.
현재 BTC/ETH gate score 0.1922는 임계값 0.7908의 24.3%, BTC/HYPE 0.2914는 임계값 0.5059의 57.6%다.</p>
<p>가격 예측은 각각 약 +11.25bp, +20.82bp다. 비용 gate의 score를 이 기대수익률과 동일시하지 않았다.
<b>현재 보존한 호가에서 gate를 제외하고 실제 planner만 호출해도 양 페어·양 방향 모두
<code>edge_below_full_cost</code>로 거절된다.</b>
따라서 지금 gate를 해제하면 수익성 있는 거래가 나온다는 주장은 성립하지 않는다.
이는 현재 한 시점의 진단이며 과거 모든 거절을 정당화하는 증거는 아니다.</p>
<p>90일 창의 각 356개 학습 행은 6시간 간격의 24시간 라벨이라 서로 겹친다.
실질 독립 표본 수를 356으로 볼 수 없다. 가격 20개 특징에는 PAS order flow·깊이·basis·펀딩이 직접 들어가지 않는다.
planner는 그 정보를 사후 경제성에 반영하지만, gate는 고정 38bp proxy 비용으로 먼저 차단한다.
현재 venue 경제성을 하나의 학습·판단 계약으로 연결할 필요가 있다.</p>
<p>추가로 pair relative-return 예측을 <code>A×(1+μ), B×(1−μ)</code>라는 두 절대 terminal 가격으로 만든다.
source 시점의 고정 hedge payoff와는 맞지만 공통 시장 방향까지 예측한 것은 아니다.
시간이 지나 current PAS 호가와 대조할 때 공통 가격 움직임·basis·beta 오차를 별도 검증해야 한다.
현재 두 모델은 BTC/ETH와 BTC/HYPE뿐이므로 EWY 등 다른 시장의 방향성 기회를 직접 찾지 않는다.</p>
<p>근거: {code("deliverables/sfinx_pairs_candidate_20261018_final/source/experiments/decision_gate_publication_20261018/signals.py",75)},
{code("deliverables/sfinx_pairs_candidate_20261018_final/source/experiments/published_pair_lifecycle_20261017/signals.py",145)},
<a href="ungated_planner_snapshot.json">gate 제외 planner 진단</a>.</p></section>

<section><h2>7. 코드에서 재현한 운영 결함</h2>
<article><h3>P1 · 오래된 신호와 재시작이 겹치면 보유 포지션의 위험 루프도 시작하지 않는다</h3>
<p>보유 BTC가 있는 checkpoint와 7시간 된 신호를 넣고 <code>main(execute=True, once=True)</code>를 실행했다.
결과는 <code>waiting_for_fresh_signal</code>이며 <code>LiveRunner</code> 생성 전에 반환했다.
따라서 이미 실행 중일 때의 stale-signal 강제 청산과 재시작 시 동작이 다르다.
신호 생산이 계속 고장나 있으면 복구·감축 로직이 계속 대기할 수 있다.</p>
<p>수정 방향: 기존 계좌 복원·체결 대조·보유 위험 관리를 신호 신선도와 분리하고,
오래된 신호는 신규 위험 증가를 차단하면서 감축 루프는 동작하게 해야 한다.
현재 실제 서비스는 신선한 신호로 정상 실행 중이고 이 장애가 발생한 기록은 없다.
근거: {code("competition_live/runner.py",274)}.</p></article>
<article><h3>P1 · 첫 다리의 체결 확인 지연으로 다음 다리의 호가가 만료된다</h3>
<p><code>LiveAccount.execute_books</code>는 처음 받은 streams를 종목 순서대로 재사용한다.
첫 주문 확인에 4초가 걸리는 기존 재현에서 두 번째 목표가 있어도 context 3초 제한 때문에 건너뛴다.
페어에서는 한쪽만 노출될 수 있고 기본 포트폴리오에서도 일부 종목 조정이 누락될 수 있다.
종목별 호가를 갱신하는 별도 후보의 같은 재현은 두 주문을 완료한다.</p>
<p>실제 첫 packet에도 SP500 이하 <code>NO_FRESH_BOOK</code> 기록이 있지만 그 종목들의 목표는 0이었다.
BTC·HYPE는 모두 체결됐다. 따라서 지금의 작은 노출을 이 결함 탓으로 돌리지 않는다.
수정 방향: 실제 필요한 종목마다 quotes와 시각을 갱신하고,
확인 지연·한쪽 부분 체결·감축을 같은 테스트에서 검증한다.
근거: {code("competition_live/live_account.py",72)}, {code("pas_strategy_audit/test_execution.py",35)}.</p></article>
<article><h3>P2 · 보유하지 않은 종목의 context가 빠져도 다른 보유 종목의 감축까지 막힌다</h3>
<p>BTC만 보유한 모의 계좌에서 USDJPY context 하나를 제거했다.
신호가 없어 강제 감축해야 하는 조건에서도 <code>missing_account_valuation</code>으로 반환하고 청산 호출에 도달하지 않았다.
같은 context만 되돌리면 목표 0의 감축 호출이 실행된다.
수정 방향: 필요한 계좌 평가와 증거 있는 reduce-only 감축을 미보유 종목의 가격 가용성과 분리해야 한다.
가상의 가격으로 손익을 메우는 방식은 피한다.
근거: {code("competition_live/coordinator.py",40)}, {code("competition_live/coordinator.py",68)}.</p></article>
<p><a href="operational_reproductions.json">새로 재현한 두 조건과 결과</a> ·
<a href="reproduce_operational_gaps.py">네트워크·서명 없이 실행하는 재현 코드</a>.
재현은 임시 디렉터리, 가짜 계좌/서비스와 mock을 사용했다. 운용 파일과 서비스는 수정하지 않았다.</p></section>

<section><h2>8. 과거 성과의 실질 근거: 전체 수익의 절반, 최근 구간의 대부분이 작은 페어 표본에 의존한다</h2>
{table(["기간 UTC","후보","순손익 USDP","기본 포트폴리오","페어","Q","σ"],history_rows)}
<p>현재 배정 50,000 + 예약 50,000의 기존 저장 평가를 다시 읽은 표다. 모두 mainnet 기준시장·재사용 역사·base 비용 조건이다.
새 독립 백테스트가 아니다. 현재판 전체 +1,296.35 중 pair +616.29, 약 47.5%다.
9/1 이후 +191.34 중 pair +168.90, 약 88.3%이며 기본 포트폴리오의 기여는 +22.44에 그친다.
전체 완료 pair는 5건, stress는 3건이다. PAS 실전의 pair는 0건이다.</p>
<p>따라서 ‘페어 결합으로 성능이 좋아진 모델’이라는 역사적 설명과 ‘현재 PAS에서 작동하는 수익 엔진’은 아직 연결되지 않았다.
전체 수익의 약 85.3%는 8월 구간에서 나왔고 이후 수익은 약했다.
단순 숏 추가는 전체에서는 악화했지만 후반은 개선했다.
규모 보정 제거도 전체 Q는 개선했지만 후반 Q는 악화했다.
이 결과는 성능 향상을 단정할 근거가 아니라 국면·규모·실행 효과를 분리할 필요의 근거다.</p>
<p>라벨 maturity의 과거 제한과 인과성 검사는 코드에 있다. 그러나 같은 구간을 반복해서 후보 선택에 썼다는 문제는 남는다.
Chronos의 외부 사전학습 corpus 전 범위를 이번 감사에서 검증하지도 않았다.
큰 신경망을 더 얹기 전에 current baseline·단순 가격 지속/무변화·PAS 펀딩 기준·양방향 단순 추세·현금과의
비용 차감 비교, nonoverlap 예측 진단, 다음 미사용 구간의 고정된 검증 계획이 필요하다.</p></section>

<section><h2>9. 운영 구조와 점수 해석</h2>
<p>실행기 import의 상당수는 동결 bundle/source, 별도 기본 신호 subprocess는 루트 소스다.
루트 파일 수정·테스트가 실제 실행기 변경을 뜻하지 않는다. 이번 감사에서 각 경로를 별도 프로세스로 확인했다.
배포 20개 파일, bundle manifest, 기본 신호의 324개 보호 파일 해시가 일치했다.</p>
<p>실제 두 서비스는 <code>active/running</code>, 재시작 횟수 0, <code>Restart=on-failure</code>, <code>Transient=yes</code>다.
호스트 재부팅 후 자동 기동을 보장하는 영구 unit은 현재 실행 경로가 아니다.
계좌·호가 수집과 주문 확인의 동기 호출, 전체 coordinator pickle을 tick마다 쓰는 방식,
계속 누적되는 snapshots/events도 장기 운용 지연과 복구 비용을 관리할 대상이다.
현재 이로 인한 메모리·성능 장애가 관측된 것은 아니다.</p>
<p>주최 측 공개 데이터 생성시각 {html.escape(board["response"]["generated_at"])} 기준 SFINX는
<code>eligible=false</code>, 이유 <code>few_daily_returns</code>, Q/Score 미산정이다.
일일 표의 수익률 {official["return"]*100:+.6f}%와 01:20 계좌의 {account["return_pct"]:+.6f}%는 서로 다른 관측이다.
12:00 UTC 일일 샘플 기반 순위와 10분 간격 실시간 Top 10 차트를 섞으면 안 된다.
평가식은 Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), Score=0.8Q+0.2J로 현재 공개 설정과 코드가 맞는다.
최종 공인 결과는 주최 측에 있다.
<a href="https://perpetual-alpha26.github.io/leaderboard">공식 리더보드</a>,
<a href="raw/official_leaderboard.json">조회 원자료</a>.</p></section>

<section><h2>10. 개선 우선순위와 통과 기준</h2>
{table(["순서","작업","완료를 판단할 근거"],[
["1 · 운영 결함","신호와 독립된 계좌 복구·감축 루프, 종목별 fresh quote, 미보유 종목 장애 격리",
 "보유 상태 재시작·신호 없음·지연·한쪽 체결·API 누락에서도 계좌 대조와 유효 감축이 작동하는 통합 재현"],
["2 · 위험 규칙","추세·되돌림·페어를 합친 뒤 최종 위험 감축의 의미를 고정",
 "국면·예측 충돌 시 최종 비중과 리스크가 어떻게 바뀌는지 수치로 설명; scalar 재확대·되돌림 우회 여부 검사"],
["3 · PAS 경제성","mainnet 방향 예측과 PAS basis·실제 펀딩·스프레드·깊이·exit 비용을 분리 기록",
 "신호가 예상한 순수익과 실제 체결 후 가격/basis/펀딩/비용 PnL의 대조"],
["4 · 알파 검증","단순 양방향·현금·캐리 기준 대비 현재 모델의 추가 가치를 평가",
 "같은 정보시각·같은 비용·같은 예산; 재사용/미사용 구간 명시; pair 거래 수와 nonoverlap 표본 수 보고"],
["5 · 규모 결정","현재 50,000 배정 안에서 규모별 Q·최대낙폭·체결 가능성 비교",
 "작은 주문 가격의 선형 복제 금지; depth·지연 stress; 예측력 변화와 exposure 효과 분리"],
["6 · 배포 단순화","활성 코드·설정 단일 식별, 의존성 고정, 영구 서비스·장애 경보",
 "재부팅·프로세스 종료·재시작 드릴과 rollback 절차; 실제 활성 해시 확인"],
])}
<p>현재 자료만으로 “바로 숏 전환”, “gate 제거”, “노출 10배 확대”, “최신 연구 후보 교체”를 정당화할 수 없다.
반대로 위 세 운영 결함은 수익 예측의 우열과 독립적으로 고쳐야 할 구체적인 문제다.
이번 요청의 범위인 검토와 진단은 수행했고, 실제 전략 교체·주문·포지션 변경은 수행하지 않았다.</p></section>

<section><h2>11. 검증 범위와 재현 자료</h2>
<ul>
<li>공개 PAS 계좌·10종목 호가·context·본대회 candle·수수료·펀딩·입금 조회, mainnet/xyz context, 공식 순위 데이터.</li>
<li>기존 실행/원장/예산/재시작/지연 재현 테스트 <b>22개 통과</b>. 수익성을 입증하는 테스트는 아니다.</li>
<li>5187개 공개 신호 목표 행의 계산 경로 정확 재구성. 저장된 모델 출력을 사용했으므로 Chronos 내부 수치 추론을 새로 검증한 것은 아니다.</li>
<li>새 운영 경계 2개 재현, gate 제외 planner 4개 방향의 비용 거절 확인, 비중 projection 감소 없음 확인.</li>
<li>현재 호가 한 시점과 누적 계좌 기록만으로 전체 PAS 구간의 역사적 L2 체결이나 미래 수익을 재현했다고 주장하지 않는다.</li>
<li>소스 검토는 현재 활성 진입점부터 실제 신호·모델·위험·실행·복구 의존 경로와 관련 평가를 대상으로 했다.
저장소의 모든 폐기 연구 파일에 무결점 검증을 했다는 뜻은 아니다.</li>
</ul>
<p><a href="analysis.json">시장·계좌 계산</a> · <a href="signal_trace.json">신호 단계</a> ·
<a href="runtime_summary.json">런타임 집계</a> · <a href="operational_reproductions.json">장애 재현</a> ·
<a href="verification.json">검증 기록</a> · <a href="capture_manifest.json">최초 수집 해시</a> ·
<a href="evidence_manifest.json">최종 증거 해시</a></p>
<p>실행 스크립트: <a href="capture_public.py">공개 API 수집</a>,
<a href="analyze.py">시장 계산</a>, <a href="trace_signal.py">기본 신호 추적</a>,
<a href="reproduce_operational_gaps.py">운영 재현</a>. 감사가 전송한 주문 0, signer 로드 0.</p>
<p>공개 계좌 API 해석 근거:
<a href="https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint">Hyperliquid 공식 Info endpoint 문서</a>.
API 정보와 현장 수치의 구분을 위해 각 응답의 request·수신시각·원문을 보존했다.</p></section>
"""
    css = """
    :root{color-scheme:light;--ink:#192c3b;--muted:#526573;--line:#d9e2e8;--accent:#186b65}
    *{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:var(--ink);font:16px/1.75 system-ui,-apple-system,sans-serif}
    main{max-width:1180px;margin:40px auto;padding:38px 48px;background:#fff}
    header{padding-bottom:28px;border-bottom:3px solid var(--accent)}h1{font-size:34px;line-height:1.4;letter-spacing:-.03em;max-width:960px}
    h2{font-size:24px;line-height:1.4;margin:0 0 18px}h3{font-size:19px;line-height:1.5}
    section{padding-top:36px}p{margin:14px 0}.lead{font-size:19px}.eyebrow{font-size:14px;color:var(--accent);font-weight:600}
    a{color:#145d87;overflow-wrap:anywhere}code{font-size:.88em;background:#eef2f4;padding:2px 4px;overflow-wrap:anywhere}
    table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.55}th,td{padding:11px 9px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
    th{background:#eef4f4;font-weight:600}th:first-child,td:first-child{text-align:left}.table-wrap{overflow:auto;margin:20px 0}
    article{border-left:3px solid #cc8a42;padding-left:20px;margin:25px 0}
    li{margin:8px 0}footer{border-top:1px solid var(--line);margin-top:40px;padding-top:18px;color:var(--muted);font-size:13px}
    @media(max-width:700px){main{padding:24px 20px;margin:0}h1{font-size:27px}h2{font-size:21px}}
    @media print{body{background:white}main{margin:0;padding:10px;max-width:none}section{break-inside:auto}a{color:inherit}table{font-size:10px}}
    """
    doc = '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' \
          f"<title>SFINX 현재 전략·시장·코드 종합 감사</title><style>{css}</style></head><body><main>{sections}" \
          '<footer>작성 기준: 2026-10-21 KST · 원자료 관측시각을 우선한다. 실제 운용 파일·서비스·포지션 변경 없음.</footer></main></body></html>'
    (OUT / "report.html").write_text(doc)
    hashes = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "evidence_manifest.json"}
    (OUT / "evidence_manifest.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n")
    print(f"report: {OUT / 'report.html'}; evidence files: {len(hashes)}")


if __name__ == "__main__":
    main()
