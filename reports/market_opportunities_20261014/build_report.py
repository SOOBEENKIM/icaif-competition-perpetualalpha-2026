"""Render the frozen research evidence and decisions; no strategy orders or model fitting."""
import csv, datetime, html, json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def esc(x):return html.escape(str(x))
def num(x,d=2):return '—' if x is None else f'{x:,.{d}f}'
def kst(ms):return datetime.datetime.fromtimestamp(ms/1000,datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def main():
    a=json.loads((HERE/'analysis.json').read_text());papers=json.loads((HERE/'literature.json').read_text());validation=json.loads((HERE/'validation.json').read_text())
    history={r['coin']:r for r in a['historical']};live={r['coin']:r for r in a['live']}
    summary=[]
    for coin,h in history.items():
        s=live[coin]
        summary.append(dict(coin=coin,spread_median_bps=s['spread_bps']['median'],spread_p05_bps=s['spread_bps']['p05'],spread_p95_bps=s['spread_bps']['p95'],
            mid_oracle_median_bps=s['mid_oracle_bps']['median'],buy_1000_to_oracle_after18fees_median_bps=s['buy_to_oracle_net18_bps_1000']['median'],
            static_taker_roundtrip_1000_median_bps=s['roundtrip_cost_bps_1000']['median'],best_bid_notional_median=s['best_bid_notional']['median'],best_ask_notional_median=s['best_ask_notional']['median'],
            historical_unique_trades=h['unique_trades'],historical_notional=h['notional'],historical_median_trade_notional=h['median_trade_notional'],historical_trades_per_hour=h['trades_per_hour'],
            top3_maker_notional_share=h['top3_maker_notional_share'],settled_funding_abs_median_bps_hour=h['abs_funding_bps_per_hour']['median'],latest_settled_funding_bps_hour=h['latest_settled_funding_bps'],
            live_new_trades=s['observed_new_trades'],hourly_abs_trade_close_return_median_bps=h['hourly_abs_move_bps']['median']))
    with (HERE/'market_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    market_table=table(['종목','스프레드 중앙값 bp','5–95% bp','중간가/오라클 bp','1,000 왕복 비용 bp','신규 체결'],[
        [r['coin'],num(r['spread_median_bps']),num(r['spread_p05_bps'])+' ~ '+num(r['spread_p95_bps']),num(r['mid_oracle_median_bps']),num(r['static_taker_roundtrip_1000_median_bps']),r['live_new_trades']] for r in summary])
    liquidity_table=table(['종목','과거 체결 수','거래금액 USDPR','체결금액 중앙값','시간당 체결','최우선 매수/매도 잔량 중앙값','상위 3 메이커 금액 비중'],[
        [r['coin'],num(r['historical_unique_trades'],0),num(r['historical_notional'],0),num(r['historical_median_trade_notional']),num(r['historical_trades_per_hour']),num(r['best_bid_notional_median'],0)+' / '+num(r['best_ask_notional_median'],0),num(100*r['top3_maker_notional_share'],1)+'%'] for r in summary])
    funding_table=table(['종목','시간당 |펀딩| 중앙값 bp','최근 확정 펀딩 bp/h','음수 비율','비용 18bp 회수 시간*'],[
        [c,num(h['abs_funding_bps_per_hour']['median'],4),num(h['latest_settled_funding_bps'],4),num(h['funding_negative_fraction']*100,1)+'%',num(h['taker_fee_only_recovery_hours_at_median_abs'],1)] for c,h in history.items()])
    price_table=table(['종목','1시간 |종가 수익률| 중앙값 bp','외부 대응시장 1시간 수익률 상관'],[
        [c,num(h['hourly_abs_move_bps']['median']),num(h['hourly_return_corr_to_reference'],3)] for c,h in history.items()])
    paper_cards=''.join(f'<article><div class="tag">{esc(p["id"])} · {esc(p["date"])} · {esc(p["evidence"])}</div><h3><a href="{esc(p["url"])}">{esc(p["title"])}</a></h3><p class="muted">{esc(p["authors"])} · 확인 범위: {esc(p["read_scope"])}</p><p>{esc(p["finding"])}</p><p><b>적용 한계:</b> {esc(p["limit"])}</p><p><b>선택에 반영:</b> {esc(p["decision"])}</p></article>' for p in papers)
    eth60=[]
    with (HERE/'trade_markouts.csv').open(encoding='utf-8-sig') as f:
        eth60=[float(r['maker_oracle_markout_after_entry3_bps']) for r in csv.DictReader(f) if r['coin']=='ETH' and r['horizon_seconds']=='60']
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Perpetual Alpha · 수익 기회 조사 · 2026-10-14</title>
<style>body{{font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;line-height:1.75;color:#172333;background:#f4f7fa;margin:0}}main{{max-width:1140px;margin:auto;padding:36px 24px 80px}}h1{{font-size:32px;line-height:1.3}}h2{{margin-top:42px;border-top:1px solid #d5deea;padding-top:26px}}h3{{line-height:1.5}}p{{max-width:100ch}}a{{color:#1659ac}}.lead{{font-size:20px}}.callout,article{{background:white;border:1px solid #dbe4ef;border-radius:12px;padding:20px;margin:16px 0}}.callout{{border-left:5px solid #22856f}}.warning{{border-left-color:#b77a16}}.muted,.tag{{color:#536579;font-size:14px}}.tag{{font-weight:600}}table{{border-collapse:collapse;background:white;font-size:14px;width:100%;margin:15px 0}}td,th{{padding:10px;border-bottom:1px solid #e2e8f0;text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}th{{background:#e9f0f8}}.scroll{{overflow:auto}}code,pre{{background:#e8eef5;border-radius:4px}}pre{{padding:16px;white-space:pre-wrap;overflow-wrap:anywhere}}li{{margin:8px 0}}nav{{display:flex;gap:16px;flex-wrap:wrap}}@media print{{body{{background:white}}main{{padding:0}}article{{break-inside:avoid}}.scroll{{overflow:visible}}td,th{{white-space:normal}}}}</style>
<main><div class="tag">전략 선정 전 조사 · 공개 데이터만 사용 · 주문·서명 없음</div><h1>무슨 수익 기회를 먼저 공략할 것인가</h1>
<p class="lead"><b>1순위는 ETH를 중심으로 한 선택적 메이커 전략이다.</b> 외부 기준가격과 오라클, 체결 방향, 펀딩을 함께 보고 불리한 체결을 피하면서 호가 차이를 얻는 가설을 먼저 검증한다. HYPE는 작은 주문으로 확장할 후보이며, SP500의 오라클 괴리·펀딩 결합은 조건부 2순위다.</p>
<div class="callout warning"><b>선정한 것은 구현·검증할 후보이지, 수익성이 입증된 전략이 아니다.</b> 이번 작업은 최근 연구와 실제 시장의 기회·비용 구조를 조사했다. 모델 학습이나 우리 주문의 전향적 수익 실험은 수행하지 않았다. 관측 구간에서 무조건적인 양방향 메이킹, 단순 시장가 역추세, 펀딩만 노린 포지션을 바로 채택할 근거는 부족하다.</div>
<nav><a href="#data">데이터 범위</a><a href="#market">시장 증거</a><a href="#candidates">전략 후보</a><a href="#models">첫 모델</a><a href="#papers">논문 근거</a><a href="#evaluation">검증·대회 평가</a><a href="#reproduce">재현</a></nav>

<h2 id="data">1. 조사 범위와 시간</h2>
<ul><li><b>과거 체결:</b> {kst(a['historical_start_ms'])}부터 {kst(a['historical_fill_end_ms'])}까지, 연습시장 par의 10종목. 지갑별 체결 {sum(h['unique_trades'] for h in history.values())*2:,}행을 양측으로 대조해 <b>{sum(h['unique_trades'] for h in history.values()):,}건</b>으로 중복 제거했다. 총 거래금액은 {num(sum(h['notional'] for h in history.values()),0)} USDPR이다.</li>
<li><b>확정 펀딩:</b> 종목별 225시간, 총 2,250개. 마지막 확정치는 2026-10-14 17:00 KST에 해당한다. 실시간 context의 funding 추정치와 구분했다.</li>
<li><b>호가·오라클·mark·외부 기준가격 동시 관측:</b> {kst(a['capture']['start_ms'])} ~ {kst(a['capture']['end_ms'])}, {a['capture']['duration_seconds']:.1f}초. par L2 메시지 1,748개, 신규 체결 34건, 외부 대응시장 BBO 메시지 32,126개. 1초 단위 유효 패널은 8,960행이며 독립 표본 8,960개를 의미하지 않는다.</li>
<li><b>가격 이력:</b> par와 메인넷 대응시장 1시간 봉. 완료된 par 봉 2,230개 구간의 체결 수가 수집한 고유 체결 수와 모두 일치했다. 최신 부분 봉은 이 검증에서 제외했다. 외부시장 종가는 과거 par 오라클이 아니다.</li>
<li><b>연결 품질:</b> 17:09:53 KST에 par 연결 종료 1회 후 재연결했다. 종목별 새 호가·context가 돌아올 때까지 약 2.2–2.6초를 패널에서 제외했다. 실시간 체결은 재구독 스냅샷까지 시간·tid로 중복 제거했다.</li></ul>
<p>10/12–13 공지의 정식 DEX는 pa/USDP다. 이번 최종 공개 API 조회에서는 perpDexs에 par만 해당 이름으로 확인됐고 pa 조회는 HTTP 500이었다. 따라서 이 보고서는 <b>par/USDPR 연습시장 증거</b>다. 정식시장 유동성·참가자·스프레드가 같을 것으로 가정하지 않는다. <a href="https://app.notion.com/p/3eb73946edee810b995de6f2fcfa2608">사용자가 보관한 주최 측 공지·답변</a></p>

<h2 id="market">2. 실제 시장에서 확인한 기회와 장애물</h2>
<p><b>수수료가 전략의 출발점이다.</b> 기본 요율은 편도 메이커 3bp, 테이커 9bp다. 과거 체결의 fee도 반올림 오차 범위에서 이 요율과 일치했다. 1bp=0.01%. 비슷한 명목금액으로 진입·종료하면 수수료만 메이커/메이커 6bp, 메이커/테이커 12bp, 테이커/테이커 18bp다. 메이커 리베이트를 받는 시장의 전략을 그대로 가져올 수 없다. <a href="https://perpetual-alpha26.github.io/docs/">공식 Participant Guide</a></p>
{market_table}
<p class="muted">모든 호가 통계는 15분 구간의 1초 패널 기준이다. ‘1,000 왕복 비용’은 오라클 기준 목표명목 1,000 USDPR에 해당하는 동일 수량을 현재 매도 호가로 사고 매수 호가로 팔 때의 VWAP 차이+테이커 수수료 18bp다. 수량은 lot에 맞춰 내림한다. 미래 청산 비용의 예측이나 주문 체결 보장이 아니며, 지연과 추가 시장 충격은 제외했다.</p>
<article><h3>① ETH에는 메이커 후보가 될 가격 여지가 있지만, 체결 후 손실을 걸러야 한다</h3>
<p>ETH 스프레드 중앙값은 {num(live['ETH']['spread_bps']['median'])}bp다. 양쪽을 같은 가격 상태에서 메이커로 모두 체결한다는 가정이면 수수료 6bp 차감 후 약 {num(live['ETH']['spread_bps']['median']-6)}bp의 가격 여지가 있다. 이는 순수익 추정치가 아니다. 중간에 가격이 움직이거나 한쪽만 체결되면 사라진다. 실제 신규 ETH 체결 중 60초 뒤 오라클을 관측할 수 있었던 {len(eth60)}건의 메이커 방향 평가손익은 진입 수수료 3bp 차감 후 평균 {num(sum(eth60)/len(eth60))}bp였다. 청산 수수료·펀딩도 아직 빼지 않은 값이다.</p>
<p>이 진단은 다른 참가자의 체결에 대한 짧은 markout이며 우리 전략 성과가 아니다. 다만 <b>스프레드만 보고 아무 때나 호가를 내는 전략을 채택하면 안 된다는 실제 경고 신호</b>다. HYPE는 더 넓은 스프레드가 있지만 작은 체결·얕은 잔량 때문에 주문 크기와 재고 위험을 더 제한해야 한다.</p></article>
<article><h3>② SP500은 오라클 대비 할인과 음의 펀딩이 함께 보인다. 실행 가능한 이익은 훨씬 작다</h3>
<p>중간가가 오라클보다 중앙값 {num(abs(live['SP500']['mid_oracle_bps']['median']))}bp 낮았다. 최근 확정 펀딩은 {num(history['SP500']['latest_settled_funding_bps'],4)}bp/h로 롱이 받는 방향이다. 따라서 ‘할인된 롱을 잡고 괴리 축소와 펀딩을 함께 얻는다’는 가설은 조사할 가치가 있다.</p>
<p>하지만 1,000 USDPR 목표명목을 실제 매도 호가에 매수한 VWAP에서 <b>현재 오라클에 청산한다고 가정</b>하고 양쪽 테이커 수수료를 빼면, 여지는 중앙값 {num(live['SP500']['buy_to_oracle_net18_bps_1000']['median'])}bp다. 약 {num(live['SP500']['any_1000_oracle_gap_after18fees_fraction']*100,1)}%의 관측 초에서는 양수였지만, 연속된 가격 상태가 반복된 비율이지 독립 거래 기회 비율이 아니다. 오라클은 청산 호가가 아니며, 괴리가 언제 좁혀지는지도 입증하지 못했다.</p>
<p>‘오라클보다 싸다’는 이유만으로 즉시 시장가 매수하지 않는다. 메이커로 진입해 비용을 줄이거나, 충분히 큰 괴리와 보수적으로 예상한 펀딩이 청산 비용·가격 위험까지 덮는 경우만 후보로 남긴다. 대회 종료 시 오라클 평가와 일반적인 장중 체결 청산을 별도로 계산한다.</p></article>

<h3>유동성: 자본 100만을 그대로 투입할 시장이 아니다</h3>
{liquidity_table}
<p>상위 3 메이커 지갑의 체결 금액 비중은 종목별 약 88.6–97.7%였다. 이는 공개 체결에서 관측한 집중도이며, 각 지갑이 주최 측 봇인지 참가팀인지는 확정하지 않는다. 기존 공급자의 대기열 뒤에 주문을 놓는 경우와 가격을 개선하는 경우의 수익·체결 확률이 다르다. 자금이 충분하다는 사실은 대량 체결이나 높은 수익금의 근거가 아니다.</p>
<h3>펀딩: 단독 주력보다 재고 보유 비용으로 먼저 사용</h3>
{funding_table}
<p class="muted">* |펀딩| 중앙값을 같은 부호로 계속 받는다는 가정 아래 수수료 18bp만 회수하는 산술 예시. 스프레드·가격변동·레버리지·펀딩 부호 변화는 제외한다. 미래 예상 회수 시간이 아니다. 음수 펀딩은 롱 수취, 양수는 숏 수취.</p>
<p>BTC·ETH·HYPE의 |시간당 확정 펀딩| 중앙값은 0.125bp다. 18bp를 회수하려면 이 값이 유지돼도 144시간이 걸린다. SP500 등의 0.0625bp에서는 288시간이다. 최신 한 시간의 큰 음수 펀딩을 여러 날 확정 수입으로 연장하면 안 된다. 메인넷에 반대 포지션을 두는 거래도 대회 채점 계좌 내 완전한 헤지가 된다고 볼 수 없다.</p>
<h3>가격 자료: 외부 기준가격은 유용하지만 선행 신호는 아직 검증되지 않았다</h3>
{price_table}
<p>BTC·ETH·HYPE의 시간별 수익률은 메인넷 대응시장과 함께 움직이는 경향이 관측됐다. 같은 시간 구간의 상관은 미래 예측력이나 선행 관계의 증거가 아니다. 1시간 가격 변동의 절댓값도 그 전부를 얻을 수 있는 수익을 뜻하지 않는다. EWJ의 중앙값 0은 체결·갱신 부족의 영향을 받을 수 있으므로 낮은 위험으로 해석하지 않는다. 외부 가격은 우선 오래된 호가를 거두고 재고를 제한하는 입력으로 사용한다.</p>

<h2 id="candidates">3. 선정한 전략 후보와 우선순위</h2>
<article><div class="tag">1순위 · 먼저 구현·검증</div><h3>A. 외부 기준가격·펀딩을 반영한 선택적 메이커</h3>
<p><b>수익원:</b> 주문을 기다려 주는 대가로 얻는 호가 차이. 외부 가격·오라클·체결 흐름으로 불리한 쪽 체결을 줄이고, 펀딩을 예상 보유 비용에 반영한다. ETH부터 검증하고, HYPE는 작은 주문의 확장 후보, BTC는 비용 조건을 통과할 때만 추가한다.</p>
<ul><li><b>진입:</b> 양방향 호가를 항상 유지하지 않는다. 매수·매도 각각에 대해 체결 조건부 예상 청산금액에서 진입·청산 수수료와 보수적 슬리피지, 예상 펀딩 비용을 빼고 양수인 쪽만 post-only 주문을 고려한다. 고정된 ‘20bp 이상’ 규칙은 현재 관측 중앙값을 과적합할 수 있으므로 채택하지 않는다.</li>
<li><b>재고 관리:</b> 매수 체결이 누적되면 추가 매수 가격·수량을 줄이고 재고 축소 쪽을 우선한다. 외부 가격 급변, 지속적 한 방향 체결, 데이터 단절에는 위험을 늘리는 주문을 취소한다. 최종 청산을 테이커로 해야 할 가능성까지 비용에 넣는다.</li>
<li><b>모델:</b> 먼저 재고 페널티와 펀딩을 포함한 해석 가능한 호가 제어를 기준으로 둔다. 그 위에 로지스틱 회귀로 불리한 체결 분류, 트리 기반 회귀로 60·300초 손익 분포를 비교한다. 체결 시간·확률은 공개 L2만으로 정답을 만들 수 없어 보수적 대기열 범위와 추후 실제 주문 기록이 필요하다.</li>
<li><b>탈락 조건:</b> 보수적 체결·청산 가정의 비용 후 성과가 기본 메이커보다 개선되지 않음, 손익이 소수의 운 좋은 체결에 집중됨, 재고 일방향 노출이 성과를 설명함, 혹은 참여율·수량 증가 시 여지가 사라짐.</li>
<li><b>논문 연결:</b> P1 펀딩·재고, P2 주문 흐름 변화, P5 체결/수익 상충, P6 대기열·충격.</li></ul></article>
<article><div class="tag">2순위 · 조건부 연구, 지금 즉시 매수 근거 아님</div><h3>B. 오라클 괴리 축소 + 유리한 펀딩</h3>
<p><b>수익원:</b> 계약 가격이 기준가격에서 벗어났다가 돌아오는 부분과 보유 중 펀딩. SP500에서 두 방향이 맞물린 사례를 관측했지만 비용 차감 후 여지는 작았다.</p>
<ul><li><b>진입:</b> 실행 가능한 진입 VWAP·예상 청산 VWAP로 계산한다. 괴리 수준뿐 아니라 해당 국면에서의 축소 확률·시간을 추정하고, 펀딩 예측의 보수적 범위까지 넣어 양수일 때만 고려한다. 메이커 진입은 미체결·역선택 비용을 함께 평가한다.</li>
<li><b>종료:</b> 괴리 축소, 보유시간 한도, 펀딩 불리 전환, 오라클·외부가격 변화, 포트폴리오 위험 한도 중 먼저 발생한 조건으로 정리한다. 오라클 가격에 언제든 체결할 수 있다고 가정하지 않는다.</li>
<li><b>모델:</b> 국면별 괴리 분포와 단순 AR/상태공간 모형, 펀딩 EWMA/AR 예측부터 비교한다. 평균회귀 계수를 고정하기 전에 정상성·구간 안정성부터 확인한다.</li>
<li><b>탈락 조건:</b> 괴리가 지속·확대됨, 한쪽 청산 비용이 수익을 흡수함, 예측한 펀딩 방향이 유지되지 않음, 수익이 오라클 자체의 방향 노출에서만 발생함.</li>
<li><b>논문 연결:</b> P1의 보유 비용 처리, P10의 미래 펀딩 불확실성. 현재 15분 자료만으로 평균회귀 속도를 추정하지 않는다.</li></ul></article>
<article><div class="tag">3순위 · 비교 기준과 위험 필터</div><h3>C. 낮은 회전율의 추세·변동성 조절</h3>
<p><b>역할:</b> BTC·ETH·HYPE의 수시간 추세 신호와 변동성 조절을 단순 기준으로 둔다. A가 추세 구간에서 재고를 쌓는 문제를 감지하고, 복잡한 모델이 이 기준보다 실제로 나은지 평가한다. 가격 움직임은 확인했지만 추세의 미래 수익성은 이번 조사에서 입증하지 않았다.</p>
<ul><li>수시간 이동평균·누적수익률 신호와 거래하지 않는 완충 구간을 사용하되, 관측한 가격 변동 전체를 예상수익으로 사용하지 않는다.</li><li>차익이 비용을 넘지 못하면 포지션 변경을 생략한다. ML을 추가한다면 연속적인 가격값보다 ‘비용을 넘는 순수익이 날 확률·분포’를 대상으로 한다.</li><li>별도 미래 구간에서 무신호·현금 기준보다 개선되지 않으면 주력으로 승격하지 않는다. P11은 아이디어 참고이며 비용·종목·펀딩 설명 차이로 성과 근거에서는 제외한다.</li></ul></article>
<p><b>첫 구현에서 제외:</b> 펀딩 수취만 노리는 큰 방향 포지션, 수수료를 자주 내는 시장가 단타, 상관만 보고 구성한 BTC/ETH 페어, 수익성이 검증되지 않은 외부 가격 지연 차익, 10종목을 일괄 처리하는 대형 RL 모델. 이들은 불가능하다는 결론이 아니라 현재 자료의 지지가 부족하다는 판단이다.</p>

<h2 id="models">4. 무엇을 예측하고, 에이전트는 무엇을 할 것인가</h2>
<p><b>예측 대상은 ‘다음 가격 상승 여부’ 하나가 아니다.</b> 주문 방향·가격·크기를 조건으로 ① 일정 시간 안 체결될 가능성/시간, ② 체결 후 60초·300초 가격 변화 및 실제 청산 비용을 반영한 손익, ③ 다음 시간 펀딩, ④ 한 방향 주문 흐름으로 바뀔 위험을 분리한다. 학습 전 데이터와 라벨이 확보됐는지부터 판단한다.</p>
<p>입력은 오라클·mark·중간가 차이, 외부가격 변화와 수신 시각, L2 깊이·호가 차이·불균형, 최근 매수/매도 체결 금액과 빈도, 확정 펀딩·진행 중 추정 펀딩, 현재 재고다. mainnet 가격은 par 오라클의 정답이 아니며 둘의 차이 자체도 관측한다.</p>
<pre>실시간 자료 수집·시간 정렬
  → 현재 시장 상태와 데이터 신선도 확인
  → 비용 후 수익·재고 위험 추정
  → 호가 가격·수량·취소/대기 선택
  → 주문·부분 체결·잔량 대조
  → 손익을 가격/수수료/펀딩/재고로 분해</pre>
<p>초기 비교군은 기본 재고 제어 메이커, +외부가격 필터, +체결 후 손익 모델, +펀딩, +주문 흐름 변화 감지로 단계적으로 구성한다. 최종 모델에서 각 요소를 하나씩 제거하는 실험으로 기여를 확인한다. 복잡한 RL·LSTM은 단순 기준보다 낫고 시뮬레이터의 체결 오차가 통제된 뒤 검토한다. 현재 신규 체결 34건으로 이런 모델을 훈련했다는 주장은 할 수 없다.</p>

<h2 id="papers">5. 최신 연구에서 가져온 것과 가져오지 않은 것</h2>
<p>2026-10-14 기준 2025–2026년 원문을 조사했다. 아래 11편은 핵심 검토 목록이며 체계적 문헌고찰의 전수 목록이 아니다. 대부분 프리프린트이고, 학술지 실증·실제 주문·과거 재생·합성 시뮬레이션을 구분했다. 전문을 확인하지 못한 두 SSRN 논문은 초록 근거라고 표시했다. 본문을 읽은 논문도 코드를 재현해 성과를 검증한 것은 아니다.</p>
{paper_cards}

<h2 id="evaluation">6. 대회 점수와 연결하는 검증 기준</h2>
<p>공식 평가는 최종 오라클 평가 순수익, 일별 equity 수익률의 표준편차, 기술보고서 심사로 구성된다. 수익 40%·위험 30%·심사 30%라는 목표에 맞춰, <b>우리 버전끼리는 같은 기간·자본·수수료·청산 가정으로 비교</b>해야 한다. 팀별 min–max 정규화 순위는 다른 팀의 결과에도 영향을 받으므로 고정된 수익률을 점수로 바로 바꾸지 않는다. <a href="https://perpetual-alpha26.github.io/about">공식 평가 기준</a>, <a href="https://perpetual-alpha26.github.io/">공식 가중치</a></p>
<ol><li><b>연구/평가 분리:</b> 오늘 본 데이터는 후보 선정용이다. 이미 본 자료를 최종 미사용 평가 구간이라고 부르지 않는다. 이후 날짜·세션 단위로 학습→검증→고정 버전 전향적 평가를 순서대로 진행하고, 겹치는 손익 라벨 사이에는 가장 긴 라벨 기간만큼 간격을 둔다.</li>
<li><b>체결 현실성:</b> 주문이 호가에 닿았다는 이유로 전량 체결하지 않는다. 대기열 뒤, 부분 체결, 취소 지연, post-only 거절, 한쪽만 체결된 재고, 테이커 청산을 포함한다. L2만으로 정할 수 없는 부분은 보수적/낙관적 범위를 나눠 보여 준다.</li>
<li><b>최소 비교:</b> 현금 기준, 단순 추세, 기본 메이커와 A/B를 동일 기간·위험 상한에서 비교한다. 수익·일별 변동성·최대 낙폭·체결 수·노출·회전율·수수료·펀딩을 같이 기록한다. 서로 다른 전략의 위험과 수익이 엇갈리면 단일 승리라고 포장하지 않는다.</li>
<li><b>용량:</b> 주문 크기를 단계적으로 바꾸고 거래금액 참여율을 제한해, 수익이 유지되는 규모를 측정한다. 자본 수익률과 주문 명목 대비 수익률을 구분한다. 거래 수가 적으면 일별 블록을 이용한 불확실성 범위를 넓게 보고한다.</li>
<li><b>타 지갑 비교:</b> 이전에 만든 공개 지갑 자료를 같은 시작·종료 시각으로 자른다. 입출금과 수수료·펀딩을 반영하고 참가자 정체·지갑 역할의 불확실성을 표시한다. 버전 개선 판단은 고정된 비교집단과 원시 지표로 한다. 심사 30점은 임의로 만들어 채우지 않는다.</li>
<li><b>정식시장 이행:</b> pa 공개 후 계약·담보·수수료·펀딩·깊이를 다시 읽고 par 가정과 비교한다. 새 시장의 초기 자료 부족을 par 성과로 덮지 않는다. 일별 equity 경계·처리 방식의 미확정 사항은 공식 확인 전에는 참고 계산으로 표시한다.</li></ol>
<p>심사에 보여 줄 결과는 모델 이름의 복잡함보다 ‘어떤 조건에서 비용 후 수익이 남았는가, 어떤 요소가 기여했는가, 실패 구간과 실행 오차는 무엇인가’를 재현할 수 있는 근거다. 현재 선택은 <b>A를 첫 실험, B를 조건부 병행 연구, C를 비교 기준</b>으로 두는 것이다.</p>

<h2 id="reproduce">7. 원자료·검증·재현</h2>
<p>데이터·산술 검증 {validation['checks_count']}개 통과. 검증 범위는 중복 제거, 양측 체결 대조, 수수료, 완료 봉 체결 수, 펀딩 시간, 과거 정보만 사용하는 시간 정렬, 연결 공백 제외, VWAP 산술이다. 전략 수익성 검증을 의미하지 않는다.</p>
<ul><li><a href="market_summary.csv">10종목 요약 CSV</a> · <a href="analysis.json">전체 통계 JSON</a> · <a href="literature.json">문헌과 적용 판단</a></li><li><a href="historical_unique_trades.csv">과거 고유 체결</a> · <a href="settled_funding.csv">확정 펀딩</a> · <a href="hourly_price_panel.csv">시간봉과 외부 대응가격</a></li><li><a href="live_panel_1s.csv">동시 관측 1초 패널</a> · <a href="live_unique_trades.csv">실시간 구간 신규 체결</a> · <a href="trade_markouts.csv">체결 후 평가가격 진단</a></li><li><a href="validation.json">검증 결과</a> · <a href="README.txt">재현·수집 설명</a> · raw/에 요청·응답 및 압축 스트림 보관</li></ul>
<pre>python analyze_market.py
python validate_research.py
python build_report.py</pre>
<p class="muted">Python 3.10+ 및 numpy 필요. 기존 원자료로 분석을 재현할 때는 네트워크·개인키가 필요 없다. 수집 스크립트를 다시 실행하면 별도 시점 자료가 되므로 이 동결 폴더의 원자료를 덮어쓰지 않는다. 자세한 파일 해시는 SHA256SUMS.txt에 있다.</p>
</main></html>'''
    (HERE/'report.html').write_text(body,encoding='utf-8')
    print('Wrote report.html and market_summary.csv')

if __name__=='__main__':main()
