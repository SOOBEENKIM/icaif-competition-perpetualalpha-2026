"""Ex ante funding-budget audit, not an additional strategy or backtest.

Use the already recorded quote preceding the registered probe. Keep quantities,
exit price hypothesis and fees fixed while varying future funding assumptions.
No sealed observations or live state is read.
"""
from datetime import datetime
import html
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo
import numpy as np
from experiments.cost_horizon_20261016.research import save, sha

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; OUT = HERE / 'artifacts'
SOURCE = ROOT / 'experiments/paired_funding_20261017/artifacts/current_20261016T192420Z.json'
HISTORY = ROOT / 'experiments/pairs_research_20261017/par_history/20261016T173824Z'
HOUR = 3600000


def offsets(start_ms, horizon):
    first = (start_ms // HOUR + 1) * HOUR
    times = np.arange(first, start_ms + horizon * HOUR + 1, HOUR, dtype=np.int64)
    return (times - start_ms) / HOUR


def receipts_with_decay(receive_per_hour, payment_offsets, half_life):
    if half_life <= 0: return 0.
    return float(receive_per_hour * np.exp2(-np.asarray(payment_offsets) / half_life).sum())


def minimum_half_life(receive_per_hour, pay_per_hour, payment_offsets, required_net_funding):
    need = required_net_funding + pay_per_hour * len(payment_offsets)
    if need <= 0: return 0.
    if receive_per_hour * len(payment_offsets) <= need: return None
    lo, hi = 0., 1e8
    for _ in range(80):
        mid = (lo + hi) / 2
        if receipts_with_decay(receive_per_hour, payment_offsets, mid) < need: lo = mid
        else: hi = mid
    result = (lo + hi) / 2
    if not math.isclose(receipts_with_decay(receive_per_hour, payment_offsets, result), need, abs_tol=1e-7):
        raise ValueError('Funding break-even equation failed')
    return result


def history_diagnostic(symbol, extra, sign):
    source = HISTORY / f'funding_{symbol}.json'
    raw = json.loads(source.read_text())['rows'] + extra[symbol]
    rows = sorted({int(r['time']) // HOUR: r for r in raw}.values(), key=lambda r: r['time'])
    buckets = np.array([int(r['time']) // HOUR for r in rows])
    if not np.all(np.diff(buckets) == 1): raise ValueError('Missing historical funding hour')
    rates = np.array([float(r['fundingRate']) for r in rows])
    income = -sign * rates > 0
    current = 0
    for good in income[::-1]:
        if not good: break
        current += 1
    return dict(symbol=symbol, historical_rows=len(rows), last_settlement=rows[-1]['time'],
        current_income_sign_run_hours=current, current_run_right_censored=True,
        income_hours_last72=int(income[-72:].sum()), mean_rate_last72=float(rates[-72:].mean()),
        mean_rate_last168=float(rates[-168:].mean()), independent_episodes_not_hours=True,
        source=str(source.relative_to(ROOT)), source_sha256=sha(source))


def main():
    if OUT.exists(): raise FileExistsError('Sensitivity audit already recorded')
    OUT.mkdir()
    protocol = dict(created_at=time.time(), source_quote=str(SOURCE.relative_to(ROOT)), source_sha256=sha(SOURCE),
        analysis_source_sha256=sha(__file__), half_lives_hours=[6,12,24,48], horizon_hours=72,
        scope='Funding-only sensitivity at the original pre-probe quantities and price/exit-cost budget. Conditional expectations, not realized PnL or new entry selection.',
        paths='Replicate original 50%-of-positive-net carry; full persistence; receipts half with payments unchanged; receipts half/payments 150%; receipts exponential decay with payments constant.',
        root='Solve the monotone funding-budget equality. Not an optimized strategy parameter or suggested early exit time.',
        oracle_assumption='Keep each original oracle fixed for funding conversion; future oracle/basis and liquidation depth remain uncertain.',
        raw_sealed_outcomes_read=False, strategy_or_trial_changed=False, orders_sent=0)
    save(OUT / 'protocol.json', protocol)
    source = json.loads(SOURCE.read_text()); meta, ctx = source['market_receipts'][0]['response']
    contexts = {a['name']: b for a,b in zip(meta['universe'],ctx)}
    extra = {r['request']['coin'].split(':')[1]:r['response'] for r in source['rate_receipts']}
    entries = [r for r in source['rows'] if r['model'] == 'published_current_persistence'
        and r['proposal']['coins'] == ['par:EWY','par:SP500'] and r['proposal']['horizon_hours'] == 72]
    rows, budgets, attribution = [], [], []
    payment_offsets = offsets(source['at_ms'], 72)
    assert len(payment_offsets) == 72 and 0 < payment_offsets[0] <= 1 and 71 < payment_offsets[-1] <= 72
    assert minimum_half_life(1.,0.,[1,2],2.) is None
    assert minimum_half_life(0.,0.,[1,2],0.) == 0.
    for entry in entries:
        q = entry['proposal']; scenario = entry['scenario']
        flow = np.array([-qty * float(contexts[c]['oraclePx']) * float(contexts[c]['funding'])
                         for c,qty in zip(q['coins'],q['target_qty'])])
        receive = float(np.maximum(flow,0).sum()); pay = float(np.maximum(-flow,0).sum())
        net = receive - pay; fixed_budget = q['predicted_fees'] - q['predicted_price_pnl']
        paths = dict(original_50_percent_net=(.5 if net > 0 else 1.) * net * 72,
            full_rate_persistence=net * 72,
            receipts_50_payments_100=(.5*receive-pay)*72,
            receipts_50_payments_150=(.5*receive-1.5*pay)*72)
        for half_life in protocol['half_lives_hours']:
            paths[f'receipts_half_life_{half_life}h'] = receipts_with_decay(receive,payment_offsets,half_life) - pay*72
        assert math.isclose(paths['original_50_percent_net'],q['predicted_funding'],abs_tol=1e-7)
        assert math.isclose(paths['original_50_percent_net']-fixed_budget,q['predicted_net_pnl'],abs_tol=1e-7)
        for name, funding in paths.items():
            result = funding-fixed_budget
            rows.append(dict(scenario=scenario,path=name,gross_notional=q['gross_notional'],
                expected_funding=funding,fixed_price_and_fee_budget=fixed_budget,
                expected_net_pnl=result,expected_net_bps=result/q['gross_notional']*10000))
        buffer = .0005*q['gross_notional']
        budgets.append(dict(scenario=scenario,receive_per_hour=receive,pay_per_hour=pay,
            funding_needed=fixed_budget,funding_needed_with_5bp_buffer=fixed_budget+buffer,
            original_forecast_per_payment=.5*net if net>0 else net,
            constant_forecast_payments_to_cover_72h_exit_budget=math.ceil(max(0.,fixed_budget/(.5*net))) if net>0 else None,
            constant_forecast_payments_to_cover_budget_plus_buffer=math.ceil(max(0.,(fixed_budget+buffer)/(.5*net))) if net>0 else None,
            receive_fraction_needed_for_break_even=(fixed_budget+pay*72)/(receive*72) if receive else None,
            receive_half_life_needed_for_break_even=minimum_half_life(receive,pay,payment_offsets,fixed_budget),
            receive_half_life_needed_for_5bp=minimum_half_life(receive,pay,payment_offsets,fixed_budget+buffer)))
        attribution.append(dict(scenario=scenario,coins=q['coins'],hourly_funding_cash=flow.tolist(),
            share_of_positive_income=(np.maximum(flow,0)/receive).tolist() if receive else None))
    history = [history_diagnostic('EWY',extra,-1),history_diagnostic('SP500',extra,1)]
    result = dict(at_unix=time.time(),quote_at_ms=source['at_ms'],rows=rows,budgets=budgets,
        attribution=attribution,historical_context=history,actual_pnl=None,actual_Q=None,
        realized_improvement_proven=False,raw_sealed_outcomes_read=False,strategy_changed=False,
        interpretation='A registered economic probe is justified, but profitability remains conditional on funding persistence and price/exit assumptions. Receipt source concentration is measured; sustained past signs do not constitute independent future episodes.')
    save(OUT/'results.json', result)
    report(protocol,result)
    print(json.dumps(dict(budgets=budgets,attribution=attribution,history=history,
        rows=[r for r in rows if 'half_life' in r['path']]),indent=2))


def report(protocol,result):
    def kst(t):return datetime.fromtimestamp(t,ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SFINX 펀딩 지속성 비용 회수 점검</title><style>body{{font:16px/1.7 system-ui;max-width:1160px;margin:40px auto;padding:0 24px;color:#183449}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:right;padding:10px;border-bottom:1px solid #c9d5dd}}th{{background:#eaf0f5}}td:first-child,th:first-child{{text-align:left}}.box{{background:#fff1d8;border-left:5px solid #ad6a00;padding:18px}}h2{{margin-top:32px}}a{{color:#05669d}}</style>
<h1>펀딩 수입의 지속성과 비용 회수 조건</h1><p>사용한 원래 호가: {kst(result['quote_at_ms']/1000)} · 이후 시험의 봉인 자료 접근 없음 · 주문 및 전략 변경 없음</p>
<div class="box"><b>이 전략의 예상 수입은 대부분 SP500 롱의 음수 펀딩에서 나옵니다.</b> EWY 숏과 SP500 롱이 모두 펀딩을 받는 당시 조건이었지만 수입은 한 종목에 집중됐습니다. 72시간 동안 펀딩을 받는다는 가정의 성립 여부가 핵심입니다. 아래 표는 가정 민감도이며 실현 성과가 아닙니다.</div>
<h2>종목별 수입 기여</h2><table><tr><th>비용 조건</th><th>EWY 숏 시간당 예상 펀딩</th><th>SP500 롱 시간당 예상 펀딩</th><th>SP500 수입 비중</th></tr>'''
    for a in result['attribution']:
        body+=f"<tr><td>{a['scenario']}</td><td>${a['hourly_funding_cash'][0]:.5f}</td><td>${a['hourly_funding_cash'][1]:.5f}</td><td>{a['share_of_positive_income'][1]*100:.2f}%</td></tr>"
    body+='</table><p>이 값은 할인 전 공개 예상률과 당시 오라클·계획 수량으로 환산했습니다. 원래 전략은 양의 합계 펀딩을 50%로 줄였습니다. 이 스냅샷에서는 두 종목 모두 수입이므로 종목별 수입을 절반으로 줄여도 원래 계산과 같습니다. 한쪽이 비용인 다른 상황에서는 두 방식이 같지 않을 수 있습니다.</p><h2>72시간 청산 비용을 회수하는 조건</h2><table><tr><th>조건</th><th>회수해야 할 비용 예산</th><th>원래 할인 가정의 필요 지급 횟수</th><th>추가 5bp 포함 횟수</th><th>손익분기 수입 반감기</th><th>5bp 확보 수입 반감기</th></tr>'
    for b in result['budgets']:
        body+=f"<tr><td>{b['scenario']}</td><td>${b['funding_needed']:.2f}</td><td>{b['constant_forecast_payments_to_cover_72h_exit_budget']}</td><td>{b['constant_forecast_payments_to_cover_budget_plus_buffer']}</td><td>{b['receive_half_life_needed_for_break_even']:.2f}시간</td><td>{b['receive_half_life_needed_for_5bp']:.2f}시간</td></tr>"
    body+='</table><p>필요 지급 횟수는 고정된 72시간 청산 가격·수수료 예산을 같은 시간당 수입으로 채우려면 필요한 횟수입니다. 해당 시간에 청산하면 이익이 된다는 최적 보유기간 계산이 아닙니다. 더 일찍 청산하면 가격·호가·수수료가 달라집니다. 반감기는 수입이 지수적으로 감소하고 지급 비용은 유지되는 가상 경로에서 비용 회수 식을 푼 값입니다. 그 값에 맞추어 전략을 재조정하지 않았습니다.</p><h2>펀딩 수입이 줄어드는 경우</h2><table><tr><th>비용</th><th>가정</th><th>72시간 예상 펀딩</th><th>예상 순손익</th><th>예상 순손익 bp</th></tr>'
    for r in result['rows']:
        body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['scenario'],r['path'],f"${r['expected_funding']:+.2f}",f"${r['expected_net_pnl']:+.2f}",f"{r['expected_net_bps']:+.3f}"])+'</tr>'
    body+='</table><p>행 간에는 같은 비용 조건의 수량·오라클·72시간 청산 가격 예산을 고정했습니다. 반감기 경로는 수입 감소를 직접 적용하므로 별도로 50% 할인을 중복 적용하지 않았습니다. 시간은 원래 스냅샷부터 실제 정각 펀딩 시각까지의 간격을 사용합니다. 미래 오라클 가격·괴리 축소·청산 호가 변화까지 보장하는 계산은 아닙니다.</p><h2>이미 관측했던 정산 이력과의 대조</h2><ul>'
    for h in result['historical_context']:
        body+=f"<li>{h['symbol']}: 원래 호가 전에 확보한 정산 {h['historical_rows']}시간. 마지막 정산 {kst(h['last_settlement']/1000)}. 해당 포지션이 펀딩을 받은 연속 구간은 마지막까지 {h['current_income_sign_run_hours']}시간이며, 최근 72회 중 수입 정산은 {h['income_hours_last72']}회였습니다.</li>"
    body+='''</ul><p>SP500의 음수 펀딩은 이미 오래 지속됐지만, 끝나지 않은 한 구간입니다. 104시간을 104개의 독립적인 성공 사례로 세거나 앞으로도 72시간 지속될 확률로 바꾸지 않았습니다. 과거 정산률과 당시 공개 예상률도 구분했습니다.</p>
<p><a href="https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding">Hyperliquid 공식 문서</a>상 펀딩은 시간마다 지급되고 가격 프리미엄에 따라 달라집니다. HIP-3는 영향 가격을 사용하는 별도 프리미엄 계산이 있고 배포자의 설정도 영향을 줄 수 있으므로, 현재 최우선 호가의 괴리만으로 미래 정산률을 확정하지 않았습니다. 실제 검증은 매 정산률과 당시 오라클을 사용합니다.</p>
<h2>운용 판단에 주는 의미</h2><p>새 모델의 이름보다 SP500 펀딩의 지속성과 페어 청산까지의 잔여 비용이 이 후보의 핵심입니다. 현재의 기대 이익은 확정 수익이 아니므로 기존보다 실현 PnL·Q가 개선됐다고 판정할 수 없습니다. 이미 등록한 72시간 기회 검증과 7일간 공통 조건 비교에서 실제 양쪽 진입·펀딩·청산을 확인합니다. 이 진단으로 봉인된 후보를 수정하거나 새 시험을 늘리지 않았습니다.</p><ul>'''
    paths=[(OUT/'protocol.json','진단 규칙과 원자료 해시'),(OUT/'results.json','정산 이력·수입 분해·민감도 수치'),
           (HERE/'analysis.py','현금 계산과 비용 회수 식 검증'),(SOURCE,'사용한 원래 호가·공개 펀딩 응답'),
           (ROOT/'reports/ensemble_carry_20261017/report.html','기존 전략과의 비교 설계')]
    body+=''.join(f'<li><a href="{p}">{label}</a></li>' for p,label in paths)+'</ul></html>'
    dest=ROOT/'reports/carry_sensitivity_20261017';dest.mkdir(exist_ok=True);(dest/'report.html').write_text(body)


if __name__=='__main__':main()
