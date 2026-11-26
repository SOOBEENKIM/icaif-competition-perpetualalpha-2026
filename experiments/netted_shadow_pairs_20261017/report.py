"""Publish the failed as-of confirmation and portfolio-netting diagnostics."""
import csv,gzip,html,json,time
from pathlib import Path
import numpy as np
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'source_state_pairs_20261017'


def document(title,body):
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+html.escape(title)+'''</title>
<style>body{font:16px/1.65 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#182434}h1{font-size:28px}h2{font-size:21px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dce3eb;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.note{padding:16px;background:#f0f5fa;border-left:4px solid #426a95}code{font-size:14px}</style>'''+body+'</html>'


def table(rows,include_venue=False):
    headings=['구분','후보','조건','순손익','수익률 %','기간 σ %','Q','수수료','펀딩','체결 조각','평균 총노출 %','종료 잔여 명목금액']
    cells=[]
    for r in rows:
        m=r['metrics'];phase=(r.get('venue','')+' / ' if include_venue else '')+r['phase']
        values=[phase,r['arm'],r['scenario'],f"{m['net_pnl']:+,.4f}",f"{100*m['roi']:+.6f}",
            f"{100*m['volatility_horizon']:.6f}",f"{m['financial_score']:.8f}",f"{m['fees']:.4f}",
            f"{m['funding']:+.4f}",m['fills'],f"{100*m['mean_gross']:.6f}",f"{m['residual_gross']:,.4f}"]
        cells.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headings)+'</tr></thead><tbody>'+''.join(cells)+'</tbody></table></div>'


def export_csv(path,rows):
    records=[{k:r.get(k,'') for k in ('venue','phase','arm','scenario')}|r['metrics'] for r in rows]
    fields=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)


def audit_orders(results):
    rows=[]
    for r in results['rows']:
        key='-'.join(r[k] for k in ('venue','phase','arm','scenario'))
        with gzip.open(HERE/'artifacts/runs'/f'{key}.json.gz','rt') as f:run=json.load(f)
        orders=run['orders'];m=run['metrics']
        rows.append(dict(key=key,order_records=len(orders),zero_filled=sum(o['filled_qty']==0 for o in orders),
            zero_capacity=sum(o['capacity']==0 for o in orders),entry_limit_blocked=sum(o.get('entry_limit_blocked',False) for o in orders),
            price_move=m['price_move_pnl'],impact=m['execution_cost'],fees=m['fees'],funding=m['funding'],net_pnl=m['net_pnl']))
    return dict(rows=rows,interpretation='Candle-volume participation is an execution proxy, not proof of zero exchange order-book liquidity. Repeated rejected requests have no fee, but resulting larger actual inventory and actual fills alter PnL.',orders_sent=0)


def main():
    guards=protected()
    for folder in (HERE,SOURCE):
        manifest=json.loads((folder/'artifacts/frozen.json').read_text())
        for rel,digest in manifest['sources'].items():assert sha(ROOT/rel)==digest,rel
    source=json.loads((SOURCE/'artifacts/results.json').read_text())
    audit=json.loads((SOURCE/'audit_artifacts/results.json').read_text())
    net=json.loads((HERE/'artifacts/results.json').read_text());order_audit=audit_orders(net)
    save(HERE/'artifacts/order_audit.json',order_audit)
    out=ROOT/'reports/source_state_pairs_20261017';out.mkdir(exist_ok=True)
    body='''<h1>페어 신호의 가격 출처 점검</h1>
<p class="note"><strong>신호 기준 가격 수정만으로는 개선되지 않았다.</strong> 실제 par 과거 캔들과 펀딩을 이용한 20개 계좌 재현에서, 원래 par 체결가 확인 신호와 기준시장 확인 신호 모두 페어 진입이 없었다. 무거래의 Q=0.1은 대회 순위 자격이나 검증된 수익성을 뜻하지 않는다. 실거래 주문·전략 변경은 0건이다.</p>
<p>각 시장의 마지막 체결가가 같은 시점의 거래 가능한 호가를 뜻하지 않는 문제를 점검했다. 완성된 기준시장 시간봉으로 진입 방향을 확인하되, 거래 가치·중단 조건·비용 계산에는 현재 par 가격을 유지했다. 기존 가격 상태와 비용 제한을 완화하지 않았다.</p>
<h2>자료와 신호 차이</h2><p>par 5분 캔들 4,465개, 2026-10-01 00:00–10-16 12:00 UTC. 첫 관측 체결 이후 추가 체결 없이 지난 최대 시간은 BTC 3.5시간, ETH 4.75시간, HYPE 4.5시간이었다. 호가가 그 시간 동안 멈췄다는 뜻은 아니다.</p>
<p>BTC/HYPE의 되돌림 신호는 par 3회·기준시장 5회, BTC/ETH는 각각 7회·5회였다. 진입 방향/시점은 달라졌지만 실제 거래 제약을 통과해 제출하는 단계까지 이어지지 않았다. 공개된 과거 호가 자료의 페어별 관측 416개에서는 어느 신호 방식도 진입 시점이 겹치지 않아 해당 후보의 호가 체결 성과를 확인할 수 없었다.</p>
<p>검증: 미래 자료를 잘라도 과거 신호가 같음, 현재 par 가격만 변경해도 기준시장 신호는 같음, 현재 par의 중단·비용 상태 보존 등 3개 검사 통과. 기존 전략과 원래 신호의 체결·계좌 이력은 이전 실험과 완전히 같았다.</p>
<h2>계좌 재현 결과</h2><p>validation: 10/01 12:00–10/08 12:00 UTC; comparison: 10/08 12:00–10/16 12:00 UTC. 모든 계좌 초기자본 1,000,000. base: 편도 수수료 9bp·가격 불리함 10bp; stress: 12bp·20bp 및 더 작은 거래량 참여율. 양쪽 체결·청산은 독립적이며 종료 포지션을 무료로 청산하지 않는다. 거래량 대용치에 의존하며 전체 과거 호가를 재현한 것이 아니다.</p>'''+table(source['rows'])+'''
<h2>판정</h2><p>채택 조건을 통과한 후보 없음. 관측 가격 출처 차이는 확인했지만 수익 개선은 확인하지 못했다. 재사용한 개발용 과거 결과이며 독립적인 이후 검증은 아니다.</p>'''
    (out/'report.html').write_text(document('페어 신호 출처 점검',body));export_csv(out/'comparison.csv',source['rows'])
    source_summary=dict(selection=source['selection'],historical_replays=20,tests_passed=3,signal_comparison=audit['signal_comparison'],
        activity=audit['activity'],quote_summary=audit['quote_summary'],orders_sent=0,automatic_live_replacement=False,protected=guards)
    save(out/'summary.json',source_summary)
    save(SOURCE/'work_state.json',dict(status='completed_diagnostic_no_candidate_qualified',report=str((out/'report.html').relative_to(ROOT)),
        completed_replays=20,tests_passed=3,orders_sent=0,next='Portfolio netting diagnostic, followed by causal intent transfer rather than post-fill inventory copying.'))

    out=ROOT/'reports/netted_shadow_pairs_20261017';out.mkdir(exist_ok=True)
    body='''<h1>페어 전략과 기존 전략의 목표 수량 합산 실험</h1>
<p class="note"><strong>새 후보는 기존 두 전략을 대체할 만큼 개선되지 않았다.</strong> 50:50 혼합 두 개 모두 8월 선정 구간에서 기존 두 전략보다 순손익과 Q가 낮았다. 이후 기준시장과 par 비교도 개선 근거를 주지 않았다. 실거래 전략은 그대로이며 주문은 보내지 않았다.</p>
<h2>무엇을 구현했는가</h2><p>동일 종목을 다른 전략이 보유하면 페어를 통째로 제외하던 구조 대신, 두 독립적인 기준시장 모의 전략의 목표 역할을 하는 보유 수량을 50:50으로 합산하고 종목별 차이만 한 계좌에서 주문했다. 가상 전략의 현금·수익을 합산하지 않으며 실제 재현 계좌의 체결·수수료·펀딩은 한 번만 계산한다. 50:50 비중은 결과를 보고 최적화하지 않았다.</p>
<p>이 실험은 각 시각까지 알려진 <em>모의 체결 이후 보유 수량</em>을 신호로 사용한다. 실제 재현 계좌는 다음 관측에서 주문을 처리하므로 기준시장에서는 한 시간, par에서는 5분의 추가 전달 지연이 있다. 기존 첫째·둘째 전략 각각의 단독 추종 계좌와 페어 단독 추종 계좌도 함께 비교해 이 차이를 드러냈다. 원래 페어의 경제적 가격 한도 대신 공통 주문 가격 한도와 두 관측 이내 추종 정책을 사용하므로, 성과 차이 전부를 지연 하나의 원인으로 단정할 수 없다.</p>
<p>반대 가상 수량은 실제 주문 전에 상쇄한다. 미완료 목표는 최대 두 관측까지 시도하고 이후 페어 몫을 억제해 기존 전략에 할당한 몫으로 되돌린다. 청산은 유동성이 없으면 계속 미완료로 남아 재시도한다. 원래 위험 한도와 회전율 한도를 적용한다. source에 이미 열려 있던 포지션을 추종하는 정책은 현재 가격에서 새 진입 가치가 있는지 다시 판단하는 정책과 다르다.</p>
<h2>선정과 비교</h2><p>기준시장 validation: 8/01 12:00–9/01 12:00 UTC, comparison: 9/01 12:00–10/15 12:00 UTC. par validation: 10/01 12:00–10/08 12:00 UTC, comparison: 10/08 12:00–10/15 12:00 UTC. 모든 계좌 초기자본 1,000,000. 기준시장 기록이 끝나는 10/15를 넘겨 가상 신호를 연장하지 않았다. 8월 결과로 진단 대표를 고정한 후 나머지 구간을 계산했으나 과거 자료는 재사용한 개발 자료이다.</p>
<p>고정된 진단 대표는 <code>first_pair_half</code>이며 <strong>채택 자격은 false</strong>다. 기준시장 base/stress의 거래량 참여율은 각각 1%/0.25%, par는 별도 고정된 공통 거래량 설정을 쓴다. 각 표 안의 후보와 대조군은 같은 시장·기간·비용·위험 한도를 공유한다. 첫째/둘째 전략의 원본, 추가 전달 지연이 있는 단독 추종, 혼합을 구분한다.</p>'''
    for venue in ('reference','par'):
        body+='<h2>'+('기준시장: 대회 거래소 성과가 아님' if venue=='reference' else 'par: 캔들·거래량 기반 재현')+'</h2>'
        body+=table([r for r in net['rows'] if r['venue']==venue])
    body+='''<h2>원인과 한계</h2><p>기준시장 이후 구간에서 페어 단독 추종은 base +329.63, stress −782.09였다. 이전의 원래 페어 실행(+1,772.51 / +432.87)을 보유 수량 전달 방식으로 그대로 얻지 못했다. 진입 시점, 주문 가격 한도, 미완료 목표를 억제하는 방식이 함께 달라진 결과다.</p>
<p>par 이후 구간의 진단 대표는 base −474.95 / stress −717.37로, 기존 전략 −229.51 / −125.44보다 나빴다. base 손익은 가격 변동 −368.73, 실행 가격 비용 −49.00, 수수료 −43.25, 펀딩 −13.98로 구성됐다. 반복 요청 자체가 수수료를 만든 것은 아니며 실제 체결과 더 큰 보유 노출이 손익을 바꾸었다.</p>
<p>같은 par 구간에서 페어 단독 추종은 주문 기록 4개, 체결 0건이었다. 이 4개는 가격 한도가 아니라 캔들 거래량 참여 한도에서 가용 수량 0으로 처리됐다. 이는 과거 실제 호가 유동성이 없었다는 증거가 아니다. 전체 과거 L2와 주문 대기열이 없으므로 실제 거래소 체결 가능성과 이 보수적 대용치는 구분해야 한다.</p>
<h2>검증과 다음 수정</h2><p>12개 검사 통과: 미래 정보 차단, 상쇄 후 무거래, 펀딩 한 번만 반영, 한쪽 미체결 유료 청산, 유동성 회복 이후 청산 재시도, 반대 포지션 전환, 위험 한도, 미완료 기한 보존. 신규 계좌 재현 48개와 변경 없는 기준시장 대조군 8개를 보존했다. 최대 원장 오차는 5.48×10⁻⁹ 미만이다. 보호된 실거래 소스 373개와 예약 검증 의존 파일 345개가 변경되지 않았다.</p>
<p>다음 후보는 모의 계좌에서 이미 체결된 수량을 늦게 따라가는 방식 대신, 그 시점에 알려진 진입 의도·예상 청산 가격·남은 비용 여유를 함께 전달해 현재 par 가격에서 유효한지를 판단해야 한다. 무체결이 반복되는 거래량 대용치와 실제 호가 자료의 범위도 분리해 검증한다. 이번 결과는 교체 근거가 아니다.</p>'''
    (out/'report.html').write_text(document('페어 목표 수량 합산 비교',body));export_csv(out/'comparison.csv',net['rows'])
    summary=dict(selection=net['selection'],actual_replays=net['actual_replays'],native_reference_source_reuses=8,tests_passed=12,
        maximum_accounting_error=max(r['metrics']['maximum_accounting_error'] for r in net['rows']),
        orders_sent=0,automatic_live_replacement=False,historical_independence=False,protected=guards)
    save(out/'summary.json',summary)
    save(HERE/'work_state.json',dict(status='completed_diagnostic_no_candidate_qualified',report=str((out/'report.html').relative_to(ROOT)),
        actual_replays=48,native_reference_source_reuses=8,tests_passed=12,orders_sent=0,
        next='Preserve economic pair intentions and available-time bounds through one portfolio executor; do not copy already-filled shadow inventories without fresh entry-value check. Audit low-volume candle proxy separately from observable L2.'))
    print('Both reports written; source hashes and live/reserved guards verified.')


if __name__=='__main__':main()
