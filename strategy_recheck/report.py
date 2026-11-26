"""Decision, attribution, and reproducibility evidence for the fixed comparison."""
import csv
import gzip
import hashlib
import html
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

from competition_live.profile import ROOT, BUNDLE, verify_deployment, verify_bundle
from .protocol import OUT, freeze

NAMES = {
    'incumbent': '기존 제출판',
    'universe_only': '전체 종목 확대',
    'signed': '전체 종목＋롱숏 수정판',
    'incumbent_original_size': '기존판＋원래 규모',
    'signed_original_size': '수정판＋원래 규모',
}


def stamp(t):
    return datetime.fromtimestamp(t, ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(str(h))+'</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows) + \
        '</tbody></table></div>'


def attribution(rows):
    data = dict(np.load(BUNDLE/'data/reference.npz', allow_pickle=False))
    result = []
    for row in rows:
        if row['phase'] != 'comparison' or row['scenario'] != 'base':
            continue
        name = row['variant']
        r = json.load(gzip.open(OUT/'reference'/f'{name}_comparison_base.json.gz', 'rt'))
        end = int(np.searchsorted(data['times'], r['end']))
        qty = np.asarray(r['snapshots'][-1]['qty'])
        pnl = qty * data['par_close'][end]
        fees = np.zeros(10)
        funding = np.zeros(10)
        for f in r['fills']:
            j = f['asset']
            pnl[j] -= f['signed_qty']*f['price'] + f['fee']
            fees[j] += f['fee']
        for f in r['funding_events']:
            funding -= np.asarray(f['qty'])*np.asarray(f['prices'])*np.asarray(f['rates'])
        pnl += funding
        assert abs(pnl.sum()-row['net_pnl']) < 1e-6
        result.append(dict(variant=name, by_asset_pnl=pnl.tolist(), fees=fees.tolist(), funding=funding.tolist(),
                           base_owned_pnl=r['metrics']['base_owned_pnl'], pair_owned_pnl=r['metrics']['pair_owned_pnl']))
    return result


def main():
    spec = freeze()
    ref = json.loads((OUT/'reference_comparison.json').read_text())
    pas = json.loads((OUT/'pas_comparison.json').read_text())
    indices = {(r['variant'], r['phase'], r['scenario']): r for r in ref}
    pi = {(r['variant'], r['scenario']): r for r in pas}
    decisions = []
    for name in spec['variants']:
        if name == 'incumbent':
            continue
        checks = []
        for phase in spec['reference_windows']:
            for scenario in spec['pas_scenarios']:
                a, b = indices['incumbent', phase, scenario], indices[name, phase, scenario]
                checks.append(dict(venue='reference', phase=phase, scenario=scenario,
                                   pnl_difference=b['net_pnl']-a['net_pnl'], Q_difference=b['Q']-a['Q'],
                                   passed=b['net_pnl']>a['net_pnl'] and b['Q']>a['Q']))
        for scenario in spec['pas_scenarios']:
            a, b = pi['incumbent', scenario], pi[name, scenario]
            diff = b['pnl_after_liquidation_reserve']-a['pnl_after_liquidation_reserve']
            checks.append(dict(venue='PAS conditional', scenario=scenario,
                               terminal_adjusted_pnl_difference=diff, passed=diff>0))
        decisions.append(dict(variant=name, passes_all=all(c['passed'] for c in checks), checks=checks))
    passed = [d['variant'] for d in decisions if d['passes_all']]
    assert len(ref) == 30 and len(pas) == 10
    assert all(r['maximum_accounting_error']<1e-6 and r['timing_pass'] for r in ref+pas)
    deployment = verify_deployment()
    assert deployment['files'] == spec['deployment_hashes']
    assert verify_bundle() == spec['bundle_hash']
    actual = None
    with (ROOT/'.runtime/pas_competition_live/account_observations.jsonl').open() as f:
        for line in f:
            x = json.loads(line)
            if x.get('snapshot_time', 0) <= spec['pas_cutoff_ms']:
                if actual is None or x['snapshot_time'] > actual['snapshot_time']:
                    actual = {k: x[k] for k in ('snapshot_time', 'equity_before_tick', 'team_pnl_before_tick',
                        'team_return_percent_before_tick', 'qty', 'fees', 'funding', 'fills', 'candidate')}
    result = dict(recommendation='keep_current_strategy' if not passed else 'candidate_requires_frozen_forward_check',
                  eligible_candidates=passed, historical_independence=False, comparisons=decisions,
                  live_strategy_changed=False, deployment_files_verified=len(deployment['files']),
                  actual_account_nearest_cutoff=actual,
                  reason='No modified candidate improves PnL and Q across the predeclared matched comparisons.',
                  scope='Strategy-replacement decision; candidate execution refresh is separately function-tested, '
                        'but its live PnL benefit is not established by candle replay.')
    (OUT/'decision.json').write_text(json.dumps(result, indent=2))
    attr = attribution(ref)
    (OUT/'attribution.json').write_text(json.dumps(attr, indent=2))
    for filename, rows in [('reference_comparison.csv', ref), ('pas_comparison.csv', pas)]:
        with (OUT/filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    page = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        '<title>SFINX 전략 교체 재검증</title><style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 22px;line-height:1.7;color:#1c293b;background:#fafbfd}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #d6dfeb;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#e9eef5}.scroll{overflow:auto}.notice{padding:18px;background:#fff0d6;border-left:5px solid #ba7700}a{color:#155bb4}code{font-size:13px}</style>',
        '<h1>수정 후보로 교체할지 재검증</h1>',
        '<p class="notice"><strong>결론: 이번 수정판으로 교체하지 않습니다.</strong> 전체 종목 및 롱숏 기능은 작동하지만, 기존 제출판보다 PnL·Q가 함께 좋아진 후보는 없었습니다. 실거래 프로그램·설정·포지션은 변경하지 않았습니다.</p>',
        '<p>모든 결과는 재사용한 과거 자료에서의 개발 성과입니다. 모델 이름이나 학습 횟수를 바꾸는 실험이 아닙니다. 기존의 과거 정보 기반 롤링 모델 상태를 재사용하고, 수정한 추세·되돌림 신호와 규모의 효과를 분리했습니다. 독립 테스트 성과 또는 미래 수익 보장이 아닙니다.</p>',
        '<h2>비교 조건</h2>',
        f'<p>팀 초기 자본 100,000 USDP, 사용자 전략 50,000, 팀원 예약 50,000. 2,000 USDP 실험 상한을 제거했습니다. 실제 사용자 예산 기준 총노출 60%, 순노출 40%, 종목 15%/7,500 USDP, 일별 예측 위험 0.5%, 회당 정상 변경 15%, 여유율 90%를 적용했습니다. 상한은 투자 목표가 아닙니다.</p>',
        '<p>원래 규모는 사전 존재하던 하방 보정 69.3879%를 제거한 1.44117배의 기준 신호입니다. 페어 모델·페어 예산은 동일합니다. 규모를 결과에 맞춰 최적화하거나 5만 전액 투자를 강제하지 않았습니다. 종목 확대만 한 후보도 진단용으로 포함했습니다.</p>',
        '<p>과거 기준시장 모의 체결은 기존 제출판의 동일 엔진·가격 제한·거래량·위험 제약을 사용했습니다. 기본은 수수료 9bp/추가 가격 충격 10bp/거래량 참여 1%; 불리한 조건은 12bp/20bp/0.25%/펀딩 수취 50%입니다. 실제 PAS 과거 체결을 재현한 데이터가 아닙니다.</p>',
        '<p>Q는 팀 전체 자본 기준, 12:00 UTC 일별 표본으로 계산했습니다. R=누적 수익률, σ=일별 수익률 표본 표준편차×√일수, Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05). J=0으로 두어 금융 기여는 0.8Q입니다. σ와 일별 표준편차는 서로 다른 수치입니다. <a href="https://perpetual-alpha26.github.io/leaderboard">공식 리더보드</a>.</p>']
    headers = ['전략', '조건', '순손익 USDP', '팀 수익률 %', '기간 변동성 σ %', '일별 표준편차 %', 'Q', '평균 투자액', '배정자본 사용 %', '체결 조각', '완료 페어']
    for phase in spec['reference_windows']:
        selected = [r for r in ref if r['phase']==phase]
        title = {'validation':'8월 개발 비교', 'comparison':'9월~10월 15일 개발 비교', 'continuous':'8월~10월 15일 연속 운용'}[phase]
        page += [f'<h2>{title}</h2><p>{stamp(selected[0]["start"])} → {stamp(selected[0]["end"])}</p>',
            table(headers, [[NAMES[r['variant']],r['scenario'],f'{r["net_pnl"]:+.2f}',f'{100*r["team_return"]:+.5f}',
                f'{100*r["volatility"]:.5f}',f'{100*r["volatility"]/math.sqrt(r["daily_returns"]):.5f}',
                f'{r["Q"]:.6f}',f'{r["mean_notional"]:,.2f}',f'{r["mean_notional"]/500:.2f}',
                r['fills'],r['completed_pairs']] for r in selected])]
    page += ['<h2>비용과 종목별 기여</h2>',
        '<p>9월~10월 15일 기본 조건입니다. 전체 종목 롱숏 후보의 가격 변동 수익은 늘었지만 비용 증가가 더 컸습니다. 페어 부문의 기여는 이 비교에서 동일합니다. 종목별 수치는 체결 현금흐름·수수료·펀딩·최종 평가를 합산해 계좌 순손익과 대조했습니다.</p>',
        table(['전략','가격 변동 수익','수수료＋가격 충격','펀딩','기준 매매 기여','페어 기여'],
            [[NAMES[r['variant']],f'{r["price_move_pnl"]:.2f}',f'{r["fees"]+r["execution_cost"]:.2f}',f'{r["funding"]:+.2f}',
              f'{next(a for a in attr if a["variant"]==r["variant"])["base_owned_pnl"]:+.2f}',
              f'{next(a for a in attr if a["variant"]==r["variant"])["pair_owned_pnl"]:+.2f}']
             for r in ref if r['phase']=='comparison' and r['scenario']=='base'])]
    symbols = ['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY']
    page.append(table(['전략']+symbols, [[NAMES[a['variant']]]+[f'{v:+.2f}' for v in a['by_asset_pnl']] for a in attr]))
    page += ['<h2>PAS 실제 자료를 이용한 조건부 모의 평가</h2>',
        f'<p>{stamp(spec["pas_start_ms"]/1000)} → {stamp(spec["pas_cutoff_ms"]/1000)}. 과거 L2·오라클·마크 전체가 없어 이후 활성 분봉의 고가/저가와 추가 가격 충격, 과거 거래량 참여 상한을 사용했습니다. 실제 거래소 호가 수량을 의미하지 않습니다. GOLD는 시작 25분 후, USDEUR는 7분 후 첫 관측 가격이 있어 그 이전 신규 목표를 막았습니다. 미래 가격으로 채우지 않았습니다.</p>',
        '<p><strong>이 모의 평가는 기존판의 실제 체결액도 재현하지 못합니다.</strong> 분봉 거래량 상한과 가격 제한 때문에 모의 노출이 실제보다 매우 작습니다. 따라서 실거래 손익 예측이나 전략 우열의 확정 근거로 사용하지 않습니다. 불리한 조건에서 손실이 더 작아진 것은 체결 감소의 영향입니다.</p>',
        '<p>미래 분봉은 체결 모의·평가 정답에만 사용했습니다. 목표는 실제 과거 신호 게시 시점 이후에만 사용했고 체결은 다음 분봉 이후입니다. 펀딩 비율은 실제 PAS 정산 기록이지만 금액은 직전 체결가격으로 근사했습니다. 종료 비용 차감은 청산 비용 충당 추정이며 실제 전량 청산 체결이 아닙니다.</p>',
        table(['전략','조건','평가손익','종료 비용 충당 후','평균 노출','체결 조각','숏 진입 조각','Q'],
              [[NAMES[r['variant']],r['scenario'],f'{r["net_pnl"]:+.3f}',
                f'{r["pnl_after_liquidation_reserve"]:+.3f}',f'{r["mean_notional"]:.2f}',
                r['fills'],r['short_opening_fills'],'계산 불가: 일별 수익률 1개'] for r in pas]),
        '<p>PAS 구간에서 캡처한 실제 페어 진입 검사는 모두 거부되었습니다. 이 구간의 비교는 기준 매매 부분을 재생하며 가상의 페어 수익을 추가하지 않았습니다. Q·σ를 0으로 대체하지 않았습니다.</p>']
    if actual:
        page.append(f'<p>참고: 종료 시각 직전 실제 계좌 {stamp(actual["snapshot_time"]/1000)}의 손익은 '
                    f'{actual["team_pnl_before_tick"]:+.4f} USDP입니다. 위 가상 전략 손익과 직접 비교하면 안 됩니다.</p>')
    page += ['<h2>기능 검증과 교체 판단</h2>',
        '<p>25개 테스트 통과: 전 종목·양방향 신호, 미래 입력 추가 시 과거 판단 불변, 후순위 종목의 호가 갱신, 청산 확인 후 반대 진입, 실제 위험 한도, 2천 상한 제거, 미관측 종목 제외, 회계 대조. 30개 과거 계좌와 10개 PAS 모의 계좌의 회계 오차는 10⁻⁶ 미만이고 신규 주문의 시점 조건을 통과했습니다. 기존 보고서의 기준판 PnL·Q를 정확히 재현했습니다.</p>',
        '<p>다만 신호 함수·주문 어댑터 테스트는 수정판 전체를 실거래에 배포했다는 뜻이 아닙니다. 역사적 분봉으로는 호가 갱신 수정의 실제 PnL 개선을 입증할 수 없습니다. 기존 위험 감축 후 재확대 경로 등 앞선 구조 검토 사항을 모두 해결했다고 주장하지 않습니다.</p>',
        '<p><strong>수정 전략으로 교체하거나 규모를 원래대로 늘리는 것을 권하지 않습니다.</strong> 규모 복원은 일부 기간 PnL을 높였으나 9월 이후 Q를 악화시켰고 불리한 조건에서는 PnL도 낮아졌습니다. 전 종목/롱숏 후보는 더 많은 자본을 사용했지만 순성과가 낮았습니다. 이는 현재판이 최적이라는 증명이 아니라 이번 후보의 교체 근거가 부족하다는 판단입니다.</p>',
        '<p>실거래 코드 20개 파일과 모델 번들 해시를 검증했습니다. 사용자 지갑에 새 주문을 보내거나 서비스 교체를 하지 않았습니다. 평가 대상 후보·조건·원자료·결과는 동결해 보존합니다. 합격 후보가 없어 새로운 독립 성과가 있는 것처럼 표시하거나 자동 승격하지 않았습니다.</p>',
        '<p><a href="protocol.json">사전 비교 조건</a> · <a href="decision.json">판단 및 비교 차이</a> · '
        '<a href="reference_comparison.csv">과거 비교 CSV</a> · <a href="pas_comparison.csv">PAS 비교 CSV</a> · '
        '<a href="attribution.json">종목 기여</a> · <a href="tests.log">테스트</a> · <a href="manifest.json">재현 해시</a></p></html>']
    (OUT/'report.html').write_text('\n'.join(page))
    files = [f for f in OUT.rglob('*') if f.is_file() and f.name!='manifest.json']
    files += list((ROOT/'strategy_recheck').glob('*.py'))
    manifest = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'decision':result['recommendation'], 'eligible':passed,
                      'reference_runs':len(ref), 'PAS_runs':len(pas), 'artifacts':len(manifest)}, indent=2))


if __name__ == '__main__':
    main()
