import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from competition_live.profile import ROOT,verify_deployment,verify_bundle
from strategy_recheck.report import table,stamp
from .protocol import OUT,freeze

NAMES=dict(incumbent='기존 제출판',risk_only='위험 감축 보존만',
           short_only='기존 종목에 숏만 추가',short_and_exit='숏＋추세 반전 청산',
           risk_short_exit='위험 수정＋숏·반전 청산',risk_short_exit_all='앞 수정＋전체 종목',
           size_only='규모 복원만 (진단용)')


def main():
    spec=freeze()
    ref=json.loads((OUT/'reference_results.json').read_text())
    pas=json.loads((OUT/'pas_results.json').read_text())
    trace=json.loads((OUT/'target_trace.json').read_text())
    nomination=json.loads((OUT/'nomination.json').read_text())
    assert len(ref)==42 and len(pas)==14
    assert all(r['timing_pass'] and r['accounting_error']<1e-6 for r in ref)
    assert all(r['timing_pass'] and r['maximum_accounting_error']<1e-6 for r in pas)
    assert all(r['risk_breach_bars']==0 for r in ref+pas)
    deployed=verify_deployment()
    assert deployed['files']==spec['deployment_hashes'] and verify_bundle()==spec['bundle_hash']
    # Verify the old benchmark remains exactly reproduced.
    prior=json.loads((ROOT/'reports/strategy_recheck_20261020/reference_comparison.json').read_text())
    for r in ref:
        if r['variant']!='incumbent':continue
        old=next(x for x in prior if x['variant']=='incumbent' and x['phase']==r['phase'] and x['scenario']==r['scenario'])
        assert abs(r['net_pnl']-old['net_pnl'])<1e-8 and abs(r['Q']-old['Q'])<1e-12
    # The freeze used by the public-quote smoke check must still match.
    frozen=json.loads((OUT/'candidate_freeze.json').read_text())
    for f,h in frozen['files'].items():
        assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
    tests=(OUT/'tests.log').read_text()
    assert '\nOK\n' in tests and 'FAILED' not in tests
    idx={(r['variant'],r['phase'],r['scenario']):r for r in ref}
    checks=[]
    for name in spec['arms']:
        if name=='incumbent':continue
        comparisons=[]
        for phase in ('validation','comparison','continuous'):
            for scenario in ('base','stress'):
                a=idx['incumbent',phase,scenario];b=idx[name,phase,scenario]
                comparisons.append(dict(phase=phase,scenario=scenario,
                    pnl_difference=b['net_pnl']-a['net_pnl'],Q_difference=b['Q']-a['Q'],
                    both_improved=b['net_pnl']>a['net_pnl'] and b['Q']>a['Q']))
        checks.append(dict(variant=name,comparisons=comparisons))
    decision=dict(engineering_fixes=['per-needed-asset fresh quote','size then attenuate trend',
        'risk cuts bypass trade buffer/turnover','one quantity ceiling per publication'],
        execution_fix_tested=True,risk_fix_tested=True,short_order_path_tested=True,
        sizing_unit_bug_found=False,capital_use_forced=False,
        august_nominee=nomination['nominee'],strategy_promotion=False,
        live_strategy_changed=False,live_orders_sent_by_this_task=0,
        reference_runs=len(ref),pas_runs=len(pas),historical_independence=False,
        note='Do not reject a correct execution/risk contract merely because it reduces historical profit. '
             'Do not describe those repairs as proof of improved alpha. Short-only improves the later reused '
             'window but fails the predeclared earlier selection; no full replacement nomination.',
        checks=checks)
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
    for name,rows in [('reference_results.csv',ref),('pas_results.csv',pas)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    baseline=trace[0]
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        '<title>SFINX 실행·위험·숏 분리 수정 검증</title><style>body{font:16px/1.75 system-ui,sans-serif;max-width:1250px;margin:32px auto;padding:0 22px;color:#172c3e;background:#fafbfd}h1,h2{line-height:1.35}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:9px;border-bottom:1px solid #d9e1ea;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#e9eff5}.box{padding:20px;background:#eaf4fb;border-left:5px solid #276d94}.warn{background:#fff1d8;padding:16px}a{color:#145d93}</style>',
        '<h1>실행·위험 관리·숏·투자 규모를 분리한 수정과 검증</h1>',
        '<div class="box"><strong>구현 결함 수정과 전략 성과 판단을 분리했습니다.</strong> 호가 만료·위험 감축 상쇄·감축 주문 생략을 재현하고 수정 후보의 통합 실행까지 확인했습니다. 숏만 추가한 후보는 최근 과거 구간에서 PnL과 Q가 개선됐지만 8월에는 악화되어, 사전 기준에 따른 전체 전략 교체 후보는 선정되지 않았습니다. 실거래 전략은 변경하지 않았습니다.</div>',
        '<h2>문제별 확인과 수정</h2>',
        table(['항목','재현한 원인','수정 및 검증'],[
            ['후순위 주문 호가 만료','첫 주문 확인에 4초가 걸리면 다음 종목 문맥의 3초 유효시간 초과','필요 종목마다 재조회. 신호·수량·가격 제한 유지. 누락 체결 응답은 중단하고 재주문 금지'],
            ['위험 감축 상쇄','normalize_risk뿐 아니라 뒤의 목표 위험 수량 계산도 50% 감축을 완전히 복원','필터 전 추세의 규모를 먼저 계산한 뒤 기존 0.5~1.0 감축률 적용. 되돌림 신호는 독립적으로 유지'],
            ['최종 감축 주문 누락','20% 미만 수량 변경 생략 및 정상 회전율 한도가 감축을 막음','정당한 최종 순노출 감축에만 별도 상한. 감축은 버퍼·회전율을 우회하고 다른 종목으로 재배분하지 않음'],
            ['불필요한 반복 감축','같은 신호를 가격 틱마다 다시 수량으로 환산하면 소량 감축 반복 가능','신호 게시별 수량 상한을 고정. 새 게시에서만 갱신. 계좌 전체의 위험 검사는 현재 가격으로 계속 수행'],
            ['숏 미진입','기존 주요 EMA·채널 추세 상태는 0/+1; 되돌림과 페어만 잠재적 숏 경로','기존 종목에서 -1 상태와 대칭 진입을 따로 추가. EMA 반전 청산과 종목 확대는 별도 비교'],
            ['낮은 자본 활용','50% 배정 한도와 실제 투자 목표를 혼동할 수 있음','금액/비율 단위 오류는 발견하지 못함. 목표 위험·고정 보정·진입 거절을 분해하고 강제 증액하지 않음']]),
        '<p>위험 수정의 범위는 기준 매매(base owner)의 추세 필터입니다. 페어 모델과 소유권은 변경하지 않았습니다. 반대 방향의 되돌림 포지션까지 무조건 줄이는 새로운 위험 정책을 섞지 않았습니다. 최종 순노출이 실제로 감소한 종목만 감축 보호 대상으로 삼았습니다.</p>',
        '<h2>과적합과 정보 누수 관리</h2>',
        '<p>7개 조합·기존 파라미터·비용 조건을 먼저 고정했습니다. 기간 성과를 본 뒤 임계값·보유기간을 탐색하지 않았습니다. 기존 학습 산출물은 유지했으며 새로운 PAS 결과를 학습시키지 않았습니다. 페어 모델의 성숙하지 않은 미래 라벨 차단, 미래 자료를 덧붙여도 과거 신호가 바뀌지 않는 검사를 통과했습니다.</p>',
        '<p>8월 결과로 후보를 선정하고 그 선택 파일을 저장한 뒤 9월~10월 15일 및 연속 운용을 평가했습니다. 규모 복원은 진단용이며 후보 선정에서 제외했습니다. 8월 양쪽 체결 조건에서 기존판 PnL·Q를 모두 넘은 전략 후보가 없어 선택값은 null입니다. 이후 구간이 좋아졌다는 이유로 선택 기준을 바꾸지 않았습니다.</p>',
        '<p class="warn">그러나 이 과거 기간은 이전 개발에서 이미 본 자료입니다. 시간순 계산과 미래 누수 검사를 통과해도 완전히 미사용한 독립 테스트가 되지는 않습니다. 모든 표는 개발 성과이며 독립 검증 성과로 표시하지 않습니다.</p>',
        '<h2>비교 조건</h2>',
        '<p>팀 자본 100,000 USDP, 사용자 예산 50,000, 팀원 예약 50,000. 2,000 USDP 고정 상한 없음. 기존 총노출·순노출·종목·회전율·위험 한도를 유지했습니다. 기존 계좌 엔진을 독립 모듈로 읽어 수정 컨트롤러만 주입했으며, 실제 운용 파일은 변경하지 않았습니다. 기존판의 과거 PnL과 Q는 앞선 보고서와 정확히 일치합니다.</p>',
        '<p>기본 비용: 수수료 9bp, 가격 충격 10bp, 거래량 참여 1%. 불리한 조건: 12bp, 20bp, 0.25%, 펀딩 수취 50%. 기준시장 과거 체결 모의이며 PAS의 당시 호가 재현은 아닙니다. Q와 σ는 팀 자본의 12:00 UTC 일별 표본으로 계산합니다. σ는 일별 수익률 표본 표준편차×√일수입니다.</p>']
    for phase,title in [('validation','8월: 사전 선정 구간'),('comparison','9월~10월 15일: 이후 비교 구간'),
                        ('continuous','8월~10월 15일: 연속 운용')]:
        rows=[r for r in ref if r['phase']==phase]
        page += [f'<h2>{title}</h2><p>{stamp(rows[0]["start"])} → {stamp(rows[0]["end"])}</p>',
            table(['후보','조건','순손익 USDP','팀 수익률 %','기간 σ %','Q','팀 최대낙폭 %','평균 투자액','숏 진입 조각','완료 페어'],
             [[NAMES[r['variant']],r['scenario'],f'{r["net_pnl"]:+.2f}',f'{100*r["team_return"]:+.5f}',
               f'{100*r["volatility"]:.5f}',f'{r["Q"]:.6f}',f'{100*r["max_drawdown"]:.4f}',
               f'{r["mean_notional"]:,.2f}',r['short_entries'],r['completed_pairs']] for r in rows])]
    page += ['<p>숏만 추가한 경우와 추가로 EMA 반전 청산을 켠 경우는 이 과거 자료에서 목표 비중이 같았습니다. 청산 기능 자체는 별도의 상승 후 하락 시나리오에서 작동함을 확인했습니다. 이 자료에서 효과가 있었다고 과장하지 않습니다.</p>',
        '<h2>현재 신호와 투자 규모의 원인</h2>',
        f'<p>고정한 입력 시각: {stamp(trace[0]["source_time"])}. 기존 목표 명목액은 5만 예산 중 {baseline["target_notional_at_50k"]:,.2f} USDP입니다. 예측 일별 변동성은 사용자 예산 기준 {baseline["predicted_user_daily_volatility"]*100:.4f}%, 계좌 위험 상한은 0.5%입니다. 상한까지 반드시 투자하도록 설정된 전략이 아닙니다.</p>',
        '<p>기준 수량 계산의 목표 위험은 일별 0.1%이고, 90% 여유·3/7일 수량 조정·최종 0.649594배 보정이 이어집니다. 기존 하방 보정은 원래 규모의 69.3879%입니다. 페어 진입이 거절되면 그 예산도 사용하지 않습니다. 낮은 비중은 이 설계의 결과이며, 추가 100분의 1 환산이나 1백만 자본 잔재 때문에 생긴 오류라는 근거는 발견하지 못했습니다.</p>',
        table(['후보','목표 명목액 (50k)','예측 일별 위험 %','BTC 목표 %','ETH 목표 %','HYPE 목표 %'],
          [[NAMES[t['variant']],f'{t["target_notional_at_50k"]:,.2f}',f'{100*t["predicted_user_daily_volatility"]:.4f}']+
           [f'{100*t["assets"][j]["final_weight"]:+.4f}' for j in range(3)] for t in trace]),
        '<p>숏 후보가 실제로 ETH·HYPE에 음수 목표를 생성하고 주문 경로가 매도 신규 진입으로 처리함을 확인했습니다. BTC의 장기 추세가 양수인 상황까지 수동으로 숏으로 뒤집지는 않았습니다. 전체 종목 후보에서 HYPE의 순방향이 달라질 수 있는 것은 추세·되돌림과 공통 위험 배분의 합성 때문입니다.</p>',
        '<h2>본대회 이후 PAS 구간: 별도 평가</h2>',
        f'<p>{stamp(spec["pas_start_ms"]/1000)} → {stamp(spec["pas_cutoff_ms"]/1000)}. 실제 공개 PAS 분봉·거래량·펀딩을 새로 수집했습니다. 미래 값은 평가 정답과 체결 모의에만 사용했고 신호의 입력·학습·설정 선택에 사용하지 않았습니다.</p>',
        '<p class="warn">과거 호가 전체와 오라클·마크 기록이 없어 PAS 평가는 조건부입니다. 분봉 거래량 상한이 실제 호가 유동성을 재현하지 못해 모의 노출은 실제보다 훨씬 작습니다. 이 결과만으로 실거래에서 더 좋아졌거나 나빠졌다고 확정할 수 없습니다. 불리한 조건에서 0인 경우는 미체결이며 좋은 전략 성과가 아닙니다.</p>',
        table(['후보','조건','평가손익','청산 비용 충당 후','평균 노출','체결 조각','숏 진입 조각'],
          [[NAMES[r['variant']],r['scenario'],f'{r["net_pnl"]:+.4f}',f'{r["pnl_after_liquidation_reserve"]:+.4f}',
            f'{r["mean_notional"]:.2f}',r['fills'],r['short_opening_fills']] for r in pas]),
        '<p>일별 수익률은 1개라 σ·Q를 계산하지 않았습니다. 펀딩 비율은 실제 정산 기록이지만 지급액은 직전 체결가격으로 근사했습니다. 종료 청산 비용 충당은 실제 청산 완료가 아닙니다. 초기 가격이 없는 GOLD·USDEUR는 관측 전 신규 목표에서 제외했고 미래 가격으로 채우지 않았습니다.</p>',
        '<p>보조적으로 기존 6시간 판단 간격의 PAS 방향을 확인했습니다. 신선한 시작·종료 가격이 있는 표본은 기존/숏 후보 각각 3개뿐입니다. 기존 방향은 0/3, 숏 후보는 2/3이었으나 표본이 너무 적어 예측력·수익성의 근거로 채택하지 않았습니다. 누락 표본과 원 라벨을 모두 저장했습니다.</p>',
        '<h2>기능·통합 검증 및 판단</h2>',
        '<p>35개 테스트 통과. 50% 위험 감축의 완전 상쇄 재현, 위험 감축의 버퍼·회전율 우회, 다른 종목 재확대 방지, 게시별 수량 고정, 실제 coordinator→확인된 모의 체결 경로의 위험 감축과 숏 진입, 미확인 체결의 재주문 금지, 청산 후 반대 진입, 미래 라벨 배제와 과거 신호 불변을 검증했습니다. 모의 체결은 시험용 응답이며 실제 주문을 전송하지 않았습니다.</p>',
        '<p>수정 소스를 동결한 뒤 공개 PAS 10개 종목의 호가를 읽어 갱신·유효시간 검사를 통과했습니다. 이는 호가 입력 기능 확인이며 미래 수익 검증이 아닙니다. 42개 기준시장 계좌와 14개 PAS 모의 계좌에서 회계 오차는 10⁻⁶ 미만, 정보 시점 검사 통과, 기록된 위험 한도 위반은 0이었습니다.</p>',
        '<p><strong>실행 오류와 위험 감축 계약의 수정은 구현·검증 완료입니다.</strong> 역사적 이익이 줄었다는 이유로 오류 수정을 무효화하지 않습니다. 다만 수정본이 이미 실거래 서비스에 반영됐다는 뜻은 아닙니다. 현재 주문을 내는 20개 운용 파일과 모델 번들은 그대로 유지했습니다.</p>',
        '<p><strong>숏 전략을 포함한 전체 교체의 성과 근거는 아직 일관되지 않습니다.</strong> 최근 비교에서는 숏 후보와 위험＋숏 후보가 PnL·Q를 개선했으나, 미리 정한 8월 선정 조건과 연속 운용에서는 기존판보다 낮았습니다. 최근 구간만 골라 승격하지 않았습니다. 규모만 키우는 안도 채택하지 않습니다.</p>',
        '<p><a href="protocol.json">사전 조건</a> · <a href="nomination.json">이후 구간 평가 전 선정 기록</a> · '
        '<a href="decision.json">최종 판단</a> · <a href="reference_results.csv">42개 과거 결과</a> · '
        '<a href="pas_results.csv">14개 PAS 결과</a> · <a href="target_trace.json">규모·방향 계산 원장</a> · '
        '<a href="pas_direction_labels.json">PAS 평가 라벨</a> · <a href="tests.log">테스트</a> · '
        '<a href="candidate_freeze.json">후보 동결 해시</a> · <a href="quote_probe.json">공개 호가 검사</a> · '
        '<a href="manifest.json">전체 산출물 해시</a></p></html>']
    (OUT/'report.html').write_text('\n'.join(page))
    files=[f for f in OUT.rglob('*') if f.is_file() and f.name!='manifest.json']
    files+=list((ROOT/'strategy_repairs').glob('*.py'))
    (OUT/'manifest.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                               for f in files},indent=2))
    print(json.dumps(dict(reference_runs=len(ref),pas_runs=len(pas),nominee=nomination['nominee'],
                          deployment_files=len(deployed['files']),orders_sent=0),indent=2))


if __name__=='__main__':
    main()
