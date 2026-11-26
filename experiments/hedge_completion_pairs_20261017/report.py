import csv,gzip,json
import numpy as np
from .research import HERE,ROOT,PREVIOUS
from experiments.adaptive_mean_pairs_20261017.attribution import episodes
from experiments.quote_anchored_pairs_20261017.report import table
from experiments.par_direct_forecast_20261017.data import save,sha
from experiments.complementary_alpha_20261017.research import protected


def main():
    data={v:json.loads((HERE/f'{v}_artifacts/results.json').read_text()) for v in ('reference','par')}
    books=json.loads((HERE/'book_artifacts/results.json').read_text())
    prior=json.loads((PREVIOUS/'attribution_artifacts/results.json').read_text())
    manifests=[HERE/f'{v}_artifacts/frozen.json' for v in ('reference','par')]+[HERE/'book_artifacts/frozen.json',
        PREVIOUS/'reference_artifacts_v2/frozen.json',PREVIOUS/'attribution_artifacts/frozen.json']
    for manifest in manifests:
        for path,h in json.loads(manifest.read_text())['sources'].items():assert sha(ROOT/path)==h,path
    max_error=0.;flat=[];episode_rows=[];terminal=[];count=0
    for venue in data:
        for path in sorted((HERE/f'{venue}_artifacts/runs').glob('*.json.gz')):
            with gzip.open(path,'rt') as f:r=json.load(f)
            m=r['metrics'];returns=np.asarray([d['return_'] for d in r['daily']]);vol=np.std(returns,ddof=1)*np.sqrt(len(returns))
            q=(max(m['roi'],0)+.005)/(abs(m['roi'])+1.5*vol+.05)
            assert abs(q-m['financial_score'])<1e-12 and abs(vol-m['volatility_horizon'])<1e-12
            assert abs(np.prod(1+returns)-1-m['roi'])<1e-10
            for e in r['events']:
                if e['event'] in ('pair_intent','completion_intent'):
                    assert e['minimum_expected_cash_at_limits']>=e['buffer_cash']-1e-8
            max_error=max(max_error,m['maximum_accounting_error']);count+=1
            flat.append(dict(venue=venue,phase=r['phase'],arm=r['label'],scenario=r['scenario'],**m))
            if venue=='reference' and r['label']=='fixed_delayed_complete_168':
                episode_rows.extend(dict(phase=r['phase'],scenario=r['scenario'],**e) for e in episodes(r))
                last=r['snapshots'][-1];terminal.append(dict(phase=r['phase'],scenario=r['scenario'],cash_net=last['cash']-1e6,
                    unrealized=last['equity']-last['cash'],residual_gross=m['residual_gross'],net_after_reserve=m['pnl_after_liquidation_reserve']))
    assert count==40 and len(books['rows'])==74
    guards=protected();target=ROOT/'reports/hedge_completion_pairs_20261017';target.mkdir(exist_ok=True)
    body='<h1>한쪽 체결 이후 헤지 완성: 구현과 비교</h1>'
    body+='<p><strong>기준시장 후반에서 기본·스트레스 모두 기존 두 전략보다 높은 PnL과 Q를 얻었습니다. 대회 시장의 개선은 아직 입증하지 못했습니다.</strong> 새 후보는 앞선 선택 구간에서도 양수가 됐지만 그 구간의 기존 전략 수익에는 못 미쳤고, 실제 par 과거 체결 모형에서는 진입이 없었습니다. 따라서 실거래 교체는 보류합니다.</p>'
    body+='<h2>원인 분해</h2><p>직전 페어 후보는 기본 +1,773.03 USDPR에서 스트레스 −59.02로 내려갔습니다. 기본 체결·보유 경로를 고정한 채 수수료·불리한 가격·펀딩 수취액만 스트레스 조건으로 바꾸면 +544.77입니다. 나머지 −603.79는 실제 스트레스에서 체결과 보유 경로가 바뀐 영향입니다. 이 계산은 원인 분해이며, 기본 체결이 스트레스 조건에서도 가능하다는 가정의 실현 수익은 아닙니다.</p>'
    ca=next(x for x in prior['cost_attribution'] if x['phase']=='comparison' and x['arm']=='fixed_delayed_168')
    body+=table(['항목','USDPR'],[['기본 순손익',f"{ca['base_pnl']:+.2f}"],['수수료 증가',f"{-ca['extra_fee_same_prices']:+.2f}"],
        ['체결가격 불리함 증가',f"{-ca['extra_adverse_execution']:+.2f}"],['가격 변경에 따른 추가 수수료',f"{-ca['fee_change_on_repriced_fills']:+.2f}"],
        ['양수 펀딩 수취액 감소',f"{-ca['lost_positive_funding_receipts']:+.2f}"],['고정 경로 스트레스 손익',f"{ca['stress_on_fixed_base_inventory']:+.2f}"],
        ['달라진 체결·보유 경로',f"{ca['changed_path_contribution']:+.2f}"],['실제 스트레스 재생',f"{ca['actual_stress_pnl']:+.2f}"]])
    body+='<h2>수정한 상태 관리</h2><p>한쪽만 체결되면 이미 체결된 수량을 유지하고, 미체결된 종목을 한 번 더 주문할 수 있게 했습니다. 처음 고정한 예상 페어 가치에서 이미 낸 수수료, 해당 페어의 펀딩, 남은 진입·청산 비용을 모두 빼고 5bp 여유가 있을 때만 주문합니다. 그 여유로 미체결 종목의 제한가를 미리 고정합니다. 목표 평균에 도달했거나 손절 조건에 걸렸으면 추가 헤지를 하지 않습니다. 추가 주문도 실패하면 유료 감축하고, 일부 체결이면 초과 수량을 줄입니다.</p>'
    body+='<p>첫 주문의 결과가 불명확하면 추가 주문을 만들지 않습니다. 같은 계좌의 다른 포지션 수익으로 이 페어의 비용 예산을 보충하지 않습니다. 추가 시도 중 한쪽 노출 시간이 늘어나는 위험도 별도로 기록합니다. 실제 거래소의 IOC 미체결 잔량 취소 규칙을 확인했으며, 주문 두 개를 원자적으로 함께 체결된 것으로 처리하지 않았습니다. <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint">공식 주문 API 설명</a>.</p>'
    def performance(venue,phase):
        rows=[];rr=data[venue]['rows']
        for r in rr:
            if r['phase']!=phase or r['scenario']!='base':continue
            a=r['metrics'];b=next(x['metrics'] for x in rr if x['arm']==r['arm'] and x['phase']==phase and x['scenario']=='stress')
            rows.append([r['arm'],f"{a['net_pnl']:+.2f}",f"{b['net_pnl']:+.2f}",f"{a['roi']*100:+.6f}%",
                f"{a['volatility_horizon']*100:.6f}%",f"{a['financial_score']:.6f}" if a['eligible'] else '— (미충족)',
                f"{b['financial_score']:.6f}" if b['eligible'] else '— (미충족)',f"{a['completed_pairs']}/{b['completed_pairs']}",
                f"{a['completion_balanced']}/{b['completion_balanced']}",f"{a['mean_gross']*100:.4f}%"])
        return table(['전략','기본 순손익','스트레스 순손익','기본 R','기본 기간 σ','기본 Q','스트레스 Q','페어 완료 기본/스트레스','추가 헤지 성공 기본/스트레스','기본 평균 투자 비중'],rows)
    body+='<h2>기준시장 비교</h2><p>초기 자본 1,000,000 USDPR. 선택: 8월 1일 21시~9월 1일 21시(KST), 후반: 9월 1일 21시~10월 15일 21시. 실제 기준시장 가격·거래량·펀딩을 사용한 시간봉 근사입니다. 기본 비용은 편도 수수료 9bp·불리한 가격 10bp, 스트레스는 12bp·20bp와 낮은 거래량 한도·절반의 양수 펀딩 수취입니다. 실거래 par 계좌의 결과가 아닙니다.</p>'
    body+='<h3>선택 구간</h3>'+performance('reference','validation')
    body+='<h3>이후 비교 구간</h3>'+performance('reference','comparison')
    body+='<p>추가 헤지형은 선택 구간 기본 +553.08·스트레스 +1,206.49, 후반 기본 +1,772.51·스트레스 +432.87입니다. 스트레스가 선택 구간에서 더 높은 이유는 단순 비용 증가만이 아니라 체결 경로와 보유 규모도 달라졌기 때문입니다. 선택 구간의 기존 1·2번보다 수익과 Q가 낮고 스트레스 페어 완료도 두 건이어서 미리 정한 선택 기준을 통과하지 못했습니다. 후반 결과를 보고 통과 판정을 바꾸지 않았습니다.</p>'
    body+='<p>8월의 양수는 BTC/ETH 한 거래의 이익에 의존합니다. 해당 거래의 +1,683.75를 제외하면 기본 손익은 음수입니다. 후반에는 BTC/HYPE 미종료 포지션도 있습니다. 따라서 표본을 충분히 확보한 일반화 성과로 판단할 수 없습니다.</p>'
    body+=table(['기간','조건','현금 순손익','미실현손익','미청산 명목액','청산비 추정 차감 후'],[
        [r['phase'],r['scenario'],f"{r['cash_net']:+.2f}",f"{r['unrealized']:+.2f}",f"{r['residual_gross']:.2f}",f"{r['net_after_reserve']:+.2f}"] for r in terminal])
    body+='<p>후반 기본 한쪽 노출 누적 시간은 직전 방식 60분에서 240분으로 증가했습니다. 추가 헤지는 공짜 위험 제거가 아닙니다. 계좌 위험 한도를 계속 적용했고, 거래량이나 가격 제한으로 즉시 줄일 수 없는 잔여량도 기록했습니다. 기존 전략과 혼합한 경우 종목 충돌 방지 규칙으로 대부분 추가 진입이 막혀 성과 개선이 없었습니다. 이 혼합은 정상적인 포트폴리오 배분 최적화의 증거가 아닙니다.</p>'
    body+='<h2>대회 par 과거 비교</h2><p>선택: 10월 1일 21시~8일 21시, 후반: 8일 21시~16일 21시(KST). 실제 par 체결 봉·펀딩을 쓰되 전체 호가가 없어 다음 5분봉 종가와 관측 거래량으로 체결을 근사합니다. 이 모형에서 두 페어 후보는 거래가 없었습니다. 거래량 0이 실제 주문장 수량 0을 뜻하지는 않습니다. Q의 거래 요건을 충족하지 않은 0 손익을 개선으로 처리하지 않습니다.</p>'
    body+=performance('par','comparison')
    body+='<h2>별도의 실제 호가·체결 진단</h2><p>이전에 고정한 일곱 예측 후보에서 나온 가정상 지정가 체결 조각 37개를 그대로 두고 추가 헤지 유무의 74개 경로를 비교했습니다. 새 평균회귀 전략의 진입 사례가 아닙니다. 추가 주문은 첫 IOC가 관측상 미체결일 때만, 1초 응답 지연과 1초 주문 지연을 거쳐 새로운 호가에서 시도합니다. 긴 보유의 청산에는 실제 호가가 없어 기존 캔들 근사가 남아 있습니다. 중복된 탐침을 하나의 계좌 PnL이나 Q로 합산하지 않습니다.</p>'
    qr=[]
    for a in books['summaries']:
        if a['completion_enabled']:continue
        b=next(x for x in books['summaries'] if x['model']==a['model'] and x['scenario']==a['scenario'] and x['completion_enabled'])
        qr.append([a['model'],a['scenario'],a['probes'],b['completion_plans'],b['completed_hedges'],
            f"{a['net_cash']:+.6f}",f"{b['net_cash']:+.6f}",f"{b['net_cash']-a['net_cash']:+.6f}",b['unknown_outcomes']])
    body+=table(['예측 후보','비용','동일 체결 조각','추가 주문','추가 완전 헤지','기존 조건부 손익','수정 조건부 손익','차이','결과 불명'],qr)
    body+='<p>헤지를 완성할 수 있어도 잘못된 예측의 손실까지 개선하지는 못했습니다. 일부 행에서 명목액 대비 bp가 좋아져도 추가 헤지로 분모가 커진 효과가 있으므로, 위 표는 동일 체결 사례의 현금 손익 차이를 함께 표시합니다. 관측 누락된 헤지 결과는 불명으로 남겼고, 미체결 가정을 사실로 확정하지 않았습니다.</p>'
    body+='<h2>검증과 다음 판단</h2>'
    body+=f'<p>시간봉·5분봉 상태 관리 검증 18개, 호가 처리 검증 14개를 통과했습니다. 기존 두 전략과 즉시 중단 대조군의 체결·계좌 평가액을 앞선 실행과 정확히 대조했습니다. 호가 진단의 추가 시도 없는 37개 경로도 기존 결과를 정확히 재현했습니다. 40개 계좌 재생의 Q·복리수익률·변동성·제한가 예산을 재검사했고, 최대 원장 오차는 {max_error:.3g} USDPR입니다. 운용 소스 {guards["active_protected_sources"]}개 및 예약 의존 {guards["reserved_dependencies"]}개는 변경하지 않았습니다.</p>'
    body+='<p>Q는 현재 공식 식 (max(R,0)+0.005)/(|R|+1.5σ+0.05), σ는 완료 일별 수익률의 표본 표준편차×√n입니다. 심사 점수는 계산하지 않았습니다. 모든 기간과 호가 기록은 재사용 과거 자료이며 독립 미래 검증은 아닙니다. <a href="https://perpetual-alpha26.github.io/leaderboard#practice">공식 평가 설명</a>.</p>'
    body+='<p><strong>결정: 개선된 헤지 절차와 후반 양수 후보를 보존하되 실거래 교체는 보류.</strong> 다음에는 기준시장 신호와 par 체결 종가 신호의 차이, 자산 충돌로 사실상 작동하지 않는 혼합 구조를 구분해 점검합니다. 추가 체결 그 자체가 성과 개선이라는 가정은 하지 않습니다.</p>'
    css='body{font-family:system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;line-height:1.7;color:#24364a}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px;border:1px solid #cbd8e3;text-align:left}th{background:#edf3f8}a{color:#1762a0}'
    (target/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 헤지 완성 비교</title><style>'+css+'</style><body>'+body+'</body></html>')
    with (target/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    save(target/'summary.json',dict(valid_replays=count,tests=32,book_paths=74,episodes=episode_rows,terminal=terminal,
        protected=guards,maximum_accounting_error=max_error,selections={k:v['selection'] for k,v in data.items()},
        historical_reference_later_improved=True,competition_improvement_established=False,
        independent_validation=False,orders_sent=0,live_replacement=False))
    save(HERE/'work_state.json',dict(status='progress',goal_complete=False,
        previous_turn_classification='progress: fixed/adaptive mean and economic-limit comparisons implemented and executed',
        this_turn='Per-episode cash attribution, cost/path decomposition and one-attempt bounded hedge completion implemented. 32 tests, 40 account replays, 74 reused L2 paths. Reference later base/stress PnL and Q above both incumbents; selection positive but below incumbents; par model zero trades and old L2 forecast probes not consistently improved.',
        report=str(target/'report.html'),
        next_action='Preserve completed positive-reference candidate. Diagnose causal source-state mismatch: reference strategy uses reference hourly spread while par adapter uses sparse par trade-close spread. Also inspect portfolio overlay that blocks all pair legs already owned by the incumbent. Declare any source or aggregation correction before new replay; do not invent historical book liquidity, count overlapping probes as Q, relax selection after seeing results, inspect sealed observations or alter live strategies.',
        orders_sent=0,protected_sources_unchanged=True,historical_independence=False,data_wait_required=False))
    print(target/'report.html',flush=True)


if __name__=='__main__':main()
