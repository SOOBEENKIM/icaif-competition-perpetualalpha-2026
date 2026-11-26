import csv,gzip,json
import numpy as np
from .research import HERE,ROOT
from experiments.quote_anchored_pairs_20261017.report import table
from experiments.par_direct_forecast_20261017.data import save,sha
from experiments.complementary_alpha_20261017.research import protected


def main():
    folders={'par_tight':'artifacts','par_economic':'economic_par_artifacts',
             'reference_tight':'reference_artifacts_v2','reference_economic':'economic_reference_artifacts'}
    data={};allrows=[];max_error=0.;terminal=[];run_count=0
    for name,folder in folders.items():
        out=HERE/folder;data[name]=json.loads((out/'results.json').read_text())
        for path,digest in json.loads((out/'frozen.json').read_text())['sources'].items():assert sha(ROOT/path)==digest,path
        for path in sorted((out/'runs').glob('*.json.gz')):
            with gzip.open(path,'rt') as f:r=json.load(f)
            m=r['metrics'];returns=np.asarray([x['return_'] for x in r['daily']])
            vol=float(np.std(returns,ddof=1)*np.sqrt(len(returns)))
            q=(max(m['roi'],0)+.005)/(abs(m['roi'])+1.5*vol+.05)
            assert abs(q-m['financial_score'])<1e-12 and abs(vol-m['volatility_horizon'])<1e-12
            assert abs(np.prod(1+returns)-1-m['roi'])<1e-10
            step=3600 if name.startswith('reference') else 300
            assert all(f['time']-f['decision_time']==step for f in r['fills'])
            for e in r['events']:
                if e['event']=='pair_intent':assert e['minimum_expected_cash_at_limits']>=e['buffer_cash']-1e-8
            max_error=max(max_error,m['maximum_accounting_error']);run_count+=1
            last=r['snapshots'][-1]
            terminal.append(dict(experiment=name,phase=r['phase'],arm=r['label'],scenario=r['scenario'],
                cash_net_pnl=last['cash']-1e6,unrealized=last['equity']-last['cash'],**m))
        allrows.extend(dict(experiment=name,phase=r['phase'],arm=r['arm'],scenario=r['scenario'],**r['metrics']) for r in data[name]['rows'])
    assert run_count==112
    equivalence=json.loads((HERE/'reference_artifacts_v2/baseline_equivalence.json').read_text());assert equivalence['passed']
    guards=protected();diag=json.loads((HERE/'artifacts/entry_diagnostics.json').read_text())
    target=ROOT/'reports/adaptive_mean_pairs_20261017';target.mkdir(exist_ok=True)
    body='<h1>페어 선정·보유 관리·주문 제한가 수정 비교</h1>'
    body+='<p><strong>기준시장 후반 비교에서 개선된 조합을 찾았지만, 대회 실거래로 교체할 검증은 통과하지 못했습니다.</strong> 고정 평균·되돌림 확인 진입·예상 수익에 따른 제한가 조합은 9월 1일~10월 15일 기본 비용 비교에서 +1,773.03 USDPR로 기존 2번 +572.88보다 높았습니다. 그러나 앞선 8월 선택 구간에서는 손실이었고, 후반 비용 스트레스도 −59.02였습니다. 실제 par 과거 체결 모형에서는 완료된 페어 거래가 없었습니다.</p>'
    body+='<h2>이번에 구현한 비교</h2><p>진입 시 헤지 비율과 위험 기준을 고정하고, 보유 중 평균을 고정하는 방식과 갱신하는 방식을 비교했습니다. 진입은 괴리가 기준을 넘어가는 시점과 되돌아오기 시작한 시점을 각각 시험했습니다. 부분 체결 감축, 손절 후 재진입 잠금, 시간 경과에 따른 손절 강화, 실제 청산 완료 확인을 포함합니다. 종목별 주문 가격은 제출 전에 정하며 다른 종목의 미래 체결 결과를 보고 소급 취소하지 않습니다.</p>'
    body+='<p>첫 비교에서 10bp의 추가 가격 제한 때문에 한쪽 체결 중단이 많았습니다. 두 번째 비교는 이 추가 제한만 제거하되, 예상 현금 수익에서 진입·청산 비용과 5bp 여유를 남기는 종목별 제한가는 유지했습니다. 미래 평균으로의 복귀는 수익 예측 가정이며 반드시 실현되는 가격이 아닙니다. 논문의 개념을 참고한 원래 구현이며 저자의 매매 코드를 복사하지 않았습니다. <a href="https://arxiv.org/html/2606.04574v1">참고 논문</a>.</p>'
    body+='<p>평균 갱신형이 항상 더 좋은 결과를 내지는 않았습니다. 이번 후반 비교에서는 고정 평균과 되돌림 확인 진입의 조합이 더 나았습니다. 따라서 최신 논문의 구성요소라는 이유만으로 갱신형을 채택하지 않습니다.</p>'
    def metrics_table(name,phase):
        rows=[];rr=data[name]['rows']
        for r in rr:
            if r['phase']!=phase or r['scenario']!='base':continue
            a=r['metrics'];b=next(x['metrics'] for x in rr if x['phase']==phase and x['arm']==r['arm'] and x['scenario']=='stress')
            rows.append([r['arm'],f"{a['net_pnl']:+.2f}",f"{b['net_pnl']:+.2f}",f"{100*a['roi']:+.6f}%",
                f"{100*a['volatility_horizon']:.6f}%",f"{a['financial_score']:.6f}" if a['eligible'] else '— (미충족)',
                f"{a['completed_pairs']} / {b['completed_pairs']}",f"{a['mean_gross']*100:.4f}%",f"{a['residual_gross']:.2f}"])
        return table(['전략','기본 순손익','스트레스 순손익','기본 R','기본 기간 σ','기본 Q','페어 완료 기본/스트레스','평균 투자 비중','잔여 명목액'],rows)
    body+='<h2>실제 par 과거 자료: 10월 비교</h2><p>선택 구간 10월 1일 21시~8일 21시(KST), 이후 비교 8일 21시~16일 21시입니다. 자본 1,000,000 USDPR. par 체결 봉·펀딩은 실제 기록이지만, 전 기간 호가가 없어 다음 5분봉 종가와 거래량으로 주문을 근사합니다. 무거래 봉은 실제 호가 수량 0이라는 증거가 아닙니다. 기본 수수료/불리한 가격은 편도 9/10bp, 스트레스 12/20bp이며 스트레스 거래량 한도와 양수 펀딩 수취액은 절반입니다.</p>'
    body+=metrics_table('par_economic','comparison')
    body+='<p>선택 구간에는 모든 후보의 거래가 없었습니다. 후반 표준 진입형은 한쪽 체결 후 유료 중단만 발생했고, 되돌림 확인형은 거래가 없었습니다. 0 손익과 이론상 Q=0.1을 개선이나 순위 획득으로 처리하지 않았습니다. 기존 2번에 더한 후보도 추가 거래가 없어 기존과 같았습니다.</p>'
    body+=table(['방식','기간','시간 신호','판단 시 유효','양쪽 과거 거래량 존재','주문 조건 통과'],[
        [x['arm'],x['phase'],x['source_pair_signals'],x['current_spread_valid'],x['both_legs_have_past_volume'],x['nonempty_economic_proposals']]
        for x in diag['rows'] if x['scenario']=='base' and x['arm'].startswith('adaptive_')])
    body+='<p>위 진입 진단은 추가 10bp 제한을 둔 최초 비교에서, 보유 상태를 제외하고 원인을 세어본 것입니다. 별도로 기존 다섯 호가 기록에서 각 방식당 416개의 주문·페어 조합을 확인했지만 진입 신호와 방향이 일치한 사례는 0개였습니다. 이 기록만으로 새 진입·청산 방식의 호가 체결 성능을 주장할 수 없습니다.</p>'
    body+='<h2>더 긴 기준시장 자료: 신호와 보유 규칙 진단</h2><p><strong>이하 표는 par 계좌 성과가 아닙니다.</strong> 실제 기준시장 시간봉·거래량·펀딩과 보관된 기존 두 전략의 신호를 같은 조건에 넣었습니다. 선택은 8월 1일 21시~9월 1일 21시, 후반은 9월 1일 21시~10월 15일 21시(KST)입니다. 매월 이전 자료로 선정한 페어만 새로 진입하며 이미 보유한 수량은 유지합니다. 다음 시간봉 종가의 체결 가정과 제한된 거래량을 사용하므로 장중 체결을 복원한 결과는 아닙니다.</p>'
    body+='<h3>선택 구간: 수익 기반 제한가</h3>'+metrics_table('reference_economic','validation')
    body+='<h3>이후 비교: 수익 기반 제한가</h3>'+metrics_table('reference_economic','comparison')
    body+='<h3>이후 비교: 추가 10bp 제한가 대조군</h3>'+metrics_table('reference_tight','comparison')
    body+='<p>기준시장 기본/스트레스 거래량 한도는 관측량의 1%/0.25%입니다. 수수료·불리한 가격은 각각 9/10bp와 12/20bp이며 스트레스에서는 양수 펀딩 수취액을 절반으로 처리했습니다. 모든 후보와 기존판의 자본·전역 위험 한도는 같지만 실제 보유 규모는 다릅니다. 따라서 평균 투자 비중을 함께 기입했습니다.</p>'
    selected=[x for x in terminal if x['experiment']=='reference_economic' and x['phase']=='comparison' and x['arm']=='fixed_delayed_168']
    body+='<h3>양수 후보의 손익 구성과 미완료 보유</h3>'
    body+=table(['비용 조건','현금 순손익','미실현손익','총 순손익','남은 포지션 명목액','예상 청산비 차감 후'],[
        [x['scenario'],f"{x['cash_net_pnl']:+.2f}",f"{x['unrealized']:+.2f}",f"{x['net_pnl']:+.2f}",f"{x['residual_gross']:.2f}",f"{x['pnl_after_liquidation_reserve']:+.2f}"] for x in selected])
    body+='<p>기본 총수익 +0.177303%에는 현금 순손익 +1,011.18과 미실현손익 +761.85가 포함됩니다. 스트레스는 비용 차감 후 음수이고 페어 완료가 두 건뿐입니다. 남은 포지션의 청산비 차감값도 추정치이며 실제 청산 완료 금액이 아닙니다. 과거 선택 구간 실패와 이 제한을 함께 고려해 교체 후보로 승인하지 않았습니다.</p>'
    body+='<h2>검증과 보존한 오류</h2>'
    body+=f'<p>전략·보유 관리 검증 12개와 수익 기반 제한가 검증 2개를 통과했습니다. 유효 비교 112회의 회계·복리수익률·기간 변동성·Q·체결 시점·최악 제한가의 예상 수익 여유를 재검사했고, 원장 최대 오차는 {max_error:.3g} USDPR입니다. 기존 1·2번의 시간봉 선택 구간은 동일 주문 제한가를 적용한 별도 기존 실행 루프와 체결 및 계좌 평가액이 정확히 일치했습니다.</p>'
    body+='<p>시간봉 변환 첫 실행에서 기준 전략의 거래량 조회 길이가 12시간으로 남아 있던 문제를 발견해 1시간으로 수정했습니다. 첫 결과와 무효 사유를 보존하고 별도 v2 폴더에서 다시 실행했습니다. 초기 독립 비교에서 제한가 유무가 달랐던 것도 같은 체결 조건을 적용해 다시 확인했습니다. 이 결과를 과거 원래 브로커와 완전히 같은 결과라고 표현하지 않습니다.</p>'
    body+=f'<p>운용 소스 {guards["active_protected_sources"]}개와 예약 의존 파일 {guards["reserved_dependencies"]}개는 변경되지 않았습니다. 연구 주문 0건. 모든 평가 기간은 재사용 자료이며 독립 검증이나 유의성 입증이 아닙니다. 후보 선택 파일은 각 후반 비교 전에 고정했으며, 후반의 좋은 조합을 선택 구간 통과로 바꾸지 않았습니다.</p>'
    body+='<p><strong>현재 결정: 실거래 교체 보류.</strong> 후반의 개선 조합은 후속 검증 대상으로 보존합니다. 다음 점검은 남은 포지션의 청산과 비용 스트레스 손실을 거래별로 분해하고, 수익이 일부 페어·날짜에 집중되는지 확인하는 것입니다. 별도 보유 목표 변경은 그 원인이 확인된 뒤 고정된 비교로 시험합니다.</p>'
    body+='<p><a href="../reference_anchor_pairs_20261017/report.html">직전 가격 예측 및 페어 선정 추정법 점검</a> · <a href="https://perpetual-alpha26.github.io/leaderboard#practice">공식 평가식</a></p>'
    css='body{font-family:system-ui,sans-serif;max-width:1360px;margin:32px auto;padding:0 24px;line-height:1.7;color:#24364a}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px;border:1px solid #ccd9e4;text-align:left}th{background:#edf3f8}a{color:#1762a0}'
    (target/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 전략 수정 비교</title><style>'+css+'</style><body>'+body+'</body></html>')
    with (target/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    save(target/'summary.json',dict(selections={k:v['selection'] for k,v in data.items()},valid_replays=run_count,
        tests=14,maximum_accounting_error=max_error,protected=guards,selected_terminal=selected,
        performance_improvement_established=False,partial_reference_improvement=True,independent_validation=False,
        par_completed_pairs=0,orders_sent=0,live_replacement=False))
    save(HERE/'work_state.json',dict(status='progress',goal_complete=False,performance_improvement_established=False,
        this_turn='Implemented and tested formation-estimator correction, fixed-entry hedge/risk with fixed/adaptive mean, standard/delayed entry, stop lock, paid lifecycle, and economic reservation limits. 112 valid comparative runs across par and longer reference histories. Reference later fixed/delayed/economic candidate +1773.03 versus second +572.88; stress -59.02 and selection losses, so rejected for replacement.',
        report=str(target/'report.html'),
        next_action='Audit per-trade contribution and stress degradation for the fixed-mean delayed economic-limit candidate, including its still-open positions; reference cash PnL is +1011.18 and unrealized +761.85, stress cash -530.62. Determine if edge is concentrated, whether entry/exit delay or funding causes stress failure, and only then predeclare a bounded correction. Do not equate reference results with par, lower costs post hoc, inspect sealed observations or modify live/reserved strategies.',
        historical_independence=False,orders_sent=0,protected_sources_unchanged=True,data_wait_required=False))
    print(target/'report.html',flush=True)


if __name__=='__main__':main()
