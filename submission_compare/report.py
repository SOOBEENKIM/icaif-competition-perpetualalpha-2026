import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import time
import numpy as np
from competition_live.profile import ROOT,BUNDLE,load_profile,verify_deployment,verify_bundle
from practice_live.infer import protected_hashes
from .run import OUT,OLD,AGREE,save
from .attribution import attribute


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    profile=load_profile()
    new=json.loads((OUT/'results.json').read_text())
    old=json.loads((OLD/'reference_results.json').read_text())
    agree=json.loads((AGREE/'reference_results.json').read_text())
    recent=json.loads((OUT/'recent_results.json').read_text())
    selected={}
    for name,source,rows in [
        ('incumbent',OUT,new),('short_only',OLD,old),
        ('confirmed_short',OUT,new),('forecast_filter',OUT,new),
        ('agreement_confirmed_short',OUT,new),('agreement',AGREE,agree),('size_only',OLD,old)]:
        selected[name]=dict(source=source,rows=[r for r in rows if r['variant']==name])
    names=dict(incumbent='현행',short_only='대칭 숏 추가',confirmed_short='Chronos·JM 확인 숏',
        forecast_filter='Chronos 반대 노출 제거',agreement_confirmed_short='방향 일치 배분 + 확인 숏',
        agreement='방향 일치 비중 조정',size_only='원래 규모 복원 (약 1.44배)')
    index={(r['variant'],r['phase'],r['scenario']):r for item in selected.values() for r in item['rows']}
    data=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
    attribution={}
    for name,item in selected.items():
        attribution[name]={}
        for phase in ['validation','comparison','continuous']:
            for scenario in ['base','stress']:
                path=item['source']/'reference'/f'{name}_{phase}_{scenario}.json.gz'
                with gzip.open(path,'rt') as f:result=json.load(f)
                attribution[name][phase+'_'+scenario]=attribute(result,data,
                    profile['configs']['reference']['scenarios'][scenario])
    save(OUT/'directional_attribution.json',attribution)
    appendix=[]
    for dirname in ['all_market_alpha_20261020','all_market_alpha_wave2_20261020',
                    'all_market_alpha_wave3_20261020','all_market_alpha_wave4_20261020',
                    'all_market_alpha_wave5_20261020','validation_rebuild_20261021',
                    'isolated_repairs_20261020']:
        path=ROOT/'reports'/dirname/'reference_results.json'
        for row in json.loads(path.read_text()):appendix.append(dict(study=dirname,**row))
    appendix.extend(dict(study='submission_compare_20261021',**row) for row in new)
    fields=['study','variant','phase','scenario','net_pnl','Q','max_drawdown','fees',
            'execution_cost','funding','base_pnl','pair_pnl','fills','risk_breach_bars']
    with (OUT/'all_historical_comparisons.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(appendix)
    counter=1/json.loads((ROOT/'strategy/profiles.json').read_text())['default']['relative_to_original']
    decision=dict(at=time.time(),recommendation='nominate_original_size_by_aggregate_backtest_pnl_and_Q',
        basis='For the stated historical-performance comparison, original size has the highest continuous '
            'base/stress PnL and Q. It changes exposure, not forecasting skill. It is not uniformly superior: '
            'later stress results are weaker and drawdown is larger. Current is the lower-exposure alternative. '
            'New directional filters fail the long-window comparison.',
        best_continuous_historical_pnl='size_only',size_multiplier=counter,
        best_continuous_non_size_candidate='agreement',
        best_recent_base_only_pnl='forecast_filter',
        new_full_replays=len(new),recent_base_only_replays=len(recent),
        later_window_independent=False,future_validation_required_for_this_decision=False,
        live_changes=False,orders_sent=0,research_service_stopped=True,
        candidate_tradeoff='No tested candidate dominates current across aggregate, later-period, and recent directional results.')
    save(OUT/'decision.json',decision)
    live=json.loads((ROOT/'.runtime/pas_competition_live/status.json').read_text())
    assert live['state']=='running' and not live['fatal']
    assert time.time()-live['at']<90
    completed=json.loads((OUT/'completed.json').read_text())
    sizing=json.loads((OUT/'sizing_submission_candidate.json').read_text())
    assert len(sizing['reproduced_windows'])==6 and sizing['reference_target_max_error']<1e-12
    assert hashlib.sha256((ROOT/sizing['adapter']).read_bytes()).hexdigest()==sizing['adapter_sha256']
    verification=dict(at=time.time(),historical=completed,attribution_accounts_checked=42*3,
        sizing_reproduced_windows=6,total_this_turn_replays=len(new)+len(recent)+6,
        maximum_attribution_error=max(abs(scope['reconciliation_error'])
          for variants in attribution.values() for run in variants.values() for scope in run.values()),
        protected_baseline_files=len(protected_hashes()),bundle_hash=verify_bundle(),
        deployment_files=len(verify_deployment()['files']),status=live,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (ROOT/'submission_compare').glob('*.py')})
    save(OUT/'verification.json',verification)
    def row(name,phase,sc='base'):return index[name,phase,sc]
    fullrows=[]
    for name in names:
        a=row(name,'continuous');b=row(name,'continuous','stress');c=row(name,'comparison')
        fullrows.append((names[name],f"{a['net_pnl']:,.2f}",f"{b['net_pnl']:,.2f}",
            f"{a['Q']:.5f}",f"{a['max_drawdown']*100:.3f}%",f"{c['net_pnl']:,.2f}",
            f"{row(name,'comparison','stress')['net_pnl']:,.2f}"))
    rec={(r['variant'],r['scenario']):r for r in recent}
    recrows=[(names[n],f"{rec[n,'base']['net_pnl']:+.2f}",f"{rec[n,'stress']['net_pnl']:+.2f}",
        f"{rec[n,'base']['max_drawdown']*100:.3f}%",rec[n,'base']['fills']) for n in names]
    side_rows=[]
    for name in names:
        a=attribution[name]['continuous_base']['base']['sides']
        b=attribution[name]['continuous_base']['pair']['sides']
        side_rows.append((names[name],f"{a['long']['net_pnl']:+.2f}",f"{a['short']['net_pnl']:+.2f}",
            f"{sum(r['net_pnl'] for r in b.values()):+.2f}",f"{row(name,'continuous')['net_pnl']:+.2f}"))
    body=f"""
<h1>지금 제출할 전략의 백테스트 비교</h1>
<p class="lead"><strong>전체 백테스트 PnL·Q 기준 제출 후보 1순위는 기존 전략의 원래 규모 복원입니다.</strong>
이는 노출을 늘리는 변경이며, 방향 예측이 좋아졌다는 뜻은 아닙니다.
이번에 추가한 방향 필터 3개는 장기 비교에서 원안을 이기지 못했습니다.
최고 누적 성과 후보의 최근 성과·위험·규모 차이까지 함께 비교했습니다.
미래 관측을 기다려야 이 판단을 내릴 수 있다는 조건은 두지 않았습니다.</p>
<p>요청을 잘못 해석해 시작했던 72시간 미래 평가는 사용자 지시에 따라 중단했습니다.
이번 작업은 이미 확보한 자료의 유한한 백테스트이며, 운용 전략·자금·주문은 변경하지 않았습니다.</p>
<h2>1. 같은 계좌와 비용으로 비교한 결과</h2>
<p>운용 50,000 USDP + 예비자금 50,000 USDP. 수익은 비용 차감 USDP, Q·최대낙폭은 전체 100,000 기준입니다.
현행의 6개 기간·시나리오 결과를 다시 실행해 기존 PnL·Q를 재현했습니다. 신규 후보 3개를 포함해 24회 재생했습니다.</p>
{table(['전략','8/1~10/15 순손익','악화 비용 순손익','전체 Q','전체 최대낙폭','9/1~10/15 순손익','해당 기간 악화 비용'],fullrows)}
<p>기준: 체결당 수수료 9bp·가격 충격 10bp·거래량 참여 한도 1%.
악화: 수수료 12bp·충격 20bp·참여 한도 0.25%·수취 펀딩 50%.
기존 신호 시점·체결 지연·20% 거래 생략 기준·자산별 위험 한도·페어 정책을 동일하게 적용했습니다.
원래 규모 복원은 약 {counter:.3f}배 목표 비중으로 <strong>규모 변경 대조군</strong>입니다.</p>
<p>8월~10월 전체 누적 손익만 고르면 <strong>규모 복원 1,601.27 USDP</strong>가 가장 높습니다.
그러나 방향을 더 잘 맞힌 결과와 구분해야 합니다. 규모 후보는 9월 이후 악화 비용에서 현행보다 낮고,
최근 구간의 최대낙폭도 커집니다. 방향 일치 비중 조정은 전체 순손익·Q가 개선되지만,
9월 이후와 가장 최근 구간에서 모두 현행보다 약했습니다.</p>
<h2>2. 지금 시장에 가까운 10/15~10/20도 별도 비교</h2>
<p><strong>이 표는 기본 방향 전략만</strong> 같은 시작 시각의 무포지션 계좌에서 비교했습니다.
페어를 제외했으므로 위 전체 전략 손익과 더하거나 같은 값으로 비교하면 안 됩니다.
종료는 10월 20일 12:00 UTC까지 이미 관측된 참조 시장 자료입니다.</p>
{table(['전략','기본 순손익','악화 비용 순손익','최대낙폭','체결 수'],recrows)}
<p>이 최근 5일만 보면 방향 필터와 숏 후보가 더 좋습니다. 사용자의 지적대로
<strong>당시 방향을 더 잘 잡으면 비용을 넘는 손익 개선이 가능합니다.</strong>
다만 5일 승자를 전체 역사에서도 우월한 모델로 제시하지 않았습니다.
같은 필터가 긴 비교 구간에서 얼마나 손익을 깎는지도 첫 표에 그대로 반영했습니다.</p>
<h2>3. 롱·숏이 실제로 얼마를 벌고 잃었나</h2>
<p>8/1~10/15 기준입니다. 보유 수량의 가격 손익, 방향별 진입·청산 비용, 펀딩을 분해했습니다.
부호가 바뀌는 체결은 기존 방향 청산과 새 방향 진입으로 나눴습니다.
기본 전략과 페어 원장을 포함한 126개 계정 분해가 기존 원장과 오차 10⁻⁶ USDP 미만으로 일치합니다.</p>
{table(['전략','기본 롱 순손익','기본 숏 순손익','페어 순손익','총순손익'],side_rows)}
<p>페어의 숏 다리는 상대가격 헤지 역할을 하므로 개별 다리의 수익만으로 없앨지 결정하면 안 됩니다.
또한 단순 적중률 대신 수량·손익 크기·보유 기간·거래 비용을 포함한 순손익으로 비교했습니다.</p>
<h2>4. 이번 판단과 남는 한계</h2>
<ul>
<li><strong>규모 복원:</strong> 전체 역사 기준·악화 PnL·Q 최고이므로 이번 기준의 제출 후보 1순위. 더 큰 노출·낙폭을 수반하며 방향 정확도 개선으로 부르지 않습니다. 실전 적용 전 검토용 어댑터를 준비했습니다.</li>
<li><strong>현행:</strong> 후반 악화 비용 성과와 낮은 노출·낙폭을 우선할 때의 선택. 현재 운용은 그대로 유지 중입니다.</li>
<li><strong>방향 일치 비중 조정:</strong> 규모만 올리지 않은 후보 중 전체 역사 개선. 후반·최근 부진 때문에 현재 교체 추천에서 제외했습니다.</li>
<li><strong>새 확인 숏·예측 필터:</strong> 최근에는 개선됐지만 긴 구간 손실이 큼. 이번 비교로 즉시 교체하지 않습니다.</li>
</ul>
<p>이번 수치는 mainnet 참조 가격에 대회 비용·실행 제약을 적용한 조건부 백테스트입니다.
PAS의 과거 호가 전체를 복원한 실제 체결 기록은 아닙니다.
여러 번 본 역사에서 후보를 비교했으므로 독립 검증 성과라고 주장하지 않습니다.
그 한계를 명시한 상태에서 지금의 제출 추천을 내렸으며, 새 미래 평가를 시작하지 않았습니다.</p>
<p>새 후보의 구체적 변경: ① 기존 대칭 숏이 음수이고 Chronos 예측도 음수이며 JM 상승 확률이 0.4 미만인 경우에만 숏,
② Chronos와 반대인 기존 BTC/ETH/HYPE 노출을 제거,
③ 기존 방향 일치 비중 조정에 ①을 결합.
최적 숫자가 나올 때까지 임계값을 재탐색하지 않았고, 세 후보의 실패 결과도 모두 보존했습니다.</p>
<h2>재현 자료</h2>
<p><a href="{OUT/'results.json'}">이번 전체 재생 24건</a> ·
<a href="{OUT/'recent_results.json'}">최근 방향 전략 재생 14건</a> ·
<a href="{OUT/'directional_attribution.json'}">롱·숏·원장 분해</a> ·
<a href="{OUT/'all_historical_comparisons.csv'}">이전 후보를 포함한 전체 비교 CSV</a> ·
<a href="{OUT/'decision.json'}">현재 제출 판단</a> ·
<a href="{OUT/'sizing_submission_candidate.json'}">규모 복원 후보 정의·재현 결과</a> ·
<a href="{ROOT/'submission_compare/model.py'}">새 후보 코드</a> ·
<a href="{OUT/'verification.json'}">재현·배포 불변 검증</a></p>
"""
    css="""body{font-family:system-ui,sans-serif;line-height:1.8;background:#f6f8fc;color:#203148;margin:0}
    main{max-width:1150px;margin:auto;padding:35px 25px 70px}h1{font-size:32px}h2{margin-top:35px;font-size:23px}
    .lead{font-size:18px}table{border-collapse:collapse;background:white;width:100%;font-size:13px}
    th,td{padding:10px 12px;border-bottom:1px solid #dde5ee;text-align:left}th{background:#e7edf6}
    .scroll{overflow:auto}a{color:#245bb1}li{margin:10px 0}"""
    (OUT/'report.html').write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>제출 전략 백테스트 비교</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
    # Correct the prior report's now-obsolete "ongoing" headline without deleting evidence.
    old_report=ROOT/'reports/pas_venue_research_20261021/report.html'
    text=old_report.read_text()
    banner='<p class="note" id="cancelled"><strong>상태 변경: 사용자의 요청에 따라 72시간 미래 평가는 중단했습니다.</strong> 아래는 중단 전 조사 기록입니다. 현재 판단은 <a href="'+str(OUT/'report.html')+'">기존 자료 백테스트 비교</a>를 참조하세요.</p>'
    if 'id="cancelled"' not in text:old_report.write_text(text.replace('<main>','<main>'+banner,1))
    save(OUT/'evidence_manifest.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.glob('*') if p.is_file() and p.name!='evidence_manifest.json'})
    print(json.dumps(dict(report=str(OUT/'report.html'),decision=decision,
        max_attribution_error=verification['maximum_attribution_error']),ensure_ascii=False))


if __name__=='__main__':main()
