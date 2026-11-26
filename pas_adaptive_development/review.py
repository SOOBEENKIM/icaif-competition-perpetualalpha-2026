"""Reproducible audit and complete experiment inventory, including failures."""
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from .study import ROOT, save, sha

OUT=ROOT/'reports/incumbent_constraints_review_20261021'
FAMILIES=[
 ('adaptive_development','20261021','예측·동적 규모'),
 ('covariance_development','20261021','공분산 헤지 v1 — 신호 수량 지연 발견'),
 ('covariance_v2','20261021','공분산 헤지 v2 — 수량 지연 수정'),
 ('trend_ensemble','20261021','10종목 추세 앙상블'),
 ('online_allocation','20261021','과거 성과 기반 전문가 배분'),
 ('sleeve_budget','20261021','기본·페어 규모 복원'),
 ('pair_headroom','20261021','페어 진입 한도 여유'),
 ('guarded_allocation','20261021','한도 도달 전 유료 감축·용량 예약'),
 ('relative_trend','20261021','45개 페어 상대 추세'),
 ('relative_ridge','20261021','45개 페어 릿지·기존 예측 앙상블'),
 ('universe_repair','20261021','동일 되돌림 규칙 10종목 확장'),
 ('signed_ema','20261022','기존 숏 EMA 활성화'),
 ('all_market_ema','20261022','동일 EMA 규칙 10종목·양방향 확장'),
]


def stamp(t):
    return datetime.fromtimestamp(t,ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def read(path):return json.loads(Path(path).read_text())
def raw(name):return read(OUT/'raw'/f'{name}.json')
def link(path,label=None):
    return '<a href="'+html.escape(str(path))+'">'+html.escape(label or str(path.relative_to(ROOT)))+'</a>'
def table(head,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    from competition_live.profile import verify_deployment,load_profile
    deployment=verify_deployment();profile=load_profile()
    benchmark=read(ROOT/'reports/pas_covariance_v2_20261021/reference_results.json')
    ref={(r['arm'],r['phase'],r['scenario']):r for r in benchmark if r['arm'] in ['incumbent','size_only']}
    allrows=[];families=[];violations=[];digests={}
    for family,date,title in FAMILIES:
        directory=ROOT/'reports'/f'pas_{family}_{date}'
        protocol=read(directory/'protocol.json');rows=read(directory/'reference_results.json')
        nominee=protocol['nominee'];details=[]
        freeze=directory/'source_freeze.json'
        if freeze.exists():
            for name,digest in read(freeze).items():
                if sha(ROOT/name)!=digest:violations.append(dict(family=family,source=name))
        assert deployment['files']==protocol['deployment']
        for r in rows:
            allrows.append(dict(family=family,**r))
            if r['phase'] in ['continuous','comparison'] and r['arm']==nominee:
                failures=[]
                for control in ['incumbent','size_only']:
                    for metric in ['net_pnl','Q']:
                        if r[metric]<=ref[control,r['phase'],r['scenario']][metric]:
                            failures.append(control+':'+metric)
                if r['risk_breach_bars']:failures.append('hard_risk')
                if not r['timing_pass']:failures.append('timing')
                if r['accounting_error']>=1e-6:failures.append('accounting')
                details.append(dict(phase=r['phase'],scenario=r['scenario'],failures=failures))
        assert len(details)==4
        families.append(dict(family=family,title=title,nominee=nominee,rows=len(rows),
                             promotion_pass=all(not x['failures'] for x in details),checks=details))
        for f in [directory/'protocol.json',directory/'reference_results.json']:
            digests[str(f.relative_to(ROOT))]=sha(f)
    assert not violations,violations
    # The retrospective winner-selection check is separate from each fixed nominee.
    any_pass=[]
    for family,_,_ in FAMILIES:
        rr=[r for r in allrows if r['family']==family]
        for arm in sorted({r['arm'] for r in rr}-{'incumbent','size_only','base_restore'}):
            selected=[r for r in rr if r['arm']==arm and r['phase'] in ['continuous','comparison']]
            passed=len(selected)==4 and all(r['timing_pass'] and r['accounting_error']<1e-6 and r['risk_breach_bars']==0
                and all(r[m]>ref[c,r['phase'],r['scenario']][m] for c in ['incumbent','size_only'] for m in ['net_pnl','Q']) for r in selected)
            if passed:any_pass.append(dict(family=family,arm=arm))
    status=read(ROOT/'.runtime/pas_competition_live/status.json')
    recovery=read(OUT/'operational_deployment_record.json') if (OUT/'operational_deployment_record.json').exists() else None
    producer=read(ROOT/'.runtime/pas_competition_live/producer_status.json')
    signal=read(ROOT/'.runtime/pas_competition_live/signal.json')
    events=[json.loads(s) for s in (ROOT/'.runtime/pas_competition_live/orders.jsonl').read_text().splitlines()]
    own_cloids={e['order']['cloid'] for e in events if e['event']=='submit_intent'}
    filldata=raw('latest_own_fills');fills=filldata['response']
    matched=[f for f in fills if f.get('cloid') in own_cloids]
    unattributed=[f for f in fills if f.get('cloid') not in own_cloids]
    positions=raw('latest_own_positions');peer=raw('latest_peer_positions')
    own_spot=next(b for b in raw('latest_own_spot')['response']['balances'] if b['coin']=='USDP')
    peer_spot=next(b for b in raw('latest_peer_spot')['response']['balances'] if b['coin']=='USDP')
    peer_positions=[x['position'] for x in peer['response']['assetPositions']]
    peer_gross=sum(float(x['positionValue']) for x in peer_positions)
    official=raw('leaderboard')['response'];teams=[t for t in official['teams'] if t['name'] in ['SFINX','Algo Hedge']]
    z=dict(np.load(ROOT/'reports/pas_adaptive_development_20261021/targets.npz'))
    scored=z['times']>=profile['start_unix']
    scope=dict(symbols=profile['trading']['symbols'],history_start=stamp(int(z['times'][0])),history_end=stamp(int(z['times'][-1])),
               positive=(z['incumbent']>1e-12).sum(0).tolist(),negative=(z['incumbent']< -1e-12).sum(0).tolist(),
               scored_positive=(z['incumbent'][scored]>1e-12).sum(0).tolist(),scored_negative=(z['incumbent'][scored]< -1e-12).sum(0).tolist())
    audit=dict(created_kst=stamp(datetime.now(timezone.utc).timestamp()),status=status,producer=producer,
        matched_filled_orders=sorted({f['oid'] for f in matched}),other_program_fills=unattributed,
        other_program_is_user_confirmed_teammate=bool(recovery),operational_recovery=recovery,
        official_snapshot=official['generated_at'],official_teams=teams,signal_source=signal['source_time'],
        base_weights=signal['publication']['data']['frame']['weights'],scope=scope,
        position_snapshot=positions['at'],own_usdp_equity=float(own_spot['total']),peer_address=peer['request']['user'],
        peer_name_mapping_verified=False,peer_usdp_equity=float(peer_spot['total']),peer_gross=peer_gross,
        peer_effective_leverage=peer_gross/float(peer_spot['total']),families=families,comparison_rows=len(allrows),
        rows_include_reused_controls=True,independent_test=False,any_arm_passes_all_criteria=any_pass,
        orders_sent=0,live_files_unchanged=True,source_hash_violations=violations)
    save(OUT/'audit.json',audit);save(OUT/'all_results.json',allrows)
    columns=['family','arm','phase','scenario','net_pnl','return_fraction','volatility','Q','score','terminal_pnl','max_drawdown',
             'mean_notional','fees','execution_cost','funding','base_pnl','pair_pnl','completed_pairs','fills','short_entries',
             'risk_breach_bars','accounting_error','timing_pass','symbols']
    with (OUT/'all_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(allrows)
    blocks=['<h1>기존 제출 전략 점검과 수정 실험</h1>', '<p>'+audit['created_kst']+' 작성. 새 알파 전략으로 교체하지 않았습니다.</p>',
      ('<p class="alert"><b>공유 지갑 오류를 수정해 기존 전략의 실행을 복구했습니다.</b> '
       '10월 22일 한국시간 00:25부터 shared_wallet_repair.runner 실행. 38개 검사와 읽기 전용 실제 계좌 대조를 통과했습니다. '
       '현재 <code>'+html.escape(status['state'])+'</code>, orders_enabled='+str(status.get('orders_enabled'))+'. '
       '팀원 ETH는 별도 원장으로 관리하고, 우리 ETH 주문·취소 및 ETH가 포함된 페어 주문 전체를 차단합니다. '
       '우리 50,000 USDP 예산과 기존 알파는 유지합니다. 보호 범위는 ETH 분리이며 임의의 동일 종목 공동 거래까지 지원한다는 뜻은 아닙니다. '
       +link(OUT/'operational_deployment_record.json','배포·검증 기록')+'</p>') if recovery else '',
      '<p><b>발견 당시 실행기는 중단 상태였습니다.</b> <code>paused_requires_review / orders_enabled=false</code>. '
      '신호 생산기는 작동하지만, 우리 주문 원장에 없는 ETH 숏과 펀딩 수량의 불일치로 주문 실행이 중단되어 있습니다. '
      '신호 모델을 바꾸거나 주문량만 늘려서는 이 중단 원인이 해결되지 않습니다.</p>',
      '<p><b>종목 제한과 롱 편향은 실제 선택된 코드에 있습니다.</b> 10개 심볼을 설정해 둔 것과 10개 모두를 같은 가격 전략으로 판단하는 것은 다릅니다. '
      '이를 수정한 격리 후보를 구현·실행했으나, 전체 기간·후반 기간·불리한 비용 조건을 함께 통과한 교체 후보는 아직 없습니다. '
      '문제 재현과 수정의 정확성, 수익 개선 여부를 구분합니다.</p>',
      '<h2>1. 계좌·실행 상태</h2>',
      '<p>공개 조회 '+stamp(positions['at'])+'. 본대회 총 자본 100,000 USDP 중 본인 실행 예산 50,000, 팀원 예비 자본 50,000입니다. '
      'USDP 총자산 '+f"{float(own_spot['total']):,.5f}"+'이며, PAS marginSummary의 accountValue를 통합 계좌 총자산으로 사용하지 않았습니다.</p>',
      table(['종목','현재 수량','평균 진입가','미실현손익','주문 원장 대조'],[
        [x['position']['coin'],x['position']['szi'],x['position']['entryPx'],x['position']['unrealizedPnl'],
         '우리 제출 주문으로 확인' if x['position']['coin']=='pas:BTC' else '사용자가 팀원 주문으로 확인'] for x in positions['response']['assetPositions']]),
      '<p>우리 프로그램이 확인한 것은 BTC 진입, HYPE 진입, HYPE 청산의 3개 주문입니다. '
      'ETH 숏 0.4064의 주문 '+str(unattributed[0]['oid'])+'는 '+stamp(unattributed[0]['time']/1000)+' 체결됐으나 우리 cloid 목록과 일치하지 않습니다. '
      '처음에는 거래 주체 미확인이었으나 사용자가 팀원 주문이라고 확인했습니다. 기존 알파 설정은 <code>teammate_enabled=false</code>이고, 이번 별도 실행 어댑터가 ETH 외부 원장을 연결합니다. '
      '50:50 예산 약속만으로 하나의 지갑에서 순포지션·펀딩 원장이 분리되지 않습니다. '
      '기존 검사를 해제하는 대신 원장을 분리해 재대조했고, 팀원 포지션을 청산하지 않았습니다.</p>',
      '<p>현재 기본 신호는 BTC '+f"{audit['base_weights'][0]*100:.4f}"+'% of 50,000 ≈ '+f"{audit['base_weights'][0]*50000:,.2f}"+' USDP입니다. '
      '5만을 배정했다는 것이 5만을 매수한다는 뜻은 아닙니다. 현재 작은 규모는 신호의 목표 비중과 여러 규모 보정의 결과이며, '
      '거래소 레버리지 설정 1x 하나로 설명할 수 없습니다.</p>',
      '<h2>2. 코드에서 확인한 제한·문제</h2>']
    findings=[
      ['페어 범위','BTC–ETH와 BTC–HYPE 두 쌍만 예측. 45쌍·10종목 전체 판단이 아님.',ROOT/'competition_live/config.json'],
      ['가격 추세 범위','EMA features·fit·add_shorts 모두 첫 3개 열을 사용. 선택된 macro는 ema_base(롱). ema_signed는 구현되어 있으나 미선택.',ROOT/'replay_lab/ema_rebuild/models.py'],
      ['되돌림 범위','crypto_d7_aligned_add50 선택. 7일 되돌림의 allowed 마스크가 암호화폐 3종목.',ROOT/'replay_lab/reversion_blend/model.py'],
      ['예외','펀딩/carry 경로는 10종목에 열려 있고 과거 BRENT 거래도 있음. 전체 시스템이 7종목 거래를 금지한다는 뜻은 아님.',ROOT/'replay_lab/portfolio_development/candidate_agent.py'],
      ['규모','하방 보정은 동적 예측이 아니라 원래 규모의 0.6938788714배 고정 축소. 이전 단계의 normalize_risk가 위험 필터 감축을 다시 키울 수 있음.',ROOT/'replay_lab/tail_calibration/agent.py'],
      ['확대 시 한도 충돌','확대된 페어·기본 보유가 같은 한도를 소모. 일부 실험에서 시세 상승→한도 위반→전체 감축·재진입 반복.',ROOT/'pas_adaptive_development/engine_diff.patch'],
      ['실거래 정지 → 복구','ETH 펀딩 수량과 우리 원장 불일치 재현. 사용자 확인 후 ETH 전용 외부 원장·주문 차단 어댑터로 복구. 임의의 같은 종목 공동 매매는 여전히 지원하지 않음.',ROOT/'shared_wallet_repair/account.py'],
      ['시장 차이','주요 예측·펀딩 특성은 외부 기준시장. PAS 가격 괴리·실제 펀딩·체결 가능성이 같은 값이라는 보장은 없음.',ROOT/'strategy/agent.py'],
    ]
    blocks.append(table(['항목','확인 내용','코드'],[[a,b,link(c)] for a,b,c in findings]))
    blocks.append('<p>원래 기본 목표의 '+scope['history_start']+'~'+scope['history_end']+' 시간별 양수/음수 개수입니다. 체결 건수와 다릅니다.</p>')
    blocks.append(table(['종목','양수 시간','음수 시간','본대회 시작 이후 양수','본대회 시작 이후 음수'],
      list(zip(scope['symbols'],scope['positive'],scope['negative'],scope['scored_positive'],scope['scored_negative']))))
    blocks.extend(['<h2>3. 다른 팀을 따라 사는 방법</h2>',
      '<p>공개 주문 흐름을 따르는 copy trading / trade-following은 구현 가능한 후보입니다. '
      '실제로 사용한 공개 데이터와 추종 규칙을 설명해야 하며, 사후에 별개의 예측 논리를 꾸며 쓰면 안 됩니다. '
      '공식 HTML·leaderboard.json·chart.json에서 Algo Hedge 이름과 지갑을 연결하는 주소는 찾지 못했습니다.</p>',
      table(['공식 팀','일일 순위','R','기간 σ','Q','0.8Q (J=0)'],[[t['name'],t['rank'],f"{t['return']*100:.6f}%",f"{t['volatility_horizon']*100:.6f}%",f"{t['q']:.8f}",f"{t['score']:.8f}"] for t in teams]),
      '<p>위 표는 '+official['generated_at']+' 공식 파일이며 12:00 UTC 일일 평가와 장중 곡선은 구분합니다. '
      '주소 비교용 '+html.escape(peer['request']['user'])+'를 Algo Hedge라고 부르지 않았습니다.</p>',
      '<p>이 공개 지갑의 '+stamp(peer['at'])+' 포지션은 8종목 모두 롱입니다. 총 명목 '+f'{peer_gross:,.2f}'+' / USDP 자산 '+f"{float(peer_spot['total']):,.2f}"+
      ' = 실제 총노출 '+f"{peer_gross/float(peer_spot['total']):.3f}"+'배입니다. 화면상 설정 레버리지 10x·20x와 다릅니다. '
      '이 사례만으로 남들은 롱·숏 방향을 더 잘 맞힌다고 결론내릴 수 없습니다. 종목·노출·체결 비용·펀딩 기여를 함께 봐야 합니다.</p>',
      table(['종목','방향','수량','명목 USDP','설정 레버리지','미실현손익'],[[x['coin'],'롱' if float(x['szi'])>0 else '숏',x['szi'],x['positionValue'],x['leverage']['value'],x['unrealizedPnl']] for x in peer_positions]),
      '<p>검증할 추종 규칙: 리더를 선택한 시각을 먼저 고정 → 공개 이벤트 수신 시각 이후 판단 → 우리 주문 도착 시점 호가로 체결 → '
      '자체 5만 예산·개별/총 한도에 맞춰 수량 조절 → 리더의 축소·청산까지 추적. '
      '리더의 체결가를 복사하거나 오늘 1등을 골라 어제부터 따랐다고 계산하면 안 됩니다. '
      '현재까지 Algo Hedge 추종의 수익성 검증이나 실제 추종 주문은 수행하지 않았습니다. '
      '공개 가이드의 외부 데이터 허용을 특정 경쟁팀 복제에 대한 명시적 승인으로 해석하지 않았습니다.</p>',
      '<h2>4. 동일 조건에서 실제 실행한 수정 실험</h2>',
      '<p>전체 비교: 2026-08-01 12:00~10-15 12:00 UTC. 후반 비교: 09-01 12:00~10-15 12:00 UTC. '
      '두 구간은 이미 반복 사용한 개발 구간입니다. 미사용 독립 테스트라고 표시하지 않습니다. '
      '외부 기준시장 백테스트이며 PAS 실거래 수익률이나 대회 순위를 뜻하지 않습니다. '
      '운용 50,000+예비 50,000으로 계좌 수익률과 공식 Q 공식을 계산했습니다. '
      '부분 체결·한쪽 체결 후 비용을 지불한 청산·펀딩·잔여 청산 비용을 보존했습니다.</p>',
      '<p>기본: 편도 수수료 9bp+impact 10bp, 거래량 참여율 1%. '
      '불리한 조건: 수수료 12bp+impact 20bp, 참여율 0.25%, 펀딩 수취 50%. '
      '비교 기준은 기존판과 기본 규모만 1.441배 복원한 대조군 모두 대비 전체/후반 PnL·Q 개선, 양 비용 조건 충족, 한도/회계/시점 위반 0입니다.</p>'])
    selected=[('covariance_v2','incumbent','현재 제출판'),('covariance_v2','size_only','규모만 복원한 대조군'),
              ('universe_repair','all_reversion','되돌림만 10종목'),('signed_ema','signed_ema','기존 숏 EMA 활성화'),
              ('all_market_ema','all_signed','EMA 10종목·양방향'),('guarded_allocation','guarded_both','규모 복원+선제 감축')]
    display=[]
    for family,arm,label in selected:
        rr={(r['phase'],r['scenario']):r for r in allrows if r['family']==family and r['arm']==arm}
        full=rr['continuous','base'];late=rr['comparison','base'];stress=rr['comparison','stress']
        display.append([label,f"{full['net_pnl']:,.2f}",f"{full['volatility']*100:.4f}%",f"{full['Q']:.6f}",
                        f"{late['net_pnl']:,.2f} / {late['Q']:.6f}",f"{stress['net_pnl']:,.2f} / {stress['Q']:.6f}"])
    blocks.append(table(['후보','전체 순손익','전체 기간 σ','전체 Q','후반 기본 PnL / Q','후반 불리한 조건 PnL / Q'],display))
    blocks.extend(['<p>EMA 10종목·양방향은 후반 기본 조건에서 +208.13으로 기존 +191.34보다 개선됐지만, '
      '후반 불리한 조건에서는 +129.63으로 기존 +161.56보다 낮았습니다. 전체 기본 순손익도 +733.36 대 +1,296.35입니다. '
      '거래 범위와 숏 기능을 올바르게 넓혔다는 증거는 있지만, 그대로 교체하면 성능이 더 좋다는 증거는 아닙니다.</p>',
      '<p>이 후보의 전체 기본 체결 조각은 127→266, 평균 명목 보유는 3,575.72→8,006.64로 늘었습니다. '
      '기본 전략 기여는 +680.06→+117.29, 페어 기여는 약 +616으로 거의 같았습니다. '
      '범위 확장으로 노출과 비용이 늘고 기본 가격 전략 기여가 줄어든 결과이며, 숏 개수 자체를 늘리는 것이 목표가 될 수 없습니다.</p>',
      '<p>한도 도달 전 감축 구조는 확대된 실행의 한도 위반을 없앴지만, 기본+페어 확대 후보의 후반 불리한 순손익은 +91.52로 낮았습니다. '
      '안전성 수정은 확인했지만 수익 개선까지 확인했다는 표현은 하지 않습니다.</p>',
      '<h2>5. 전체 실험 기록</h2>',
      '<p>'+str(len(FAMILIES))+'개 실험군, '+str(len(allrows))+'개 결과 행입니다. 같은 대조군 재사용과 중복 진단을 포함하므로 '
      '독립 실험 '+str(len(allrows))+'회라는 뜻이 아닙니다. 각 군 후보·규칙은 실행 전에 저장했지만, 같은 역사 자료로 다음 연구를 진행했으므로 '
      '연구 전체의 선택 편향은 사라지지 않습니다.</p>'])
    rows=[]
    for f in families:
        full=next(r for r in allrows if r['family']==f['family'] and r['arm']==f['nominee'] and r['phase']=='continuous' and r['scenario']=='base')
        failed=sum(bool(x['failures']) for x in f['checks'])
        rows.append([f['title'],html.escape(f['nominee']),f"{full['net_pnl']:,.2f}",f"{full['Q']:.6f}",str(failed)+'/4 비교 실패','통과' if f['promotion_pass'] else '미통과'])
    blocks.append(table(['실험군','사전 지정 후보','전체 기본 PnL','Q','비교 조건','승격'],rows))
    blocks.extend(['<p>공분산 v1은 원래 목표 수량 변화 41건을 지연시키는 계약 위반이 발견되어 v2로 수정했습니다. '
      'v1의 더 좋은 수치를 최종 성능으로 채택하지 않았습니다. 상대 릿지 24h 추가 후보는 기존 결과와 같았으며 새 예측이 추가 체결을 만들지 못했습니다. '
      '72h add 표기는 같은 72h 단독 진단과 중복입니다. 모든 미통과 결과를 보존했습니다.</p>',
      '<h2>6. 구현 검증과 PAS 한계</h2>',
      '<p>13개 검사 통과: 미래 가격 변조·자료 절단에 대한 과거 결정 불변, 학습 정답 성숙 및 구간 경계 제외, '
      '종목 순서를 바꿔도 동일한 자산별 EMA 판단, 상승·하락 입력에서 10종목 모두 롱·숏 작동, '
      '공유 한도 예약, 실제 비용을 지불한 감축, 감축 후 기존 손실·손절 기준 보존, '
      '한쪽 감축 미체결을 주입한 실제 리플레이에서 유료 청산, 표시 호가 수량 중복 소비 방지 등입니다. '
      '이는 미래 수익성이나 과적합 부재를 보증하는 검사가 아닙니다.</p>',
      '<p>PAS 분봉 리플레이는 실제 첫 체결보다 약 12.016시간 늦게 체결을 만들었습니다. 거래량은 당시 잔존 호가가 아니므로 '
      '이 도구만으로 실제 체결 개선을 증명할 수 없습니다. 과거 mark/oracle도 완전하지 않아 분봉 종가 평가를 공식 Q로 대체하지 않았습니다. '
      '동기화 수집은 30분 구간 중 PAS 연결 끊김을 기록했고, 재연결 후 5분 수집은 완료했습니다. '
      '5분 호가에 대한 1,000 USDP·120초 실행 시험은 40건 진입, 2건 부분 진입, 미청산 0이었으나 '
      '이는 수량·비용 처리 점검이지 24/72시간 전략 수익성 시험이 아닙니다.</p>',
      '<p>현재 판단: 공유 지갑 거래 귀속을 해결한 실행 구조, 종목별 판단 근거 기록, 한도 도달 전 감축은 개선 대상으로 분리할 가치가 있습니다. '
      '단순 종목 확장·숏 활성화·일괄 규모 확대를 수익 개선판으로 배포할 근거는 이번 결과에서 확보하지 못했습니다. '
      '추종 후보는 이름-지갑 확인과 수신 시각이 남는 데이터부터 필요합니다. '
      '성능 후보로 전략을 교체하지 않고 기존 전략의 운영 연결만 공유 지갑 수정안으로 재개했습니다. '
      '00:07 체결·펀딩·잔고 대조: 우리 +51.236142, 팀원 ETH −21.106502, 합계 +30.129640 USDP. '
      '이는 해당 시각의 비용 차감·미실현 평가 포함 순손익이며 이후 가격에 따라 변합니다.</p>',
      '<h2>원본·재현 자료</h2><ul>'])
    for path in [OUT/'audit.json',OUT/'all_results.csv',OUT/'all_results.json',OUT/'tests.txt',
                 ROOT/'pas_adaptive_development/universe_repair.py',ROOT/'pas_adaptive_development/signed_ema.py',
                 ROOT/'pas_adaptive_development/all_market_ema.py',ROOT/'pas_adaptive_development/engine_diff.patch',
                 ROOT/'reports/pas_execution_probe_20261021/summary.json',ROOT/'reports/team_comparison_20261021/report.html',
                 OUT/'ownership_attribution.json',OUT/'shared_wallet_tests.txt',OUT/'operational_deployment_record.json']:
        blocks.append('<li>'+link(path)+'</li>')
    blocks.extend(['</ul><p>공식 자료: <a href="https://perpetual-alpha26.github.io/leaderboard">리더보드와 계산 기준</a> · '
      '<a href="https://perpetual-alpha26.github.io/docs/">참가자 가이드</a> · <a href="https://perpetual-alpha26.github.io/faq/">FAQ</a>. '
      '이번 최소 수정 후보는 기존 코드의 범위·분기 변경이며 새 논문 재현이라고 주장하지 않습니다.</p>'])
    style='body{font-family:system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;line-height:1.7;color:#18302d}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}th,td{border:1px solid #c7d8d3;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#e9f4ee}h1,h2{line-height:1.3}h2{margin-top:36px}.alert{background:#fff2d4;padding:20px}a{color:#086d6b}code{background:#eaf1ef;padding:2px 5px}'
    (OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>기존 전략 점검 및 수정 실험</title><style>'+style+'</style><body>'+''.join(blocks)+'</body></html>')
    digests[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    save(OUT/'verification.json',dict(created_kst=audit['created_kst'],deployment_hashes=deployment['files'],
        experiment_hashes=digests,live_files_unchanged=True,source_freezes_verified=True,nominees_pass=sum(f['promotion_pass'] for f in families),
        any_candidate_passes=any_pass,comparison_rows=len(allrows),orders_sent=0))
    print(json.dumps(dict(report=str(OUT/'report.html'),rows=len(allrows),families=len(families),passes=any_pass,
                          status=status['state'],matched_orders=len(audit['matched_filled_orders']),unattributed_fills=len(unattributed))))


if __name__=='__main__':main()
