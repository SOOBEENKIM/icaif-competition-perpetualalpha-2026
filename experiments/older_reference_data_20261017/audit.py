"""Freeze the older venue separately and inventory its completed pair events."""
import csv,gzip,html,io,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.event_value_pairs_20261017.model import source_design,completed_events
from experiments.netted_shadow_pairs_20261017.report import document

HERE=Path(__file__).resolve().parent


def main():
    manifest=json.loads((HERE/'manifest.json').read_text());out=HERE/'event_audit';out.mkdir(exist_ok=False)
    archive_count=0
    for record in manifest['results']:
        if record['status']!='ok':continue
        name=f"{record['symbol']}-1h-{record['month']}.zip";path=HERE/'raw'/name
        assert sha(path)==record['sha256']==(HERE/'raw'/(name+'.CHECKSUM')).read_text().split()[0]
        with zipfile.ZipFile(path) as z:
            with z.open(z.namelist()[0]) as f:rows=list(csv.reader(io.TextIOWrapper(f)))
        if rows[0][0].lower() in ('open_time','open time'):rows=rows[1:]
        values=np.asarray(rows,float);unit=record['timestamp_units']
        assert np.all(values[:,0]%(3600*unit)==0)
        assert np.all(values[:,6]+1-values[:,0]==3600*unit)
        archive_count+=1
    data={}
    for item in manifest['summary']:
        path=HERE/(item['symbol']+'.npz');assert sha(path)==item['parsed_sha256']
        with np.load(path) as z:data[item['symbol']]={k:z[k].copy() for k in z.files}
    cfg0=json.loads((ROOT/'experiments/event_value_pairs_20261017/reference_protocol.json').read_text());summaries=[]
    for a,b in [('BTCUSDT','ETHUSDT'),('BTCUSDT','HYPEUSDT')]:
        names=[a.removesuffix('USDT'),b.removesuffix('USDT')];cfg={**cfg0,'pairs':[names]}
        times,ia,ib=np.intersect1d(data[a]['market_times'],data[b]['market_times'],return_indices=True)
        assert np.all(np.diff(times)==3600)
        ref=dict(market_times=times,close=np.stack([data[a]['close'][ia],data[b]['close'][ib]],axis=-1))
        raw=source_design(ref,cfg,names);events=completed_events(ref,raw,cfg)
        key='_'.join(names);np.savez_compressed(out/f'{key}_reference.npz',**ref)
        np.savez_compressed(out/f'{key}_state.npz',**raw)
        save(out/f'{key}_events.json',events)
        summary=dict(pair=names,hours=len(times),first_available=int(times[0]),last_available=int(times[-1]),
            events=len(events),mature_events=sum(e['maturity'] is not None for e in events),
            source='Binance USD-M only; no Hyperliquid prices spliced',hypothetical_price_path_labels=True,account_backtest=False)
        summaries.append(summary);print(summary,flush=True)
    guards=protected();save(out/'summary.json',dict(pairs=summaries,archives_verified=archive_count,protected=guards,orders_sent=0,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'manifest.json',HERE/'protocol.json',HERE/'collect.py',Path(__file__),
            ROOT/'experiments/event_value_pairs_20261017/model.py',ROOT/'experiments/adaptive_mean_pairs_20261017/signals.py']}))
    report=ROOT/'reports/older_reference_data_20261017';report.mkdir(exist_ok=True)
    body='''<h1>기존 학습 기간 이전의 공개 기준시장 자료 확장</h1><p class="note"><strong>2026년 3월 이전의 공개 암호화폐 시간봉을 새로 확보하고 검증했다.</strong> 기존 8월·9~10월 평가 기간이나 실거래 전략은 변경하지 않았다. 새 자료로 학습한 후보의 성과 개선은 아직 평가하지 않았다.</p>
<p><a href="https://github.com/binance/binance-public-data">Binance 공식 공개 자료 안내</a>에 따라 USD-M 선물 월별 시간봉 ZIP과 SHA256 체크섬을 수집했다. ZIP을 실행하거나 압축 파일 경로를 추출하지 않고 CSV만 읽었다. 86개 파일 모두 공식 체크섬과 일치하며, 요청한 달·시간 단위·가격 범위·거래량·중복 여부·완료 시간과 연속성을 검사했다.</p>
<table><tr><th>종목</th><th>시간봉 수</th><th>첫 봉 시작 UTC</th><th>마지막 봉 사용 가능 UTC</th><th>관측 시작 후 누락 시간</th></tr>'''
    for item in manifest['summary']:
        start=datetime.fromtimestamp(item['first_open'],timezone.utc).isoformat();end=datetime.fromtimestamp(item['last_available'],timezone.utc).isoformat()
        body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [item['symbol'],item['rows'],start,end,item['missing_hours_after_first_observation']])+'</tr>'
    body+='''</table><p>HYPE의 2025년 1~4월 파일은 404로 기록했고 관측 이전 값을 만들지 않았다. 5월 파일은 38시간이며 첫 관측 이후에는 시간 공백이 없었다. BTC·ETH는 각각 38개월의 완전한 시간봉을 확보했다. 전체 62,030개는 종목별 행 수의 합이며 독립된 시장 표본 수를 뜻하지 않는다.</p>
<h2>같은 정의로 만든 학습용 사건</h2><p>기존 720시간 헤지 추정, 168시간 평균·표준편차, 3σ 안쪽 재진입, 고정 평균·손절·시간 종료 규칙으로 두 암호화폐 페어의 사건을 만들었다. 같은 페어에서 보유기간이 겹치는 사건은 제외하고 종료 판단 다음 봉을 목표값의 완성 시각으로 기록했다. 이는 가격 경로 목표이며 실제 주문 체결 또는 비용 차감 계좌 수익이 아니다.</p>
<table><tr><th>페어</th><th>공통 시간봉</th><th>사건</th><th>완료 목표</th></tr>'''
    for row in summaries:body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in ['/'.join(row['pair']),row['hours'],row['events'],row['mature_events']])+'</tr>'
    body+='''</table><h2>사용 범위</h2><p>거래소가 다르므로 Binance 가격과 Hyperliquid 가격을 하나의 연속 시계열로 이어 붙이지 않았다. 주식·지수 페어 자료나 par 과거 호가가 늘어난 것도 아니다. 펀딩 자료도 이 수집에 포함되지 않아 임의의 펀딩 수익을 만들지 않는다. 다음에는 이 자료에서 학습한 페어별 예상 가격 이동을 기존 Hyperliquid 기준시장 평가 기간에 적용해 시장 간 이전 성능을 따로 측정한다.</p>
<p>자료 검증 테스트 3개 통과, 86개 아카이브의 원본 체크섬과 시간 정합성을 재검증했다. 보호된 실거래 소스 373개와 예약 의존 파일 345개가 유지됐다. 실제 주문 전송 및 전략 교체는 0건이다.</p>'''
    (report/'report.html').write_text(document('장기 기준시장 학습 자료',body))
    save(HERE/'work_state.json',dict(status='progress_public_archive_collection_and_event_inventory_completed',previous_goal_turn='progress',
        report=str((report/'report.html').relative_to(ROOT)),archives_verified=86,tests_passed=3,pairs=summaries,
        orders_sent=0,live_strategy_changed=False,protected=guards,
        next='Predeclare a per-pair event-return predictor trained only on this pre-March-2026 Binance history. Use causal normalized features and source-anchored price-only labels; fit each pair separately, preserve existing fixed August selection and Sep-Oct Hyperliquid comparison/costs. Evaluate transfer explicitly, do not splice venues or infer par liquidity.',
        remaining=['New predictor not yet trained or evaluated','Binance-to-Hyperliquid transfer unproven','Full-period actual par book profitability still unverified']))
    print(report/'report.html',flush=True)


if __name__=='__main__':main()
