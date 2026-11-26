"""Record the failed execution-only route without manufacturing account ROI."""
import gzip
import html
import json
import time
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from experiments.cost_horizon_20261016.research import data
from experiments.pairs_research_20261017.research import save,sha
from experiments.complementary_alpha_20261017.research import protected
from .probe import HERE,ROOT


def main():
    out=ROOT/'reports/pairs_maker_20261017';out.mkdir(exist_ok=True)
    quotes=json.loads((HERE/'artifacts/results.json').read_text())
    pairs=json.loads((HERE/'paired_artifacts/results.json').read_text())
    side=json.loads((HERE/'trade_side_verification.json').read_text())
    _,_,_,market=data()
    rows=[]
    for label,path in [('OU 60일','pairs_research_20261017/artifacts/runs/comparison-ou_1440-base.json.gz'),
                       ('트리 72시간','pairs_forecast_20261017/artifacts/runs/comparison-tree_72-base.json.gz'),
                       ('트리 72시간·보유기간 일치','pairs_horizon_20261017/artifacts/runs/comparison-tree_72-base.json.gz')]:
        with gzip.open(ROOT/'experiments'/path,'rt') as f:r=json.load(f)
        impact=sum(f['signed_qty']*(f['price']-market.close[market.exact(f['time']),f['asset']]) for f in r['fills'])
        turnover=sum(abs(f['signed_qty'])*market.close[market.exact(f['time']),f['asset']] for f in r['fills'])
        net=r['metrics']['net_pnl'];free=net+r['metrics']['fees']+impact
        rows.append(dict(name=label,net_pnl=net,fixed_fill_zero_fee_zero_impact=free,turnover=turnover,
            required_extra_capture_bps_at_3bp_fee=(2757.8245187480934-free)/turnover*10000+3))
    save(out/'cost_ceiling.json',dict(rows=rows,baseline_second_net_pnl=2757.8245187480934,
        interpretation='Fixed-fill accounting sensitivity only, not a zero-cost rerun or a general upper bound on another strategy.'))
    def table(headers,rs):
        return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rs)+'</table>'
    qrows=[]
    for r in quotes['summaries']:
        if r['flow_filtered']:continue
        qrows.append(['표시 대기량 전부' if r['queue_model'].startswith('all_') else '앞 대기량 0 가정',r['ttl_seconds'],r['quotes'],r['quotes_with_fill'],r['statuses'].get('filled',0),f"{r['filled_notional']:,.2f}"])
    prows=[]
    for r in pairs['summaries']:
        if not r['queue_model'].startswith('all_'):continue
        prows.append([r['ttl_seconds'],'적용' if r['flow_filtered'] else '없음',r['horizon'],r['closed'],r['observations']-r['closed'],f"{r['weighted_net_bps']:.2f}" if r['weighted_net_bps'] is not None else '미측정'])
    ntrades=sum(sum(c['within_capture_trade_counts'].values()) for c in quotes['coverage'])
    kst=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 지정가·주문 흐름 검증</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#183042}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}td,th{{padding:9px;border:1px solid #ccd6df;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#ecf2f6}}.verdict{{padding:18px;background:#fff3d5;border-left:4px solid #b77800}}a{{color:#12649a}}</style>
<h1>페어 지정가·주문 흐름 검증</h1><p>{kst}. 기존 실거래·예약 실험 변경 없음. 연구 주문 0건.</p>
<div class="verdict"><strong>판단: 지정가 비용 절감만으로 기존 두 전략을 넘었다는 근거 없음.</strong> 실제 공개 체결 방향을 검증하고, 대기 물량·주문/취소 지연·헤지·양쪽 청산 비용을 반영했다. 베이지안 주문 흐름 필터를 추가했으나 평가된 체결 진단은 평균적으로 비용을 넘지 못했다.</div>
<h2>고정 체결에서 비용만 제거해도 부족</h2><p>9/1 21:00~10/13 21:00 KST의 기존 기준시장 백테스트. 두 번째 전략의 실제 가정 비용 차감 순손익은 +2,757.82다. 아래는 페어의 체결 시각·수량을 고정한 회계 민감도이며 새로운 체결 전략의 성과 상한을 뜻하지 않는다.</p>
{table(['후보','기존 순손익','수수료·슬리피지 0 가정','기존 둘째를 넘기 위해 필요한 추가 체결가격 이익 bp/거래대금(수수료 3bp 포함)'],[[r['name'],f"{r['net_pnl']:,.2f}",f"{r['fixed_fill_zero_fee_zero_impact']:,.2f}",f"{r['required_extra_capture_bps_at_3bp_fee']:.2f}"] for r in rows])}
<h2>실제 par 체결로 방향 해석 확인</h2><p>BTC·ETH·HYPE에서 매수·매도 6개 공개 체결을 각각 양쪽 지갑의 동일 시각·tid 체결과 대조했다. 6개 모두 공개 trades의 side가 crossed=true인 테이커의 side와 일치했다. 초기 구독에 포함되는 과거 체결과 재접속 중복 체결은 신규 지정가 체결로 계산하지 않았다.</p>
<h2>지정가 대기와 체결</h2><p>이미 사용했던 약 4.48시간의 공개 호가·체결 기록에서 관측 기간 안의 고유 체결 {ntrades}개를 확인했다. 5분마다 자산별 양쪽에 500 USDPR 상당의 독립적인 주문을 가정하고 30/120초를 기다렸다. 체결은 주문 도착 뒤 실제 반대 방향 체결량이 관측된 경우에만 계산했다. 앞 대기량 취소를 유리하게 추정하지 않았으며, 취소 요청 뒤 1초 동안의 체결도 반영했다.</p>
{table(['대기순위 가정','대기 초','주문 진단 수','일부라도 체결','전량 체결','체결 명목금액'],qrows)}
<p>두 대기순위 가정은 L3 자료가 없는 상황의 민감도다. 실제 대기순위나 체결을 입증한 것은 아니다. 시점·방향·대기시간·가정별 진단은 서로 겹치므로 표의 금액을 한 계좌의 거래량이나 수익으로 합산하지 않는다. 관측 종료·연결 끊김은 별도로 남겼다.</p>
<h2>반대편 자산 헤지와 청산까지 붙인 결과</h2><p>체결된 지정가 조각에 당시까지 계산한 168시간 베타로 반대편 수량을 정하고, 1초 뒤 받은 호가에서 헤지를 시도했다. 진입 수수료는 메이커 3bp·헤지 테이커 9bp, 양쪽 청산은 각 9bp다. 시간별 실제 펀딩도 반영했다. 최소 주문 금액 미달, 부족한 호가 수량, 누락된 청산·펀딩은 수익 0이 아니라 미완료로 표시했다.</p>
{table(['지정가 대기 초','주문 흐름 필터','보유 초','관측된 양쪽 청산','미완료/자료 부족','관측 완료분 명목가중 순 bp'],prows)}
<p><strong>계좌 수익률·변동성·Q는 산출하지 않았다.</strong> 독립 체결 진단끼리 노출과 호가 수량이 겹치고, 미완료 거래의 손익도 남아 있기 때문이다. 완료된 것만 골라 계좌 수익률을 만들 수 없다. 필터를 넣은 일부 조건의 손실 감소도 수익성 확인이나 전략 교체 근거가 아니다.</p>
<h2>차용한 모델과 검증 범위</h2><p><a href="https://arxiv.org/html/2609.11614v1">Moret·Lillo, 2026</a>의 베타–베르누이 변화점 필터를 구현했다. 주문 흐름 방향과 지속 기간을 과거 공개 체결만으로 갱신한다. 공개 자료에서 부모 주문을 완전히 식별할 수 없어 같은 코인·시각·트랜잭션·방향의 체결은 하나로 묶었다. 원 논문의 DQN 전체를 재현한 것은 아니다. 그 논문의 주요 결과는 모의 호가장에서 얻었다.</p>
<p><a href="https://arxiv.org/html/2610.16524v1">2026년 Hawkes-Structured Network 연구</a>도 검토했으나, 현재의 짧고 성긴 체결 자료로 주문 도착 과정을 학습해 실거래 수익성을 주장할 근거가 부족하다. 먼저 실제 관측으로 체결 가정과 가격 변화의 관계를 확인했다.</p>
<p>대기량 차감·중복 제거·공격 방향·도착 시각·취소 지연·온라인 필터·수수료·숏 손익·헤지 최소금액을 검증한 테스트 9개 통과. 자료는 재사용한 개발 구간이다.</p>
<h2>남은 검증과 일정</h2><p>72시간 오라클/펀딩 페어의 별도 원시 자료 수집은 계속된다. <a href="https://perpetual-alpha26.github.io/schedule">공식 본대회 시작은 10월 19일</a>이다. 10/24 종료 예정의 새 모의 연구는 본대회 시작 전에 성과 개선을 인증하는 실험이 아니며, 기존 운용의 자동 교체도 수행하지 않는다. 현재 공개 참가자 문서는 par를 안내하고 있으므로 추후 별도 시장·펀딩 변경 공지가 있으면 실제 대상 시장을 재확인해야 한다.</p>
<p><a href="https://perpetual-alpha26.github.io/docs/">공식 참가자 문서</a>는 미실현손익·증거금에 mark를 사용하고, 펀딩은 oracle로 계산한다고 명시한다. 새 72시간 모의 비교의 기본 성과는 mark 계좌가치이고 oracle 민감도는 별도 기록한다.</p>
<h2>검증 자료</h2><ul><li><a href="../../experiments/pairs_maker_20261017/artifacts/results.json">지정가 진단 원본</a></li><li><a href="../../experiments/pairs_maker_20261017/paired_artifacts/results.json">헤지·청산 진단 원본</a></li><li><a href="../../experiments/pairs_maker_20261017/trade_side_verification.json">공개 체결 방향 대조 영수증</a></li><li><a href="../../experiments/pairs_maker_20261017/verification_all.log">테스트 결과</a></li><li><a href="../pairs_development_20261017/report.html">앞선 56회 계좌 비교와 새 모의 연구 계획</a></li></ul></html>'''
    (out/'report.html').write_text(body)
    save(out/'manifest.json',dict(at=time.time(),report_sha256=sha(out/'report.html'),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [HERE/'artifacts/results.json',HERE/'paired_artifacts/results.json',HERE/'trade_side_verification.json']},
        protected=protected(),improvement_proven=False,live_orders_sent=0))
    print(out/'report.html')


if __name__=='__main__':main()
