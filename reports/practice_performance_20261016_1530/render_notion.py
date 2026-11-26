"""Render only the two requested Notion tables from verified cutoff metrics."""
from decimal import Decimal
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def number(value, places=6, signed=False):
    return format(Decimal(str(value)), f'{"+" if signed else ""},.{places}f')


def table(headers, rows):
    def row(cells):
        return '<tr>\n'+'\n'.join('<td>'+str(c)+'</td>' for c in cells)+'\n</tr>'
    return '<table fit-page-width="true" header-row="true">\n'+row(headers)+'\n'+'\n'.join(row(r) for r in rows)+'\n</table>'


def main():
    m=json.loads((OUT/'metrics.json').read_text())
    r=m['risk']
    nd='산출 불가 — 완료된 일별 수익률 0개'
    nc='산출 불가 — 완결 거래 0건'
    summary=table([
        '지갑 주소','측정 시작 (KST)','측정 종료 (KST)','종료 순자산·mark (USDPR)',
        '순손익·수수료/펀딩 반영·mark (USDPR)','누적 수익률·mark (%)','일별 수익률 STD (%)','관측 최대 낙폭 MDD·mark (%)'], [[
        m['measurement']['team_wallet'],
        '<mention-date start="2026-10-15" startTime="21:02" timeZone="Asia/Seoul"/> (11.086초, 최초 체결 직전)',
        '<mention-date start="2026-10-16" startTime="15:30" timeZone="Asia/Seoul"/> (00초)',
        number(m['mark_equity']),number(m['mark_net_pnl'],signed=True),number(m['mark_return_percent'],signed=True),
        nd,number(r['sampled_mdd_percent'])+' (낙폭 크기)']])
    rows=[]
    def add(group,label,value):rows.append([group,label,str(value)])
    add('측정 기준','측정 기간','18시간 27분 48.914초 — 최초 체결 직전부터 종료시각까지')
    add('측정 기준','완료된 일별 수익률 관측 수 (일)',m['complete_daily_returns'])
    add('측정 기준','일별 평가 기준시각·시간대','이 표의 집계 기준 UTC 00:00 = KST 09:00; 공식 집계 시간대 확정값 아님')
    add('측정 기준','실제 거래일 수 (일)',f"{m['trading_days_kst']} (KST 달력일 기준) / {m['trading_days_utc']} (UTC 기준)")
    add('측정 기준','정상 리밸런싱 판단 라운드 수',f"{m['normal_rounds']}회")
    add('자본','기초 순자산 (USDPR)',number(m['initial_equity']))
    add('자본','순입출금 (USDPR)',number(m['net_capital_flows']))
    add('손익','실현 매매손익·수수료/펀딩 차감 전 (USDPR)',number(m['realized_gross_pnl']))
    add('손익','미실현손익·mark (USDPR)',number(m['unrealized_mark'],signed=True))
    add('손익','누적 거래 수수료·비용 (USDPR)',number(m['fees']))
    add('손익','누적 펀딩 순수취·지급 (USDPR)',number(m['net_funding'],signed=True)+' (+수취 / −지급)')
    add('손익','순손익 검산·mark (USDPR)',f"{number(m['realized_gross_pnl'])} + ({number(m['unrealized_mark'])}) − {number(m['fees'])} + ({number(m['net_funding'])}) = {number(m['mark_net_pnl'])}")
    add('손익','종료 순자산·oracle (USDPR)',number(m['oracle_equity']))
    add('손익','순손익·oracle・수수료/펀딩 반영 (USDPR)',number(m['oracle_net_pnl'],signed=True))
    add('손익','누적 수익률·oracle (%)',number(m['oracle_return_percent'],signed=True))
    for label in ('일평균 수익률 (%)','일별 하방편차 (%)','최악의 일별 수익률 (%)','Sharpe 비율 (일별·비연율화·기준수익률 0)','Sortino 비율 (일별·비연율화·목표수익률 0)'):
        add('위험',label,nd)
    add('위험','평균 총노출 / 순자산 (%)',number(r['sampled_average_gross_percent'])+' (관측값 시간가중 근사)')
    add('위험','최대 총노출 / 순자산 (%)',number(r['sampled_maximum_gross_percent'])+' (관측값)')
    add('위험','참고: 시간별 수익률 표본 STD (%)',number(r['hourly_sample_std_percent'])+f" — {len(r['hourly_returns'])}개, ddof=1, 비연율화; 일별 STD 대체값 아님")
    add('위험','위험 지표의 관측 간격',f"실계좌 평가액 {r['equity_observations']:,}개 + 시작/종료점; 통상 약 30초, 최대 공백 {r['maximum_sample_gap_seconds']:.3f}초. 시간별 값은 정각 직전 60초 이내 관측값")
    add('거래','주문 전송 수 (건)',m['orders'])
    add('거래','실제 체결 수 (건)',f"{m['fills']} (개별 매칭 기준; 주문 {m['orders']}건이 분할 매칭됨)")
    add('거래','완결 거래 수 (건)',m['completed_trades'])
    for label in ('승률·완결 거래 기준 (%)','평균 이익 / 평균 손실 절댓값','Profit Factor·총이익 / 총손실 절댓값'):
        add('거래',label,nc)
    add('거래','누적 체결 거래대금 (USDPR)',number(m['trade_notional']))
    add('거래','메이커 체결대금 비중 (%)',number(m['maker_notional_percent']))
    add('거래','실제 체결대금 가중 수수료 (bp)',number(m['effective_fee_bps'])+' (편도)')
    add('거래','부분 체결로 종료된 주문 수 (건)',m['partial_orders'])
    add('거래','거절 주문 수 (건)',m['rejected_orders'])
    pos=[]
    for p in m['positions']:
        pos.append(f"{p['coin']} 롱 {p['qty']} / 평균 진입가 {number(p['entry_vwap'])} / {p['leverage']['value']}배 {p['leverage']['type']}")
    add('종료 상태','보유 종목·롱/숏·수량·평균 진입가','<br>'.join(pos))
    for p in m['positions']:
        add('종료 상태',p['coin']+' 가격·평가손익',f"mark {number(p['mark'])}; oracle {number(p['oracle'])}; 미실현손익(mark) {number(p['unrealized_mark'],signed=True)} USDPR")
    add('종료 상태','총 포지션 명목금액·mark (USDPR)',number(m['gross_notional']))
    add('종료 상태','순 포지션 명목금액·mark (USDPR)',number(m['net_notional']))
    from calculate import iso
    add('종료 상태','사용 증거금 (USDPR)',number(m['margin_used_exchange_snapshot'])+' — '+iso(m['margin_snapshot_ms']).split('T')[1]+' 직전 계좌 스냅샷')
    add('종료 상태','미체결 주문 수 (건)',m['open_orders'])
    add('측정 기준','마감 가격의 수신 시각 (KST)','<br>'.join(p['coin']+': '+iso(p['price_received_ms']).split('T')[1] for p in m['positions']))
    add('대회 평가','순수익 평가점수·oracle 기준','미산출 — 연습기간이며 전체 팀 정규화 자료 없음')
    add('대회 평가','일별 변동성 평가점수','미산출 — 일별 표본 및 전체 팀 정규화 자료 없음')
    add('대회 평가','알고리즘 심사 점수','미산출 — 심사 전')
    add('대회 평가','공식 종합점수','미산출 — 연습기간')
    add('대회 평가','공식 순위','해당 없음 — 연습기간')
    note=('일별 수익률 STD는 [대회 평가항목](https://perpetual-alpha26.github.io/about)이므로 포함했습니다. '
          '이번 측정은 24시간 미만이어서 일별 STD를 0%로 쓰지 않습니다. '
          '요약은 mark 기준, 대회 종료 평가와 같은 oracle 기준 수익은 세부 표에 별도 기재했습니다. '
          '최대낙폭은 기록된 평가액에서 관측한 값입니다. 순손익에는 미실현손익이 포함되며 향후 청산 수수료는 포함하지 않습니다.')
    content='# **성과 요약**\n'+summary+'\n'+note+'\n# **세부 결과**\n'+table(['구분','지표','결과'],rows)
    (OUT/'notion_target.txt').write_text(content)
    print(json.dumps({'detail_rows':len(rows),'characters':len(content)},ensure_ascii=False))


if __name__=='__main__':main()
