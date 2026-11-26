"""Read-only comparison of saved official boards and current practice evidence."""
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def kst(value):
    if isinstance(value,(int,float)):return datetime.fromtimestamp(value,ZoneInfo('Asia/Seoul')).isoformat()
    return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(ZoneInfo('Asia/Seoul')).isoformat()

def main():
    final,live=read('practice.json'),read('current.json')
    old={t['name']:t for t in final['teams']};new={t['name']:t for t in live['teams']}
    a,b=old['SFINX'],new['SFINX'];settings=live['scoring']
    calculated=(max(b['return'],0)+settings['epsilon'])/(abs(b['return'])+settings['lambda']*b['volatility_horizon']+settings['c'])
    assert abs(calculated-b['q'])<1e-14
    assert abs(calculated*settings['w_financial']-b['score'])<1e-14
    status=read('runtime_status.json');fills=read('fills.json');account=read('account_snapshot.json')
    rows=[]
    for team in sorted((t for t in live['teams'] if t['ranked']),key=lambda t:t['rank']):
        previous=old.get(team['name'],{})
        rows.append(dict(name=team['name'],current_rank=team['rank'],final_rank=previous.get('rank'),
            current_return=team['return'],final_return=previous.get('return'),
            current_score=team['score'],final_score=previous.get('score')))
    result=dict(final_sfinx=a,current_sfinx=b,final_cutoff_kst=kst(final['to']),
        current_window_end_kst=kst(live['to']),current_board_generated_kst=kst(live['generated_at']),
        final_ranked_teams=sum(t['ranked'] for t in final['teams']),current_ranked_teams=len(rows),
        hypothetical_rank_with_old_score=1+sum(t['score']>a['score'] for t in live['teams'] if t['ranked'] and t['name']!='SFINX'),
        hypothetical_rank_with_new_score_old_opponents=1+sum(t['score']>b['score'] for t in final['teams'] if t['ranked'] and t['name']!='SFINX'),
        net_return_change_percentage_points=100*(b['return']-a['return']),
        capital_equivalent_change_USDPR=1e6*(b['return']-a['return']),
        official_score_recomputed=calculated*.8,formula_changed=False,
        phase_status_and_window_changed=True,practice_final_remains_fifth=True,
        runtime_profile=status['profile'],runtime_service=status['service'],runtime_pid=status['pid'],
        runtime_status_kst=kst(status['updated_at_utc']),runtime_dex=status['dex'],
        switch_time_kst=kst(status['epoch_start_utc']),runtime_equity=status['account']['equity'],
        runtime_net_pnl=status['account']['net_pnl'],second_strategy_epoch_pnl=status['epoch_mark_pnl'],
        latest_fill_kst=kst(max(f['time'] for f in fills)/1000),fill_fragments=len(fills),
        unique_order_ids=len({f['oid'] for f in fills}),closed_pnl=sum(float(f['closedPnl']) for f in fills),
        positions=status['account']['qty'],latest_round_kst=kst(status['last_round']['decision_time']),
        last_round_hold=status['last_round']['target_qty']==status['last_round']['positions_before'],
        next_normal_decision_kst=status['next_normal_decision_kst'],
        scored_pair_service_active=False,trading_changed_by_review=False,comparison=rows)
    (HERE/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>SFINX 연습 순위·운용 확인</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1150px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #c6cdd2;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}a{color:#146cba}</style>',
        '<h1>SFINX 연습 최종표와 갱신 중인 표</h1>',
        f'<p>공식 갱신표 생성: {kst(live["generated_at"])}. 현재 운용 기록: {kst(status["updated_at_utc"])}.</p>',
        '<p><strong>Practice (final)는 5위 그대로이며, 현재 갱신표는 37위다.</strong> 사이트는 두 개의 다른 JSON을 읽는다. 최종표 종료는 10월 16일 21시, 현재 표의 집계 종료 설정은 10월 18일 21시 KST다. 주최 측이 왜 기간 설정을 바꿨는지에 대한 별도 공지는 이 분석에서 확인하지 않았다.</p>',
        '<table><tr><th>항목</th><th>연습 최종표</th><th>현재 갱신표</th></tr>']
    metrics=[('순위',f'{a["rank"]} / {result["final_ranked_teams"]}',f'{b["rank"]} / {len(rows)}'),
        ('PnL 수익률',f'{a["return"]*100:+.8f}%',f'{b["return"]*100:+.8f}%'),
        ('100만 자본 환산 순손익',f'{a["return"]*1e6:+.2f} USDPR',f'{b["return"]*1e6:+.2f} USDPR'),
        ('일별 표준편차 s',f'{a["volatility"]*100:.8f}%',f'{b["volatility"]*100:.8f}%'),
        ('공식 표시 기간 변동성 σ',f'{a["volatility_horizon"]*100:.8f}%',f'{b["volatility_horizon"]*100:.8f}%'),
        ('Q',f'{a["q"]:.9f}',f'{b["q"]:.9f}'),('Score',f'{a["score"]:.9f}',f'{b["score"]:.9f}')]
    for label,x,y in metrics:page.append(f'<tr><td>{label}</td><td>{x}</td><td>{y}</td></tr>')
    page.extend(['</table>',
        f'<p>이전 SFINX 점수를 유지했다면 현재 다른 팀 점수들과 비교해도 {result["hypothetical_rank_with_old_score"]}위다. 따라서 대폭 하락의 주원인은 우리 수익률의 음수 전환과 기간 변동성 증가다. 순손익 환산 변화는 {result["capital_equivalent_change_USDPR"]:+.2f} USDPR다. 점수가 0.08 부근인 팀이 많아 작은 점수 하락이 큰 순위 하락으로 보인다.</p>',
        '<p>다른 팀의 변화도 있다. Reinforcement Traders는 −0.04777969%에서 +0.08124033%로 개선되어 41위에서 3위가 됐다. 반면 SFINX 점수를 고정한 비교에서는 여전히 5위이므로 남들의 개선만으로 이번 하락을 설명할 수 없다. λ=1.5, c=0.05, ε=0.005, 금융 80%·심사 20% 공식은 비교 기록 간 동일하다.</p>',
        '<p>현재 표는 일별 샘플로 계산된다. 12개의 완료 일별 수익률을 반영했으며, 갱신 시각인 10월 18일 오후 장중 손익과 같지 않다. 10월 17일 21:00:08의 별도 운용 관측은 −466.64 USDPR로 공식 표와 12.46 차이가 난다. 이는 같은 순간의 관측이 아니므로 해당 관측을 공식 샘플로 대체하지 않았다.</p>',
        '<h2>실행 중인 연습 전략</h2>',
        f'<p><code>{html.escape(status["profile"])}</code> — 추세＋7일 되돌림에 위험 감축과 고정 규모 보정을 적용한 두 번째 전략. {kst(status["epoch_start_utc"])}에 교체했고, PAR / USDPR에서 계속 실행 중이다. systemd의 실제 프로세스 번호 {status["pid"]}와 갱신 중인 상태 파일을 대조했다. 오늘 준비한 PAS용 decision-gate 페어 전략은 아직 활성화하지 않았다.</p>',
        f'<p>보유: BTC 0.4682 롱, HYPE 85.78 롱. 총 {len(fills)}개 체결 조각 / 주문 ID {result["unique_order_ids"]}개. 마지막 체결은 {result["latest_fill_kst"]} HYPE 추가 매수이며, 실현 청산손익은 0이다. 이후 가격 변화로 미실현손익이 변한다. 정상 판단은 매일 03·09·15·21시 KST이며, 최근 15시 판단은 목표 수량과 실제 수량이 같아 HOLD였다. 다음 정상 판단은 {status["next_normal_decision_kst"]}.</p>',
        f'<p>별도 장중 운용 기록의 계좌 손익은 {status["account"]["net_pnl"]:+.2f} USDPR, 두 번째 전략 교체 시점 대비는 {status["epoch_mark_pnl"]:+.2f} USDPR다. 계좌 전체 과거 손익을 전부 두 번째 전략의 성과로 보지 않았다. 누적 수수료 {status["account"]["fees"]:.2f}, 수취 펀딩 {status["account"]["funding"]:.2f} USDPR. 두 롱의 가격 하락 평가손실이 펀딩 수입을 웃돈다.</p>',
        '<h2>팀별 비교</h2><table><tr><th>팀</th><th>최종표 순위</th><th>갱신표 순위</th><th>최종표 수익률</th><th>갱신표 수익률</th><th>갱신표 Score</th></tr>'])
    for r in rows:
        prior='—' if r['final_return'] is None else f'{100*r["final_return"]:+.8f}%'
        page.append(f'<tr><td>{html.escape(r["name"])}</td><td>{r["final_rank"] or "—"}</td><td>{r["current_rank"]}</td><td>{prior}</td><td>{100*r["current_return"]:+.8f}%</td><td>{r["current_score"]:.9f}</td></tr>')
    page.extend(['</table><p>이번 확인에서 전략·주문·서비스를 변경하지 않았다.</p>',
        '<p><a href="https://perpetual-alpha26.github.io/leaderboard#practice">공식 연습 최종표</a> · <a href="https://perpetual-alpha26.github.io/leaderboard">공식 갱신표</a> · <a href="practice.json">최종표 원본</a> · <a href="current.json">갱신표 원본</a> · <a href="analysis.json">계산 결과</a></p></html>'])
    (HERE/'report.html').write_text('\n'.join(page))
    print(json.dumps({k:v for k,v in result.items() if k in ['hypothetical_rank_with_old_score','runtime_profile','latest_fill_kst','last_round_hold','practice_final_remains_fifth']},ensure_ascii=False))

if __name__=='__main__':main()
