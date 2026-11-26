import csv
import gzip
import hashlib
import html
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from competition_live.profile import ROOT,verify_deployment,verify_bundle
from .protocol import OUT as FIRST
from .wave2 import OUT as SECOND
from .wave3 import OUT as THIRD
from .wave4 import OUT as FOURTH
from .wave5 import OUT


def read(path):
    return json.loads(path.read_text())


def daily(folder,name,scenario):
    with gzip.open(folder/"reference"/f"{name}_continuous_{scenario}.json.gz","rt") as f:
        run=json.load(f)
    equity=np.array([s["equity"]+50000 for s in run["snapshots"] if s["time"]%86400==43200])
    return equity[1:]/equity[:-1]-1


def q(returns):
    r=np.prod(1+returns,axis=-1)-1
    v=np.std(returns,axis=-1,ddof=1)*np.sqrt(returns.shape[-1])
    return (np.maximum(r,0)+.005)/(abs(r)+1.5*v+.05)


def uncertainty():
    rng=np.random.default_rng(20261020);rows=[]
    for scenario in ("base","stress"):
        a=daily(OUT,"incumbent",scenario);b=daily(OUT,"cost_qualified_reverse",scenario)
        n=len(a);length=7;draws=5000
        start=rng.integers(0,n,(draws,(n+length-1)//length))
        ids=((start[:,:,None]+np.arange(length))%n).reshape(draws,-1)[:,:n]
        aa,bb=a[ids],b[ids]
        dr=np.prod(1+bb,axis=1)-np.prod(1+aa,axis=1);dq=q(bb)-q(aa)
        rows.append(dict(scenario=scenario,completed_returns=n,block_days=length,draws=draws,
            incremental_return_95pct=np.quantile(dr,[.025,.975]).tolist(),
            incremental_Q_95pct=np.quantile(dq,[.025,.975]).tolist(),
            fraction_positive_return=float(np.mean(dr>0)),fraction_positive_Q=float(np.mean(dq>0)),
            interpretation="Paired circular block bootstrap of reused returns, conditional on this chosen candidate. "
                           "Does not correct adaptive candidate selection, establish independence, or prove future superiority."))
    (OUT/"uncertainty.json").write_text(json.dumps(rows,indent=2));return rows


def table(headers,rows):
    def c(x):return html.escape(str(x))
    return "<table><thead><tr>"+"".join("<th>"+c(h)+"</th>" for h in headers)+"</tr></thead><tbody>"+\
        "".join("<tr>"+"".join("<td>"+c(v)+"</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table>"


def main():
    folders=[FIRST,SECOND,THIRD,FOURTH,OUT];all_rows=[]
    for batch,folder in enumerate(folders,1):
        for row in read(folder/"reference_results.json"):
            all_rows.append(dict(batch=batch,**row))
    assert len(all_rows)==126
    assert all(r["accounting_error"]<1e-6 and r["timing_pass"] and r["risk_breach_bars"]==0 for r in all_rows)
    # Every baseline must reproduce the prior incumbent, including all windows/costs.
    prior=read(ROOT/"reports/isolated_repairs_20261020/reference_results.json")
    expected={(r["phase"],r["scenario"]):r for r in prior if r["variant"]=="incumbent"}
    for r in all_rows:
        if r["variant"]=="incumbent":
            a=expected[r["phase"],r["scenario"]]
            assert abs(r["net_pnl"]-a["net_pnl"])<1e-8 and abs(r["Q"]-a["Q"])<1e-12
    with (OUT/"all_trials.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    unc=uncertainty();last=read(OUT/"reference_results.json")
    selected=[r for r in last if r["variant"] in ("incumbent","cost_qualified_reverse")]
    pas=read(OUT/"pas_results.json");manifest=verify_deployment();bundle=verify_bundle()
    decision=dict(
        candidate="cost_qualified_reverse",stage="research_candidate_not_live",
        total_reference_runs=126,new_strategy_variants=15,size_diagnostic_variants=1,
        new_pas_runs=6,live_orders_sent=0,live_deployment_changed=False,
        source_target_parity=read(OUT/"candidate_parity.json"),
        development_august_pass=True,continuous_base_and_stress_PnL_Q_improve=True,
        later_september_period_improvement=False,pas_improvement=False,
        untouched_independent_validation=False,overfitting_ruled_out=False,
        no_future_input_in_tests=True,all_market_prediction_eligibility=True,
        actual_reference_traded_symbols=["BTC","ETH","HYPE","BRENT"],
        legacy_pair_universe_expanded=False,
        caution="Retained core remains a crypto-focused expert. New forecast layer evaluates all ten symbols, "
                "but did not cause noncrypto trades beyond BRENT in these windows. Do not say that every strategy component "
                "or every pair now covers every asset. Continuous improvement is observed development performance only.",
        deployment_files_checked=len(manifest["files"]),bundle_sha256=bundle,
    )
    (OUT/"decision.json").write_text(json.dumps(decision,indent=2))
    names={"incumbent":"기존 제출판","cost_qualified_reverse":"비용 조건＋추세 일치도＋확인 후 반전",
           "cost_qualified":"반전 제외 비교판"}
    periods={"validation":"8/1~9/1","comparison":"9/1~10/15","continuous":"8/1~10/15"}
    comparisons=table(["기간","조건","전략","순손익","팀 수익률 %","기간 σ %","Q","MDD %","평균 노출"],
        [[periods[r["phase"]],r["scenario"],names[r["variant"]],f'{r["net_pnl"]:+,.4f}',
          f'{100*r["return_fraction"]:.6f}',f'{100*r["volatility"]:.6f}',f'{r["Q"]:.8f}',
          f'{100*r["max_drawdown"]:.6f}',f'{r["mean_notional"]:,.2f}'] for r in selected])
    pas_table=table(["전략","조건","모의 손익","청산 비용 충당 후","체결 조각","숏 진입 조각","평균 노출"],
        [[names[r["variant"]],r["scenario"],f'{r["net_pnl"]:+.6f}',f'{r["pnl_after_liquidation_reserve"]:+.6f}',
          r["fills"],r["short_opening_fills"],f'{r["mean_notional"]:.2f}'] for r in pas])
    trials=table(["차수","후보","기간","비용","순손익","Q","기준 매매 체결 종목"],
        [[r["batch"],r["variant"],periods[r["phase"]],r["scenario"],f'{r["net_pnl"]:+.4f}',
          f'{r["Q"]:.8f}',", ".join(r["base_traded_symbols"])] for r in all_rows])
    intervals=table(["조건","수익률 차이 95% 구간 (퍼센트포인트)","Q 차이 95% 구간"],
        [[r["scenario"]," ~ ".join(f"{x*100:+.5f}" for x in r["incremental_return_95pct"]),
          " ~ ".join(f"{x:+.6f}" for x in r["incremental_Q_95pct"])] for r in unc])
    doc=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>SFINX 전체 종목 개선 실험</title>
<style>body{{font:16px/1.65 system-ui;max-width:1500px;margin:36px auto;padding:0 24px;color:#172337}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}th,td{{padding:8px;border:1px solid #d9e0e8;text-align:right}}
th{{background:#edf3f8}}td:first-child{{text-align:left}}.warn{{background:#fff1dc;padding:16px}}a{{color:#1754a2}}</style>
<h1>전체 종목 평가·비용 조건·추세 일치도 개선 실험</h1>
<p>작성: {datetime.now(timezone.utc).isoformat()}. 팀 초기 자본 100,000 USDP, 사용자 배정 50,000, 팀원 예약 50,000.
현재 실거래는 변경하지 않았으며 이 작업의 주문 전송은 0건입니다.</p>
<p><b>전체 기간의 PnL·Q는 기본/불리한 비용 모두 개선. 9월 이후 부분 기간과 PAS 구간에서는 개선하지 못했습니다.</b>
새 코드와 모든 실패 후보를 보존했습니다. 관측된 개발 성과이며 미래 우월성이나 과적합 부재를 입증한 결과가 아닙니다.</p>
<h2>구현</h2><ol><li>기존 페어 모델과 신호를 유지하며, 3·7·14일 변동성 보정 추세와 포지션 방향의 일치도로 비중을 0.5~1.5배 조절합니다.</li>
<li>7/1~7/29 자료로만 위험을 보정한 고정 계수 0.9670688053을 적용합니다. 평가 기간 수익으로 규모를 맞추지 않습니다.</li>
<li>새 진입층은 10종목을 동일한 릿지 모델로 평가합니다. 24시간 뒤 결과가 확정된 자료만 90일 창에서 학습하며,
왕복 비용 38bp와 기존에 고정한 확신 조건을 넘을 때만 기존 비중이 없는 종목에 보조 목표를 만듭니다.</li>
<li>24시간 연속으로 강한 반대 추세가 확인되면, 기존 목표와 새 추세 목표 중 작은 규모로 반전합니다. 실제 실행 경로는 먼저 기존 포지션을 청산합니다.</li></ol>
<p class="warn"><b>종목 범위:</b> 새 예측층에는 BTC·ETH만 허용하는 조건이 없습니다. 다만 기존 핵심 전략은 암호화폐 중심이고,
페어 후보도 BTC–ETH·BTC–HYPE 두 쌍을 유지합니다. 최종 후보의 이 기간 실제 기준 매매 체결은 BTC·ETH·HYPE·BRENT였습니다.
모든 층의 종목 제한을 없앴다거나 모든 종목을 거래했다고 주장하지 않습니다.
10종목을 실제로 거래한 다른 후보들은 성과가 나빠 최종 후보로 선정하지 않았습니다.</p>
<h2>같은 실행 엔진·자본·위험 한도에서 비교</h2>{comparisons}
<p>기본 비용: 편도 수수료 9bp＋가격 충격 10bp, 거래량 참여 1%. 불리한 비용: 12bp＋20bp, 참여 0.25%, 수취 펀딩 50%.
일별 12:00 UTC 평가액에 팀원 예약 자본을 포함해 대회 Q를 계산했습니다. 종료 청산 비용 충당 값은 원본 결과에도 보존했습니다.</p>
<h2>본대회 PAS 별도 모의평가</h2><p>10/19 21:00~10/20 22:44 KST. PAS 분봉·체결량·펀딩 자료를 사용했습니다.</p>{pas_table}
<p class="warn">세 후보의 이 구간 모의 체결은 같았습니다. 분봉 거래량 한도가 목표 수량보다 작아 비중 변경 효과가 체결로 연결되지 않았습니다.
과거 전체 호가/마크 기록이 없어 실제 체결 규모를 재현하지 못합니다. 하루 수익률 1개로 σ·Q를 만들지 않았고, 실거래 성과라고 부르지 않습니다.
이 결과를 보고 모델을 다시 조정하지 않았습니다.</p>
<h2>재현·오류·과적합 확인</h2><p>126회 기준시장 재생과 6회 PAS 재생에서 회계 오차 10⁻⁶ 미만, 기록된 위험 위반 0,
미래 체결가를 앞선 판단에 사용한 시간 위반 0. 기존판은 모든 차수에서 앞선 결과와 일치합니다.
최종 재사용 함수와 연구 목표값의 최대 차이는 0이며, 미래 48시간을 제거해도 이전 예측·목표가 유지됩니다.
각 학습의 마지막 정답 시각은 학습 실행 시각보다 엄격히 이릅니다.</p>
<p>15개 전략 변경과 규모 진단 1개를 순차적으로 탐색했습니다. 각 차수 규칙을 먼저 기록하고 8월 선택을 고정한 뒤 이후 구간을 평가했으나,
차수 사이에는 이미 결과를 봤습니다. 따라서 전체 과정을 독립 검증이라고 할 수 없습니다.
다음 미사용 구간 검증의 시작은 최종 release_manifest 생성 이후이며, 그 전 기간은 모두 개발 자료입니다.</p>
{intervals}<p>75개 일별 수익률을 7일 묶음으로 함께 재표집한 조건부 95% 구간, 5,000회.
후보 탐색에 따른 선택 편향까지 보정하는 검정이 아닙니다.</p>
<h2>근거 연구와 적용 범위</h2><p>
<a href="https://arxiv.org/abs/2607.00475">Pollok·Robik (2026)</a>의 자산군 간 정책 비교와 비용 민감도,
<a href="https://arxiv.org/abs/2302.10175">Tan·Roberts·Zohren (2023)</a>의 시계열·자산 간 특징 결합을 참고했습니다.
논문의 모델이나 수익률을 그대로 재현했다고 주장하지 않습니다. 이 후보의 학습 모델은 자체 구현한 릿지 회귀입니다.</p>
<details><summary>모든 차수·후보·실패 결과 126행</summary>{trials}</details>
<p><a href="all_trials.csv">전체 CSV</a> · <a href="decision.json">판단 기록</a> ·
<a href="latest_target_trace.json">고정된 시점의 10종목 판단</a> · <a href="uncertainty.json">불확실성 분석</a></p></html>"""
    (OUT/"report.html").write_text(doc)
    print(json.dumps(decision,indent=2))


if __name__=="__main__":
    main()
