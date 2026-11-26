import hashlib,json
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/ema_pairs_pas_20261026_v2'
OLD=ROOT/'reports/execution_revision_20261024/dataset'
START=1792411200 # Oct19 12UTC
SPLIT=1792843200 # Oct24 12UTC
END=1793016000 # Oct26 12UTC
SYMBOLS=['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY']
ARMS=['incumbent','basis_reversion','relative_trend','ema_basis_blend']
SCENARIOS=dict(base=dict(fee_bps=9.,impact_bps=10.,participation=.01,receipt_multiplier=1.),
 stress=dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=.5),
 no_receipts=dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=0.))

def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze():
 from adaptive_leverage.study import live_manifest
 if (OUT/'protocol.json').exists():raise FileExistsError('Do not replace previous protocol')
 save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),start=START,split=SPLIT,end=END,
  revision='v2: correct over-restrictive contiguous-history requirement; valid adjacent transitions on both sides of gaps retained, gaps never bridged. v1 all12 outcomes preserved. Add matured forecast economic diagnostics because native OHLC production replay produced nearly no incumbent fills.',
  parent='reports/ema_pairs_pas_20261026/protocol.json',
  arms=ARMS,scenarios=SCENARIOS,own_capital=50000,reserve=50000,pair_max_gross=6000,
  hypothesis='Retain actual EMA/trend/breakout/Chronos core. Add one independently accounted two-leg relative-value sleeve. Reference correlations estimate factor hedge; PAS/reference basis predicts venue-specific correction; relative EMA continuation is a separate expert. Neither price correlation nor basis implies guaranteed convergence.',
  formation='All45 unordered pairs, no name whitelist; trailing720 completed reference hours, return correlation>=0.2 and positive OLS beta0.1..10,>=120 joint actual observations. Each new source fixes its hedge.',
  models='24h horizon. Basis: >=36 valid adjacent hourly PAS/reference observations,<=96h, local AR with zero-change ridge prior24 observations; rho clipped0.8..1.02, forecast shrunk by coefficient uncertainty. Trend: past-only6h/24h EMA relative reference return, conservative quarter of24h extrapolation. Blend: equal price forecasts. PAS funding:6h/24h EWM signed agreement, settlements known after60s, receipts half/payments full, capped horizon24h.',
  lifecycle='Production pair entry, independent sequential IOC receipts, actual partial quantities, completion/repair/paid sticky exits. Max24h holding; no extension beyond original24h. Global and one-leg risk limits retained. Current reverse depth is an exit proxy only.',
  costs='Historical PAS OHLC conditional fills via unchanged production path, min(previous,current) volume participation, adverse next-minute extrema and latency; later actual received-book replay separately. No reference-market PnL represented as PAS.',
  selection='First same development execution comparison. If no arm has pair fills, use fully matured24h forecasts on19-24Oct only: max minimum average signed pair return net of38bp base/64bp stress and settled funding, across all admitted signals. Diagnostic economic labels are NOT executable PnL. Require >=3 distinct source days; otherwise designate ema_basis_blend as explicitly unqualified primary. Freeze choice before later48h and actual-book outcomes. No tuning on later outcomes.',
  evaluation='Only selected candidate vs incumbent and same-inventory switch on later48h, base/stress/no receipts. No retuning after this evaluation. Report full terminal liquidation reserve, price/fees/funding and daily12UTC Q. Compare exposure/return; do not imply equal realized risk.',
  independence='Old history extensively reused. Later aggregate live outcomes and parts of previous experiments already seen, so chronological held-back comparison, NOT a certified untouched test. Source-maturity tests prevent lookahead, not all selection bias.',
  sources=['https://doi.org/10.1186/s40854-024-00702-7','https://arxiv.org/abs/2403.07998','https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding'],
  live_manifest=live_manifest(),orders_sent=0,live_changed=False))

if __name__=='__main__':freeze()
