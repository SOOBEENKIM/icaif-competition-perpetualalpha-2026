Perpetual Alpha market-opportunity research — 2026-10-14

Read report.html first. This folder freezes public market observations and research
decisions. It contains no trading keys, account changes, order submissions, fitted
trading model, prospective strategy test, or claim of verified trading profits.

Venue and scope
  Practice: Hyperliquid testnet HIP-3 par, collateral USDPR, ten markets.
  Official live venue announced in the user's copied Discord notices: pa / USDP.
  Final venue probe found par in perpDexs; pa metaAndAssetCtxs returned HTTP 500.
  A point-in-time probe does not establish permanent nonexistence of pa.

Data windows (not all files share the same cutoff)
  Historical fills: 2026-10-05 00:00:00 UTC through 2026-10-14 07:10:06.938 UTC.
  Two wallet sides per execution; 39,474 rows -> 19,737 unique executions.
  Completed 1h par candle buckets reconciled: 2,230. Partial latest hour excluded.
  Settled funding: 225 observations per asset from Oct 5 00:00 through Oct 14
    08:00 UTC (settlement timestamps may be milliseconds after the hour).
  Simultaneous WS capture: Oct 14 07:59:11.904–08:14:11.958 UTC, 900.054 seconds.
    One par disconnect at 08:09:53 UTC; per-asset reconnection gaps excluded.
    34 unique new executions, 8,960 one-second asset rows. These rows are serially
    dependent, not independent trading trials. Historical subscription snapshots
    before capture are excluded; reconnect duplicates use (coin, time, tid).

Public endpoints used
  https://api.hyperliquid-testnet.xyz/info — read-only POST info requests
  wss://api.hyperliquid-testnet.xyz/ws — public subscriptions
  https://api.hyperliquid.xyz/info — external reference candles
  wss://api.hyperliquid.xyz/ws — external reference BBO and asset context
  No calls to /exchange.

Files
  report.html                 Korean research report, decisions and limitations
  literature.json             11 primary-source papers, reading scope, caveats
  market_summary.csv          Compact ten-market summary
  analysis.json               Full descriptive statistics and coverage
  historical_unique_trades.csv One taker-side representation per historical trade
  hourly_price_panel.csv      Practice OHLCV and corresponding mainnet close
  settled_funding.csv         Settled hourly rates, decimal units as returned
  live_panel_1s.csv           As-of receive-time synchronized public market panel
  live_unique_trades.csv      All deduplicated executions within live window
  trade_markouts.csv          Observed counterparty-side diagnostic, not strategy PnL
  validation.json            Data/math validation results, not profit validation
  raw/                       Frozen public requests/responses and compressed WS
  SHA256SUMS.txt             Integrity hashes excluding itself and Python caches

Units and interpretation
  1 bp = 0.0001 = 0.01%.
  Maker fee 3bp, taker fee 9bp at the observed base tier. Historical fee records
    agree within exchange rounding. Similar-notional round trips cost 6/12/18bp
    in fees for maker-maker/maker-taker/taker-taker, before other costs.
  Spread = (ask - bid) / mid * 10,000.
  Mid/oracle basis = (mid / oracle - 1) * 10,000.
  Target notionals 100/1,000/10,000 are converted to quantity at current oracle
    and rounded down to the asset lot size. VWAP walks the displayed L2 levels.
    If depth is insufficient, VWAP is missing rather than extrapolated.
  Static roundtrip cost = (buyVWAP - sellVWAP) / mean(buyVWAP,sellVWAP) * 10,000 + 18.
    The 18bp term is exact with the mean-notional normalization and equal quantity.
    This is a same-book diagnostic, not a realized or forecast roundtrip.
  Buy-to-oracle net = (oracle / buyVWAP - 1) * 10,000 - 9 - 9*oracle/buyVWAP.
    It assumes hypothetical exit at CURRENT oracle and both taker fees. An oracle
    is not an executable exit quote. No convergence probability is established.
  FundingHistory values are settled; activeAssetCtx funding is a running estimate.
    Positive funding: long pays short. Negative: short pays long.
  Markouts use later oracle/mid and an assumed entry fee; they do not include exit
    execution, exit fee or funding. They describe others' observed fills, not ours.
  Top-three maker share is by executed notional, not wallet identity or team role.
  Mainnet reference instruments are those listed in the official participant guide.
    Same-hour price correlations are descriptive, not causal lead-lag evidence.

Reproduce from frozen files, without network or keys
  Python >=3.10 and numpy are sufficient for analysis and report generation.
  From this directory:
    python analyze_market.py
    python validate_research.py
    python build_report.py
  Numerical outputs can differ minutely across library versions. Hashes certify
  the delivered files; timestamps in a newly generated validation file will differ.

Collection scripts (do not rerun into this frozen folder)
  collect_public.py: bounded 900-second public capture and historical funding/
    candles. Requires websocket-client. Reconnection/errors are logged. Public
    REST request pacing is intentionally conservative. No periodic automation.
  finalize_public.py: final settled funding/venue probe, plus local copy of the
    earlier leaderboard's public fill rows. It references that earlier report
    only during collection; frozen-file analysis has all inputs within raw/.
  analyze_market.py retains a legacy input fallback if frozen historical files
    are missing, but delivered files include the complete frozen inputs.

Official sources
  https://perpetual-alpha26.github.io/docs/
  https://perpetual-alpha26.github.io/about
  https://perpetual-alpha26.github.io/
  User-provided copied Discord notices (Oct 12–13):
  https://app.notion.com/p/3eb73946edee810b995de6f2fcfa2608

Selection: A selective funding-aware maker first; B basis/funding conditional;
C slower trend as control/risk filter. Findings are for practice venue par.
No full-market census or long-run profitability claim is made by the 15-minute
book sample. Unknown queue positions, latency, endogenous impact and future
official-venue participation remain material limitations.
