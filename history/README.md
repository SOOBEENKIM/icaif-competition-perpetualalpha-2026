# Perpetual Alpha: hourly price history

Hourly candles for every competition market, taken from the Hyperliquid mainnet
market it copies. Every hour that ended before 2026-10-05T16:00:00Z is included.

| Market | File | Hyperliquid mainnet | First hour (UTC) | Last hour (UTC) | Candles | Hours without trades |
| --- | --- | --- | --- | --- | --- | --- |
| pa:BTC | `BTC.csv` | BTC | 2026-03-11T07:00:00Z | 2026-10-05T15:00:00Z | 5001 | 0 |
| pa:ETH | `ETH.csv` | ETH | 2026-03-11T07:00:00Z | 2026-10-05T15:00:00Z | 5001 | 0 |
| pa:HYPE | `HYPE.csv` | HYPE | 2026-03-11T05:00:00Z | 2026-10-05T15:00:00Z | 5003 | 0 |
| pa:SP500 | `SP500.csv` | xyz:SP500 | 2026-03-18T13:00:00Z | 2026-10-05T15:00:00Z | 4827 | 0 |
| pa:EWY | `EWY.csv` | xyz:EWY | 2026-03-11T09:00:00Z | 2026-10-05T15:00:00Z | 4999 | 0 |
| pa:EWJ | `EWJ.csv` | xyz:EWJ | 2026-03-11T09:00:00Z | 2026-10-05T15:00:00Z | 4999 | 0 |
| pa:BRENT | `BRENT.csv` | xyz:BRENTOIL | 2026-03-11T09:00:00Z | 2026-10-05T15:00:00Z | 4999 | 0 |
| pa:GOLD | `GOLD.csv` | xyz:GOLD | 2026-03-11T07:00:00Z | 2026-10-05T15:00:00Z | 5001 | 0 |
| pa:USDEUR | `USDEUR.csv` | xyz:EUR | 2026-03-11T08:00:00Z | 2026-10-05T15:00:00Z | 5000 | 0 |
| pa:USDJPY | `USDJPY.csv` | xyz:JPY | 2026-03-11T07:00:00Z | 2026-10-05T15:00:00Z | 5001 | 0 |

## Columns

- `time`: the hour's open, in UTC (ISO 8601). Each row covers `time` to `time` + 1h.
- `open`, `high`, `low`, `close`: trade prices on Hyperliquid mainnet.
- `volume`: contracts traded, in units of the asset.
- `trades`: number of fills.

An hour with no trades has no row.

## What these prices are

These are Hyperliquid mainnet trade prices, a reference rather than the
competition's price. Each competition market's oracle is a SEDA Signal composite
(a median across several venues, Hyperliquid included), so the two differ by a
few basis points. They can differ more on equity, commodity and FX markets while
the underlying exchange is closed.

Hyperliquid serves only the most recent 5000 candles, so the history reaches
back about 208 days, or to the market's listing if that is later.
