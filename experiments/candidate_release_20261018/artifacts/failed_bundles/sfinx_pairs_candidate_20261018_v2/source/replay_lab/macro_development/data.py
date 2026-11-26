"""Read immutable receipts; never update a previous experiment's manifest."""
import json
from pathlib import Path
import numpy as np
from ..pairs_rl.public_data import load_market
from ..common_comparison.models import Market, timestamp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'replay_lab/output/macro_development/20261013-v6'
PREVIOUS = ROOT/'replay_lab/output/pairs_restart/20261013-v5'

def load():
    m, a, manifest = load_market()
    oldn = len(m.times)
    times = np.arange(m.times[0], timestamp('2026-10-13T03:00:00Z')+1, 3600)
    n = len(times)
    cols = {k: np.zeros((n, 10)) for k in ('open','high','low','close','volume','trades','funding','observed')}
    for k in ('open','high','low','close','volume','trades'): cols[k][:oldn] = a[k]
    cols['funding'][:oldn] = m.funding; cols['observed'][:oldn] = m.observed
    index = {int(t):i for i,t in enumerate(times)}
    for j,s in enumerate(m.symbols):
        rows = json.loads((PREVIOUS/'extension_receipts'/f'{s}_candleSnapshot.json').read_text())['response']
        for r in rows:
            i = index.get(int(r['t'])//1000+3600, -1)
            if i < oldn: continue
            for k,v in [('open','o'),('high','h'),('low','l'),('close','c'),('volume','v'),('trades','n')]: cols[k][i,j] = float(r[v])
        rows = json.loads((PREVIOUS/'extension_receipts'/f'{s}_fundingHistory.json').read_text())['response']
        for r in rows:
            i = index.get(int(r['time'])//3600000*3600, -1)
            if i >= oldn: cols['funding'][i,j] = float(r['fundingRate']); cols['observed'][i,j] = 1
    assert (cols['close']>0).all() and cols['observed'].all()
    market = Market(times,cols['close'],cols['volume'],cols['trades'],cols['funding'],cols['observed'].astype(bool),m.symbols)
    original = {**a, **{k:cols[k] for k in ('open','high','low','close','volume','trades')}, 'times':times-3600, 'known':times}
    original['returns'] = np.diff(np.log(cols['close']),axis=0,prepend=np.log(cols['close'][:1]))
    original['sigma'] = np.zeros_like(original['returns'])
    original['sigma'][:oldn] = a['sigma']
    for i in range(oldn,n): original['sigma'][i] = np.maximum(original['returns'][i-167:i+1].std(0,ddof=1),1e-6)
    return market, original, manifest
