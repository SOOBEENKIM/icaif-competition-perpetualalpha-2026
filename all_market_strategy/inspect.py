"""Produce the all-market candidate from fresh, verified inference inputs.

Read-only comparison. No exchange client, signer, runtime checkpoint mutation,
or writing to the active strategy's signal path.
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np

from competition_live.profile import ROOT, load_profile
from competition_live.producer import read_reference
from .signals import build, SYMBOLS


def main():
    data, incumbent, digest=read_reference(ROOT,load_profile())
    pointer=json.loads((ROOT/'practice_live/latest.json').read_text())
    with np.load(ROOT/pointer['run']/'targets.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['times'],data['market_times'])
        forecast=z['chronos_forecast'].copy(); probability=z['frozen_regime_probability'].copy()
    weights,parts=build(data,forecast,probability)
    result=dict(candidate='all_market_trend_reversion_v1',mode='read_only',orders_sent=0,
        source_time=int(data['market_times'][-1]),source_hash=digest,
        source_utc=datetime.fromtimestamp(int(data['market_times'][-1]),timezone.utc).isoformat(),
        assets=[dict(symbol=s,allowed=True,incumbent_weight=float(incumbent[-1,j]),
                     proposed_weight=float(weights[-1,j]),ema_state=float(parts['ema'][-1,j]),
                     channel_weight=float(parts['channel'][-1,j]),reversion_state=float(parts['reversion'][-1,j]),
                     carry_weight=float(parts['carry'][-1,j])) for j,s in enumerate(SYMBOLS)])
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
