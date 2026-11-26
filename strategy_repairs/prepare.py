import hashlib
import json
import numpy as np
from competition_live.profile import ROOT, BUNDLE
from replay_lab.common_comparison.engine import risk_values
from all_market_strategy.signals import SYMBOLS
from .protocol import OUT, freeze
from .model import build


def main():
    spec=freeze();pointer=spec['pointer']
    for name,digest in spec['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    d=dict(np.load(ROOT/pointer['inputs'],allow_pickle=False))
    old=dict(np.load(ROOT/pointer['run']/'targets.npz',allow_pickle=False))
    forecast=old['chronos_forecast'];probability=old['frozen_regime_probability']
    archive={'times':d['market_times']};traces=[];source_checks={}
    for name,arm in spec['arms'].items():
        w,part=build(d,forecast,probability,arm)
        if name=='incumbent':
            np.testing.assert_allclose(w,old['default'],rtol=1e-11,atol=1e-13)
            source_checks['original_max_error']=float(abs(w-old['default']).max())
        if name in ('risk_only','risk_short_exit_all'):
            n=len(w)-48
            prefix,_=build({k:v[:n] for k,v in d.items()},forecast[:n],probability[:n],arm)
            np.testing.assert_allclose(w[:n],prefix,rtol=1e-11,atol=1e-13)
            source_checks[name+'_prefix_error']=float(abs(w[:n]-prefix).max())
        archive[name]=w;archive[name+'_protected']=part['protected']
        archive['sigma']=part['sigma']
        np.savez_compressed(OUT/f'{name}_trace.npz',**part)
        rv=risk_values(w[-1]/d['close'][-1],1.,d['close'][-1],part['sigma'][-1])
        traces.append(dict(variant=name,source_time=int(d['market_times'][-1]),
            final_scale=part['final_scale'],gross_weight=float(abs(w[-1]).sum()),
            target_notional_at_50k=float(abs(w[-1]).sum()*50000),
            predicted_user_daily_volatility=rv['daily_volatility'],
            daily_vol_limit=spec['risk']['daily_volatility'],
            assets=[dict(symbol=s,ema_state=float(part['ema'][-1,j]),
                reversion_state=float(part['reversion_state'][-1,j]),
                raw_base=float(part['raw_base'][-1,j]),
                gate_ratio=float(part['gate_ratio'][-1,j]),
                legacy_sized_trend=float(part['legacy_sized_trend'][-1,j]),
                ungated_sized_trend=float(part['ungated_sized_trend'][-1,j]),
                repaired_sized_trend=float(part['repaired_sized_trend'][-1,j]),
                reversion_weight=float(part['reversion_weight'][-1,j]),
                final_weight=float(w[-1,j]),protected=bool(part['protected'][-1,j]))
                for j,s in enumerate(SYMBOLS)]))
        print(name,'target',round(traces[-1]['target_notional_at_50k'],2),flush=True)
    np.savez_compressed(OUT/'targets.npz',**archive)
    (OUT/'target_trace.json').write_text(json.dumps(traces,indent=2))
    (OUT/'causality.json').write_text(json.dumps(source_checks,indent=2))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'strategy_repairs').glob('*.py'))}
    (OUT/'frozen_source.json').write_text(json.dumps(hashes,indent=2))


if __name__=='__main__':
    main()
