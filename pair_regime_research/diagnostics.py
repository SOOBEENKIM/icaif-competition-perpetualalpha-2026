"""Fixed-source PAS markouts and replay parity; never fit to PAS outcomes."""
import copy
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .study import ROOT, OUT, INPUT, read, save, sha, verify
from pair_regime_adaptive.study import OUT as SECOND, verify as verify_second
from auto_pair_selection.selection import SYMBOLS, PAIRS, IDS

AUDIT = ROOT / 'reports/market_candidate_audit_20261023'
OLD = ROOT / 'reports/pair_model_research_20261023/wave9'


def csvsave(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows):
    result = dict(observations=len(rows), pairs=len({r['pair'] for r in rows}),
                  source_hours=len({r['source'] for r in rows}))
    if not rows:
        return dict(result, accuracy=None, mae_bps=None, zero_mae_bps=None,
                    directed_gross_bps=None, fee_only_bps=None)
    y = np.array([r['real_pas_relative_return'] for r in rows])
    p = np.array([r['mu'] for r in rows])
    directed = np.sign(p) * y
    return dict(result, accuracy=float(np.mean(p*y > 0)),
                mae_bps=float(np.mean(abs(p-y))*1e4),
                zero_mae_bps=float(np.mean(abs(y))*1e4),
                directed_gross_bps=float(directed.mean()*1e4),
                fee_only_bps=float(directed.mean()*1e4-18))


def pas():
    verify(); verify_second()
    d = dict(np.load(INPUT)); t = d['market_times']
    close = np.full_like(d['close'], np.nan)
    inputs = {str(INPUT.relative_to(ROOT)): sha(INPUT)}
    for j, symbol in enumerate(SYMBOLS):
        path = AUDIT/'raw/markets'/f'pas_{symbol}.json'
        inputs[str(path.relative_to(ROOT))] = sha(path)
        data = read(path)['candles']
        received = datetime.fromisoformat(data['received_kst']).timestamp()*1000
        for candle in data['response']:
            end = (int(candle['t'])//3600000 + 1)*3600
            i = int(np.searchsorted(t, end))
            if (i < len(t) and t[i] == end and int(candle['T']) < received
                    and int(candle['n']) > 0):
                close[i, j] = float(candle['c'])
    paths = [OLD/'adaptive_hedge_72.npz']
    paths += [OUT/f'{a}.npz' for a in read(OUT/'protocol.json')['arms']]
    paths += [SECOND/f'{a}.npz' for a in read(SECOND/'protocol.json')['arms']]
    baseline = list(csv.DictReader((AUDIT/'matured_pas_markouts.csv').open(encoding='utf-8-sig')))
    cohort = {(int(r['source']), r['pair']) for r in baseline}
    results = []; allrows = []; current = []
    for path in paths:
        inputs[str(path.relative_to(ROOT))] = sha(path)
        state = dict(np.load(path)); st = state['times']; arm = path.stem; rows = []
        for si, source in enumerate(st):
            if source < 1792411200:  # Scored phase start, 2026-10-19 12:00 UTC.
                continue
            i = int(np.searchsorted(t, source)); end = int(np.searchsorted(t, source+72*3600))
            if end >= len(t) or t[i] != source or t[end] != source+72*3600:
                continue
            for k, ids in enumerate(IDS):
                p = state['mu'][si, k]; b = state['beta'][si, k]
                prices = close[[i, end]][:, list(ids)]
                if not np.isfinite(p+b) or not np.isfinite(prices).all():
                    continue
                relative = prices[1]/prices[0]-1
                y = float((relative[0]-b*relative[1])/(1+b))
                row = dict(arm=arm, source=int(source), maturity=int(t[end]),
                           pair='/'.join(PAIRS[k]), beta=float(b), mu=float(p),
                           real_pas_relative_return=y,
                           permitted=bool(state['economic_action'][si, k] != 0),
                           original_cohort=(int(source), '/'.join(PAIRS[k])) in cohort)
                rows.append(row)
        allrows.extend(rows)
        selected = [r for r in rows if r['permitted']]
        results.append(dict(arm=arm, all_available=metrics(rows),
                            permitted=metrics(selected),
                            original_cohort_available=metrics([r for r in rows if r['original_cohort']]),
                            original_cohort_permitted=metrics([r for r in selected if r['original_cohort']])))
        if arm == 'adaptive_hedge_72':
            assert len(selected) == len(baseline) == 197
            original = read(AUDIT/'candidate_audit.json')['history_summary']
            assert abs(metrics(selected)['mae_bps']-original['mae_bps']) < 1e-9
        last = len(st)-1
        for k, pair in enumerate(PAIRS):
            p = state['mu'][last,k]; b = state['beta'][last,k]
            current.append(dict(arm=arm, source=int(st[last]), pair='/'.join(pair),
                                available=bool(state['economic_available'][last,k]),
                                permitted=bool(state['economic_action'][last,k]),
                                forecast_bps=float(p)*1e4 if np.isfinite(p) else None,
                                beta=float(b) if np.isfinite(b) else None,
                                regime=int(state['regime'][last,k]) if 'regime' in state else None))
    csvsave(OUT/'pas_forecasts.csv', allrows)
    csvsave(OUT/'current_pair_permissions.csv', current)
    save(OUT/'pas_diagnostics.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), results=results,
        horizon_hours=72, original_baseline_reproduced=True,
        limitations=['Overlapping forecasts are not independent trades.',
                     'Trade-close prices, not executable quotes or portfolio PnL.',
                     '18bp is fees only; no spread, depth, funding or lifecycle.',
                     'Available cohorts and hedge ratios differ by model; show coverage alongside errors.',
                     'No PAS labels fitted; the period was previously inspected and is development data.',
                     'Missing or unfinished candles are excluded. Zero permitted predictions is abstention, not proof of accuracy.'],
        inputs=inputs, orders_sent=0, live_changed=False))
    print(json.dumps(results, indent=2), flush=True)


def parity():
    from competition_live.profile import BUNDLE, load_profile
    from pas_performance_research.selected import controller
    from .replay import replay
    verify(); verify_second()
    profile = load_profile(); p = read(OLD/'protocol.json')
    window = next(w for w in read(BUNDLE/'expected.json')['runs'] if
                  w['venue']=='reference' and w['arm']==profile['candidate'] and
                  w['phase']=='continuous' and w['scenario']=='base')
    d = dict(np.load(BUNDLE/'data/reference.npz'))
    state = dict(np.load(OLD/'adaptive_hedge_72.npz')); t=state.pop('times')
    ix=np.searchsorted(t,d['times']); assert np.array_equal(t[ix],d['times'])
    state={k:(v[ix] if v.ndim and len(v)==len(t) else v) for k,v in state.items()}
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'))
    zi=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'))
    fi=np.searchsorted(f['times'],d['times']); raw=dict(np.load(ROOT/p['input']))
    cfg=copy.deepcopy(profile['configs']['reference']); cfg['dense_pairs']=[list(x) for x in PAIRS]
    cfg['conditional_completion']={'risk_aversion':0.}
    c=controller(d['times'],f['forecast'][fi],f['carry'][fi],
                 z['all_market_signed_core_v1'][zi],z['all_market_signed_core_v1_protected'][zi],
                 price_times=raw['market_times'],prices=raw['close'])
    result=replay(d,state,None,cfg,profile['trading'],'first_dense_hourlyhedge_individual_72',
                  'base',window['start'],window['end'],base_policy=c)
    with gzip.open(OLD/'reference/adaptive_hedge_72_continuous_base.json.gz','rt') as fp:
        original=json.load(fp)
    normalized=json.loads(json.dumps(result,default=lambda x:x.tolist(),allow_nan=False))
    checks={key:normalized[key]==value for key,value in original.items()}
    assert all(checks.values()), [k for k,v in checks.items() if not v]
    assert set(normalized)==set(original)
    verify(); verify_second()
    save(OUT/'parity.json',dict(exact_full_result_match=True,checks=checks,
                               message='New lifecycle hook without regime flags exactly reproduces original full-period results, all events, fills, accounting and risk.'))
    print('PASS full-period frozen baseline exact result parity',flush=True)


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
