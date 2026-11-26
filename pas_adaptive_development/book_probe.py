"""Standardized 1000 USDP entry/paid-exit probes, NOT strategy performance."""
import gzip,json,math
from pathlib import Path
from .study import ROOT,OUT as PARENT,sha,save
from pas_venue_research.shadow import DisplayedLiquidity
from pas_venue_research.capture import SYMBOLS

OUT=ROOT/'reports/pas_execution_probe_20261021'


def main():
    source=PARENT/'book_capture_reconnect';manifest=json.loads((source/'capture_manifest.json').read_text())
    tape=source/'synchronized_public.jsonl.gz';assert sha(tape)==manifest['sha256']
    assert int(manifest['start']//3600)==int(manifest['end']//3600),'Funding interval must be explicitly modeled'
    meta=json.loads((ROOT/'reports/pas_venue_research_20261021/raw/pas_meta.json').read_text())['response'][0]
    decimals={m['name'].split(':')[-1]:m['szDecimals'] for m in meta['universe']}
    plan=dict(source_sha=sha(tape),source_start=manifest['start'],source_end=manifest['end'],
        purpose='Standardized execution mechanics only; forced 120s hold is NOT any tested strategy horizon.',
        notional=1000,latency_seconds=1,holding_seconds=120,book_max_age_seconds=10,
        entry_limit_bps=10,exit_limit_bps=50,source_errors=manifest['errors'],
        scenarios={'base':dict(depth_fraction=.25,fee_bps=9),'stress':dict(depth_fraction=.1,fee_bps=12)},
        directions=[1,-1],orders_sent=0,independent_profitability_test=False)
    save(OUT/'protocol.json',plan)
    rows=[];fills=[]
    for scenario,cfg in plan['scenarios'].items():
        states={};liquidity={(s,side):DisplayedLiquidity(cfg['depth_fraction']) for s in SYMBOLS for side in [1,-1]}
        with gzip.open(tape,'rt') as f:
            for line in f:
                e=json.loads(line);now=e['received']
                if e['network']!='pas' or e['channel']!='l2Book':continue
                b=e['data'];s=b['coin'].split(':')[-1];server=b['time']/1000
                if s not in SYMBOLS or not 0<=now-server<=10 or not all(b['levels']):continue
                for side in [1,-1]:
                    key=s,side;liq=liquidity[key];liq.update(b)
                    def opposite(direction):return float(b['levels'][1 if direction>0 else 0][0]['px'])
                    if key not in states and now>=manifest['start']+30:
                        px=opposite(side);q=math.floor(1000/px*10**decimals[s])/10**decimals[s]
                        states[key]=dict(symbol=s,side=side,phase='entry',decision=now,arrival=now+1,
                            limit=px*(1+side*.001),requested=q,qty=0.,entry_notional=0.,cash_pnl=0.,fees=0.,entry_filled=0.)
                        continue
                    st=states.get(key)
                    if not st or st['phase']=='done':continue
                    if st['phase']=='entry' and server>=st['arrival']:
                        q,px=liq.consume(side,st['requested'],st['limit'])
                        st.update(qty=q,entry_filled=q,entry_notional=q*(px or 0),entry_time=now,phase='hold' if q else 'done')
                        st['fees']+=q*(px or 0)*cfg['fee_bps']/10000
                        if q:fills.append(dict(scenario=scenario,symbol=s,side=side,kind='entry',time=now,qty=q,price=px,decision=st['decision']))
                    if st['phase']=='hold' and now>=st['entry_time']+120:
                        st.update(phase='exit',decision=now,arrival=now+1,limit=opposite(-side)*(1-side*.005))
                        continue
                    if st['phase']=='exit' and server>=st['arrival']:
                        q,px=liq.consume(-side,st['qty'],st['limit'])
                        if q:
                            average=st['entry_notional']/st['entry_filled']
                            st['cash_pnl']+=side*q*(px-average);st['fees']+=q*px*cfg['fee_bps']/10000;st['qty']-=q
                            fills.append(dict(scenario=scenario,symbol=s,side=-side,kind='exit',time=now,qty=q,price=px,decision=st['decision']))
                        if st['qty']<1e-12:st['phase']='done'
                        else:st.update(decision=now,arrival=now+1,limit=opposite(-side)*(1-side*.005))
        for st in states.values():
            rows.append(dict(scenario=scenario,**st,fill_fraction=st['entry_filled']/st['requested'],
                fully_closed=st['qty']<1e-12,net_pnl_closed=st['cash_pnl']-st['fees'] if st['qty']<1e-12 else None))
    assert all(f['decision']<f['time'] for f in fills)
    save(OUT/'results.json',rows);save(OUT/'fills.json',fills)
    save(OUT/'summary.json',dict(probes=len(rows),entered=sum(r['entry_filled']>0 for r in rows),
        partial_entries=sum(0<r['fill_fraction']<.9999 for r in rows),unclosed=sum(not r['fully_closed'] for r in rows),
        timing_pass=True,source_errors=manifest['errors'],orders_sent=0,
        caveat='Historical displayed capacity is counterfactual, no market impact or fill guarantee. No 24/72h strategy-PnL or Q inferred.'))


if __name__=='__main__':main()
