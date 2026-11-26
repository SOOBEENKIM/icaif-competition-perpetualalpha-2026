"""Public funded-wallet activity; never infer real identities from addresses."""
import concurrent.futures
import json
import time
from .capture import OUT,query,save


def main():
    ledger=json.loads((OUT/'raw/funder_ledger.json').read_text())['response']
    grants=[r for r in ledger if r['delta'].get('type')=='spotTransfer'
            and r['delta'].get('token')=='USDP' and float(r['delta'].get('amount',0))==100000
            and r['time']>=1792411200000]
    wallets=sorted({r['delta']['destination'].lower() for r in grants})
    cutoff=int(time.time()*1000);results=[]
    jobs=[]
    for wallet in wallets:
        jobs += [(wallet+'_positions','pas',dict(type='clearinghouseState',dex='pas',user=wallet)),
                 (wallet+'_fills_0','pas',dict(type='userFillsByTime',user=wallet,startTime=1792411200000,endTime=cutoff))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(query,job):job[0] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:results.append(dict(name=future.result(),ok=True))
            except Exception as error:results.append(dict(name=futures[future],ok=False,error=str(error)))
    # Overlap timestamps and deduplicate; a saturated single timestamp is not
    # evidence of completeness and is disclosed instead of silently skipped.
    saturated=[]
    for wallet in wallets:
        path=OUT/'raw'/f'{wallet}_fills_0.json'
        if not path.exists():continue
        rows=json.loads(path.read_text())['response'];page=0
        while len(rows)>=2000 and page<20:
            cursor=max(r['time'] for r in rows)
            if min(r['time'] for r in rows)==cursor:
                saturated.append(wallet);break
            page+=1
            name=wallet+f'_fills_{page}'
            query((name,'pas',dict(type='userFillsByTime',user=wallet,startTime=cursor,endTime=cutoff)))
            rows=json.loads((OUT/'raw'/f'{name}.json').read_text())['response']
        if len(rows)>=2000 and page>=20:saturated.append(wallet)
    save(OUT/'participants_capture.json',dict(cutoff_ms=cutoff,wallets=wallets,
        grants=len(grants),unique_grant_recipients=len(wallets),funder_ledger_rows=len(ledger),
        funder_ledger_may_be_truncated=len(ledger)>=500,results=results,
        saturated_wallets=saturated,orders_sent=0))
    print(json.dumps(dict(wallets=len(wallets),requests=len(jobs),saturated=saturated,
                         failures=[r for r in results if not r['ok']])),flush=True)


if __name__=='__main__':
    main()
