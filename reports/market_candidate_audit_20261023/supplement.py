"""Additional public read-only market freshness and grant roster check."""
import json,time
from pathlib import Path
import collect as c
root=c.RAW/'current';root.mkdir(exist_ok=True)
def get(payload):
 time.sleep(4);return c.api(payload)
m=get({'type':'metaAndAssetCtxs','dex':'pas'});c.save(root/'market.json',m)
for u in m['response'][0]['universe']:
 c.save(root/(u['name'].replace(':','_')+'.json'),get({'type':'l2Book','coin':u['name']}))
f=get({'type':'userNonFundingLedgerUpdates','user':'0x304199b126a2bb77dedda4c608a9f1dadfada521','startTime':c.START-600000,'endTime':int(time.time()*1000)});c.save(root/'funder.json',f)
print('COMPLETE',c.now())
