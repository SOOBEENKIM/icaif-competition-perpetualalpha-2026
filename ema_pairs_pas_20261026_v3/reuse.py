"""Reuse unchanged controls only after checking forecast/market identity."""
import json,shutil
from .protocol import *

def main():
 parent=ROOT/'reports/ema_pairs_pas_20261026_v2';a=parent/'development_data';b=OUT/'development_data'
 compared=['packets.json','dataset.json','funding.json']
 compared += [str(p.relative_to(a)) for p in (a/'raw').glob('*.json')]
 for f in compared:assert json.loads((a/f).read_text())==json.loads((b/f).read_text()),f
 copied=[]
 for arm in ARMS[:4]:
  if arm!='incumbent':assert json.loads((a/(arm+'_forecasts.json')).read_text())==json.loads((b/(arm+'_forecasts.json')).read_text())
  for sc in SCENARIOS:
   name=f'{arm}_{sc}_development'
   for folder,ext in [('rows','.json'),('runs','.json.gz')]:
    src=parent/folder/(name+ext);dst=OUT/folder/src.name;dst.parent.mkdir(exist_ok=True)
    shutil.copyfile(src,dst);copied.append(dict(source=str(src.relative_to(ROOT)),destination=str(dst.relative_to(ROOT)),sha256=sha(src)))
 save(OUT/'reused_control_runs.json',dict(unchanged_arms=ARMS[:4],condition_comparisons=12,compared_inputs=compared,files=copied,
  rationale='Only two new forecast branches differ. Same native engine, market packets, fees, funding, risk, initial state and original4 arm forecast bytes. Not12 new experiments.'))

if __name__=='__main__':main()
