"""Bounded numerical jobs, not autonomous research agents."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,sys
from .protocol import *
from .study import run_one,verify

def main():
 later='--later' in sys.argv;verify()
 arms=['incumbent',json.loads((OUT/'selection.json').read_text())['candidate']] if later else ARMS
 tasks=[(a,s,later,False) for a in arms for s in SCENARIOS]
 if later:tasks += [(a,s,True,True) for a in arms for s in SCENARIOS]
 tasks=[t for t in tasks if not (OUT/'rows'/f'{t[0]}_{t[1]}_{"switch" if t[3] else "later" if t[2] else "development"}.json').exists()]
 with ProcessPoolExecutor(max_workers=2) as pool:
  pending=[pool.submit(run_one,*task) for task in tasks]
  for f in as_completed(pending):f.result()
 verify()

if __name__=='__main__':main()
