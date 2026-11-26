"""Fresh replay into a NEW research directory; never reuses cached outcomes."""
import argparse
import shutil
from datetime import datetime,timezone
import numpy as np
from .wave9 import OUT,verify
from .horizon_evaluation import job
from .study import read,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['validation','comparison','continuous'],default='continuous');parser.add_argument('--scenario',choices=['base','stress'],default='base');args=parser.parse_args()
    verify();destination=OUT/'reproductions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination.mkdir(parents=True,exist_ok=False)
    for name in ['protocol.json','adaptive_hedge_72.npz']:shutil.copyfile(OUT/name,destination/name)
    result=job(destination,'adaptive_hedge_72',args.phase,args.scenario)
    expected=read(OUT/'rows'/f'adaptive_hedge_72_{args.phase}_{args.scenario}.json')
    keys=['net_pnl','Q','volatility','fees','execution_cost','funding','completed_pairs','risk_breach_bars','accounting_error']
    for key in keys:np.testing.assert_allclose(result[key],expected[key],rtol=0,atol=1e-10)
    save(destination/'check.json',dict(fresh_replay=True,matched_metrics=keys,tolerance=1e-10,not_independent_test=True,orders_sent=0));verify()
    print(destination)


if __name__=='__main__':main()
