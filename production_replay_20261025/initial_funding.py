"""Public funding schedule before the frozen test, no account or outcome read."""
from competition_live.profile import load_profile
from .retrospective import request
from .protocol import OUT,save


def main():
    start=1792897200000;counts={}
    for symbol in load_profile()['trading']['symbols']:
        rows=request(dict(type='fundingHistory',coin='pas:'+symbol,startTime=start,endTime=start+59999),
                     OUT/'initial_funding'/f'{symbol}.json')
        assert all(start<=r['time']<start+60000 for r in rows)
        counts[symbol]=len(rows)
    save(OUT/'initial_funding/manifest.json',dict(counts=counts,period='03UTC before prospective start03:05',orders_sent=0))
    print(counts)


if __name__=='__main__':main()
