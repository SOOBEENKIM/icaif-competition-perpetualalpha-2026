"""Record worker heartbeat timestamps only, with no trading outcome fields."""
import json,time
from .protocol import ROOT,OUT,save


def main():
    end=1792915560;rows=[];seen=set()
    source=ROOT/'.runtime/pas_budget_lifecycle/status.json'
    while time.time()<end:
        r=json.loads(source.read_text());key=(r.get('at'),r.get('state'))
        if key not in seen:
            seen.add(key)
            rows.append(dict(observed_at=time.time(),published_at=r.get('at'),state=r.get('state'),
                signal_source=r.get('signal_source'),fatal_count=len(r.get('fatal',[]))))
            save(OUT/'worker_heartbeat_timestamps.json',dict(note='Status publication intervals, not exact order evaluation timestamps. No PnL or quantities read into this log.',rows=rows,orders_sent=0))
        time.sleep(5)


if __name__=='__main__':main()
