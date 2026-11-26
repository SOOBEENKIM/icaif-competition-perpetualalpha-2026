"""Bounded read-only archive of published signals for the frozen shadow test."""
import json,time,hashlib
from datetime import datetime
from .protocol import ROOT,OUT,save

def main():
    contract=json.loads((OUT/'forward_contract.json').read_text())
    end=datetime.fromisoformat(contract['end_utc'].replace('Z','+00:00')).timestamp()+60
    target=OUT/'forward_signals';target.mkdir(exist_ok=True)
    path=ROOT/'.runtime/pas_budget_lifecycle/signal.json';seen=set();count=0
    while time.time()<end:
        try:
            data=path.read_bytes();packet=json.loads(data);digest=hashlib.sha256(data).hexdigest()
            if digest not in seen:
                at=time.time();seen.add(digest);source=packet['body']['frame']['source_time']
                save(target/f'{source}_{digest[:16]}.json',dict(received_at=at,file_sha256=digest,packet=packet))
                count+=1
            save(OUT/'forward_signal_health.json',dict(at=time.time(),count=count,orders_sent=0))
        except (ValueError,OSError,KeyError) as e:
            save(OUT/'forward_signal_health.json',dict(at=time.time(),count=count,error=str(e),orders_sent=0))
        time.sleep(15)
    save(OUT/'forward_signal_finished.json',dict(at=time.time(),count=count,orders_sent=0))

if __name__=='__main__':main()
