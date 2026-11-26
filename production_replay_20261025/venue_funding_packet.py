"""Port the already-declared Oct22 venue funding rule to production replay.

Research envelopes only. Original publication is immutable and retained. The
legacy field name is retained for API compatibility; audit metadata declares
that its value is an estimate of PAS settled funding, not reference funding.
"""
import copy
import numpy as np
from competition_live.runtime import fingerprint
from pas_performance_research.funding import forecast,filtered_anchor


class VenueFundingPacket:
    def __init__(self,funding,start,*,filter_core=True):
        self.start=start;self.filter_core=filter_core;self.cache={}
        self.observations=[dict(known=r['time']/1000+60,symbol=r['coin'][4:],rate=r['rate']) for r in funding]

    def __call__(self,packet,now):
        if now<self.start:return packet
        key=packet['sha256']
        if key not in self.cache:
            p=copy.deepcopy(packet);body=p['body'];f=body['frame'];source=f['source_time']
            carry,count=forecast([source],self.observations,f['symbols'],horizon=168,decay=False)
            if self.filter_core:
                anchor,guard=filtered_anchor(np.asarray(f['anchor_weights']),np.asarray(f['protected'],bool),
                    np.asarray(f['price_forecast_168h']),carry[0],'venue_funding')
                f['anchor_weights']=anchor.tolist();f['protected']=guard.tolist()
            f['reference_funding_forecast_168h']=carry[0].tolist()
            body['research_funding_input']=dict(venue='pas',horizon=168,filter_core=self.filter_core,
                counts=count[0].tolist(),known_through=source,
                rule='Unchanged Oct22 EWMA6h/24h, minimum6observations, confidence24, full expected payments/half receipts in allocator. Weekly persistence is an assumption, not observed cash.',
                publication_delay_assumption_seconds=60)
            p['sha256']=fingerprint(body);self.cache[key]=p
        return self.cache[key]
