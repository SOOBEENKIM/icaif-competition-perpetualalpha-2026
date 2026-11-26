import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .signals import PublishedSignals, validate_envelope, load_snapshot

ROOT=Path(__file__).resolve().parents[2]


def envelope(source, digest='a'*64):
    frame=dict(source_time=int(source), published_time=float(source+2), weights=[0.]*10,
               retention=[1.]*10, sigma=(np.eye(10)*1e-7).tolist(), volume=[1.]*10,
               trades=[1.]*10, source_hash=digest)
    hashed=hashlib.sha256(json.dumps(frame,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return dict(received_ms=int((source+3)*1000), channel='strategySignal',
                data=dict(frame=frame,publication_sha256=hashed,source_run='practice_live/updates/test'))


class SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads((ROOT/'experiments/rolling_feature_pairs_20261017/protocol.json').read_text())
        cls.bias=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/model_protocol.json').read_text())
        cls.p=json.loads((ROOT/'experiments/paired_venue_20261017/artifacts/forward/plan.json').read_text())['trading_protocol']
        cls.models={}
        for pair in cls.bias['pairs']:
            cls.models[tuple(pair),24]=json.loads((ROOT/cls.bias['model_directory']/('_'.join(pair)+'_24.json')).read_text())
        with np.load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz') as z:
            cls.reference={k:z[k].copy() for k in ['market_times','close','volume','trades']}
        cls.day=int(cls.reference['market_times'][-1]//86400*86400)
        cls.expected={}
        for kind,folder in [('direct','rolling_feature_pairs_20261017/artifacts'),('bias','hourly_refresh_pairs_20261017/prepared')]:
            with np.load(ROOT/'experiments'/folder/f'reference_{kind}_24_state.npz') as z:
                cls.expected[kind]={k:z[k].copy() for k in ['mu','signal_times','pairs','source_prices','beta','vol']}

    def instance(self):
        return PublishedSignals(self.p['symbols'],self.bias['pairs'],self.models,self.spec,self.bias)

    def prefix(self, source):
        n=int(np.searchsorted(self.reference['market_times'],source,side='right'))
        return {k:v[:n].copy() for k,v in self.reference.items()}

    def test_published_hour_matches_frozen_batch_direct_and_bias_forecasts(self):
        model=self.instance()
        for source in [self.day,self.day+3600,self.day+7200]:
            out=model.ingest(envelope(source),self.prefix(source))
            self.assertEqual(len(out),4)
            for row in out:
                expected=self.expected[row['model']]
                k=next(i for i,v in enumerate(expected['pairs']) if list(v)==row['ids'])
                ii=np.flatnonzero(expected['signal_times'][:,k]==source)
                self.assertTrue(len(ii))
                self.assertAlmostEqual(row['mu'],expected['mu'][ii[0],k],places=12)
                np.testing.assert_array_equal(row['source_prices'],expected['source_prices'][ii[0],row['ids']])
                self.assertEqual(row['available_time'],source+8)
                self.assertLessEqual(row['coefficient_fit_source'],source)
        self.assertEqual(len(model.audit),1)
        self.assertTrue(all(r['last_maturity']<self.day for r in model.daily.values()))

    def test_later_revision_cannot_refit_or_rewrite_daily_coefficients(self):
        model=self.instance();source=self.day
        original=model.ingest(envelope(source),self.prefix(source));frozen=copy.deepcopy(model.daily)
        revised=self.prefix(source+3600);revised['close'][-500:-300,0]*=1.2
        model.ingest(envelope(source+3600,'b'*64),revised)
        self.assertEqual(model.daily,frozen)
        self.assertEqual(len(model.audit),1)
        self.assertEqual(original[0]['forecast_source'],source)

    def test_missing_midnight_cannot_be_backfilled_from_a_later_snapshot(self):
        model=self.instance();source=self.day+3600
        out=model.ingest(envelope(source),self.prefix(source))
        self.assertEqual({r['model'] for r in out},{'bias'})
        self.assertEqual(model.audit,[])

    def test_duplicate_revised_and_out_of_order_publications(self):
        model=self.instance();source=self.day;event=envelope(source)
        out=model.ingest(event,self.prefix(source))
        self.assertEqual(model.ingest(event,self.prefix(source)),out)
        with self.assertRaisesRegex(ValueError,'revised'):
            model.ingest(envelope(source,'b'*64),self.prefix(source))
        with self.assertRaisesRegex(ValueError,'Out-of-order'):
            model.ingest(envelope(source-3600),self.prefix(source-3600))

    def test_future_snapshot_and_tampered_or_unavailable_publication_rejected(self):
        model=self.instance();source=self.day
        with self.assertRaisesRegex(ValueError,'end exactly'):
            model.ingest(envelope(source),self.prefix(source+3600))
        changed=envelope(source);changed['data']['frame']['source_hash']='b'*64
        with self.assertRaisesRegex(ValueError,'digest'):
            validate_envelope(changed)
        changed=envelope(source);changed['received_ms']=(source+1)*1000
        with self.assertRaisesRegex(ValueError,'unavailable'):
            validate_envelope(changed)

    def test_snapshot_file_hash_and_scoped_path_checked_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'practice_live/updates/test/inputs.npz';path.parent.mkdir(parents=True)
            np.savez_compressed(path,**self.prefix(self.day))
            digest=hashlib.sha256(path.read_bytes()).hexdigest();event=envelope(self.day,digest)
            _,loaded=load_snapshot(root,event)
            self.assertEqual(loaded['market_times'][-1],self.day)
            with path.open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'input digest'):
                load_snapshot(root,event)
            event['data']['source_run']='../outside'
            with self.assertRaisesRegex(ValueError,'outside'):
                load_snapshot(root,event)


if __name__=='__main__':unittest.main()
