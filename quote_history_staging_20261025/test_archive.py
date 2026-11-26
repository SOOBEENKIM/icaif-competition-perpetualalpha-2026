import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from production_replay_20261025.quote_history import quote_history
from .archive import PointCache, read_point, UnflushedBoundary


class ArchiveTests(unittest.TestCase):
    cfg = dict(book_max_age_ms=15000, context_max_age_ms=3000)

    def records(self, point, price=100., epoch=1):
        at = int(point * 1000 - 1000)
        return [dict(received_ms=at, epoch=epoch, channel='l2Book',
                     data=dict(coin='pas:BTC', time=at, levels=[
                         [dict(px=str(price-.1),sz='10')], [dict(px=str(price+.1),sz='10')]])),
                dict(received_ms=at, epoch=epoch, channel='activeAssetCtx',
                     data=dict(coin='pas:BTC', ctx=dict(markPx=str(price), oraclePx=str(price))))]

    def write(self, path, rows, *, flushed=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
            if rows and flushed:
                f.write(json.dumps(dict(channel='capture_note', epoch=1, data={},
                    received_ms=max(r['received_ms'] for r in rows)+2000))+'\n')

    def test_three_points_survive_restart_without_reopening_raw(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw=root/'capture'/'old.gz'; source=200*3600
            points=[source-86400,source-10800,source]
            self.write(raw,[r for p in points for r in self.records(p)])
            cache=PointCache(root/'cache',['BTC'],self.cfg)
            history, hashes=cache.history(source,iter([raw]),now=source+60)
            prices,_=quote_history(history,['BTC'],source,self.cfg)
            np.testing.assert_array_equal(prices,np.full((3,1),100.))
            raw.unlink()
            other=PointCache(root/'cache',['BTC'],self.cfg)
            again,again_hashes=other.history(source,[raw],now=source+120)
            self.assertEqual(history,again); self.assertEqual(hashes,again_hashes)

    def test_revision_cannot_change_a_cached_point_including_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'old.gz';point=100*3600
            self.write(raw,[dict(channel='capture_note',received_ms=point*1000+1,epoch=1,data={})])
            cache=PointCache(root/'cache',['BTC'],self.cfg)
            before=cache.get(point,[raw],now=point+10)
            self.assertIsNone(before['body']['quotes']['BTC'])
            self.write(raw,self.records(point,1000))
            self.assertEqual(before,cache.get(point,[raw],now=point+20))

    def test_future_receive_does_not_complete_a_historical_quote(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'old.gz';point=100*3600;rows=self.records(point)
            rows[1]['received_ms']=point*1000+1
            self.write(raw,rows)
            self.assertIsNone(read_point([raw],point,['BTC'],self.cfg)['quotes']['BTC'])

    def test_independent_capture_epochs_cannot_form_a_quote(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);point=100*3600;a,b=self.records(point)
            first=root/'a'/'old.gz';second=root/'b'/'old.gz'
            self.write(first,[a]);self.write(second,[b])
            self.assertIsNone(read_point([first,second],point,['BTC'],self.cfg)['quotes']['BTC'])

    def test_disconnect_invalidates_prior_book_context(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'old.gz';point=100*3600;rows=self.records(point)
            rows.append(dict(channel='connection',received_ms=point*1000-500,epoch=1,data={'state':'disconnected'}))
            self.write(raw,rows)
            self.assertIsNone(read_point([raw],point,['BTC'],self.cfg)['quotes']['BTC'])

    def test_corruption_and_changed_contract_fail_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'old.gz';point=100*3600;self.write(raw,self.records(point))
            cache=PointCache(root/'cache',['BTC'],self.cfg);cache.get(point,[raw],now=point+10)
            with self.assertRaises(ValueError):
                PointCache(root/'cache',['ETH'],self.cfg).get(point,[raw],now=point+20)
            p=root/'cache'/f'{point}.json';v=json.loads(p.read_text());v['body']['point']+=3600;p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):cache.get(point,[raw],now=point+20)

    def test_future_cache_creation_not_backdated(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'old.gz';point=100*3600;self.write(raw,self.records(point))
            cache=PointCache(root/'cache',['BTC'],self.cfg)
            with self.assertRaises(ValueError):cache.get(point,[raw],now=point-1)
            cache.get(point,[raw],now=point+10)
            with self.assertRaises(ValueError):cache.get(point,[raw],now=point+5)

    def test_conflicting_same_timestamp_and_wrong_venue_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'old.gz';point=100*3600;rows=self.records(point)
            other=copy.deepcopy(rows[0]);other['data']['levels'][0][0]['px']='98'
            self.write(raw,rows+[other])
            with self.assertRaisesRegex(ValueError,'Ambiguous'):
                read_point([raw],point,['BTC'],self.cfg)
            rows[0]['data']['coin']='par:BTC';self.write(raw,rows)
            with self.assertRaisesRegex(ValueError,'venue'):
                read_point([raw],point,['BTC'],self.cfg)

    def test_unflushed_boundary_is_retried_without_permanent_missing_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'old.gz';source=100*3600
            self.write(raw,self.records(source),flushed=False)
            cache=PointCache(root/'cache',['BTC'],self.cfg)
            with self.assertRaises(UnflushedBoundary):cache.get(source,[raw],now=source+10)
            self.assertFalse((root/'cache'/f'{source}.json').exists())
            history, hashes=cache.history(source,[raw],now=source+10)
            self.assertIsNone(hashes[-1])
            prices,_=quote_history(history,['BTC'],source,self.cfg)
            self.assertTrue(np.isnan(prices[-1,0]))
            self.write(raw,self.records(source))
            history, hashes=cache.history(source,[raw],now=source+20)
            self.assertIsNotNone(hashes[-1])
            prices,_=quote_history(history,['BTC'],source,self.cfg)
            self.assertEqual(prices[-1,0],100.)

    def test_future_watermark_cannot_be_used_before_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'old.gz';point=100*3600
            rows=self.records(point)+[dict(channel='capture_note',epoch=1,data={},received_ms=(point+100)*1000)]
            self.write(raw,rows)
            cache=PointCache(root/'cache',['BTC'],self.cfg)
            with self.assertRaisesRegex(ValueError,'Future receive'):
                cache.get(point,[raw],now=point+10)
            self.assertFalse((root/'cache'/f'{point}.json').exists())


if __name__=='__main__':unittest.main()
