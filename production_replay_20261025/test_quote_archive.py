import copy
import unittest
import numpy as np
from . import test_quote_history as fixtures
from .quote_archive import make_record,validate_record,merge_features
from .quote_history import quote_history


class QuoteArchiveTests(unittest.TestCase):
    cfg=dict(book_max_age_ms=15000,context_max_age_ms=3000)
    def test_compact_archive_reproduces_full_received_stream_feature(self):
        full=fixtures.QuoteHistoryTests().stream()
        records=[make_record(full,['BTC'],t,self.cfg,15000) for t in [3600,14400]]
        merged=merge_features({},records,['BTC'],14400,self.cfg,15000)
        np.testing.assert_array_equal(quote_history(full,['BTC'],14400,self.cfg)[0],quote_history(merged,['BTC'],14400,self.cfg)[0])

    def test_record_cannot_be_backdated_or_misidentified(self):
        full=fixtures.QuoteHistoryTests().stream();r=make_record(full,['BTC'],14400,self.cfg,15000)
        with self.assertRaises(ValueError):validate_record(r,['BTC'],self.cfg,14999)
        bad=copy.deepcopy(r);bad['values']['BTC']['book']['coin']='par:BTC'
        with self.assertRaises(ValueError):validate_record(bad,['BTC'],self.cfg,15000)

    def test_stale_archive_cannot_be_marked_as_fresh(self):
        r=make_record(fixtures.QuoteHistoryTests().stream(),['BTC'],14400,self.cfg,15000)
        r['values']['BTC']['ctx_received']-=10000
        with self.assertRaises(ValueError):validate_record(r,['BTC'],self.cfg,15000)

    def test_merge_does_not_change_live_stream(self):
        full=fixtures.QuoteHistoryTests().stream();saved=copy.deepcopy(full)
        r=make_record(full,['BTC'],14400,self.cfg,15000)
        merge_features(full,[r],['BTC'],14400,self.cfg,15000)
        self.assertEqual(full,saved)


if __name__=='__main__':unittest.main()
