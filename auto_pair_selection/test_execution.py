import copy
import json
import unittest
from pathlib import Path
import numpy as np
from .selection import PAIRS, IDS
from .staging import expanded_profile, validate_forecast, preview_entries
from experiments.basis_carry_20261017.probe import asof
from experiments.published_pair_lifecycle_20261017.account import BookAccount
from experiments.published_pair_lifecycle_20261017.lifecycle import PairLifecycle


class ExecutionCompatibilityTests(unittest.TestCase):
    def setUp(self):
        profile=expanded_profile();self.p=profile['trading']
        self.cfg=json.loads(Path('competition_live/strategy_plan.json').read_text())['book_protocol']
        self.cfg.update(pair_max_gross=6000.,exit_lead_seconds=6)
        self.sc=dict(self.cfg['scenarios']['base'],impact_bps=0.)
        k=PAIRS.index(('EWY','EWJ'));self.ids=list(IDS[k]);self.marks=np.full(10,100.);self.sigma=np.eye(10)*1e-8
        self.intent=dict(model='decision_gate',pair=list(PAIRS[k]),ids=self.ids,horizon=24,forecast_source=0,
            available_time=0,source_prices=[100.,100.],source_beta=1.,source_vol=.01,mu=.02,
            terminal_prices=[102.,98.],coefficient_fit_source=0,selection_source=0,
            selection_eligible=True,economic_available=True,economic_action=1,stable_pair_index=k,
            direction=1,account_marks=self.marks.tolist())
        self.streams={s:dict(book=[],ctx=[],book_times=[],ctx_times=[],resets=[]) for s in self.intent['pair']}
        self.window(range(5))

    def window(self,seconds,empty_second=False):
        for symbol in self.intent['pair']:
            stream=self.streams[symbol]
            for sec in seconds:
                at=sec*1000;size='0' if empty_second and symbol==self.intent['pair'][1] else '100'
                book=dict(time=at,levels=[[dict(px='99.99',sz=size)],[dict(px='100.01',sz=size)]])
                ctx=dict(oraclePx='100',markPx='100',funding='0')
                stream['book'].append((at,0,book));stream['ctx'].append((at,0,ctx))
            for name in ('book','ctx'):
                stream[name].sort(key=lambda row:row[0]);stream[name+'_times']=[r[0] for r in stream[name]]

    def test_noncrypto_pair_enters_both_legs_and_pays_exit_without_touching_base(self):
        account=BookAccount(50000.,self.p['symbols'])
        account.physical.fill(0,1.,100.,0.,0)
        account.owner_fill('base',0,1.,100.,0.,0,'fixture')
        session=PairLifecycle(account,self.intent,self.cfg,self.p,self.sc,on_time=lambda *args:None)
        session.enter(0,self.streams,self.marks,self.sigma)
        self.assertEqual(session.state,'open')
        self.assertGreater(account.owners['pair'].qty[self.ids[0]],0)
        self.assertLess(account.owners['pair'].qty[self.ids[1]],0)
        # Falling off tomorrow's entry list cannot erase held inventory.
        now=24*3600-6;self.window(range(now,now+5))
        session.advance(now,self.streams,self.marks,self.sigma,forecast=None)
        self.assertEqual(session.state,'closed')
        self.assertFalse(account.owners['pair'].qty.any())
        self.assertEqual(account.owners['base'].qty[0],1.)
        self.assertGreater(account.physical.fees,0.)
        account.reconcile(self.marks)

    def test_bad_asset_mapping_is_rejected(self):
        bad=copy.deepcopy(self.intent);bad['ids']=[0,1]
        with self.assertRaises(ValueError):validate_forecast(bad,0)

    def test_noncrypto_entry_preview_and_current_book_veto(self):
        quotes={s:asof(v,0,self.cfg) for s,v in self.streams.items()}
        args=(quotes,0,self.cfg,self.p,self.sc,np.zeros(10),50000.,self.sigma,self.marks,np.zeros(10))
        preview=preview_entries([self.intent],*args)
        self.assertEqual(preview['selected_intent']['pair'],['EWY','EWJ'])
        self.assertEqual(preview['orders_sent'],0)
        quotes['EWJ']=None
        preview=preview_entries([self.intent],*args)
        self.assertIsNone(preview['selected_intent'])
        self.assertEqual(preview['candidates'][0]['reason'],'unobserved_book')

    def test_active_pair_blocks_replacement_and_duplicate_forecast_rejected(self):
        quotes={s:asof(v,0,self.cfg) for s,v in self.streams.items()}
        args=(quotes,0,self.cfg,self.p,self.sc,np.zeros(10),50000.,self.sigma,self.marks,np.zeros(10))
        result=preview_entries([self.intent],*args,active_pair=('BTC','ETH'))
        self.assertIsNone(result['selected_intent'])
        with self.assertRaises(ValueError):preview_entries([self.intent,self.intent],*args)


if __name__=='__main__':unittest.main()
