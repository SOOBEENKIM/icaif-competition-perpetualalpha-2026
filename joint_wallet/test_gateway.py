import unittest
from types import SimpleNamespace
from competition_live.profile import load_profile
from .book import JointBook,ReconciliationError
from .pas_gateway import PASGateway
from .test_book import fill


class GatewayTests(unittest.TestCase):
    def test_unused_market_can_be_set_to_1x_without_disabling_other_markets(self):
        p=load_profile();b=JointBook(p['trading']['symbols'],{'ours':50000.,'teammate':50000.});calls=[]
        ready=dict(leverage={'type':'cross','value':1},maxTradeSzs=['100','100'])
        public=SimpleNamespace(post=lambda _:ready)
        def update(*a,**kw):calls.append((a,kw));return {'status':'ok'}
        client=SimpleNamespace(base_url=p['rest_url'],account_address=p['team_wallet'],orders_enabled=True,update_leverage=update)
        gateway=PASGateway(p,public,client)
        self.assertEqual(gateway.leverage_for_entry('pas:SP500',b,dict(leverage={'type':'cross','value':20})),ready)
        self.assertEqual(calls,[((1,'pas:SP500'),{'is_cross':True})])

    def test_net_flat_opposing_owners_do_not_allow_leverage_change(self):
        p=load_profile();b=JointBook(p['trading']['symbols'],{'ours':50000.,'teammate':50000.})
        b.apply_fill('teammate',fill(1,'A',1,0));b.apply_fill('ours',fill(2,'B',1,-1))
        gateway=PASGateway(p,None)
        with self.assertRaisesRegex(ReconciliationError,'owner is exposed'):
            gateway.leverage_for_entry('pas:ETH',b,dict(leverage={'type':'cross','value':20}))

    def test_unsigned_preview_cannot_change_unused_market_settings(self):
        p=load_profile();b=JointBook(p['trading']['symbols'],{'ours':50000.,'teammate':50000.})
        with self.assertRaisesRegex(ReconciliationError,'read-only gateway'):
            PASGateway(p,None).leverage_for_entry('pas:SP500',b,dict(leverage={'type':'cross','value':20}))


if __name__=='__main__':unittest.main()
