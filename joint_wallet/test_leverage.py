import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from competition_live.profile import load_profile
from .book import JointBook, ReconciliationError
from .leverage import DEFAULT, choose, decide, load_policy, leverage_value


def gateway_class():
    from .pas_gateway import PASGateway
    return PASGateway


def setup(own=100., other=0., current=1, delta=500.):
    p = load_profile(); b = JointBook(p['trading']['symbols'], {'ours': 50000., 'teammate': 50000.})
    for name, qty in [('ours', own), ('teammate', other)]:
        b.owners[name].qty[0] = qty; b.owners[name].entry[0] = 100.
    b.physical.qty[0] = own + other; b.physical.entry[0] = 100.
    plan = dict(owner='ours', coin='pas:BTC', delta=delta, price=100., version=0,
                expected_physical=own+other, expected_owned=own, cloid='test', reduce_only=False)
    marks = np.full(10, 100.); lev = [1]*10; lev[0] = current
    return p, b, plan, marks, lev


class MarginDecisionTests(unittest.TestCase):
    def decision(self, **kw):
        p,b,plan,marks,lev = setup(**kw)
        return choose(plan=plan, book=b, marks=marks, leverages=lev, exchange_max=40, policy=DEFAULT)

    def test_selects_two_when_one_cannot_fund_fixed_quantity(self):
        r=self.decision(); self.assertEqual(r['selected_leverage'],2)
        self.assertEqual(r['delta_unchanged'],500.); self.assertFalse(r['resized_positions'])

    def test_small_target_selects_one(self):
        self.assertEqual(self.decision(delta=10.)['selected_leverage'],1)

    def test_short_symmetric(self):
        self.assertEqual(self.decision(own=-100.,delta=-500.)['selected_leverage'],2)

    def test_shared_setting_is_never_changed(self):
        r=self.decision(other=-100.,current=2)
        self.assertEqual(r['selected_leverage'],2);self.assertFalse(r['changed'])
        with self.assertRaisesRegex(ReconciliationError,'No permitted leverage'):
            self.decision(other=-100.,current=1)

    def test_existing_teammate_budget_breach_does_not_block_our_reduction(self):
        r=self.decision(own=100.,other=-800.,delta=-10.)
        self.assertEqual(r['selected_leverage'],1)
        self.assertTrue(r['shared_market_locked'])

    def test_our_capital_not_lent_from_teammate(self):
        p,b,plan,marks,lev=setup();b.owners['teammate'].cash=1000000.
        self.assertEqual(choose(plan=plan,book=b,marks=marks,leverages=lev,exchange_max=40,policy=DEFAULT)['selected_leverage'],2)

    def test_five_is_a_hard_choice_cap(self):
        with self.assertRaisesRegex(ReconciliationError,'No permitted leverage'):
            self.decision(delta=2000.)

    def test_four_and_five_are_available_without_resizing(self):
        for delta, expected in [(1300., 4), (1800., 5)]:
            r=self.decision(delta=delta)
            self.assertEqual(r['selected_leverage'],expected)
            self.assertEqual(r['delta_unchanged'],delta)
            self.assertFalse(r['resized_positions'])

    def test_exchange_ceiling_still_takes_precedence(self):
        p,b,plan,marks,lev=setup(delta=1800.)
        with self.assertRaisesRegex(ReconciliationError,'No permitted leverage'):
            choose(plan=plan,book=b,marks=marks,leverages=lev,exchange_max=4,policy=DEFAULT)

    def test_contract_prevents_silent_config_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.assertIsNone(load_policy(p))
            (p/'team_execution_contract.json').write_text(json.dumps({'required_leverage_policy':DEFAULT['mode']}))
            with self.assertRaisesRegex(ReconciliationError,'missing'):load_policy(p)
            (p/'leverage_policy.json').write_text(json.dumps(DEFAULT));self.assertEqual(load_policy(p),DEFAULT)

    def test_invalid_mode_or_precision_fails(self):
        for v in [{'type':'isolated','value':1},{'type':'cross','value':1.5}]:
            with self.assertRaises(ReconciliationError):leverage_value(v)


class GatewayIntegrationTests(unittest.TestCase):
    def gateway(self, *, current=1, delta=500., client=True, response=None, other=0.):
        p,b,plan,marks,lev=setup(current=current,delta=delta,other=other)
        state=dict(orders=[],positions=dict(time=int(time.time()*1000),assetPositions=[dict(position=dict(
            coin='pas:BTC',szi=str(b.physical.qty[0]),leverage={'type':'cross','value':current}))]))
        meta=dict(collateralToken=p['collateral_token'],universe=[dict(name='pas:'+s,maxLeverage=40) for s in b.symbols])
        ctx=[dict(markPx='100') for _ in b.symbols]
        active=dict(leverage={'type':'cross','value':current},maxTradeSzs=['10000','10000'])
        self.calls=[];self.events=[];self.state=state
        def update(level,coin,is_cross):
            self.calls.append((level,coin,is_cross))
            if response is not None:return response
            active['leverage']={'type':'cross','value':level}
            return {'status':'ok'}
        public=SimpleNamespace(profile=p,account=lambda:copy.deepcopy(self.state),market=lambda:(meta,ctx),post=lambda q:copy.deepcopy(active))
        client_obj=SimpleNamespace(base_url=p['rest_url'],account_address=p['team_wallet'],orders_enabled=True,update_leverage=update)
        gw=gateway_class()(p,public,client_obj if client else None,leverage_policy=DEFAULT,audit=self.events.append)
        return gw,b,plan,copy.deepcopy(active),dict(plan=plan,state=state,meta=meta,contexts=ctx)

    def test_live_boundary_sets_then_confirms_without_modifying_order(self):
        gw,b,plan,a,kw=self.gateway();old=copy.deepcopy(plan)
        got=gw.leverage_for_entry(plan['coin'],b,a,**kw)
        self.assertEqual(got['leverage']['value'],2);self.assertEqual(plan,old)
        self.assertEqual(self.calls,[(2,'pas:BTC',True)])
        self.assertEqual(self.events[-1]['kind'],'leverage_update_confirmed')

    def test_existing_one_needs_no_setting_transaction(self):
        gw,b,plan,a,kw=self.gateway(delta=10.)
        gw.leverage_for_entry(plan['coin'],b,a,**kw)
        self.assertEqual(self.calls,[])

    def test_readonly_cannot_mutate_leverage(self):
        gw,b,plan,a,kw=self.gateway(client=False)
        with self.assertRaisesRegex(ReconciliationError,'read-only'):gw.leverage_for_entry(plan['coin'],b,a,**kw)
        self.assertEqual(self.calls,[])

    def test_rejected_setting_does_not_mark_order_preflight_complete(self):
        gw,b,plan,a,kw=self.gateway(response={'status':'err'})
        with self.assertRaisesRegex(ReconciliationError,'rejected'):gw.leverage_for_entry(plan['coin'],b,a,**kw)
        self.assertEqual(gw.checked,{})

    def test_stale_or_changed_inventory_prevents_update(self):
        gw,b,plan,a,kw=self.gateway();self.state['positions']['assetPositions'][0]['position']['szi']='101'
        with self.assertRaisesRegex(ReconciliationError,'Inventory changed'):gw.leverage_for_entry(plan['coin'],b,a,**kw)
        self.assertEqual(self.calls,[])

    def test_shared_market_stays_as_is(self):
        gw,b,plan,a,kw=self.gateway(current=2,other=-100.)
        gw.leverage_for_entry(plan['coin'],b,a,**kw);self.assertEqual(self.calls,[])

    def test_inspection_never_changes_settings(self):
        gw,b,plan,a,kw=self.gateway(current=2)
        r=gw.inspect_current_leverage(b)
        self.assertEqual(self.calls,[]);self.assertEqual(r['setting_updates'],0)
        self.assertEqual(r['decisions'][0]['selected_leverage'],1)

    def test_complete_preflight_uses_dynamic_selection_and_preserves_size(self):
        gw,b,plan,a,kw=self.gateway(); original=copy.deepcopy(plan)
        replacements=dict(phase_state=lambda *a:'trading',account_equity=lambda *a:{'equity':100000.},
            prepare_pas_order=lambda *a,**k:SimpleNamespace(size='500.0',price='100.0'),check_oi=lambda *a:None)
        with patch.dict(gw.preflight.__globals__,replacements):gw.preflight(plan,b)
        self.assertEqual(self.calls,[(2,'pas:BTC',True)])
        self.assertEqual(plan,original);self.assertIn(plan['cloid'],gw.checked)

    def test_physical_reduction_bypasses_leverage_mutation(self):
        gw,b,plan,a,kw=self.gateway(delta=-10.);plan['reduce_only']=True
        replacements=dict(phase_state=lambda *a:'trading',account_equity=lambda *a:{'equity':100000.},
            prepare_pas_order=lambda *a,**k:SimpleNamespace(size='10.0',price='100.0'),check_oi=lambda *a:None)
        with patch.dict(gw.preflight.__globals__,replacements):gw.preflight(plan,b)
        self.assertEqual(self.calls,[]);self.assertIn(plan['cloid'],gw.checked)


if __name__=='__main__':unittest.main()
