import copy
import json
from pathlib import Path
import unittest
import numpy as np
from .account import BookAccount
from .forward import TapeWindow, FundingClock, assert_ready, metrics
from .portfolio import Portfolio
from . import test_lifecycle as fixture
from .test_signals import envelope

ROOT=Path(__file__).resolve().parents[2]


def book_events(symbols, seconds, *, size=100., price=100.):
    events=[]
    for sec in seconds:
        for symbol in symbols:
            events.extend([
                dict(received_ms=sec*1000,channel='l2Book',data=dict(coin='par:'+symbol,time=sec*1000,
                    levels=[[dict(px=str(price-.01),sz=str(size))],[dict(px=str(price+.01),sz=str(size))]])),
                dict(received_ms=sec*1000,channel='activeAssetCtx',data=dict(coin='par:'+symbol,
                    ctx=dict(oraclePx=str(price),markPx=str(price),funding='0')))])
    return sorted(events,key=lambda r:r['received_ms'])


class ForwardTests(unittest.TestCase):
    def setUp(self):
        f=fixture.LifecycleTests();f.setUp()
        self.cfg,self.p,self.sc=f.cfg,f.p,f.sc
        self.cfg['scenarios']={'base':{**self.sc,'funding_receipt_multiplier':1.}}
        self.plan=dict(paper_start_unix=0,capture_end_unix=7*86400,end_liquidation_seconds=600,
            book_protocol=self.cfg,trading_protocol=self.p)

    def test_release_guard_runs_before_any_source_or_raw_access(self):
        with self.assertRaisesRegex(ValueError,'remain sealed'):
            assert_ready({'outcome_release_unix':1000},999)

    def test_lookahead_does_not_publish_future_signal_or_quote(self):
        signal=envelope(0);events=book_events(['BTC'],range(0,12))+[signal]
        events.sort(key=lambda r:r['received_ms']);window=TapeWindow(events)
        streams=window.ensure(0)
        self.assertEqual(list(window.publications(0)),[])
        from experiments.basis_carry_20261017.probe import asof
        self.assertEqual(asof(streams['BTC'],0,self.cfg)['received']['book'],0)
        self.assertEqual(list(window.publications(3000)),[signal])
        self.assertGreater(streams['BTC']['book_times'][-1],0)

    def test_funding_boundary_charges_pre_fill_inventory_and_owner_sum_once(self):
        tape=TapeWindow(book_events(self.p['symbols'],range(3598,3603)))
        tape.ensure(3599000)
        rates={3600000:{s:.001 for s in self.p['symbols']}}
        clock=FundingClock(3500,tape,rates,self.p['symbols'],self.cfg,1.)
        account=BookAccount(1e6,self.p['symbols']);account.physical.fill(0,10.,100.,0,3500)
        account.owner_fill('base',0,10.,100.,0,3500,'fixture')
        target=np.zeros(10);target[1]=2.
        packet=account.seal(dict(packet_id='fund',decision_time=3599,source_close=3599,source_available=3599,
            source_prices=[100.]*10,risk_forced=False,owner_targets={'base':account.owners['base'].qty.tolist(),
                'pair':target.tolist()}),self.p,{**self.sc,'impact_bps':0.})
        account.execute_books(packet,tape.streams,np.ones(10)*100,np.eye(10)*1e-7,self.p,self.sc,self.cfg,on_time=clock)
        self.assertAlmostEqual(account.physical.funding,-1.)
        self.assertAlmostEqual(account.owners['pair'].funding,0.)
        clock(3601000,account);self.assertAlmostEqual(account.physical.funding,-1.)

    def test_missing_actual_funding_is_not_replaced_by_zero(self):
        tape=TapeWindow(book_events(self.p['symbols'],range(3598,3603)));tape.ensure(3599000)
        clock=FundingClock(3500,tape,{},self.p['symbols'],self.cfg,1.)
        account=BookAccount(1e6,self.p['symbols']);account.physical.fill(0,1.,100.,0,3500)
        account.owner_fill('base',0,1.,100.,0,3500,'fixture')
        with self.assertRaisesRegex(ValueError,'Missing settled funding'):
            clock(3600000,account)

    def test_stress_receipt_haircut_applies_once_to_physical_account(self):
        tape=TapeWindow(book_events(self.p['symbols'],range(3598,3603)));tape.ensure(3599000)
        rates={3600000:{s:.001 for s in self.p['symbols']}}
        clock=FundingClock(3500,tape,rates,self.p['symbols'],self.cfg,.5)
        account=BookAccount(1e6,self.p['symbols'])
        account.physical.fill(0,-5.,100.,0,3500)
        account.owner_fill('base',0,5.,100.,0,3500,'fixture')
        account.owner_fill('pair',0,-10.,100.,0,3500,'fixture')
        clock(3600000,account)
        self.assertAlmostEqual(account.physical.funding,.25)
        self.assertAlmostEqual(account.owners['base'].funding,-.25)
        self.assertAlmostEqual(account.owners['pair'].funding,.5)
        account.reconcile(np.ones(10)*100)

    def publication(self, now, raw):
        event=envelope(now);event['received_ms']=now*1000
        event['data']['frame']['published_time']=now
        event['data']['incumbent']=raw;event['data']['candidate']=raw
        return event

    def test_baseline_and_pair_share_one_account_and_net_opposite_intents(self):
        tape=TapeWindow(book_events(self.p['symbols'],range(900,912)))
        streams=tape.ensure(900000)
        clock=FundingClock(0,tape,{},self.p['symbols'],self.cfg,1.)
        arm=Portfolio('first_plus_direct','base',self.plan,clock)
        arm.last_decision=0
        raw=np.zeros(10);raw[0]=.001
        # A baseline task is already eligible at this pair's 15-minute clock.
        arm.base_task={'target':np.array([10.]+[0.]*9),'expires':1200}
        signal=self.publication(0,raw.tolist())
        forecasts=[dict(model='direct',pair=['ETH','BTC'],ids=[1,0],direction=1,
            source_beta=1.,source_prices=[100.,100.],terminal_prices=[102.,98.],source_vol=.01,
            forecast_source=0,available_time=8,horizon=24)]
        arm.step(900,streams,signal,forecasts)
        self.assertIsNotNone(arm.pair);self.assertEqual(arm.pair.state,'open')
        self.assertTrue(arm.account.transfers)
        self.assertEqual(arm.account.owners['base'].qty[0],10.)
        self.assertLess(arm.account.physical.qty[0],10.)
        self.assertLess(arm.account.owners['pair'].qty[0],0.)
        arm.account.reconcile(np.ones(10)*100)
        self.assertEqual(arm.snapshots[0]['equity'],1e6)

    def test_terminal_flatten_closes_both_owners_without_new_pair(self):
        end=600;self.plan.update(capture_end_unix=end,end_liquidation_seconds=60)
        tape=TapeWindow(book_events(self.p['symbols'],range(540,552)));streams=tape.ensure(540000)
        clock=FundingClock(0,tape,{},self.p['symbols'],self.cfg,1.)
        arm=Portfolio('first_strategy','base',self.plan,clock)
        arm.account.physical.fill(0,2.,100.,0,1);arm.account.owner_fill('base',0,2.,100.,0,1,'fixture')
        arm.step(540,streams,self.publication(0,[0.]*10),[])
        self.assertFalse(np.any(arm.account.physical.qty))
        self.assertGreater(arm.account.physical.fees,0.)

    def test_daily_q_uses_exact_12utc_equity_and_invalidates_coverage_gaps(self):
        tape=TapeWindow([]);clock=FundingClock(0,tape,{},self.p['symbols'],self.cfg,1.)
        arm=Portfolio('first_strategy','base',self.plan,clock)
        start=43200;end=start+2*86400
        arm.snapshots=[dict(time=t,equity=1e6*(1+.01*(t-start)/86400)) for t in range(start,end+1,30)]
        result=metrics(arm,start,end)
        daily=np.array([.01,1.02/1.01-1]);sigma=np.std(daily,ddof=1)*np.sqrt(2)
        self.assertAlmostEqual(result['financial_score'],.025/(.02+1.5*sigma+.05))
        self.assertFalse(result['eligible']) # no real/paper trades
        arm.snapshots=arm.snapshots[:1]+arm.snapshots[5:]
        result=metrics(arm,start,end)
        self.assertFalse(result['valid']);self.assertIsNone(result['interpretable_financial_score'])

    def test_two_day_received_book_account_lifecycle_and_funding_reconcile(self):
        start=43200;end=start+2*86400
        self.plan.update(paper_start_unix=start,capture_end_unix=end,step_seconds=30)
        def events():
            for base in range(start,end+11,30):
                offsets=sorted({0,1,3,8,*([29] if (base+30)%3600==0 else [])})
                for offset in offsets:
                    at=base+offset
                    for symbol in self.p['symbols']:
                        price=100.+(at-start)*(.00001 if symbol=='ETH' else (-.00001 if symbol=='BTC' else 0.))
                        yield from book_events([symbol],[at],price=price)
        tape=TapeWindow(events())
        rates={t*1000:{s:(.0002 if s=='BTC' else .0001) for s in self.p['symbols']}
               for t in range(start+3600,end+1,3600)}
        clock=FundingClock(start,tape,rates,self.p['symbols'],self.cfg,1.)
        arm=Portfolio('first_plus_direct','base',self.plan,clock)
        publication=None;forecasts=[];last_source=-1
        for now in range(start,end+1,30):
            streams=tape.ensure(now*1000);source=now//3600*3600
            if source!=last_source and now>=source+8:
                raw=np.zeros(10);raw[0]=.001
                publication=self.publication(source,raw.tolist())
                prices=[100+(source-start)*.00001,100-(source-start)*.00001]
                forecasts=[dict(model='direct',pair=['ETH','BTC'],ids=[1,0],source_beta=1.,
                    source_prices=prices,terminal_prices=(np.array(prices)*[1.02,.98]).tolist(),
                    source_vol=.01,forecast_source=source,available_time=source+8,horizon=24)]
                last_source=source
            arm.step(now,streams,publication,forecasts)
        result=metrics(arm,start,end)
        self.assertTrue(result['valid'],result['fatal'])
        self.assertGreaterEqual(result['completed_pairs'],1)
        self.assertEqual(result['completed_days'],2)
        self.assertFalse(result['residual_inventory'])
        self.assertGreater(result['fees'],0.)
        self.assertGreater(len(arm.account.physical.funding_events),20)
        self.assertLess(result['maximum_ownership_error'],1e-6)
        self.assertTrue(all(f['decision_time']<f['time'] for f in arm.account.physical.fills))
        self.assertTrue(all(s.state=='closed' for s in arm.sessions))
        for session in arm.sessions:
            if session.closed_reason=='forecast_maturity':
                closing=[e for e in session.events if e['kind']=='close']
                self.assertTrue(closing)
                self.assertTrue(all(o.get('arrival_ms',0)<session.maturity*1000
                                    for e in closing for o in e['result']['orders']))


if __name__=='__main__':unittest.main()
