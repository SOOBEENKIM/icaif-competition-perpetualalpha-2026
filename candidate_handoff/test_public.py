import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import requests
from .public import ReadBudget,PublicPAS,PublicReadUnavailable
from competition_live.profile import load_profile

class PublicTests(unittest.TestCase):
    def test_shared_instances_obey_weighted_rolling_limit(self):
        now=[1000.];slept=[]
        def sleep(s):slept.append(s);now[0]+=s
        with tempfile.TemporaryDirectory() as d:
            a=ReadBudget(Path(d)/'quota.json',limit=100,burst=40,clock=lambda:now[0],sleep=sleep)
            b=ReadBudget(Path(d)/'quota.json',limit=100,burst=40,clock=lambda:now[0],sleep=sleep)
            stamps=[]
            for i in range(13):
                (a if i%2 else b).acquire(20);stamps.append(now[0])
            self.assertTrue(slept)
            for t in stamps:self.assertLessEqual(sum(20 for at in stamps if t-60<at<=t),100)
            a.cooldown(61);before=now[0];b.acquire(2);self.assertGreaterEqual(now[0]-before,61)

    def test_429_is_one_http_attempt_and_shared_cooldown(self):
        class Budget:
            def __init__(self):self.weights=[];self.delays=[]
            def acquire(self,w):self.weights.append(w)
            def cooldown(self,s):self.delays.append(s)
        budget=Budget();api=PublicPAS(load_profile(),budget=budget)
        response=requests.Response();response.status_code=429;response.headers['Retry-After']='90'
        with patch('competition_live.venue.PublicPAS.post',side_effect=requests.HTTPError(response=response)) as post:
            with self.assertRaises(PublicReadUnavailable):api.post(dict(type='userFunding'))
            self.assertEqual(post.call_count,1)
        self.assertEqual(budget.delays,[90]);self.assertEqual(api.session.get_adapter('https://').max_retries.total,0)

    def test_response_weight_is_charged_for_history(self):
        class Budget:
            def __init__(self):self.weights=[]
            def acquire(self,w):self.weights.append(w)
        budget=Budget();api=PublicPAS(load_profile(),budget=budget)
        with patch('competition_live.venue.PublicPAS.post',return_value=[{}]*41):api.post(dict(type='userFunding'))
        self.assertEqual(budget.weights,[20,3])

if __name__=='__main__':unittest.main()
