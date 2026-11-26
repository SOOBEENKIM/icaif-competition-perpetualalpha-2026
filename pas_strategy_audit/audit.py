"""Evidence-only review of frozen deployment and public datasets."""
import json,collections,hashlib,datetime
import numpy as np
from .data import ROOT,OUT,load
from competition_live.runtime import private_load
from competition_live.profile import verify_deployment,verify_bundle
from replay_lab.competition_rebuild.causal_risk import normalize_risk

def main():
 cfg,d=load();raw=OUT/'raw'
 def loadraw(name):return json.loads((raw/(name+'.json')).read_text())['data']
 state=private_load(ROOT/'.runtime/pas_competition_live/coordinator.pkl')['portfolio']
 events=[e for e in state.events if e['time']*1000<=cfg['cutoff_ms']]
 statuses=[]
 with (ROOT/'.runtime/pas_competition_live/account_observations.jsonl').open() as f:
  for line in f:
   x=json.loads(line)
   if x.get('snapshot_time',0)<=cfg['cutoff_ms']:statuses.append(x)
 actual=max(statuses,key=lambda x:x['snapshot_time'])
 own={'observation_time_ms':actual['snapshot_time'],'pnl':actual['team_pnl_before_tick'],'return_percent':actual['team_return_percent_before_tick'],'equity':actual['equity_before_tick'],'fees':actual['fees'],'funding_received':actual['funding'],'qty':actual['qty'],'fill_fragments':actual['fills'],'pair_state':actual['pair_state']}
 ctx={m['name']:c for m,c in zip(loadraw('pas_context')[0]['universe'],loadraw('pas_context')[1])}
 quality=[];start=int(np.searchsorted(d['times'],cfg['start_ms']+60000))
 for j,s in enumerate(cfg['symbols']):
  ids=np.flatnonzero(d['active'][:,j]&(d['times']>cfg['start_ms']));first,last=ids[[0,-1]]
  book=loadraw('pas_book_'+s);bid,ask=[float(level[0]['px']) for level in book['levels']]
  rates=[x['rate'] for x in d['funding_events'] if x['j']==j]
  quality.append({'symbol':s,'first_trade_minute_ms':int(d['times'][first]),'last_trade_minute_ms':int(d['times'][last]),'first_observed_price':float(d['open'][first,j]),'last_trade_price':float(d['close'][last,j]),'first_to_last_trade_return_percent':float((d['close'][last,j]/d['open'][first,j]-1)*100),'active_minutes':len(ids),'total_minutes':len(d['times'])-start,'no_trade_minutes_percent':float(100*(1-len(ids)/(len(d['times'])-start))),'final_trade_age_minutes':float(d['age_ms'][-1,j]/60000),'volume_notional_proxy':float(np.nansum(d['volume'][start:,j]*d['close'][start:,j])),'book_observation_ms':json.loads((raw/('pas_book_'+s+'.json')).read_text())['received_ms'],'bid':bid,'ask':ask,'spread_bps':(ask-bid)/((ask+bid)/2)*10000,'top_bid_notional':bid*float(book['levels'][0][0]['sz']),'top_ask_notional':ask*float(book['levels'][1][0]['sz']),'settled_funding_sum_bps':sum(rates)*10000})
 baseline=[e for e in events if e['kind']=='baseline_packet'];checks=collections.Counter(e.get('reason') for e in events if e['kind']=='entry_check')
 times=1775001600+np.arange(1441)*3600;base=np.zeros((1441,10));base[:,0]=.04;reduced=base*.5;sigma=np.repeat((np.eye(10)*1e-6)[None,:,:],1441,axis=0)
 restored=normalize_risk(base,reduced,sigma,times)
 risk_probe={'raw':float(base[-1,0]),'risk_filtered':float(reduced[-1,0]),'after_risk_matching':float(restored[-1,0]),'applies_to':'Current v10_frozen baseline normalization; this example is not a measured cause of current PnL.'}
 leaderboard=loadraw('leaderboard');top=sorted([x for x in leaderboard['teams'] if x['return'] is not None],key=lambda x:-x['return'])[0]
 peer_summaries=[]
 for peer in json.loads((OUT/'peer_details.json').read_text()):
  fills=[f for f in peer['fills'] if f['coin'].startswith('pas:')];funds=[x for x in peer['funding'] if x['delta']['coin'].startswith('pas:')]
  assert len(peer['fills'])<2000 and len(peer['funding'])<500
  total=sum(float(f['sz'])*float(f['px']) for f in fills);fees=sum(float(f['fee']) for f in fills)
  positions={x['position']['coin']:x['position'] for x in peer['positions']['assetPositions']}
  asset=[]
  for coin in sorted({f['coin'] for f in fills}):
   fs=[f for f in fills if f['coin']==coin];fp=sum(float(x['delta']['usdc']) for x in funds if x['delta']['coin']==coin);p=positions.get(coin,{})
   price_realized=sum(float(f['closedPnl']) for f in fs);fee=sum(float(f['fee']) for f in fs);unrealized=float(p.get('unrealizedPnl',0))
   asset.append({'symbol':coin[4:],'fills':len(fs),'directions':dict(collections.Counter(f['dir'] for f in fs)),'realized_price_pnl':price_realized,'unrealized_price_pnl':unrealized,'fees':fee,'funding':fp,'net_component':price_realized+unrealized-fee+fp,'current_qty':float(p.get('szi',0)),'current_notional':float(p.get('positionValue',0))})
  peer_summaries.append({'wallet':peer['wallet'],'balance_observed_ms':peer['at_ms'],'usdp_total':peer['usdp_total'],'funding_grants':[x for x in peer['ledger'] if x['delta'].get('token')=='USDP'],'fill_fragments':len(fills),'orders':len({f['oid'] for f in fills}),'maker_fill_fragments':sum(not f['crossed'] for f in fills),'maker_notional_fraction':sum(float(f['sz'])*float(f['px']) for f in fills if not f['crossed'])/total,'effective_fee_bps':fees/total*10000,'gross_notional':sum(x['current_notional'] for x in asset),'fees':fees,'funding':sum(x['funding'] for x in asset),'assets':asset,'official_team_name':None,'identity_note':'Public leaderboard omits wallet addresses. Similar balance trajectory and rank are insufficient to prove a name/address mapping. Balance, fills and positions were fetched sequentially; mark changes can cause a small reconciliation difference.'})
 report={'period':cfg,'actual':own,'pair_entry_reasons':dict(checks),'baseline_packets':baseline,'market_quality':quality,'risk_normalization_probe':risk_probe,'top_pnl_daily_snapshot':top,'top_snapshot_generated':leaderboard['generated_at'],'peers':peer_summaries,'deployment_files_verified':len(verify_deployment()['files']),'source_bundle_hash':verify_bundle(),'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [ROOT/'pas_strategy_audit',ROOT/'all_market_strategy'] for p in folder.glob('*.py')}}
 (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({'actual':own,'pair_entry_reasons':dict(checks),'deployment_files_verified':report['deployment_files_verified'],'peer_count':len(peer_summaries),'risk_probe':risk_probe},indent=2))
if __name__=='__main__':main()
