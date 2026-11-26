import {OURS,START} from './domain.mjs';
const WINDOW=30*60000,BUCKET=30*60000;
const average=a=>a.reduce((s,v)=>s+v,0)/a.length;
export function nearestPoint(rows,at){
 let l=0,r=rows.length;while(l<r){const m=(l+r)>>1;if(rows[m].at<at)l=m+1;else r=m}
 return [rows[l-1],rows[l]].filter(Boolean).sort((a,b)=>Math.abs(a.at-at)-Math.abs(b.at-at))[0]||null;
}
let cached;
function series(evidence,chart,board){
 if(cached?.evidence===evidence&&cached.chart===chart&&cached.board===board)return cached.rows;
 const maps=Object.fromEntries(Object.entries(evidence.teams||{}).map(([n,rows])=>[n,new Map(rows.map(r=>[r.at,r]))]));
 const put=(name,row)=>{(maps[name]??=new Map()).set(row.at,row)};
 if(chart?.start&&Date.parse(chart.start)===START)for(const t of chart.teams||[]){
  t.values?.forEach((v,i)=>{if(v!==null)put(t.name,{at:START+i*chart.step_s*1000,returnPct:v*100,kind:'chart',source:'live official chart'})});
  if(t.pnl!==null&&t.last_at)put(t.name,{at:Date.parse(t.last_at),returnPct:t.pnl*100,kind:'latest',source:'live official chart'});
 }
 if(board?.generated_at&&Date.parse(board.from)===START){const at=new Date(board.generated_at);at.setUTCHours(12,0,0,0);if(at.getTime()>Date.parse(board.generated_at))at.setUTCDate(at.getUTCDate()-1);for(const t of board.teams||[])if(t.return!==null)put(t.name,{at:at.getTime(),returnPct:t.return*100,kind:'daily',source:'live official daily standing'})}
 const rows=Object.fromEntries(Object.entries(maps).map(([n,m])=>[n,[...m.values()].sort((a,b)=>a.at-b.at)]));cached={evidence,chart,board,rows};return rows;
}
export function compareTrack(walletRows,teamRows){
 const buckets=new Map();
 for(const w of walletRows){const t=nearestPoint(teamRows,w.at);if(!t||Math.abs(t.at-w.at)>WINDOW)continue;const p={walletAt:w.at,teamAt:t.at,walletReturn:w.returnPct,teamReturn:t.returnPct,errorPp:Math.abs(w.returnPct-t.returnPct),lagMinutes:Math.abs(t.at-w.at)/60000,source:w.source,teamSource:t.source};const k=Math.floor(w.at/BUCKET),old=buckets.get(k);if(!old||p.lagMinutes<old.lagMinutes)buckets.set(k,p)}
 const points=[...buckets.values()].sort((a,b)=>a.walletAt-b.walletAt);if(!points.length)return null;
 const mae=average(points.map(p=>p.errorPp));const changes=points.slice(1).map((p,i)=>Math.abs((p.walletReturn-points[i].walletReturn)-(p.teamReturn-points[i].teamReturn)));
 return {points,count:points.length,days:new Set(points.map(p=>new Date(p.walletAt).toISOString().slice(0,10))).size,maePp:mae,changeErrorPp:changes.length?average(changes):null,score:mae+(changes.length?.25*average(changes):0),lastAt:points.at(-1).walletAt};
}
export function commonComparison(a,b){
 const bm=new Map(b.points.map(p=>[Math.floor(p.walletAt/BUCKET),p]));
 const pairs=a.points.map(p=>[p,bm.get(Math.floor(p.walletAt/BUCKET))]).filter(p=>p[1]);
 if(pairs.length<2)return null;
 return {count:pairs.length,a:average(pairs.map(p=>p[0].errorPp)),b:average(pairs.map(p=>p[1].errorPp))};
}
export function inferTeams(seed,evidence,summaries){
 if(!evidence||!seed)return {};
 const tracks=series(evidence,seed.official?.chart?.data,seed.official?.leaderboard?.data),result={};
 for(const s of summaries){
  if(s.address===OURS){result[s.address]={name:'SFINX',kind:'confirmed',label:'확인됨',candidates:[]};continue}
  const points=[...(evidence.wallets?.[s.address]||[]),...(seed.identityObservations?.[s.address]||[])];
  const w=seed.wallets?.[s.address];if(s.comparable&&s.returnPct!==null&&w?.position&&w?.spot&&Math.abs(w.position.at-w.spot.at)<120000)points.push({at:s.at,returnPct:s.returnPct,source:'live wallet account'});
  let candidates=Object.entries(tracks).filter(([name])=>name!=='SFINX').map(([name,rows])=>{const c=compareTrack(points,rows);return c?{name,...c}:null}).filter(Boolean);
  const multi=candidates.filter(c=>c.count>=2);if(multi.length)candidates=multi;
  // Compare rivals on common times. A team with just two quiet-day samples must
  // not win merely because another team also exposes volatile intraday samples.
  for(const c of candidates)c.wins=0;
  for(let i=0;i<candidates.length;i++)for(let j=i+1;j<candidates.length;j++){
   const c=commonComparison(candidates[i],candidates[j]);if(!c)continue;
   if(c.a<c.b-.0001){candidates[i].wins++;candidates[j].wins--}
   else if(c.b<c.a-.0001){candidates[j].wins++;candidates[i].wins--}
  }
  candidates.sort((a,b)=>b.wins-a.wins||a.score-b.score||b.count-a.count||a.name.localeCompare(b.name));
  if(!candidates.length){result[s.address]={kind:'missing',label:'대조 자료 없음',candidates:[]};continue}
  const best=candidates[0],runner=candidates[1],shared=runner?commonComparison(best,runner):null,gap=shared?shared.b-shared.a:runner?runner.score-best.score:null,ambiguous=runner&&gap<=Math.max(.003,(shared?.a??best.score)*.35);
  const label=ambiguous?'후보 경합':best.count===1?'단일 시점 후보':best.maePp>.15?'후보 · 오차 큼':'추정';
  result[s.address]={...best,kind:'inferred',label,ambiguous:!!ambiguous,candidates:candidates.slice(0,3),gapPp:gap};
 }
 const names=new Map();for(const [a,r]of Object.entries(result))if(r.kind==='inferred'){const ids=names.get(r.name)||[];ids.push(a);names.set(r.name,ids)}
 for(const r of Object.values(result))if(r.kind==='inferred'&&(names.get(r.name)?.length||0)>1){r.conflictingWallets=names.get(r.name).length;if(r.maePp<=.15)r.label='중복 후보'}
 return result;
}
export function rememberIdentitySnapshot(seed,s){
 const w=seed.wallets[s.address];if(!s.comparable||s.returnPct===null||Math.abs(w.position.at-w.spot.at)>120000)return;
 const obs=seed.identityObservations??={},rows=obs[s.address]??=[];const last=rows.at(-1);
 if(last&&s.at-last.at<10*60000)return;
 rows.push({at:s.at,returnPct:s.returnPct,source:'browser public observation'});obs[s.address]=rows.slice(-500);
}
