import {rememberIdentitySnapshot} from './team-matching.mjs';
import {START,OURS,FUNDER,INITIAL,mergeRows,summarize} from './domain.mjs';
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function cache(write){
 return new Promise(resolve=>{try{const req=indexedDB.open('pas-public-observer',1);req.onupgradeneeded=()=>req.result.createObjectStore('data');req.onerror=()=>resolve(null);req.onsuccess=()=>{const db=req.result,tx=db.transaction('data',write?'readwrite':'readonly'),s=tx.objectStore('data'),r=write?s.put(write,'latest'):s.get('latest');r.onsuccess=()=>resolve(r.result);r.onerror=()=>resolve(null);tx.oncomplete=()=>db.close()}}catch{resolve(null)}});
}
export async function restore(seed){const old=await cache();if(!old)return seed;seed.identityObservations=old.identityObservations||{};for(const [a,w] of Object.entries(old.wallets||{})){if(!seed.wallets[a])seed.wallets[a]=w;else for(const [k,v]of Object.entries(w))if(v?.at>(seed.wallets[a][k]?.at||0))seed.wallets[a][k]=v}for(const k of ['chart','leaderboard'])if(old.official?.[k]?.at>seed.official?.[k]?.at)seed.official[k]=old.official[k];return seed}
export class LiveData{
 constructor(seed,changed){this.seed=seed;this.changed=changed;this.selected=OURS;this.queue=[];this.running=false;this.closed=false;this.paused=false;this.errors=[];this.intervals=[];this.last=0;this.selectedBusy=false;}
 emit(){this.changed()}
 fail(e,label){this.errors.unshift({at:Date.now(),text:label+': '+e.message});this.errors=this.errors.slice(0,8);this.emit()}
 async info(body,priority=5){return new Promise((resolve,reject)=>{this.queue.push({body,priority,resolve,reject});this.queue.sort((a,b)=>a.priority-b.priority);this.pump()})}
 async pump(){if(this.running)return;this.running=true;while(this.queue.length&&!this.closed){const q=this.queue.shift();await delay(Math.max(0,1800-(Date.now()-this.last)));this.last=Date.now();try{const r=await fetch('/api/info',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(q.body),signal:AbortSignal.timeout(26000)});const v=await r.json();if(!r.ok){if(r.status===429)this.last=Date.now()+60000;throw Error(v.error||'HTTP '+r.status)}q.resolve(v)}catch(e){q.reject(e)}}this.running=false}
 async resource(w,key,type,priority){w[key]=await this.info({type,user:w.address},priority);this.emit()}
 async history(w,key,type,priority=5){const old=w[key]||{rows:[],through:START,complete:true},cap=type==='userFillsByTime'?2000:500,end=Date.now();let cursor=Math.max(START-600000,(old.through||START)-1000),rows=[],done=false;
  for(let i=0;i<12;i++){const r=await this.info({type,user:w.address,startTime:cursor,endTime:end,aggregateByTime:false},priority);if(!Array.isArray(r.data))throw Error('History response invalid');rows=mergeRows(rows,r.data);if(r.data.length<cap){done=true;break}const next=Math.max(...r.data.map(x=>x.time));if(next<=cursor)break;cursor=next}
  w[key]={rows:mergeRows(old.rows,rows),through:done?end:cursor,complete:old.complete!==false&&done,retention:old.retention||key==='fills'&&rows.length>=10000,at:Date.now()};this.emit();
 }
 async refreshWallet(address,full=false){const w=this.seed.wallets[address];if(!w)return;const p=address===this.selected?1:5;try{
  if(!w.mode||Date.now()-w.mode.at>600000)await this.resource(w,'mode','userAbstraction',p);
  await this.resource(w,'position','clearinghouseState',p);await this.resource(w,'spot','spotClearinghouseState',p);await this.history(w,'fills','userFillsByTime',p);
  if(full||!w.ledger||Date.now()-w.ledger.at>600000)await this.history(w,'ledger','userNonFundingLedgerUpdates',p);
  if(full){await this.history(w,'funding','userFunding',p);const orders=await this.info({type:'historicalOrders',user:address},p);const m=new Map((w.orders?.data||[]).map(o=>[o.order.coin+'|'+o.order.oid,o]));for(const o of orders.data)m.set(o.order.coin+'|'+o.order.oid,o);w.orders={...orders,data:[...m.values()]};await this.resource(w,'openOrders','frontendOpenOrders',p)}
  rememberIdentitySnapshot(this.seed,summarize(w));this.emit();
 }catch(e){this.fail(e,address.slice(0,10))}}
 async refreshSelected(){if(this.selectedBusy||this.paused||this.closed)return;const address=this.selected;this.selectedBusy=true;await this.refreshWallet(address,true);this.selectedBusy=false;if(address!==this.selected)this.refreshSelected()}
 async official(){if(this.paused)return;await Promise.all(['chart','leaderboard'].map(async name=>{try{const r=await fetch('/api/official?name='+name,{signal:AbortSignal.timeout(24000)}),v=await r.json();if(!r.ok)throw Error(v.error);this.seed.official[name]=v;this.emit()}catch(e){this.fail(e,'공식 '+name)}}))}
 async market(){if(this.paused)return;try{this.seed.market=await this.info({type:'metaAndAssetCtxs'},1);this.emit()}catch(e){this.fail(e,'시장')}}
 async candles(coin){try{const r=await this.info({type:'candleSnapshot',req:{coin,interval:'1h',startTime:START-86400000,endTime:Date.now()}},2);return r.data}catch(e){this.fail(e,'가격 이력');return null}}
 async discover(){const w={address:FUNDER};try{await this.history(w,'ledger','userNonFundingLedgerUpdates',8);for(const r of w.ledger.rows){const d=r.delta||{},a=d.destination?.toLowerCase();if(d.user?.toLowerCase()===FUNDER&&d.token==='USDP'&&Number(d.amount)===INITIAL&&a&&!this.seed.wallets[a])this.seed.wallets[a]={address:a,grants:[r],errors:[]}}this.emit()}catch(e){this.fail(e,'지급 지갑 갱신')}}
 async background(){while(!this.closed){if(this.paused){await delay(1000);continue}const candidates=Object.values(this.seed.wallets).filter(w=>w.address!==this.selected).sort((a,b)=>(a.position?.at||0)-(b.position?.at||0));if(candidates.length){const w=candidates[0];await this.refreshWallet(w.address,!w.orders||Date.now()-w.orders.at>1200000)}await delay(200)}}
 start(){this.refreshSelected();this.official();this.market();this.background();this.intervals=[setInterval(()=>this.refreshSelected(),45000),setInterval(()=>this.official(),60000),setInterval(()=>this.market(),60000),setInterval(()=>cache(this.seed),60000),setInterval(()=>this.discover(),3600000)];}
 stop(){this.closed=true;this.intervals.forEach(clearInterval);for(const q of this.queue)q.reject(Error('Closed'));this.queue=[];cache(this.seed)}
}
