export const START=1792411200000, INITIAL=100000;
export const OURS='0x125778065a578d598d88bddaa3fdd635280d0e47';
export const FUNDER='0x304199b126a2bb77dedda4c608a9f1dadfada521';
export const SYMBOLS=['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY'];
export const displayCoin=c=>({SP500:'S&P500',BRENT:'BRENTOIL',USDEUR:'EURUSD'}[String(c).replace('pas:','')]||String(c).replace('pas:',''));
export const isPas=x=>String(x?.coin||'').startsWith('pas:');
export const num=x=>x===null||x===undefined||x===''?null:Number.isFinite(Number(x))?Number(x):null;
export const sum=(a,f)=>a.reduce((s,x)=>s+(num(f(x))||0),0);
export const data=(w,k)=>w?.[k]?.data;
export function mergeRows(a=[],b=[]){
 const m=new Map();for(const x of [...a,...b]){const key=x.tid!==undefined?[x.coin,x.tid,x.oid,x.side,x.time].join('|'):JSON.stringify(x);m.set(key,x)}
 return [...m.values()].sort((x,y)=>x.time-y.time);
}
export function cashflows(wallet){
 const address=wallet.address.toLowerCase();let net=0,other=0,unknown=0,grants=0;
 for(const r of wallet.ledger?.rows||[]){if(r.time<START-600000)continue;const d=r.delta||{};if(d.token!=='USDP')continue;
  const from=(d.user||'').toLowerCase(),to=(d.destination||'').toLowerCase(),amount=num(d.amount);
  if(from===address&&to===address)continue;
  if(amount===null){unknown++;continue}
  let signed;if(to===address||d.type==='spotGenesis')signed=amount;else if(from===address)signed=-amount;else{unknown++;continue}
  net+=signed;if(from===FUNDER&&to===address&&amount===INITIAL)grants++;else other++;
 }
 return {net,other,unknown,grants,complete:wallet.ledger?.complete===true};
}
export function positionRows(w){
 return (data(w,'position')?.assetPositions||[]).map(x=>x.position).filter(isPas).map(p=>{
  const qty=num(p.szi),notional=num(p.positionValue),entry=num(p.entryPx);
  return {...p,address:w.address,qty,notional,entry,mark:qty?notional/Math.abs(qty):null,
   side:qty>0?'롱':'숏',upnl:num(p.unrealizedPnl),margin:num(p.marginUsed),leverageValue:num(p.leverage?.value),
   leverageType:p.leverage?.type,at:w.position?.at};
 });
}
export function fillRows(w){
 return (w.fills?.rows||[]).filter(f=>isPas(f)&&f.time>=START).map(f=>{
  const qty=num(f.sz),price=num(f.px),before=num(f.startPosition),delta=(f.side==='B'?1:-1)*qty,after=before+delta;
  const closing=before*delta<0?Math.min(Math.abs(before),Math.abs(delta)):0,opening=Math.abs(delta)-closing;
  const action=before*after<0?(before>0?'롱 → 숏':'숏 → 롱'):opening>1e-12?(delta>0?'롱 진입·추가':'숏 진입·추가'):(before>0?'롱 축소·청산':'숏 축소·청산');
  return {...f,address:w.address,qty,price,before,after,notional:qty*price,opening,closing,action,
   initialPct:qty*price/INITIAL*100,afterInitialPct:after*price/INITIAL*100,
   closed:num(f.closedPnl),feeValue:num(f.fee),feeBps:qty*price?Number(f.fee)/(qty*price)*10000:null};
 });
}
export function orderRows(w){
 const latest=new Map(),groups=new Map();
 for(const x of data(w,'orders')||[]){const o=x.order;if(!isPas(o)||o.timestamp<START)continue;const key=o.coin+'|'+o.oid;const prior=latest.get(key);
  if(!prior||x.statusTimestamp>prior.statusTimestamp||x.statusTimestamp===prior.statusTimestamp&&x.status!=='open')latest.set(key,x)}
 for(const o of data(w,'openOrders')||[]){if(!isPas(o)||o.timestamp<START)continue;const key=o.coin+'|'+o.oid;if(!latest.has(key))latest.set(key,{order:o,status:'open',statusTimestamp:o.timestamp})}
 for(const f of fillRows(w)){const key=f.coin+'|'+f.oid;const g=groups.get(key)||[];g.push(f);groups.set(key,g)}
 const keys=new Set([...latest.keys(),...groups.keys()]);
 return [...keys].map(key=>{const h=latest.get(key),o=h?.order,fs=groups.get(key)||[],orig=num(o?.origSz),filled=sum(fs,f=>f.qty),notional=sum(fs,f=>f.notional);
  return {address:w.address,coin:o?.coin||fs[0]?.coin,oid:o?.oid||fs[0]?.oid,side:o?.side||fs[0]?.side,
   submitted:o?.timestamp??null,firstFill:fs.length?Math.min(...fs.map(f=>f.time)):null,lastFill:fs.length?Math.max(...fs.map(f=>f.time)):null,
   original:orig,filled,fillPct:orig>0?filled/orig*100:null,notional,average:filled?notional/filled:null,limit:num(o?.limitPx),
   status:h?.status||'주문 원본 미확인',orderType:o?.orderType||'—',tif:o?.tif||'—',pieces:fs.length,
   waitSeconds:o?.timestamp&&fs.length?(Math.min(...fs.map(f=>f.time))-o.timestamp)/1000:null,
   complete:w.fills?.complete===true&&!w.fills?.retention,reduceOnly:o?.reduceOnly??null};
 }).sort((a,b)=>(b.submitted||b.lastFill)-(a.submitted||a.lastFill));
}
export function summarize(w){
 const positions=positionRows(w),fills=fillRows(w),mode=data(w,'mode'),spot=data(w,'spot'),perp=data(w,'position');
 const coin=spot?.balances?.find(x=>x.coin==='USDP');
 let equity=null;
 if(spot&&perp){if(mode==='unifiedAccount')equity=num(coin?.total)||0;else if(['disabled','default','standard'].includes(mode))equity=(num(coin?.total)||0)+(num(perp.marginSummary?.accountValue)||0)}
 const flows=cashflows(w),pnl=equity!==null&&flows.complete&&!flows.unknown?equity-flows.net:null;
 const long=sum(positions,p=>p.qty>0?p.notional:0),short=sum(positions,p=>p.qty<0?p.notional:0),gross=long+short;
 const closed=sum(fills,f=>f.closed),fees=sum(fills,f=>f.feeToken==='USDP'?f.feeValue:0),funding=sum((w.funding?.rows||[]).filter(r=>r.time>=START&&isPas(r.delta)),r=>r.delta.usdc),upnl=sum(positions,p=>p.upnl);
 const histComplete=['fills','funding','ledger'].every(k=>w[k]?.complete)&&!w.fills?.retention;
 const recon=histComplete?closed-fees+funding+upnl:null;
 const volume=sum(fills,f=>f.notional),maker=sum(fills,f=>f.crossed?0:f.notional),orders=orderRows(w);
 const known=orders.filter(o=>o.original>0),den=sum(known,o=>o.original*(o.average??o.limit??0));
 return {address:w.address,name:w.address===OURS?'SFINX':null,positions,fills,orders,mode,equity,pnl,
  returnPct:pnl!==null?pnl/INITIAL*100:null,comparable:flows.complete&&!flows.unknown&&flows.grants===1&&flows.other===0,
  long,short,gross,net:long-short,longPct:gross?long/gross*100:null,shortPct:gross?short/gross*100:null,
  exposurePct:equity>0?gross/equity*100:null,netPct:equity>0?(long-short)/equity*100:null,effectiveLeverage:equity>0?gross/equity:null,
  margin:sum(positions,p=>p.margin),marginPct:equity>0?sum(positions,p=>p.margin)/equity*100:null,
  leverages:[...new Set(positions.map(p=>p.leverageValue))].sort((a,b)=>a-b),closed,fees,funding,upnl,reconstructed:recon,
  residual:pnl!==null&&recon!==null?pnl-recon:null,volume,makerPct:volume?maker/volume*100:null,
  lastFill:fills.length?Math.max(...fills.map(f=>f.time)):null,filledOrders:new Set(fills.map(f=>f.coin+'|'+f.oid)).size,
  openOrders:(data(w,'openOrders')||[]).filter(isPas).length,flows,histComplete,knownOrderCount:known.length,
  orderFillValuePct:den>0?sum(known,o=>o.filled*(o.average??o.limit??0))/den*100:null,
  at:Math.min(w.position?.at||0,w.spot?.at||0),historyAt:w.fills?.at||0,errors:w.errors||[]};
}
export function assetPnl(w){
 const s=summarize(w),m=new Map();const row=c=>{if(!m.has(c))m.set(c,{coin:c,closed:0,fees:0,funding:0,upnl:0,volume:0,fills:0});return m.get(c)};
 for(const f of s.fills){const r=row(f.coin);r.closed+=f.closed;r.fees+=f.feeToken==='USDP'?f.feeValue:0;r.volume+=f.notional;r.fills++}
 for(const x of w.funding?.rows||[])if(x.time>=START&&isPas(x.delta))row(x.delta.coin).funding+=Number(x.delta.usdc);
 for(const p of s.positions)row(p.coin).upnl=p.upnl;
 return [...m.values()].map(r=>({...r,net:r.closed-r.fees+r.funding+r.upnl})).sort((a,b)=>b.net-a.net);
}
export function priorMove(candles,time,hours){
 const rows=candles.filter(c=>c.T<time).sort((a,b)=>a.t-b.t);const latest=rows.at(-1);if(!latest||time-latest.T>3600000)return null;
 const prior=rows.filter(c=>c.T<=latest.T-hours*3600000).at(-1);
 return prior&&latest.T-prior.T<=(hours+1)*3600000?(Number(latest.c)/Number(prior.c)-1)*100:null;
}
export function officialRows(seed){
 const chart=new Map((seed.official?.chart?.data?.teams||[]).map(x=>[x.name,x]));
 return (seed.official?.leaderboard?.data?.teams||[]).map(t=>({...t,live:chart.get(t.name)||null}));
}
