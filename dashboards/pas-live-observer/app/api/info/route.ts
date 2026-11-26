const types=new Set(['userAbstraction','clearinghouseState','spotClearinghouseState','frontendOpenOrders','historicalOrders','userFillsByTime','userFunding','userNonFundingLedgerUpdates','metaAndAssetCtxs','recentTrades','candleSnapshot']);
export async function POST(request:Request){
 try{
  if(Number(request.headers.get('content-length'))>5000)return Response.json({error:'Request too large'},{status:400});
  const raw=await request.json();
  if(!raw||typeof raw!=='object')return Response.json({error:'Invalid request'},{status:400});
  const b=raw as {type:string;user?:string;dex?:string;coin?:string;req?:{coin:string;interval:string}};
  if(!types.has(b.type))return Response.json({error:'Read-only endpoint'},{status:400});
  if(b.user&&!/^0x[0-9a-fA-F]{40}$/.test(b.user))return Response.json({error:'Invalid address'},{status:400});
  if(['clearinghouseState','frontendOpenOrders','metaAndAssetCtxs'].includes(b.type))b.dex='pas';
  if(b.type==='recentTrades'&&!/^pas:[A-Z0-9]+$/.test(b.coin||''))return Response.json({error:'PAS markets only'},{status:400});
  if(b.type==='candleSnapshot'&&(!/^pas:[A-Z0-9]+$/.test(b.req?.coin||'')||!['1h','1m'].includes(b.req?.interval||'')))return Response.json({error:'Invalid candle request'},{status:400});
  const r=await fetch('https://api.hyperliquid-testnet.xyz/info',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b),signal:AbortSignal.timeout(22000)});
  if(!r.ok)return Response.json({error:'Hyperliquid HTTP '+r.status,retryAfter:r.status===429?60:5},{status:r.status===429?429:502});
  return Response.json({data:await r.json(),at:Date.now()},{headers:{'Cache-Control':'no-store'}});
 }catch(e){return Response.json({error:String(e)},{status:502})}
}
