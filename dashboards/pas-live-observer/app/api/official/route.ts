const ORIGIN='https://d23m2l9bjnbgul.cloudfront.net/public/';
export async function GET(request:Request){
 const name=new URL(request.url).searchParams.get('name');
 if(!['chart','leaderboard'].includes(name||''))return Response.json({error:'Unknown official source'},{status:400});
 try{const r=await fetch(ORIGIN+name+'.json?t='+Math.floor(Date.now()/30000),{signal:AbortSignal.timeout(20000)});
  if(!r.ok)return Response.json({error:'Official source HTTP '+r.status},{status:502});
  return Response.json({data:await r.json(),at:Date.now()},{headers:{'Cache-Control':'private, max-age=15'}});
 }catch(e){return Response.json({error:String(e)},{status:502})}
}
