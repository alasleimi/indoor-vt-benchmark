// Browser implementation of science/matching.py, checked against Python fixtures.
const indices = n => Array.from({length:n}, (_, i) => i);
function subset(cost) {
  const m=cost.length, n=cost[0].length;
  const dp=Array.from({length:m+1}, () => new Float64Array(n+1).fill(Infinity));
  const take=Array.from({length:m+1}, () => new Uint8Array(n+1)); dp[0].fill(0);
  for(let i=1;i<=m;i++) for(let j=1;j<=n;j++) {
    const choose=dp[i-1][j-1]+cost[i-1][j-1];
    if(choose<=dp[i][j-1]) {dp[i][j]=choose;take[i][j]=1;} else dp[i][j]=dp[i][j-1];
  }
  const result=new Array(m); let i=m,j=n;
  while(i) {if(take[i][j]) {result[i-1]=j-1;i--;}j--;}
  return result;
}
function windows(a,p) {
  const tiny=Number.EPSILON*Math.max(a.at(-1)-a[0],...a.map(Math.abs),1)*16;
  return a.map((v,i)=>indices(2*p+1).map(k=>{
    const j=i+k-p;
    return (j<0?a[0]+j*tiny:j>=a.length?a.at(-1)+(j-a.length+1)*tiny:a[j])-v;
  }));
}
export function delta(a,b,p,alpha,normalize=false,separation=-1,tolerance=0) {
  if(alpha===1)p=0;
  const wa=windows(a,p),wb=windows(b,p),width=2*p+1;
  const divisor=normalize&&p?width*(width-1)/2:1;
  return a.map((v,i)=>b.map((w,j)=>{
    if(separation>=0&&(Math.abs(v-w)>separation+tolerance||separation>v+w+tolerance))return Infinity;
    const d=wa[i].map((x,k)=>x-wb[j][k]).sort((x,y)=>x-y);
    let pattern=0;
    for(let k=1;k<width;k++)pattern+=k*(width-k)*(d[k]-d[k-1]);
    return alpha*Math.abs(v-w)+(1-alpha)*pattern/divisor;
  }));
}
export function match(a0,b0,p,separation=-1) {
  let ia=indices(a0.length).sort((i,j)=>a0[i]-a0[j]),ib=indices(b0.length).sort((i,j)=>b0[i]-b0[j]);
  if(p.unify&&ia.length&&ib.length) {
    if(ia.length>ib.length)ia=subset(ib.map(j=>ia.map(i=>Math.abs(b0[j]-a0[i])))).map(i=>ia[i]);
    else if(ib.length>ia.length)ib=subset(ia.map(i=>ib.map(j=>Math.abs(a0[i]-b0[j])))).map(j=>ib[j]);
  }
  const left_indices=ia.slice(),right_indices=ib.slice(),trace=[],matches=[];
  const distance=p.triangle_tolerance_m==null?-1:separation;
  const allow=p.matcher==='CrossingAllowedMPUrgeMatcher'; let round=0;
  while(ia.length&&ib.length) {
    round++;
    const d=delta(ia.map(i=>a0[i]),ib.map(j=>b0[j]),p.half_window,p.alpha,false,distance,p.triangle_tolerance_m||0);
    const left=d.map(row=>row.indexOf(Math.min(...row)));
    const right=ib.map((_,j)=>{let best=0;for(let i=1;i<ia.length;i++)if(d[i][j]<d[best][j])best=i;return best;});
    const candidates=ia.flatMap((ai,i)=>right[left[i]]===i&&Number.isFinite(d[i][left[i]])?[[d[i][left[i]],ai,ib[left[i]],i,left[i]]]:[]);
    candidates.sort((a,b)=>a[0]-b[0]||left_indices.indexOf(a[1])-left_indices.indexOf(b[1])||right_indices.indexOf(a[2])-right_indices.indexOf(b[2]));
    if(!candidates.length)break;
    const removeA=new Set(),removeB=new Set();
    for(const [cost,ai,bi,i,j] of candidates) {
      removeA.add(i);removeB.add(j);
      const good=allow||!matches.some(([a,b])=>(a0[ai]-a0[a])*(b0[bi]-b0[b])<0);
      trace.push([ai,bi,cost,round,good?1:0]);if(good)matches.push([ai,bi,cost]);
    }
    ia=ia.filter((_,i)=>!removeA.has(i));ib=ib.filter((_,i)=>!removeB.has(i));
  }
  return {matches,trace,left_indices,right_indices};
}
export function quality(a,b,edges) {
  const available=a.source_ids.reduce((s,x)=>s+b.source_ids.filter(y=>x===y).length,0);
  const tp=edges.filter(([i,j])=>a.source_ids[i]===b.source_ids[j]).length,n=edges.length;
  return {tp,fp:n-tp,fn:available-tp,available,precision:n?tp/n:0,recall:available?tp/available:0,f1:n+available?2*tp/(n+available):0};
}
export function pairData(a,b,p,sources) {
  const separation=Math.hypot(...a.xyz.map((v,k)=>v-b.xyz[k]));
  const r=match(a.ranges,b.ranges,p,separation),ia=r.left_indices,ib=r.right_indices;
  const values=new Array(a.ranges.length*b.ranges.length).fill(null),pattern=values.slice();
  if(ia.length&&ib.length) {
    const args=[ia.map(i=>a.ranges[i]),ib.map(j=>b.ranges[j]),p.half_window];
    const dist=p.triangle_tolerance_m==null?-1:separation;
    const d=delta(...args,p.alpha,false,dist,p.triangle_tolerance_m||0),pat=delta(...args,0,false,dist,p.triangle_tolerance_m||0);
    ia.forEach((i,x)=>ib.forEach((j,y)=>{values[i*b.ranges.length+j]=Number.isFinite(d[x][y])?d[x][y]:null;pattern[i*b.ranges.length+j]=Number.isFinite(pat[x][y])?pat[x][y]:null;}));
  }
  const first=r.trace.filter(t=>t[3]===1&&t[4]),final=r.trace.filter(t=>t[4]);
  return {a,b,sources,matrix:{ia,ib,values,pattern,matches:first.map(t=>t.slice(0,2)),rejected:r.trace.filter(t=>t[3]===1&&!t[4]).map(t=>t.slice(0,2))},
    final:{matches:final.map(t=>t.slice(0,2)),rounds:final.map(t=>t[3]),rejected:r.trace.filter(t=>!t[4]).map(t=>[t[0],t[1],t[3]])},
    quality:quality(a,b,first),finalQuality:quality(a,b,final)};
}
function correlation(m) {const[n,pos,sx,sxx,sxy]=m,den=(n*sxx-sx*sx)*(n*pos-pos*pos);return den>0?(n*sxy-sx*pos)/Math.sqrt(den):null;}
export function summary(data,init,stage) {
  const candidate=[0,0,0,0,0],accepted=[0,0,0,0,0],bins=new Map(),counts=[];let tp=0,n=0,available=0;
  const add=(m,x,y)=>{m[0]++;m[1]+=+y;m[2]+=x;m[3]+=x*x;m[4]+=y?x:0;};
  for(const[i,j]of init.scene.pairs) {
    const pair=pairData(data.profiles[i],data.profiles[j],init.meta,data.sources),q=stage==='final'?pair.finalQuality:pair.quality;
    tp+=q.tp;n+=q.tp+q.fp;available+=q.available;counts.push([q.tp,q.tp+q.fp,q.available]);
    pair.matrix.values.forEach((x,k)=>{if(x==null||!Number.isFinite(x))return;const y=pair.a.source_ids[Math.floor(k/pair.b.ranges.length)]===pair.b.source_ids[k%pair.b.ranges.length];
      add(candidate,x,y);const bin=Math.floor(x/.25),v=bins.get(bin)||[0,0,0];v[0]++;v[1]+=+y;v[2]+=x;bins.set(bin,v);
    });
    for(const[a,b]of(stage==='final'?pair.final.matches:pair.matrix.matches)) {const x=pair.matrix.values[a*pair.b.ranges.length+b];if(x!=null)add(accepted,x,pair.a.source_ids[a]===pair.b.source_ids[b]);}
  }
  return {summary:{tp,fp:n-tp,fn:available-tp,precision:n?tp/n:0,recall:available?tp/available:0,f1:n+available?2*tp/(n+available):0,pairs:counts.length,candidate_cells:candidate[0],candidate_same_source:candidate[1],candidate_correlation:correlation(candidate),accepted_correlation:correlation(accepted)},
    pair_counts:counts,bins:[...bins].sort((a,b)=>a[0]-b[0]).map(([bid,v])=>[bid*.25,v[2]/v[0],v[0],v[1]])};
}
export function spatial(data,init) {
  const pairs=[],distance_m=[],neighborhood=[],first_counts=[],final_counts=[],local=new Set(init.scene.pairs.map(x=>x.join(',')));
  for(let i=0;i<data.profiles.length;i++)for(let j=i+1;j<data.profiles.length;j++) {
    const a=data.profiles[i],b=data.profiles[j],distance=Math.hypot(...a.xyz.map((v,k)=>v-b.xyz[k]));
    const r=match(a.ranges,b.ranges,init.meta,distance),edges=r.trace.filter(t=>t[4]),q=quality(a,b,edges),f=quality(a,b,edges.filter(t=>t[3]===1));
    pairs.push([i,j]);distance_m.push(distance);neighborhood.push(local.has(i+','+j));
    first_counts.push([f.tp,f.tp+f.fp,f.available,a.ranges.length*b.ranges.length]);final_counts.push([q.tp,q.tp+q.fp,q.available,a.ranges.length*b.ranges.length]);
  }
  return {pairs,distance_m,neighborhood,count_columns:['correct','selected','true','candidate_cells'],first_counts,final_counts};
}
