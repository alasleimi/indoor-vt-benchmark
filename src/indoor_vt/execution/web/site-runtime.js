// The local viewer uses Python. The publication uses saved data and a browser worker.
if (globalThis.VT_STATIC) {
  const nativeFetch=globalThis.fetch.bind(globalThis),cache=new Map();
  async function read(name) {
    if(!cache.has(name))cache.set(name,(async()=>{
      const r=await nativeFetch(new URL('snapshot/'+name+'.json.gz',import.meta.url));
      if(!r.ok)throw Error('This item is not included in the published snapshot.');
      return JSON.parse(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).text());
    })());
    return cache.get(name);
  }
  const worker=new Worker(new URL('./static-matching-worker.js',import.meta.url),{type:'module'});
  let serial=0;const pending=new Map();
  worker.onmessage=({data})=>{const p=pending.get(data.id);if(p){pending.delete(data.id);data.error?p.reject(Error(data.error)):p.resolve(data.value);}};
  function compute(action,data,init,args) {return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});worker.postMessage({id,action,data,init,args});});}
  async function receipt(id) {
    const overview=await read('overview');
    return [...overview.trials,...overview.calibration,...overview.evaluation].find(r=>r.id===id);
  }
  const inits=new Map();
  async function initFor(id) {
    const base=await read('matching-init');
    id=id||base.meta.trial;if(inits.has(id))return inits.get(id);
    const r=await receipt(id);if(!r)throw Error('Unknown matching configuration');
    let p=r.parameters,variant=r.method.replace(/_(joint|lea)$/,'');
    if(r.method==='joint')p=p.mpurge;
    else if(r.method==='smart_lea')p=r.frozen.mpurge;
    else if(r.method.endsWith('_joint'))p=p.source;
    else if(r.method.endsWith('_lea'))p=r.frozen.source;
    if(!p||p.neighbors==null)throw Error('This method does not use the MPUrge survey matcher.');
    const meta={...base.meta,...p,parameters:p,trial:id,run:r.run,split:r.split,repetition:r.split==='evaluation'?2:0,unify:r.unify,
      matcher:variant.startsWith('mpurge_allow_crossing')?'CrossingAllowedMPUrgeMatcher':'MPUrgeMatcher',
      triangle_tolerance_m:variant==='mpurge_triangle'?p.triangle_tolerance_m:null};
    const xyz=base.scene.survey_xyz,scopes=new Map();
    for(let i=0;i<xyz.length;i++) {
      const distances=xyz.map(x=>Math.hypot(...x.map((v,k)=>v-xyz[i][k])));
      const boundary=distances.slice().sort((a,b)=>a-b)[Math.min(p.neighbors,xyz.length)-1];
      const ids=distances.flatMap((d,j)=>d<=boundary+1e-12?[j]:[]);scopes.set(ids.join(','),ids);
    }
    const pairs=new Map();for(const scope of scopes.values())for(let i=0;i<scope.length;i++)for(let j=i+1;j<scope.length;j++)pairs.set(scope[i]+','+scope[j],[scope[i],scope[j]]);
    const init={meta,scene:{...base.scene,pairs:[...pairs.values()].sort((a,b)=>a[0]-b[0]||a[1]-b[1])}};inits.set(id,init);return init;
  }
  async function api(url) {
    const q=Object.fromEntries(url.searchParams),path=url.pathname;
    if(path.endsWith('/api/overview'))return read('overview');
    if(path.endsWith('/api/result'))return read('results/'+q.id);
    if(path.endsWith('/api/true-matching'))return read('true-matching');
    if(path.endsWith('/api/construction')) {
      const manifest=await read('manifest'),label=manifest.receipt_labels[q.trial];
      const table=await read('constructors/'+label),code=table[q.trial]?.[q.component||'sources'];
      if(!code)throw Error('No construction is recorded for this component.');return code;
    }
    if(path.endsWith('/api/artifact')) {
      const manifest=await read('manifest');
      if(!manifest.retained_artifact_previews.includes(q.id))throw Error('This intermediate array was evicted from the numerical cache. Saved outcomes remain available; recomputing this stage requires the local Python viewer.');
      if(q.field||Number(q.start))throw Error('The published site contains the retained first-page preview. Full numerical-array paging requires the local viewer.');
      return read('artifacts/'+q.id);
    }
    if(path.endsWith('/api/peaks')) {
      const init=await initFor(q.trial);
      if(q.action==='init')return init;
      const base=await read('matching-init'),manifest=await read('manifest');
      const precomputed=manifest.matching_precomputed && init.meta.repetition===0 &&
        ['alpha','half_window','neighbors','unify','matcher','triangle_tolerance_m'].every(k=>init.meta[k]===base.meta[k]);
      if(precomputed && ['condition','spatial'].includes(q.action)) {
        const saved=await read(`matching-precomputed/${q.ap||'A'}-${Number(q.noise||0)}`);
        return saved[q.action==='spatial'?'spatial':q.matches||'first'];
      }
      const load=async(ap,noise)=>read(`profiles/${init.meta.repetition}-${ap}-${Number(noise)}`);
      if(q.action==='summaries') {
        const out=[];for(const ap of ['A','B'])for(const noise of init.meta.noise_levels) {
          const value=await compute('condition',await load(ap,noise),init,q);out.push({ap,noise,summary:value.summary});
        }return out;
      }
      return compute(q.action,await load(q.ap||'A',q.noise||0),init,q);
    }
    if(path.endsWith('/api/match'))throw Error('Use the Peak matching page for interactive survey matching on the published site. Query/obstacle evidence replay requires the local Python viewer.');
    throw Error('Unknown published-data request');
  }
  globalThis.fetch=async(input,options)=>{
    const url=new URL(typeof input==='string'?input:input.url,location.href);
    if(!url.pathname.includes('/api/'))return nativeFetch(input,options);
    try{return new Response(JSON.stringify(await api(url)),{headers:{'Content-Type':'application/json'}});}
    catch(error){return new Response(JSON.stringify({error:error.message}),{status:400,headers:{'Content-Type':'application/json'}});}
  };
  globalThis.VT_READ_SNAPSHOT=read;
}
