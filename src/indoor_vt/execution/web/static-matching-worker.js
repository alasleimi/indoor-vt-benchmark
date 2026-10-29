import {pairData,summary,spatial} from './static-matcher.mjs';
import {distributions} from './static-distributions.mjs';
const cache=new Map();
self.onmessage=({data:message})=>{
  const {id,action,data,init,args}=message;
  try {
    const key=JSON.stringify([action,init.meta.trial,args.ap,args.noise,args.matches,args.left,args.right]);
    if(!cache.has(key)) {
      let value;
      if(action==='pair')value=pairData(data.profiles[Number(args.left)],data.profiles[Number(args.right)],init.meta,data.sources);
      else if(action==='condition')value=summary(data,init,args.matches||'first');
      else if(action==='spatial')value=spatial(data,init);
      else if(action==='distributions')value=distributions(data,init);
      else throw Error('Unknown matching operation');
      cache.set(key,value);while(cache.size>16)cache.delete(cache.keys().next().value);
    }
    self.postMessage({id,value:cache.get(key)});
  }catch(error){self.postMessage({id,error:error.message});}
};
