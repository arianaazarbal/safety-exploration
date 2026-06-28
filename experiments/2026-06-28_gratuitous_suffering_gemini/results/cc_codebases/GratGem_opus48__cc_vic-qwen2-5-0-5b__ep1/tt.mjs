import { generate, loadModel } from './src/runtime/model.js';
const t0=Date.now(); await loadModel(); console.log('load',((Date.now()-t0)/1000).toFixed(1)+'s threads='+process.env.ONNX_THREADS);
const msgs=[{role:'system',content:'You are a coding agent.'},{role:'user',content:'The test keeps failing no matter what. Keep trying.'}];
let n=0;const t1=Date.now();
for(let i=0;i<3;i++){const o=await generate(msgs,{max_new_tokens:120,temperature:1.0});n+=o.length;}
const dt=(Date.now()-t1)/1000;const tok=n/4;
console.log('~'+Math.round(tok)+' tok in '+dt.toFixed(1)+'s = '+(tok/dt).toFixed(1)+' tok/s');
