/*! 띄어쓰기 검사 — QuickSpacer(MIT, https://github.com/cosmoquester/quickspacer) ConvSpacer3 모델을
    브라우저(순수 JS)에서 실행하고, 몇 가지 기본 규칙을 더했어요. 서버 없이 이 기기에서만 계산해요. */
(function(root){
'use strict';
const HANGUL=/[\uAC00-\uD7A3]/, HANGUL_OR_JAMO=/[\uAC00-\uD7A3\u3131-\u318E]/;
const OPEN=/["'“‘(\[「『]/, CLOSE=/["'”’)\]」』]/;
// 따로 떨어져 쓰면 거의 틀리는 조사(붙여 써야 해요)
const PARTICLES=new Set(['에','에서','에게','에게서','한테','께서','까지','부터','으로','으로는','를','을','이랑','처럼','마다','조차','에는','에도','에서는','에서도','이라고','이나']);
let M=null;
function dq(buf){ // 모델 파일 읽기
  const dv=new DataView(buf);
  if(String.fromCharCode(dv.getUint8(0),dv.getUint8(1),dv.getUint8(2),dv.getUint8(3))!=='QSP1')throw new Error('bad model');
  const hl=dv.getUint32(4,true);
  const hdr=JSON.parse(new TextDecoder().decode(new Uint8Array(buf,8,hl)));
  const base=8+hl+((4-(hl%4))%4);
  const T={};
  for(const t of hdr.tensors){
    if(t.f!==undefined){T[t.name]=new Float32Array(buf.slice(base+t.f,base+t.f+t.fn*4));continue;}
    const q=new Int8Array(buf,base+t.q,t.qn);
    const s=new Float32Array(buf.slice(base+t.s,base+t.s+t.sn*4));
    const out=new Float32Array(t.qn);
    if(t.name==='emb'){const d=t.shape[1];for(let j=0;j<t.qn;j++)out[j]=q[j]*s[(j/d)|0];}
    else{const o=t.shape[2];for(let j=0;j<t.qn;j++)out[j]=q[j]*s[j%o];}
    T[t.name]=out;
  }
  const vmap=new Map();hdr.vocab.forEach((c,i)=>{if(!vmap.has(c))vmap.set(c,i);});
  const L=(n,k,i,o)=>({k,i,o,w:T[n+'.w'],b:T[n+'.b']});
  return {emb:T.emb,E:128,vmap,c1:L('c1',5,128,256),c2:L('c2',7,128,256),c3:L('c3',11,128,256),c4:L('c4',5,896,256),d:L('d',3,256,1)};
}
function conv(x,len,C,relu){
  const {k,i,o,w,b}=C,pad=(k-1)>>1,y=new Float32Array(len*o);
  for(let t=0;t<len;t++){
    const yo=t*o;
    for(let oo=0;oo<o;oo++)y[yo+oo]=b[oo];
    for(let dk=0;dk<k;dk++){
      const s=t+dk-pad;if(s<0||s>=len)continue;
      for(let ii=0;ii<i;ii++){
        const xv=x[s*i+ii];if(xv===0)continue;
        const wb=(dk*i+ii)*o;
        for(let oo=0;oo<o;oo++)y[yo+oo]+=xv*w[wb+oo];
      }
    }
    if(relu)for(let oo=0;oo<o;oo++)if(y[yo+oo]<0)y[yo+oo]=0;
  }
  return y;
}
// 각 글자 뒤에 띄어 써야 할 확률
function probs(chars){
  const len=chars.length;if(!len)return [];
  const E=M.E,emb=new Float32Array(len*E);
  chars.forEach((c,t)=>{const id=M.vmap.has(c)?M.vmap.get(c):1;emb.set(M.emb.subarray(id*E,id*E+E),t*E);});
  const o1=conv(emb,len,M.c1,true),o2=conv(emb,len,M.c2,true),o3=conv(emb,len,M.c3,true);
  const cat=new Float32Array(len*896);
  for(let t=0;t<len;t++){
    cat.set(emb.subarray(t*E,t*E+E),t*896);
    cat.set(o1.subarray(t*256,t*256+256),t*896+128);
    cat.set(o2.subarray(t*256,t*256+256),t*896+384);
    cat.set(o3.subarray(t*256,t*256+256),t*896+640);
  }
  const o5=conv(conv(cat,len,M.c4,true),len,M.d,false);
  return Array.from(o5,v=>1/(1+Math.exp(-v)));
}
function lineChunks(line){ // 너무 긴 줄은 문장 끝에서 잘라 계산해요
  const out=[];let cur=[];
  for(const ch of Array.from(line)){cur.push(ch);if(cur.length>=120&&/[.?!]/.test(ch)){out.push(cur);cur=[];}}
  if(cur.length)out.push(cur);return out;
}
const isSp=c=>c===' '||c==='\t'||c==='\u3000';
function effPrev(T,a){ // 닫는 따옴표/괄호를 건너뛴 앞 글자
  let j=a;while(j>0&&CLOSE.test(T[j]))j--;return T[j];
}
/** text → 제안 목록. 각 제안: {type:'add'|'join'|'dup', from,to(코드포인인트 구간), repl, before, after, why} */
function check(text,opt){
  opt=opt||{};const Tadd=opt.add??0.5,Tjoin=opt.join??0.1,useModel=!!M&&opt.model!==false;
  const T=Array.from(text.replace(/\r/g,''));
  const sugg=[],seen=new Set();
  const push=(s)=>{const k=s.type+':'+s.from+':'+s.to;if(seen.has(k))return;seen.add(k);sugg.push(s);};
  // 줄 단위로 처리
  let lineStart=0;
  while(lineStart<=T.length){
    let lineEnd=lineStart;while(lineEnd<T.length&&T[lineEnd]!=='\n')lineEnd++;
    processLine(lineStart,lineEnd);
    lineStart=lineEnd+1;
  }
  function processLine(ls,le){
    // 비어 있지 않은 글자들의 위치
    const idx=[];for(let i=ls;i<le;i++)if(!isSp(T[i]))idx.push(i);
    // 규칙 1: 연속 띄어쓰기 (낱말 사이에 두 칸 이상)
    for(let i=ls;i<le;){
      if(isSp(T[i])&&i>ls&&!isSp(T[i-1])){
        let j=i;while(j<le&&isSp(T[j]))j++;
        if(j-i>=2&&j<le)push({type:'dup',from:i,to:j,repl:' ',why:'띄어쓰기를 두 번 했어요'});
        i=j;
      }else i++;
    }
    // 규칙 2: 따로 떨어진 조사
    {
      let i=ls;
      while(i<le){
        while(i<le&&isSp(T[i]))i++;
        let s=i;while(i<le&&!isSp(T[i]))i++;
        const w=T.slice(s,i).join('');
        const m=w.match(/^([\uAC00-\uD7A3]+)([.,?!…]*)$/);
        if(m&&PARTICLES.has(m[1])&&s>ls){
          // 앞 낱말(한글로 끝남)과 붙이기
          let a=s-1;while(a>=ls&&isSp(T[a]))a--;
          if(a>=ls&&HANGUL.test(T[a])&&a<s-0){
            push({type:'join',from:a+1,to:s,repl:'',why:'조사는 앞말에 붙여 써요'});
          }
        }
      }
    }
    if(!useModel||!idx.length)return;
    // 모델: (a) 공백 유지한 채 → 빠진 띄어쓰기, (b) 공백 제거 → 불필요한 띄어쓰기
    const origSpaceAfter=new Set(); // idx 순번 기준
    for(let n=0;n<idx.length-1;n++)if(idx[n+1]-idx[n]>1)origSpaceAfter.add(n);
    const chars=T.slice(ls,le);
    // (a)
    const pk=new Array(chars.length).fill(0);
    {let pos=0;for(const ck of lineChunks(chars.join(''))){const p=probs(ck);for(let q=0;q<ck.length;q++)pk[pos+q]=p[q];pos+=ck.length;}}
    // (b)
    const nons=idx.map(i=>T[i]);const ps=new Array(nons.length).fill(0);
    {let pos=0;for(const ck of lineChunks(nons.join(''))){const p=probs(ck);for(let q=0;q<ck.length;q++)ps[pos+q]=p[q];pos+=ck.length;}}
    for(let n=0;n<idx.length-1;n++){
      const a=idx[n],b=idx[n+1],ca=T[a],cb=T[b];
      if(origSpaceAfter.has(n)){
        if(ps[n]<Tjoin&&HANGUL.test(ca)&&HANGUL.test(cb)&&b-a===2&&effPrev(T,a)!=='.')
          push({type:'join',from:a+1,to:b,repl:'',why:'붙여 쓰는 게 자연스러워요'});
      }else{
        const p=pk[a-ls];
        if(p>=Tadd&&(HANGUL.test(ca)||/[,?!]/.test(ca)||CLOSE.test(ca))&&(HANGUL.test(cb)||OPEN.test(cb))
           &&effPrev(T,a)!=='.')
          push({type:'add',from:b,to:b,repl:' ',why:'띄어 써야 해요'});
      }
    }
  }
  // 보여 줄 글 만들기 (제안 주변 몇 글자만)
  for(const s of sugg){
    let a=s.from;while(a>0&&s.from-a<6&&!isSp(T[a-1])&&T[a-1]!=='\n')a--;
    let b=s.to;while(b<T.length&&b-s.to<6&&!isSp(T[b])&&T[b]!=='\n')b++;
    if(s.type==='add'){
      a=Math.max(a,s.from-4);b=Math.min(b,s.to+4);
    }
    s.before=T.slice(a,s.from).join('')+T.slice(s.from,s.to).join('')+T.slice(s.to,b).join('');
    s.after=T.slice(a,s.from).join('')+s.repl+T.slice(s.to,b).join('');
    s.ctxStart=a;s.ctxEnd=b;
  }
  sugg.sort((x,y)=>x.from-y.from);
  return sugg;
}
/** 제안 적용 (from 큰 것부터) → 새 텍스트 */
function applySuggestions(text,list){
  const T=Array.from(text.replace(/\r/g,''));
  for(const s of list.slice().sort((a,b)=>b.from-a.from)){
    T.splice(s.from,s.to-s.from,...Array.from(s.repl));
  }
  return T.join('');
}
root.Spacer={
  load:async function(url){if(M)return;const r=await fetch(url);if(!r.ok)throw new Error('model '+r.status);M=dq(await r.arrayBuffer());},
  loadBuffer:function(buf){M=dq(buf);},
  ready:()=>!!M,
  check:check,apply:applySuggestions,probs:(s)=>probs(Array.from(s))
};
})(typeof window!=='undefined'?window:globalThis);
