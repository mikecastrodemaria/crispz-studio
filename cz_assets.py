"""crispz-studio - UI/static assets (pure strings, no logic).

Extrait de app.py pour alleger le fichier principal:
  - ASSET_BROWSER_HTML : la SPA de l'Asset Browser (deposee dans le dossier de sortie).
  - CZ_JS              : JS injecte au chargement (theme sombre + preview de style au survol).
  - FOOOCUS_CSS        : CSS de l'interface (facon Fooocus).
"""

ASSET_BROWSER_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>crispz-studio - Asset Browser</title>
<style>
:root{--bg:#0b1018;--panel:#1a2233;--line:#2a3346;--fg:#e6ebf2;--mut:#8b98ad}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font-family:system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:#0b1018ee;backdrop-filter:blur(6px);
padding:10px 14px;display:flex;gap:12px;align-items:center;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:600}
input,button,select{background:#141b29;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:7px 10px;font-size:13px}
button{cursor:pointer}#count{color:var(--mut);font-size:12px}
#grid{flex:1;min-width:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;padding:12px}
#wrap{display:flex;align-items:flex-start}
#folders{width:190px;flex:0 0 190px;padding:8px;border-right:1px solid var(--line);
position:sticky;top:53px;max-height:calc(100vh - 53px);overflow:auto}
#folders .f{display:flex;justify-content:space-between;align-items:center;gap:6px;padding:6px 8px;
border-radius:6px;cursor:pointer;font-size:13px}
#folders .f:hover{background:#1c2740}#folders .f.active{background:#3b4356}
#folders .f.hidden-f{opacity:.55}
#folders .f .cnt{color:var(--mut);font-size:11px;margin-right:4px}
#folders .f .hb{display:none;background:#5a2230;border:1px solid #7a2e40;color:#fff;border-radius:4px;
font-size:10px;padding:1px 6px;cursor:pointer}
#folders .f:hover .hb{display:inline-block}
#hiddenbtn.on{background:#3b4356;border-color:#5d6884}
.src.active{background:#7a5cff2b;border-color:#7a5cff;color:#fff}
.cell.ph{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;
padding:8px;background:#141b29;animation:none}
.cell.ph .ic{font-size:30px;opacity:.5;margin-bottom:6px}
.cell.ph .n{font-size:11px;color:#cfd8e6;word-break:break-word;line-height:1.25}
@keyframes cz-shim{0%{background-position:200% 0}100%{background-position:-200% 0}}
.cell{position:relative;aspect-ratio:1;border-radius:8px;overflow:hidden;cursor:zoom-in;border:1px solid var(--line);
background:#11182a linear-gradient(100deg,#11182a 30%,#1c2740 50%,#11182a 70%);background-size:200% 100%;animation:cz-shim 1.3s linear infinite}
.cell.loaded{animation:none;background:#11182a}
.cell img{width:100%;height:100%;object-fit:cover;display:block;opacity:0;transition:opacity .35s,transform .15s}
.cell img.loaded{opacity:1}
.cell:hover img{transform:scale(1.04)}
.cell .cap{position:absolute;left:0;right:0;bottom:0;font-size:10px;padding:3px 5px;color:#cfd8e6;
background:linear-gradient(transparent,#000b);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
body.blur .cell img{filter:blur(14px)}body.blur .cell:hover img{filter:none}
#lb{position:fixed;inset:0;background:#000d;z-index:10;display:none;grid-template-columns:1fr 340px}
#lb.open{display:grid}
#lbimg{display:flex;align-items:center;justify-content:center;padding:16px;min-width:0}
#lbimg img{max-width:100%;max-height:96vh;object-fit:contain}
#side{background:var(--panel);border-left:1px solid var(--line);padding:16px;overflow:auto;font-size:13px}
#side h3{margin:.2em 0 .4em;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
#side .v{margin:0 0 10px;word-break:break-word;white-space:pre-wrap}
#side button{margin:4px 6px 4px 0}
.ex{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:6px}
.ex img{width:100%;border-radius:4px;display:block;background:#11182a}
.nav{position:fixed;top:50%;transform:translateY(-50%);font-size:40px;color:#fff;cursor:pointer;
user-select:none;padding:0 14px;opacity:.7}.nav:hover{opacity:1}#prev{left:0}#next{right:344px}
#close{position:fixed;top:10px;right:352px;font-size:30px;color:#fff;cursor:pointer;z-index:11}
.ex img{cursor:zoom-in;transition:transform .12s}.ex img:hover{transform:scale(1.05)}
/* Statut du fetch CivitAI (barre + phase) dans le panneau lateral */
.cvstatus{margin:8px 0;padding:10px;border:1px solid var(--line);border-radius:8px;background:#141b29}
.cvstatus .lbl{display:flex;align-items:center;gap:8px;font-size:12px;color:#cfd8e6}
.cvstatus.err .lbl{color:#ff9db0}.cvstatus.ok .lbl{color:#8fe0a5}
.spin{width:14px;height:14px;border:2px solid #3a6ea5;border-top-color:transparent;border-radius:50%;
display:inline-block;animation:cvspin .7s linear infinite;flex:0 0 auto}
@keyframes cvspin{to{transform:rotate(360deg)}}
.cvbar{height:6px;margin-top:8px;border-radius:4px;background:#0e1626;overflow:hidden}
.cvbar>i{display:block;height:100%;background:linear-gradient(90deg,#3a6ea5,#6aa6e0);width:0;
border-radius:4px;transition:width .25s}
.cvbar.indet>i{width:35%;animation:cvindet 1.1s ease-in-out infinite}
@keyframes cvindet{0%{margin-left:-35%}100%{margin-left:100%}}
button:disabled{opacity:.6;cursor:default}
/* Badge "nouvelle version dispo" sur les cartes modeles + toast batch */
.cell .upd{position:absolute;top:6px;left:6px;z-index:2;font-size:10px;font-weight:600;
padding:2px 6px;border-radius:6px;background:#7a4a12ee;color:#ffd27a;border:1px solid #b6803a;
pointer-events:none}
.cell.ph .upd{position:absolute}
#batchstatus{position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:30;
display:none;max-width:80vw;padding:10px 16px;border-radius:10px;
background:#141b29;border:1px solid var(--line);color:#cfd8e6;font-size:13px;
box-shadow:0 6px 24px #000a}
#batchstatus.ok{color:#8fe0a5;border-color:#2f6b45}#batchstatus.err{color:#ff9db0;border-color:#7a2e40}
#batchstatus .spin{margin-right:8px;vertical-align:-2px}
/* Visionneuse d'exemples (grand format + prompt), au-dessus de la lightbox */
#exlb{position:fixed;inset:0;background:#000e;z-index:20;display:none;
grid-template-columns:1fr 360px}#exlb.open{display:grid}
#eximg{display:flex;align-items:center;justify-content:center;padding:16px;min-width:0}
#eximg img{max-width:100%;max-height:96vh;object-fit:contain;border-radius:6px}
#exside{background:var(--panel);border-left:1px solid var(--line);padding:16px;overflow:auto;font-size:13px}
#exside h3{margin:.2em 0 .4em;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
#exside .v{margin:0 0 10px;word-break:break-word;white-space:pre-wrap;line-height:1.4}
#exclose{position:fixed;top:10px;right:372px;font-size:30px;color:#fff;cursor:pointer;z-index:21}
#exlb .nav{z-index:21}#exprev{left:0}#exnext{right:364px}
#excount{color:var(--mut);font-size:12px;margin-bottom:8px}
</style></head><body>
<header><h1>🖼️ crispz-studio</h1>
<input id="q" placeholder="Search metadata (prompt, style, model, seed, sampler...)" style="flex:1;min-width:160px">
<button class="src active" data-s="outputs">Outputs</button>
<button class="src" data-s="loras">LoRAs</button>
<button class="src" data-s="models">Models</button>
<button id="hiddenbtn" title="Show hidden folders">Hidden</button>
<button id="blurbtn">Blur</button>
<button id="fetchall" title="Fetch missing CivitAI info for every model in this tab" style="display:none">🔄 Fetch all missing</button>
<span id="count"></span></header>
<div id="wrap"><aside id="folders"></aside><div id="grid"></div></div>
<div id="lb"><span id="close">&times;</span><span class="nav" id="prev">&#10094;</span>
<span class="nav" id="next">&#10095;</span><div id="lbimg"><img id="big"></div>
<div id="side"></div></div>
<div id="exlb"><span id="exclose">&times;</span><span class="nav" id="exprev">&#10094;</span>
<span class="nav" id="exnext">&#10095;</span><div id="eximg"><img id="exbig"></div>
<div id="exside"></div></div>
<script>
let DATA=[],VIEW=[],cur=0,EX=[],excur=0;
var CZ_BATCH=("__CZ_BATCH__"==="1");   // bouton "Fetch all missing" (injecte par le serveur)
const grid=document.getElementById('grid'),lb=document.getElementById('lb'),big=document.getElementById('big'),
side=document.getElementById('side'),q=document.getElementById('q'),cnt=document.getElementById('count'),
folders=document.getElementById('folders'),
exlb=document.getElementById('exlb'),exbig=document.getElementById('exbig'),exside=document.getElementById('exside');
function esc(s){return (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function render(){grid.innerHTML='';VIEW.forEach((e,i)=>{const c=document.createElement('div');
const label=e.name||e.file;const hasImg=!!(e.thumb||e.img)||curSource==='outputs';
if(!hasImg){c.className='cell ph loaded';
c.innerHTML='<div class="ic">'+(e.mode==='lora'?'🧩':'📦')+'</div><div class="n">'+esc(label)+'</div>';
if(e.update)c.innerHTML+='<div class="upd" title="Newer version on CivitAI'+(e.latest?': '+esc(e.latest):'')+'">⚠ update</div>';
c.onclick=()=>open(i);grid.appendChild(c);return;}
c.className='cell';
const im=document.createElement('img');im.loading='lazy';
const thumb=encodeURI(e.thumb||e.img||e.file),full=encodeURI(e.img||e.file);let tries=12;
im.onload=function(){im.classList.add('loaded');c.classList.add('loaded');};
im.onerror=function(){if(thumb!==full&&tries>0){tries--;
setTimeout(function(){im.src=thumb+(thumb.indexOf('?')<0?'?r=':'&r=')+Date.now();},2500);}
else{im.onerror=null;im.src=full;}};
im.src=thumb;
const cap=document.createElement('div');cap.className='cap';cap.textContent=label;
c.appendChild(im);c.appendChild(cap);
if(e.update){var ub=document.createElement('div');ub.className='upd';ub.textContent='⚠ update';
ub.title='Newer version on CivitAI'+(e.latest?': '+e.latest:'');c.appendChild(ub);}
c.onclick=()=>open(i);grid.appendChild(c);});
cnt.textContent=VIEW.length+' / '+DATA.length;}
function hay(e){return (e.file+' '+(e.prompt||'')+' '+(e.negative||'')+' '+(e.mode||'')+' '+
(e.seed||'')+' '+(e.steps||'')+' '+(e.guidance||'')+' '+(e.size||'')+' '+(e.model||'')+' '+
((e.loras||[]).join(' '))+' '+((e.styles||[]).join(' '))+' '+(e.sampler||'')+' '+(e.day||'')).toLowerCase();}
function filter(){const s=q.value.toLowerCase().trim();
VIEW=DATA.filter(function(e){const d=e.day||'(root)';
if(!showHidden&&hidden.has(d))return false;
if(curFolder&&d!==curFolder)return false;
if(s&&!hay(e).includes(s))return false;return true;});render();}
function open(i){cur=i;const e=VIEW[i];const isOut=curSource==='outputs';
const imgUrl=e.img||(isOut?e.file:'');
if(imgUrl){big.style.display='';big.src=encodeURI(imgUrl);}else{big.style.display='none';}
let h='';
if(e.name&&e.name!==e.file)h+='<h3>Name</h3><div class="v">'+esc(e.name)+'</div>';
h+='<h3>'+(isOut?'Prompt':'Trigger words')+'</h3><div class="v">'+esc(e.prompt||'(none)')+'</div>';
if(e.negative)h+='<h3>Negative</h3><div class="v">'+esc(e.negative)+'</div>';
h+='<h3>Info</h3><div class="v">';
['mode','seed','steps','guidance','size','model','sampler','day','date'].forEach(k=>{if(e[k]!=null&&e[k]!=='')h+=k+': '+esc(e[k])+'\n';});
if(e.styles&&e.styles.length)h+='styles: '+esc(e.styles.join(', '))+'\n';
if(e.loras&&e.loras.length)h+='loras: '+esc(e.loras.join(', '))+'\n';
h+='file: '+esc(e.file)+'</div>';
h+='<button onclick="cp(\''+'prompt'+'\')">Copy '+(isOut?'prompt':'triggers')+'</button>';
h+='<button onclick="cp(\''+'all'+'\')">Copy all</button>';
if(isOut){h+='<a href="'+encodeURI(e.file)+'" download="'+esc(e.file.split('/').pop())+'" style="margin-left:6px;color:#9fb3d6">Download</a>';
h+='<button onclick="delAsset()" style="margin-left:6px;background:#5a2230;border-color:#7a2e40">Delete</button>';}
else{if(e.update)h+='<div class="v" style="color:#ffd27a;margin-bottom:8px">⚠️ Newer version on CivitAI'+(e.latest?': '+esc(e.latest):'')+'</div>';
h+='<button id="cvbtn" onclick="civitaiFetch()" style="margin-left:6px;background:#274b6d;border-color:#3a6ea5">🔎 Fetch from CivitAI</button>';
if(e.civitai)h+='<a href="'+encodeURI(e.civitai)+'" target="_blank" style="margin-left:6px;color:#9fb3d6">CivitAI page</a>';
EX=(e.examples||[]).map(function(x){return (typeof x==='string')?{url:x,prompt:''}:x;}).filter(function(x){return x&&x.url;});
if(EX.length){h+='<h3>Examples <span style="color:var(--mut);font-weight:400">(click to enlarge)</span></h3><div class="ex">'+
EX.map(function(x,ix){return '<img loading="lazy" src="'+encodeURI(x.url)+'" onclick="exOpen('+ix+')" title="'+esc((x.prompt||'').slice(0,140))+'">';}).join('')+'</div>';}}
side.innerHTML=h;lb.classList.add('open');}
function cp(what){const e=VIEW[cur];let t=e.prompt||'';if(what==='all')t=JSON.stringify(e,null,2);
navigator.clipboard.writeText(t).catch(()=>{});}
async function delAsset(){const e=VIEW[cur];if(!e||!confirm('Delete '+e.file+' ?'))return;
try{const r=await fetch('/gradio_api/call/delete_asset',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({data:[e.file]})});const j=await r.json();const eid=j.event_id||j.hash;
if(eid){await fetch('/gradio_api/call/delete_asset/'+eid);}
DATA=DATA.filter(x=>x.file!==e.file);close();filter();}catch(err){alert('Delete failed: '+err);}}
// Appel generique d'un endpoint Gradio (POST event_id -> GET stream -> 1re sortie)
async function gcall(name,data){
const r=await fetch('/gradio_api/call/'+name,{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({data:data})});const j=await r.json();const eid=j.event_id||j.hash;if(!eid)return null;
const t=await (await fetch('/gradio_api/call/'+name+'/'+eid)).text();
const m=t.match(/data:\s*(\[[\s\S]*?\])/);if(m){try{return JSON.parse(m[1])[0];}catch(_){}}return null;}
function _sleep(ms){return new Promise(function(r){setTimeout(r,ms);});}
function cvBar(frac){var pct=(frac==null?'':(Math.max(0,Math.min(1,frac))*100).toFixed(0)+'%');
return '<div class="cvbar'+(frac==null?' indet':'')+'"><i style="width:'+pct+'"></i></div>';}
function cvStatus(text,frac,cls,spin){var s=document.getElementById('cvstatus');
if(!s){s=document.createElement('div');s.id='cvstatus';side.appendChild(s);}
s.className='cvstatus'+(cls?' '+cls:'');
s.innerHTML='<div class="lbl">'+(spin?'<span class="spin"></span>':'')+'<span>'+esc(text)+'</span></div>'+
(cls?'':cvBar(frac));return s;}
async function civitaiFetch(){const e=VIEW[cur];if(!e)return;
const btn=document.getElementById('cvbtn');if(btn)btn.disabled=true;
cvStatus('Starting…',null,'',true);
try{const key=await gcall('civitai_fetch',[e.file, e.mode==='lora'?'loras':'models']);
if(!key||(''+key).indexOf('error:')===0)throw new Error(key||'no job started');
var st=null;
for(var i=0;i<1200;i++){var raw=await gcall('civitai_progress',[key]);
st=(typeof raw==='string')?(function(){try{return JSON.parse(raw);}catch(_){return null;}})():raw;
if(st){cvStatus(st.text||st.phase||'Working…',(st.frac==null?null:st.frac),'',true);
if(st.done)break;}
await _sleep(400);}
var ok=!!(st&&st.ok),msg=(st&&(st.message||st.text))||'done';
cvStatus((ok?'✅ ':'⚠️ ')+msg,null,ok?'ok':'err',false);
if(ok)setTimeout(function(){close();loadSource(curSource);},1000);
}catch(err){cvStatus('⚠️ CivitAI fetch failed: '+err,null,'err',false);}
finally{if(btn)btn.disabled=false;}}
// --- Batch: recuperer toutes les infos CivitAI manquantes de l'onglet courant ---
function batchStatus(html,cls){var s=document.getElementById('batchstatus');
if(!s){s=document.createElement('div');s.id='batchstatus';document.body.appendChild(s);}
s.className=(cls||'');s.innerHTML=html||'';s.style.display=html?'block':'none';return s;}
async function fetchAll(){var src=curSource;if(src!=='loras'&&src!=='models')return;
var btn=document.getElementById('fetchall');if(btn)btn.disabled=true;
batchStatus('<span class="spin"></span>Starting batch…');
try{var key=await gcall('civitai_fetch_all',[src]);
if(!key||(''+key).indexOf('error:')===0)throw new Error(key||'batch not started');
var st=null;
for(var k=0;k<200000;k++){var raw=await gcall('civitai_progress',[key]);
st=(typeof raw==='string')?(function(){try{return JSON.parse(raw);}catch(_){return null;}})():raw;
if(st){var c=(st.n?(' '+(st.i||0)+'/'+st.n):'');
batchStatus('<span class="spin"></span>Batch'+c+' — '+esc(st.text||'working…'));
if(st.done)break;}
await _sleep(500);}
var sm=(st&&st.summary)||{};
batchStatus('✅ Batch done: '+(sm.enriched||0)+' enriched · '+(sm.updated||0)+' with newer version · '+
(sm.skipped||0)+' skipped · '+(sm.failed||0)+' failed',(st&&st.ok)?'ok':'err');
loadSource(src);setTimeout(function(){batchStatus('');},7000);
}catch(err){batchStatus('⚠️ Batch failed: '+esc(''+err),'err');}
finally{if(btn)btn.disabled=false;}}
// --- Visionneuse d'exemples (grand format + prompt + navigation) ---
function exRender(){var x=EX[excur];if(!x)return;exbig.src=encodeURI(x.url);
var h='<div id="excount">'+(excur+1)+' / '+EX.length+'</div>';
h+='<h3>Prompt</h3><div class="v">'+(x.prompt?esc(x.prompt):
'<span style="color:var(--mut)">(the uploader did not publish the generation parameters for this image)</span>')+'</div>';
if(x.width&&x.height)h+='<h3>Size</h3><div class="v">'+esc(x.width)+' × '+esc(x.height)+'</div>';
h+='<button onclick="exCopy()">Copy prompt</button>';
h+='<a href="'+encodeURI(x.url)+'" target="_blank" style="margin-left:6px;color:#9fb3d6">Open image</a>';
exside.innerHTML=h;}
function exOpen(ix){if(!EX.length)return;excur=(ix+EX.length)%EX.length;exRender();exlb.classList.add('open');}
function exClose(){exlb.classList.remove('open');}
function exNav(d){if(!EX.length)return;excur=(excur+d+EX.length)%EX.length;exRender();}
function exCopy(){var x=EX[excur];if(x)navigator.clipboard.writeText(x.prompt||'').catch(function(){});}
document.getElementById('exclose').onclick=exClose;
document.getElementById('exprev').onclick=function(){exNav(-1);};
document.getElementById('exnext').onclick=function(){exNav(1);};
exlb.onclick=function(ev){if(ev.target===exlb||ev.target===exbig.parentNode)exClose();};
function close(){lb.classList.remove('open');}
document.getElementById('close').onclick=close;
document.getElementById('prev').onclick=()=>open((cur-1+VIEW.length)%VIEW.length);
document.getElementById('next').onclick=()=>open((cur+1)%VIEW.length);
lb.onclick=ev=>{if(ev.target===lb||ev.target===big.parentNode)close();};
document.addEventListener('keydown',ev=>{
if(exlb.classList.contains('open')){if(ev.key==='Escape')exClose();
if(ev.key==='ArrowLeft')exNav(-1);if(ev.key==='ArrowRight')exNav(1);return;}
if(!lb.classList.contains('open'))return;
if(ev.key==='Escape')close();if(ev.key==='ArrowLeft')document.getElementById('prev').click();
if(ev.key==='ArrowRight')document.getElementById('next').click();});
q.oninput=filter;
document.getElementById('blurbtn').onclick=()=>document.body.classList.toggle('blur');
document.getElementById('fetchall').onclick=fetchAll;
function _today(){const d=new Date(),m=String(d.getMonth()+1).padStart(2,'0'),da=String(d.getDate()).padStart(2,'0');return d.getFullYear()+'-'+m+'-'+da;}
// --- Sous-dossiers (sidebar) + hide, persistant en localStorage ---
let curFolder='',showHidden=false,_folderUserSet=false,curSource='outputs',hidden=new Set();
try{hidden=new Set(JSON.parse(localStorage.getItem('cz_ab_hidden')||'[]'));}catch(e){}
function saveHidden(){try{localStorage.setItem('cz_ab_hidden',JSON.stringify([...hidden]));}catch(e){}}
function renderFolders(){const c={};DATA.forEach(function(e){const d=e.day||'(root)';c[d]=(c[d]||0)+1;});
const names=Object.keys(c).sort().reverse();
if(!_folderUserSet&&!curFolder&&names.indexOf(_today())>=0)curFolder=_today();
let h='<div class="f'+(curFolder===''?' active':'')+'" data-f=""><span>All</span><span class="cnt">'+DATA.length+'</span></div>';
names.forEach(function(d){const isH=hidden.has(d);if(isH&&!showHidden)return;
h+='<div class="f'+(curFolder===d?' active':'')+(isH?' hidden-f':'')+'" data-f="'+esc(d)+'">'+
'<span>'+esc(d)+'</span><span><span class="cnt">'+c[d]+'</span>'+
'<button class="hb" data-h="'+esc(d)+'">'+(isH?'show':'hide')+'</button></span></div>';});
folders.innerHTML=h;}
folders.onclick=function(ev){const hb=ev.target.closest('.hb');
if(hb){const d=hb.getAttribute('data-h');if(hidden.has(d))hidden.delete(d);else hidden.add(d);saveHidden();
if(curFolder===d&&hidden.has(d)&&!showHidden)curFolder='';renderFolders();filter();return;}
const f=ev.target.closest('.f');if(!f)return;curFolder=f.getAttribute('data-f')||'';_folderUserSet=true;
renderFolders();filter();};
document.getElementById('hiddenbtn').onclick=function(){showHidden=!showHidden;
this.classList.toggle('on',showHidden);renderFolders();filter();};
var _gen='',_focus='';var _srcUrls={outputs:'_index/manifest.json',loras:'_index/loras.json',models:'_index/models.json'};
function _tryFocus(){if(!_focus)return;curFolder='';_folderUserSet=true;renderFolders();filter();
var ix=-1;for(var k=0;k<VIEW.length;k++){if(VIEW[k].file===_focus||VIEW[k].name===_focus){ix=k;break;}}
if(ix>=0){open(ix);_focus='';}}
function _apply(m){DATA=m.images||[];if(m.blur)document.body.classList.add('blur');renderFolders();filter();
if(!DATA.length&&m&&m.building){grid.innerHTML='<p style="padding:20px;color:#8b98ad">Indexing your output folder… '+
'(first run — can take ~30 s for large folders; it will fill in automatically)</p>';}}
function loadSource(src){curSource=src;_folderUserSet=false;curFolder='';
[].slice.call(document.querySelectorAll('.src')).forEach(function(b){b.classList.toggle('active',b.getAttribute('data-s')===src);});
var _fa=document.getElementById('fetchall');
if(_fa)_fa.style.display=(CZ_BATCH&&(src==='loras'||src==='models'))?'':'none';
fetch(_srcUrls[src]+'?t='+Date.now()).then(function(r){return r.ok?r.json():null;}).then(function(m){
if(m){if(src==='outputs')_gen=m.generated||'';_apply(m);_tryFocus();}
else{DATA=[];renderFolders();grid.innerHTML='<p style="padding:20px;color:#8b98ad">No '+src+' catalog yet (building in background). Reopen the Asset Browser in a few seconds.</p>';cnt.textContent='0 / 0';}});}
[].slice.call(document.querySelectorAll('.src')).forEach(function(b){b.onclick=function(){loadSource(b.getAttribute('data-s'));};});
function _poll(n){if(n<=0)return;setTimeout(function(){
fetch('_index/manifest.json?t='+Date.now()).then(r=>r.ok?r.json():null).then(m=>{
if(m&&m.generated&&m.generated!==_gen&&curSource==='outputs'){_gen=m.generated;_apply(m);}_poll(n-1);}).catch(()=>_poll(n-1));},2000);}
function _load(tries){fetch('_index/manifest.json?t='+Date.now()).then(r=>{if(!r.ok)throw 0;return r.json();})
.then(m=>{_gen=m.generated||'';_apply(m);_poll(90);})
.catch(e=>{if(tries>0){grid.innerHTML='<p style="padding:20px;color:#8b98ad">Indexing…</p>';setTimeout(()=>_load(tries-1),1200);}
else grid.innerHTML='<p style="padding:20px;color:#8b98ad">No manifest. Click Reindex in crispz-studio.</p>';});}
(function(){try{var u=new URL(location.href);var s=u.searchParams.get('src');
_focus=u.searchParams.get('focus')||'';
if(s&&_srcUrls[s]){loadSource(s);return;}}catch(e){}
_load(25);})();
</script></body></html>
"""


CZ_JS = """
() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme', 'dark'); window.location.replace(u.toString()); return;
  }
  const SAMPLES = __MAP__;

  // --- Preview de style au survol ---
  let tip = null;
  const ensureTip = () => {
    if (!tip) { tip = document.createElement('div'); tip.className = 'cz-style-preview';
      tip.style.display = 'none'; tip.innerHTML = '<img>'; document.body.appendChild(tip); }
    return tip;
  };
  document.addEventListener('mouseover', (e) => {
    const lbl = e.target.closest && e.target.closest('#cz_styles label');
    if (!lbl) return;
    const name = (lbl.innerText || '').trim();
    const url = SAMPLES[name];
    if (!url) return;
    const t = ensureTip(); const im = t.querySelector('img');
    im.onerror = () => { t.style.display = 'none'; };
    im.src = url; t.style.display = 'block';
  });
  document.addEventListener('mousemove', (e) => {
    if (tip && tip.style.display === 'block') {
      let x = e.clientX + 18, y = e.clientY + 18;
      if (x + 240 > window.innerWidth) x = e.clientX - 240;
      if (y + 240 > window.innerHeight) y = e.clientY - 240;
      tip.style.left = x + 'px'; tip.style.top = y + 'px';
    }
  });
  document.addEventListener('mouseout', (e) => {
    const lbl = e.target.closest && e.target.closest('#cz_styles label');
    if (lbl && tip) tip.style.display = 'none';
  });

  // Le plein ecran + les fleches sont gerees nativement par la galerie Gradio
  // (preview / fullscreen). Pas de lightbox custom (evite le doublon au clic).
}
"""


FOOOCUS_CSS = """
.gradio-container { max-width: 100% !important; width: 100% !important; padding: 0 1rem !important; }
.dark, :root {
  --body-background-fill: #0b1018;
  --background-fill-primary: #0b1018;
  --background-fill-secondary: #11182400;
  --block-background-fill: #1a2233;
  --block-border-color: #2a3346;
  --border-color-primary: #2a3346;
  --input-background-fill: #141b29;
}
/* Rendu homothetique: image entierement visible (contain), centree, jamais plus
   grande que la zone -> pas de scroll, pas de cover. La galerie se dimensionne a
   l'image (plafond 78vh), donc plus de bande vide ni d'image coupee. */
#cz_result { min-height: 60vh !important; }
#cz_result .grid-wrap, #cz_result .grid-container { min-height: 58vh !important; max-height: 82vh !important; }
#cz_result .empty, #cz_result .image-container { min-height: 56vh !important; }
#cz_result img {
  object-fit: contain !important;
  max-width: 100% !important;
  max-height: 78vh !important;
  width: auto !important;
  height: auto !important;
  margin-left: auto !important;
  margin-right: auto !important;
  cursor: zoom-in;
}
#cz_result .thumbnail-item, #cz_result .thumbnail-item img, #cz_result button img {
  object-fit: contain !important; }
#cz_prompt textarea, #cz_neg textarea { font-size: 1.04rem; }
#cz_generate, #cz_edit_generate { font-size: 1.12rem; font-weight: 600;
  background: linear-gradient(180deg,#5a6376,#3b4356) !important; color: #fff !important;
  border: 1px solid #5d6884 !important; box-shadow: none !important; }
#cz_generate { min-height: 96px !important; height: 100% !important; }
#cz_edit_generate { min-height: 48px !important; }
#cz_generate:hover, #cz_edit_generate:hover { background: linear-gradient(180deg,#69738a,#454e63) !important; }
/* Spinner anime pendant la generation (classe .generating posee/retiree en JS) */
@keyframes cz-spin { to { transform: rotate(360deg); } }
#cz_generate.generating, #cz_edit_generate.generating { opacity: .85; }
#cz_generate.generating::after, #cz_edit_generate.generating::after { content: ""; display: inline-block;
  width: 15px; height: 15px; margin-left: 10px; vertical-align: middle;
  border: 2px solid rgba(255,255,255,.35); border-top-color: #fff; border-radius: 50%;
  animation: cz-spin .7s linear infinite; }
/* Bloc styles: scroller interne */
#cz_styles { max-height: 340px; overflow-y: auto; padding-right: 6px; }
/* Preview de style au survol */
.cz-style-preview { position: fixed; z-index: 10000; pointer-events: none;
  border: 1px solid #2a3346; border-radius: 8px; overflow: hidden;
  box-shadow: 0 6px 24px rgba(0,0,0,.6); background: #0b1018; }
.cz-style-preview img { display: block; width: 110px; height: auto; }
/* Galerie avancee: flou NSFW optionnel */
#cz_gallery.cz-blur img { filter: blur(18px); transition: filter .15s; }
#cz_gallery.cz-blur img:hover { filter: none; }
/* Lightbox plein ecran */
.cz-lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.93); z-index: 10001;
  display: flex; align-items: center; justify-content: center; cursor: zoom-out; }
.cz-lightbox img { max-width: 95vw; max-height: 95vh; object-fit: contain; }
.cz-lightbox .cz-close { position: fixed; top: 14px; right: 26px; color: #fff;
  font-size: 44px; line-height: 1; cursor: pointer; font-weight: 300; }
"""


# JS autonome du tag-autocomplete (injecte via gr.Blocks(head=...) UNIQUEMENT si la
# feature est activee -> zero JS/fetch quand off). Placeholders remplaces au build:
# __SRC__ (URLs des CSV), __LOCAL__ (assets locaux, ex. __wildcards__), __MAX__.
# Index: tri global par popularite une fois, dedoublonnage entre sources, buckets par
# prefixe de 2 caracteres, sortie anticipee a MAX resultats.
TAG_AC_JS = r"""
(() => {
  const SRC = __SRC__, LOCAL = __LOCAL__, MAXR = __MAX__;
  const SEL = '#cz_prompt textarea, #cz_neg textarea';
  let E = [], BUCKET = new Map(), READY = false;
  let box = null, mirror = null, items = [], sel = -1, curTA = null, tok = null;
  let nq = 0, tq = 0;

  function parseText(text) {
    const out = [];
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith('#')) continue;
      if (line.indexOf(',') < 0) { out.push({t: line, n: 0, a: []}); continue; }
      const m = line.match(/^([^,]+),([^,]*),?([^,"]*),?"?(.*?)"?\s*$/);
      if (!m || !m[1].trim()) continue;
      const a = (m[4] || '').split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
      out.push({t: m[1].trim(), n: parseInt(m[3], 10) || 0, a: a});
    }
    return out;
  }

  function buildIndex(lists) {
    const t0 = performance.now();
    const best = new Map();
    for (const w of LOCAL) best.set(w.toLowerCase(), {t: w, n: 1e12, a: []});
    for (const list of lists) for (const e of list) {
      const k = e.t.toLowerCase(), p = best.get(k);
      if (!p) best.set(k, e);
      else { if (e.n > p.n) p.n = e.n; if (e.a.length && !p.a.length) p.a = e.a; }
    }
    E = [...best.values()].sort((x, y) => y.n - x.n);
    BUCKET = new Map();
    E.forEach((e, i) => {
      const keys = new Set([e.t.toLowerCase().slice(0, 2)]);
      for (const al of e.a) keys.add(al.slice(0, 2));
      for (const k of keys) { let b = BUCKET.get(k); if (!b) BUCKET.set(k, b = []); b.push(i); }
    });
    READY = true;
    console.log('[tagac] ready in ' + (performance.now() - t0).toFixed(0) + ' ms - '
                + E.length + ' entries, ' + SRC.length + ' source file(s)');
  }

  function query(q) {
    const t0 = performance.now();
    q = q.toLowerCase();
    const out = [], b = BUCKET.get(q.slice(0, 2)) || [];
    for (const i of b) {
      const e = E[i];
      if (e.t.toLowerCase().startsWith(q)) out.push({e: e, via: null});
      else { const al = e.a.find(x => x.startsWith(q)); if (al) out.push({e: e, via: al}); }
      if (out.length >= MAXR) break;
    }
    tq += performance.now() - t0; nq += 1;
    if (nq % 50 === 0)
      console.debug('[tagac] avg query ' + (tq / nq).toFixed(3) + ' ms over ' + nq + ' keystrokes');
    return out;
  }

  function tokenAt(v, caret) {
    let s = caret;
    while (s > 0 && v[s - 1] !== ',' && v[s - 1] !== '\n') s -= 1;
    while (s < caret && (v[s] === ' ' || v[s] === '\t')) s += 1;
    return {start: s, text: v.slice(s, caret)};
  }

  function ensureUI() {
    if (box) return;
    const st = document.createElement('style');
    st.textContent = '.czac{position:fixed;z-index:10001;background:#141b2e;border:1px solid #3c4864;'
      + 'border-radius:8px;font-size:13px;color:#dfe6f2;box-shadow:0 8px 24px rgba(0,0,0,.5);'
      + 'max-width:420px;overflow:hidden}'
      + '.czac div{padding:5px 10px;cursor:pointer;display:flex;gap:10px;justify-content:space-between}'
      + '.czac div.on{background:#2b3a5c}';
    document.head.appendChild(st);
    box = document.createElement('div');
    box.className = 'czac';
    box.style.display = 'none';
    document.body.appendChild(box);
    mirror = document.createElement('div');
    mirror.style.cssText = 'position:absolute;visibility:hidden;left:-9999px;top:0;'
      + 'white-space:pre-wrap;word-wrap:break-word;overflow:hidden;';
    document.body.appendChild(mirror);
  }

  function caretXY(ta) {
    ensureUI();
    const cs = getComputedStyle(ta);
    const props = ['fontFamily', 'fontSize', 'fontWeight', 'lineHeight', 'letterSpacing',
                   'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
                   'borderTopWidth', 'borderLeftWidth', 'boxSizing'];
    for (const p of props) mirror.style[p] = cs[p];
    mirror.style.width = ta.clientWidth + 'px';
    mirror.textContent = ta.value.slice(0, ta.selectionStart);
    const mk = document.createElement('span');
    mk.textContent = '​';
    mirror.appendChild(mk);
    const r = ta.getBoundingClientRect();
    let lh = parseFloat(cs.lineHeight);
    if (!lh || isNaN(lh)) lh = parseFloat(cs.fontSize) * 1.3;
    return {x: Math.min(r.left + mk.offsetLeft - ta.scrollLeft, window.innerWidth - 430),
            y: r.top + mk.offsetTop - ta.scrollTop + lh};
  }

  function fmt(n) {
    if (n >= 1e12) return '';
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(0) + 'k';
    return n > 0 ? String(n) : '';
  }

  function render(res, xy) {
    ensureUI();
    box.textContent = '';
    items = res; sel = 0;
    res.forEach((r, i) => {
      const d = document.createElement('div');
      const l = document.createElement('span');
      l.textContent = r.via ? r.e.t + '  (' + r.via + ')' : r.e.t;
      const c = document.createElement('span');
      c.textContent = fmt(r.e.n);
      c.style.opacity = '.55';
      d.appendChild(l); d.appendChild(c);
      if (i === 0) d.classList.add('on');
      d.addEventListener('mousedown', ev => { ev.preventDefault(); sel = i; pick(); });
      box.appendChild(d);
    });
    box.style.left = xy.x + 'px';
    box.style.top = xy.y + 'px';
    box.style.display = 'block';
  }

  function move(d) {
    if (!items.length) return;
    sel = (sel + d + items.length) % items.length;
    Array.prototype.forEach.call(box.children, (c, i) => c.classList.toggle('on', i === sel));
  }

  function close() { if (box) box.style.display = 'none'; items = []; sel = -1; }

  function pick() {
    if (sel < 0 || !items[sel] || !curTA) return;
    const e = items[sel].e, ta = curTA, v = ta.value, caret = ta.selectionStart;
    const text = e.t.indexOf('__') === 0 ? e.t : e.t.replace(/_/g, ' ');
    const before = v.slice(0, tok.start), after = v.slice(caret);
    const sep = after.trim() === '' ? ', ' : (/^\s*,/.test(after) ? '' : ', ');
    ta.value = before + text + sep + after;
    const pos = (before + text + sep).length;
    ta.setSelectionRange(pos, pos);
    ta.dispatchEvent(new Event('input', {bubbles: true}));
    close();
  }

  function onInput(ta) {
    if (!READY) return;
    curTA = ta;
    tok = tokenAt(ta.value, ta.selectionStart);
    if (tok.text.trim().length < 2) { close(); return; }
    const res = query(tok.text.trim());
    if (!res.length) { close(); return; }
    render(res, caretXY(ta));
  }

  document.addEventListener('input', ev => {
    const ta = ev.target;
    if (ta && ta.matches && ta.matches(SEL)) onInput(ta);
  }, true);
  document.addEventListener('keydown', ev => {
    if (!box || box.style.display === 'none') return;
    const ta = ev.target;
    if (!ta || !ta.matches || !ta.matches(SEL)) return;
    if (ev.key === 'ArrowDown') move(1);
    else if (ev.key === 'ArrowUp') move(-1);
    else if (ev.key === 'Tab' || ev.key === 'Enter') pick();
    else if (ev.key === 'Escape') close();
    else return;
    ev.preventDefault(); ev.stopPropagation();
  }, true);
  document.addEventListener('click', ev => { if (box && !box.contains(ev.target)) close(); }, true);
  window.addEventListener('resize', close);

  Promise.all(SRC.map(u => fetch(u).then(r => r.ok ? r.text() : '').catch(() => '')))
    .then(txts => buildIndex(txts.map(parseText)))
    .catch(e => console.warn('[tagac] init failed:', e));
})();
"""
