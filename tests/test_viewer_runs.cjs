const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');

const html=fs.readFileSync(path.join(__dirname,'../src/jevsceneminer/viewer/static/index.html'),'utf8');
const main=html.match(/^const mainRun = [\s\S]*?(?=^const pretty)/m)[0];
function functionSource(name,async=false) {
  const expression=new RegExp('^'+(async?'async ':'')+'function '+name+'\\([^]*?^}\\n','m');
  return html.match(expression)[0];
}
function viewer(runs,documents) {
  const elements={};
  const context={S:{runs:[],data:{runs:documents,tags:[],indicator:[],meta:{}},gtScenes:[],probs:[]},
    LAT:['keep_lane'],LON:[],location:{search:'',hash:''},URLSearchParams,
    Option:function(text,value){this.text=text;this.value=value;},
    fetch:async()=>({json:async()=>({runs,sessions:[{name:'sample',date:'2025-12-29',sid:'sample'}]})}),
    $:id=>elements[id]||(elements[id]={add(){},checked:false,value:'sample'}),
    load:async()=>{},scenesFrom:doc=>doc?.scenes||[],cameraMock:null,
    CameraDecisions:{mock(){throw Error('Unexpected mock input');}}};
  vm.createContext(context);
  vm.runInContext(main+functionSource('init',true)+functionSource('rowSpec')+functionSource('cameraResultsSource'),context);
  return {context,elements,evaluate:code=>vm.runInContext(code,context)};
}
const document={scenes:[{start:0,end:4,lateral:'turn_right',phases:[]}]};

test('custom run supplied by the API drives bars, primary scenes and camera labels',async()=>{
  const v=viewer(['candidate'],{candidate:document});
  await v.evaluate('init()');
  assert.deepEqual(Array.from(v.context.S.runs),['candidate']);
  assert.equal(v.evaluate('mainRun()[0]'),'candidate');
  assert.deepEqual(Array.from(v.evaluate('rowSpec().map(r=>r.id)')),['overview','candidate:lat','gt:lat','ruler']);
  assert.equal(v.evaluate('cameraResultsSource().scenes[0].lateral'),'turn_right');
});

test('explicit comparison runs keep their lateral and detailed speed rows',async()=>{
  const v=viewer(['jev','baseline'],{jev:document,baseline:document});
  await v.evaluate('init()');
  let rows=Array.from(v.evaluate('rowSpec().map(r=>r.id)'));
  assert.ok(rows.includes('jev:lat')&&rows.includes('baseline:lat'));
  v.elements.showDetails.checked=true;
  rows=Array.from(v.evaluate('rowSpec().map(r=>r.id)'));
  assert.ok(rows.includes('jev:lon')&&rows.includes('baseline:lon')&&rows.includes('gt:lon'));
});

test('a missing first run does not hide an available later document',async()=>{
  const v=viewer(['missing','candidate'],{missing:null,candidate:document});
  await v.evaluate('init()');
  assert.equal(v.evaluate('mainRun()[0]'),'candidate');
  assert.equal(v.evaluate('cameraResultsSource().scenes.length'),1);
});
